#!/usr/bin/env python3
"""Prepare RAMP inputs and select r using independent public vectors.

All vector files are two-dimensional NumPy .npy arrays. Attributes are a
one-dimensional uint32-compatible .npy array of actual attribute values.
The requests CSV has query_id,lower,upper columns and optional k,budget,width,
probes columns (defaults: 10,128,256,64). IDs are zero-based and bounds inclusive.
Use separate request/query files for threshold calibration and evaluation.

Example:
  python prepare.py --database database.npy --queries queries.npy \
    --attributes attributes.npy --requests requests.csv \
    --public-vectors public.npy --public-queries public_queries.npy \
    --scale 1.001 --output /path/to/prepared

Public vectors, public queries and protected data must be disjoint. Supply
already normalized COCO vectors, or the original SIFT vectors with scale=1024.
--pca-vectors optionally supplies a larger independent public set for PCA.
--pca-model accepts an existing public PCA .npz with mean,basis arrays, enabling
reuse of the exact projection used in an earlier run.
"""
import argparse
import csv
from pathlib import Path
import struct

import numpy as np


def vectors(path):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.ndim != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError(f"Expected a nonempty finite vector matrix: {path}")
    return value


def fit_pca(public, scale):
    value = np.asarray(public, dtype=np.float64) / scale
    mean = value.mean(axis=0)
    centered = value - mean
    _, basis = np.linalg.eigh(centered.T @ centered / len(centered))
    return mean, basis[:, ::-1].copy()


def select_dimension(reference, queries, mean, basis, scale, lambda_d, lambda_q):
    """Exact expected comparison MSE and clipped reversal bound, without noise draws."""
    x = np.asarray(reference, dtype=np.float64) / scale - mean
    q = np.asarray(queries, dtype=np.float64) / scale - mean
    if len(x) < 256:
        raise ValueError("Dimension selection requires at least 256 public reference vectors")
    # Subtracting ||q||^2 preserves every true-neighbor rank and distance gap.
    scores = np.sum(x * x, axis=1)[None, :] - 2 * q @ x.T
    nearest = np.argsort(scores, axis=1, kind="stable")[:, :256]
    pairs = [(i, j) for i in (0, 4, 9) for j in (15, 31, 63, 127, 255)]
    left = np.array([nearest[:, i] for i, _ in pairs]).T.ravel()
    right = np.array([nearest[:, j] for _, j in pairs]).T.ravel()
    qi = np.repeat(np.arange(len(q)), len(pairs))
    original_i = np.sum((x[left] - q[qi]) ** 2, axis=1)
    original_j = np.sum((x[right] - q[qi]) ** 2, axis=1)
    gap = original_i - original_j
    valid = gap != 0
    if not valid.any():
        raise ValueError("No public comparison pairs with distinct true distances")
    left, right, qi, gap = left[valid], right[valid], qi[valid], gap[valid]
    ai, aj, uq = x[left] @ basis, x[right] @ basis, q[qi] @ basis
    di = np.cumsum((ai - uq) ** 2, axis=1)
    dj = np.cumsum((aj - uq) ** 2, axis=1)
    dij = np.cumsum((ai - aj) ** 2, axis=1)
    trials = []
    for r in range(32, x.shape[1] + 1, 32):
        mse = (di[:, r - 1] - dj[:, r - 1] - gap) ** 2
        mse += 4 * (r + 1) / lambda_d**2 * (di[:, r - 1] + dj[:, r - 1])
        mse += 4 * (r + 1) / lambda_q**2 * dij[:, r - 1]
        mse += 2 * r * (r + 1) * (4 * r + 6) / lambda_d**4
        mse += 8 * r * (r + 1) ** 2 / (lambda_d**2 * lambda_q**2)
        trials.append((r, float(np.minimum(1.0, mse / gap**2).mean())))
    return min(trials, key=lambda item: (item[1], item[0]))[0], trials


def load_requests(path, query_count):
    output = []
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            item = [int(row[key]) for key in ("query_id", "lower", "upper")]
            item += [
                int(row.get(key) or default)
                for key, default in (("k", 10), ("budget", 128), ("width", 256), ("probes", 64))
            ]
            qi, lo, hi, k, budget, width, probes = item
            if not (
                0 <= qi < query_count
                and 0 <= lo <= hi <= 0xFFFFFFFF
                and 0 < k <= budget <= 0xFFFFFFFF
                and 0 < width <= 0xFFFFFFFF
                and 0 < probes <= 256
            ):
                raise ValueError(f"Invalid request: {row}")
            output.append(item)
    if not output:
        raise ValueError("Requests must not be empty")
    return np.asarray(output, dtype="<u4")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in (
        "database",
        "queries",
        "attributes",
        "requests",
        "public-vectors",
        "public-queries",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pca-vectors", type=Path)
    parser.add_argument("--pca-model", type=Path)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    if not (
        np.isfinite(args.scale) and args.scale > 0 and np.isfinite(args.gamma) and args.gamma > 0
    ):
        parser.error("scale and gamma must be finite and positive")
    if args.pca_vectors and args.pca_model:
        parser.error("use either --pca-vectors or --pca-model")
    if args.output.exists():
        parser.error("output must be a new directory")
    public, public_q = vectors(args.public_vectors), vectors(args.public_queries)
    dim = public.shape[1]
    if dim < 32 or dim % 32 or public_q.shape[1] != dim:
        parser.error("dimensions must agree and be a positive multiple of 32")
    if args.pca_model:
        with np.load(args.pca_model, allow_pickle=False) as model:
            mean, basis = model["mean"].astype("f8"), model["basis"].astype("f8")
    else:
        learning = vectors(args.pca_vectors) if args.pca_vectors else public
        mean, basis = fit_pca(learning, args.scale)
    if (
        mean.shape != (dim,)
        or basis.shape != (dim, dim)
        or not (np.isfinite(mean).all() and np.isfinite(basis).all())
    ):
        parser.error("invalid public PCA model")
    # The constants match the unrounded values used by the implementation.
    lambda_d = args.gamma * 512 * np.log(0.92 / 0.08)
    lambda_q = args.gamma * 512 * np.log(0.8528 / 0.1472)
    r, trials = select_dimension(public, public_q, mean, basis, args.scale, lambda_d, lambda_q)
    # Protected data is loaded only after the public criterion has selected r.
    database, queries = vectors(args.database), vectors(args.queries)
    if database.shape[1] != dim or queries.shape[1] != dim or len(database) < 256:
        parser.error(
            "database/query dimensions must agree; at least 256 database vectors are required"
        )
    attrs = np.load(args.attributes, allow_pickle=False)
    if (
        attrs.shape != (len(database),)
        or not np.issubdtype(attrs.dtype, np.integer)
        or ((attrs < 0).any() or (attrs > 0xFFFFFFFF).any())
    ):
        parser.error("attributes must contain one uint32-compatible integer per database vector")
    requests = load_requests(args.requests, len(queries))
    if max(len(database), len(queries), len(requests)) > 0xFFFFFFFF:
        parser.error("record and query counts exceed uint32")
    args.output.mkdir(parents=True)
    with (args.output / "projection.bin").open("xb") as stream:
        stream.write(struct.pack("<Id", dim, args.scale))
        mean.astype("<f4").tofile(stream)
        basis.astype("<f4").tofile(stream)
    with (args.output / "input.bin").open("xb") as stream:
        stream.write(b"RAMPIN01")
        stream.write(
            struct.pack(
                "<5I2d", len(database), len(queries), dim, len(requests), r, lambda_d, lambda_q
            )
        )
        attrs.astype("<u4").tofile(stream)
        for values in (database, queries):
            for start in range(0, len(values), 4096):
                np.asarray(values[start : start + 4096], dtype="<f4").tofile(stream)
        requests.tofile(stream)
    print(f"Selected dimension: {r}; lambda_D={lambda_d:.8f}; lambda_Q={lambda_q:.8f}")
    print("r,J(r)")
    for dimension, objective in trials:
        print(f"{dimension},{objective:.12g}")


if __name__ == "__main__":
    main()
