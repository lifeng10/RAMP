#!/usr/bin/env python3
"""Convert SIFT, SIFT1M, COCO, or FANNS100K into RAMP input files.

Sources:
  sift / sift1m: directory containing sift_base/query/learn.fvecs
  coco: coco-i2i-512-angular.hdf5 (requires h5py)
  fanns100k: directory containing database_vectors.fvecs,
            query_vectors.fvecs, and database_attributes.jsonl
            --public-source points to arxiv-for-fanns-large/database_vectors.fvecs

All outputs use the same filenames accepted by prepare.py. Use scale=1024 for
SIFT/SIFT1M and scale=1.001 for normalized COCO/FANNS100K vectors. Random ranks
are generated for SIFT and COCO; FANNS100K retains its update_date values.
Date ties can make actual matching counts differ from the requested fractions.

--matching accepts one or more fractions. --budget, --width, and --probes accept
one value for all fractions or one value per fraction, in the same order.
Defaults provide a runnable workload; they do not tune search parameters.

Calibration queries are held out from evaluation queries. To select thresholds,
prepare a separate input using calibration_queries.npy and calibration_requests.csv,
run ramp --calibrate, and pass its queries.csv to select_thresholds.py. Use the
same search settings and public data when preparing the evaluation input.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def fvecs(path, dimension):
    row_bytes = (dimension + 1) * 4
    if not path.stat().st_size or path.stat().st_size % row_bytes:
        raise ValueError(f"Incomplete {dimension}-dimensional fvecs file: {path}")
    return np.memmap(path, dtype="<i4", mode="r").reshape(-1, dimension + 1)


def read_fvecs(path, count, dimension=128):
    rows = fvecs(path, dimension)
    if len(rows) < count or not np.all(rows[:count, 0] == dimension):
        raise ValueError(f"Expected {count} vectors of dimension {dimension}: {path}")
    return rows[:count, 1:].view("<f4").copy()


def normalize(values):
    values = np.asarray(values, dtype=np.float32).copy()
    for start in range(0, len(values), 4096):
        block = values[start : start + 4096]
        norms = np.linalg.norm(block.astype(np.float64), axis=1)
        if not np.isfinite(block).all() or not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError("Expected finite, nonzero vectors")
        block[:] = block / norms[:, None]
    return values


def load_sift(args):
    database = read_fvecs(args.source / "sift_base.fvecs", args.records)
    queries = read_fvecs(args.source / "sift_query.fvecs", args.query_count)
    learning_count = max(50_000, args.public_count)
    rows = fvecs(args.source / "sift_learn.fvecs", 128)
    if not np.all(rows[:, 0] == 128):
        raise ValueError("Inconsistent dimension in SIFT learning vectors")
    learning = independent_public(
        rows[:, 1:].view("<f4"), (database, queries), learning_count + args.public_query_count
    )
    return database, queries, learning[:learning_count], learning[learning_count:], None


def load_coco(args):
    try:
        import h5py
    except ImportError as error:
        raise ValueError("COCO conversion requires h5py") from error
    with h5py.File(args.source, "r") as source:
        if "train" not in source or "test" not in source:
            raise ValueError("COCO HDF5 must contain train and test matrices")
        train, test = source["train"], source["test"]
        if (
            train.ndim != 2
            or test.ndim != 2
            or train.shape[1] != 512
            or test.shape[1] != 512
            or len(train) < args.records + args.public_count + args.public_query_count
            or len(test) < args.query_count
        ):
            raise ValueError("COCO does not contain enough 512-dimensional vectors")
        database = normalize(train[: args.records])
        queries = normalize(test[: args.query_count])
        learning = independent_public(normalize(train[args.records :]), (database, queries))
        if len(learning) < args.public_count + args.public_query_count:
            raise ValueError("Not enough distinct public COCO vectors")
    # The train tail is public and never used as evaluation database vectors.
    public_queries = learning[args.public_count : args.public_count + args.public_query_count]
    return database, queries, learning, public_queries, None


def vector_digest(row):
    return hashlib.sha256(np.asarray(row, dtype="<f4").tobytes()).digest()


def independent_public(candidates, protected, count=None):
    """Exclude evaluation vectors and duplicates, preserving source order."""
    seen = {vector_digest(row) for matrix in protected for row in matrix}
    public = []
    for vector in candidates:
        digest = vector_digest(vector)
        if digest not in seen:
            public.append(vector.copy())
            seen.add(digest)
            if len(public) == count:
                break
    if count is not None and len(public) < count:
        raise ValueError(f"Need {count} public vectors distinct from evaluation data")
    return np.asarray(public, dtype=np.float32)


def load_fanns(args):
    if args.public_source is None:
        raise ValueError("FANNS100K requires --public-source with independent public fvecs")
    database = read_fvecs(args.source / "database_vectors.fvecs", args.records, 4096)
    queries = read_fvecs(args.source / "query_vectors.fvecs", args.query_count, 4096)
    attributes = []
    with (args.source / "database_attributes.jsonl").open() as stream:
        for line in stream:
            value = json.loads(line)["update_date"]
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError("update_date must be a uint32-compatible integer")
            attributes.append(value)
            if len(attributes) == args.records:
                break
    if len(attributes) != args.records:
        raise ValueError("Missing update_date values for database vectors")

    # The medium set can overlap with the large set. Exclude evaluation vectors
    # and duplicate public vectors before selecting the independent public split.
    public_rows = fvecs(args.public_source, 4096)

    def candidates():
        for row in public_rows:
            if row[0] != 4096:
                raise ValueError("Inconsistent dimension in public fvecs")
            yield row[1:].view("<f4")

    public = normalize(
        independent_public(
            candidates(), (database, queries), args.public_count + args.public_query_count
        )
    )
    return (
        normalize(database),
        normalize(queries),
        public[: args.public_count],
        public[args.public_count :],
        np.asarray(attributes, dtype=np.uint32),
    )


def per_range(values, count, name):
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} needs one value or one value per matching fraction")
    return values


def write_requests(path, attributes, queries, args, rng):
    ordered = np.sort(attributes)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("query_id", "lower", "upper", "k", "budget", "width", "probes"))
        for fraction, budget, width, probes in zip(
            args.matching, args.budget, args.width, args.probes
        ):
            matching = max(1, int(len(attributes) * fraction))
            for query_id in range(queries):
                start = int(rng.integers(len(attributes) - matching + 1))
                lower, upper = int(ordered[start]), int(ordered[start + matching - 1])
                writer.writerow((query_id, lower, upper, args.k, budget, width, probes))


def parse_args(argv, dataset=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    if dataset is None:
        parser.add_argument("dataset", choices=("sift", "sift1m", "coco", "fanns100k"))
    else:
        parser.set_defaults(dataset=dataset)
    parser.add_argument("source", type=Path, help="Downloaded dataset file or directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--public-source", type=Path, help="FANNS public database_vectors.fvecs")
    parser.add_argument("--records", type=int, help="Database size (default: paper dataset size)")
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--calibration-queries", type=int, default=200)
    parser.add_argument("--public-count", type=int, default=10_000)
    parser.add_argument("--public-query-count", type=int, default=160)
    parser.add_argument("--matching", type=float, nargs="+", default=[0.1])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--budget", type=int, nargs="+", default=[64])
    parser.add_argument("--width", type=int, nargs="+", default=[256])
    parser.add_argument("--probes", type=int, nargs="+", default=[64])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.records is None:
        args.records = 1_000_000 if args.dataset == "sift1m" else 100_000
    if (
        args.records < 256
        or args.records > 0xFFFFFFFF
        or args.queries <= 0
        or args.calibration_queries < 0
        or args.public_count < 256
        or args.public_query_count <= 0
        or args.seed < 0
        or not 0 < args.k <= 0xFFFFFFFF
    ):
        parser.error("Invalid record, query, public sample, k, or seed setting")
    if args.output.exists():
        parser.error("--output must be a new directory")
    if not all(np.isfinite(value) and 0 < value <= 1 for value in args.matching):
        parser.error("--matching fractions must be in (0, 1]")
    try:
        for name in ("budget", "width", "probes"):
            setattr(args, name, per_range(getattr(args, name), len(args.matching), name))
    except ValueError as error:
        parser.error(str(error))
    if (
        any(not args.k <= value <= 0xFFFFFFFF for value in args.budget)
        or any(not 0 < value <= 0xFFFFFFFF for value in args.width)
        or any(not 0 < value <= 256 for value in args.probes)
    ):
        parser.error("Require budget >= k, width > 0, and probes in [1, 256]")
    args.query_count = args.queries + args.calibration_queries
    return args


def main(argv=None, dataset=None):
    args = parse_args(argv, dataset)
    loader = {"sift": load_sift, "sift1m": load_sift, "coco": load_coco, "fanns100k": load_fanns}
    database, queries, learning, public_queries, attributes = loader[args.dataset](args)
    rng = np.random.default_rng(args.seed)
    if attributes is None:
        attributes = rng.permutation(len(database)).astype(np.uint32)
    arrays = {
        "database": database,
        "queries": queries[: args.queries],
        "pca": learning,
        "public": learning[: args.public_count],
        "public_queries": public_queries,
        "attributes": attributes,
    }
    if args.calibration_queries:
        arrays["calibration_queries"] = queries[args.queries :]
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {name}")
    args.output.mkdir(parents=True)
    for name, values in arrays.items():
        np.save(args.output / f"{name}.npy", values)
    write_requests(args.output / "requests.csv", attributes, args.queries, args, rng)
    if args.calibration_queries:
        write_requests(
            args.output / "calibration_requests.csv",
            attributes,
            args.calibration_queries,
            args,
            rng,
        )
    scale = 1024 if args.dataset in ("sift", "sift1m") else 1.001
    print(
        f"Converted {len(database):,} database vectors and {args.queries} queries to {args.output}"
    )
    print(f"Use prepare.py with --scale {scale}; see convert_data.py --help for calibration.")


if __name__ == "__main__":
    main()
