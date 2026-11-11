#!/usr/bin/env python3
"""Select RAMP path thresholds from independent calibration measurements.

Run ramp with --calibrate on held-out calibration requests first. Then:
  python select_thresholds.py /path/to/calibration/queries.csv --records 100000

Search parameters remain fixed per range size. Each range size receives equal
weight; there is no minimum-recall constraint. Candidate cut points are 0, n,
and integer geometric means of adjacent matching counts. This script prints
the thresholds to pass to the subsequent evaluation run.
"""
import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import statistics


def select_thresholds(rows, records):
    grouped = defaultdict(list)
    budgets = {}
    cases = defaultdict(set)
    for row in rows:
        count, budget = int(row["matching"]), int(row["budget"])
        path = row["path"]
        if count in budgets and budgets[count] != budget:
            raise ValueError("Use a fixed candidate budget within each range size")
        if not 0 <= count <= records or path not in {"Scan", "IVF", "HNSW"}:
            raise ValueError("Invalid calibration row")
        elapsed = float(row["vector_ms"])
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid calibration time")
        budgets[count] = budget
        grouped[count, path].append(elapsed)
        case = (row["request"], row["query_id"], row["budget"], row["width"], row["probes"])
        cases[count, path].add(case)
    counts = sorted(budgets)
    if not counts:
        raise ValueError("No calibration measurements")
    for count in counts:
        if any(not grouped[count, path] for path in ("Scan", "IVF", "HNSW")):
            raise ValueError("Calibration must measure all three RAMP paths at every range size")
        if not cases[count, "Scan"] == cases[count, "IVF"] == cases[count, "HNSW"]:
            raise ValueError("Paths must use the same queries and search settings")
    mean = {key: statistics.mean(value) for key, value in grouped.items()}
    candidates = sorted({0, records, *(math.isqrt(a * b) for a, b in zip(counts, counts[1:]))})
    trials = []
    for scan in candidates:
        for hnsw in candidates:
            if scan >= hnsw:
                continue
            times = []
            for count in counts:
                path = (
                    "Scan"
                    if count <= budgets[count] or count <= scan
                    else ("IVF" if count <= hnsw else "HNSW")
                )
                times.append(mean[count, path])
            trials.append((statistics.mean(times), scan, hnsw))
    if not trials:
        raise ValueError("No valid threshold pair")
    return min(trials)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--records", type=int, required=True)
    args = parser.parse_args()
    if args.records <= 0:
        parser.error("records must be positive")
    with args.measurements.open(newline="") as stream:
        elapsed, scan, hnsw = select_thresholds(list(csv.DictReader(stream)), args.records)
    print(f"--scan-threshold {scan} --hnsw-threshold {hnsw}")
    print(f"Mean calibration vector-search time: {elapsed:.6f} ms")


if __name__ == "__main__":
    main()
