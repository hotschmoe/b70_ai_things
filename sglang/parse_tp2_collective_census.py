#!/usr/bin/env python3
"""Count Kineto record_param_comms events and enforce a TP=2 decode census."""

import argparse
import collections
import gzip
import json
from pathlib import Path


def load_trace(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    return payload.get("traceEvents", payload)


def first_tensor_dims(args):
    dims = args.get("Input Dims", [])
    if not dims:
        return ()
    dims = dims[0]
    if dims and isinstance(dims[0], list):
        dims = dims[0]
    return tuple(int(value) for value in dims)


def census(path: Path):
    counts = collections.Counter()
    metadata = set()
    for event in load_trace(path):
        if event.get("name") != "record_param_comms":
            continue
        args = event.get("args", {})
        name = str(args.get("Collective name", "unknown"))
        if "allgather" in name:
            kind = "allgather"
        elif "allreduce" in name:
            kind = "allreduce"
        else:
            kind = name
        key = (kind, str(args.get("dtype", "unknown")), first_tensor_dims(args))
        counts[key] += 1
        metadata.add(
            (
                kind,
                str(args.get("dtype", "unknown")),
                str(args.get("Process Group Ranks", "unknown")),
                int(args.get("Is asynchronized op", -1)),
            )
        )
    return counts, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--arm", choices=("baseline", "replicated"), required=True)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    paths = sorted(args.trace_dir.glob("*DECODE.trace.json.gz"))
    if len(paths) != 2:
        raise SystemExit(f"expected two rank decode traces, found {len(paths)}")

    # The 11 embedding calls per scheduler iteration are nine single-row
    # draft lookups and two 11-row target/verify lookups.  Replication removes
    # those exact signatures: 27/132 becomes 18/130.
    per_step_small = 27 if args.arm == "baseline" else 18
    per_step_verify = 132 if args.arm == "baseline" else 130
    expected = {
        ("allreduce", "BFloat16", (1, 5120)): per_step_small * args.steps,
        ("allreduce", "BFloat16", (11, 5120)): per_step_verify * args.steps,
        ("allgather", "BFloat16", (1, 124160)): 9 * args.steps,
        ("allgather", "BFloat16", (11, 124160)): 2 * args.steps,
    }
    expected_total = (170 if args.arm == "baseline" else 159) * args.steps

    reference = None
    for path in paths:
        counts, metadata = census(path)
        print(f"TRACE -> {path.name}")
        for key, count in sorted(counts.items(), key=lambda item: str(item[0])):
            print(f"COUNT -> {count:4d} {key}")
        print(f"METADATA -> {sorted(metadata)}")
        print(f"TOTAL -> {sum(counts.values())}")

        if counts != expected:
            print(f"EXPECTED -> {expected}")
            raise SystemExit(f"collective census mismatch in {path.name}")
        if sum(counts.values()) != expected_total:
            raise SystemExit(f"collective total mismatch in {path.name}")
        if metadata != {
            ("allgather", "BFloat16", "[0, 1]", 1),
            ("allreduce", "BFloat16", "[0, 1]", 1),
        }:
            raise SystemExit(f"collective metadata mismatch in {path.name}")
        if reference is not None and counts != reference:
            raise SystemExit("rank collective censuses differ")
        reference = counts

    ar_per_step = 159 if args.arm == "baseline" else 148
    print(
        f"VERDICT -> PASS arm={args.arm} allreduce_per_step={ar_per_step} "
        "allgather_per_step=11"
    )


if __name__ == "__main__":
    main()
