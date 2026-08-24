#!/usr/bin/env python3
"""Produce a shaped host/device census from TP=2 Kineto traces.

The profiler records one CPU event per logical operator and correlates every
XPU kernel launch with that event through ``External id``.  This parser keeps
the recorded input dimensions, so repeated GEMM and materialization costs can
be assigned to concrete model shapes instead of broad kernel-name buckets.
"""

import argparse
import collections
import gzip
import json
import re
from pathlib import Path


DEVICE_CATEGORIES = {
    "Kernel",
    "gpu_memcpy",
    "gpu_memset",
    "gpu_op",
    "gpu_user_annotation",
    "kernel",
    "xpu_op",
}

DEFAULT_OP_PATTERN = "|".join(
    re.escape(name)
    for name in (
        "aten::copy_",
        "aten::mm",
        "aten::reshape",
        "ChunkGatedDeltaRuleFunction",
        "_xpu_C::dynamic_per_token_int8_quant",
        "_xpu_C::int8_gemm_w8a8",
        "_xpu_C::int8_gemm_w8a16",
        "sgl_kernel::gemma_fused_add_rmsnorm",
        "sgl_kernel::silu_and_mul",
    )
)


def load_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    return payload.get("traceEvents", payload)


def normalized_dims(event):
    dims = (event.get("args") or {}).get("Input Dims") or []
    return tuple(tuple(value) if isinstance(value, list) else value for value in dims)


def format_dims(dims):
    if not dims:
        return "-"
    return ";".join("x".join(str(value) for value in tensor) or "scalar" for tensor in dims)


def trace_census(path, op_pattern):
    events = load_trace(path)
    cpu_by_external_id = {}
    rows = collections.defaultdict(
        lambda: {"calls": 0, "host_us": 0.0, "kernels": 0, "device_us": 0.0}
    )

    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "cpu_op":
            continue
        external_id = (event.get("args") or {}).get("External id")
        if external_id is not None:
            cpu_by_external_id[external_id] = event
        if op_pattern.search(str(event.get("name", ""))):
            key = (event.get("name", "?"), normalized_dims(event))
            rows[key]["calls"] += 1
            rows[key]["host_us"] += float(event.get("dur", 0.0) or 0.0)

    matched_device_us = 0.0
    total_device_us = 0.0
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in DEVICE_CATEGORIES:
            continue
        duration = float(event.get("dur", 0.0) or 0.0)
        total_device_us += duration
        external_id = (event.get("args") or {}).get("External id")
        cpu_event = cpu_by_external_id.get(external_id)
        if cpu_event is None or not op_pattern.search(str(cpu_event.get("name", ""))):
            continue
        key = (cpu_event.get("name", "?"), normalized_dims(cpu_event))
        rows[key]["kernels"] += 1
        rows[key]["device_us"] += duration
        matched_device_us += duration

    return rows, total_device_us, matched_device_us


def print_census(path, rows, total_device_us, matched_device_us, steps, minimum_ms):
    print(f"TRACE\t{path}")
    print(
        "SUMMARY\t"
        f"total_device_ms={total_device_us / 1000.0:.3f}\t"
        f"selected_device_ms={matched_device_us / 1000.0:.3f}\t"
        f"selected_share_pct={100.0 * matched_device_us / max(total_device_us, 1.0):.2f}"
    )
    print(
        "OP\tCALLS\tCALLS_PER_STEP\tKERNELS\tDEVICE_MS\tDEVICE_US_PER_CALL\t"
        "HOST_MS\tINPUT_DIMS"
    )
    ordered = sorted(
        rows.items(),
        key=lambda item: (-item[1]["device_us"], item[0][0], str(item[0][1])),
    )
    for (name, dims), values in ordered:
        device_ms = values["device_us"] / 1000.0
        if device_ms < minimum_ms:
            continue
        calls = values["calls"]
        print(
            f"{name}\t{calls}\t{calls / steps:.3f}\t{values['kernels']}\t"
            f"{device_ms:.3f}\t{values['device_us'] / max(calls, 1):.3f}\t"
            f"{values['host_us'] / 1000.0:.3f}\t{format_dims(dims)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, nargs="+")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--op-pattern", default=DEFAULT_OP_PATTERN)
    parser.add_argument("--minimum-device-ms", type=float, default=0.1)
    args = parser.parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    op_pattern = re.compile(args.op_pattern)

    for index, path in enumerate(args.trace):
        if index:
            print()
        rows, total_device_us, matched_device_us = trace_census(path, op_pattern)
        print_census(
            path,
            rows,
            total_device_us,
            matched_device_us,
            args.steps,
            args.minimum_device_ms,
        )


if __name__ == "__main__":
    main()
