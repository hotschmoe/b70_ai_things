#!/usr/bin/env python3
"""Attribute Ornith XPU kernels to semantic record_function ranges.

Kineto records an XPU launch as a CPU runtime event and the eventual device
kernel as a separate event joined by ``correlation``.  This analyzer assigns
the actual kernel duration to the narrowest enclosing ``b70::`` annotation at
the launch site.  It therefore avoids per-operation XPU synchronizations while
still producing a device-time stage ledger.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("traceEvents", payload)


def interval_contains(event: dict[str, Any], timestamp: float) -> bool:
    return event["ts"] <= timestamp <= event["ts"] + event.get("dur", 0.0)


def analyze(path: Path) -> dict[str, Any]:
    events = load_events(path)
    annotations_by_thread: dict[tuple[Any, Any], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    runtime_by_correlation: dict[Any, dict[str, Any]] = {}
    kernels: list[dict[str, Any]] = []

    for event in events:
        if event.get("ph") != "X":
            continue
        category = str(event.get("cat", "")).lower()
        if category == "user_annotation" and str(event.get("name", "")).startswith(
            "b70::"
        ):
            annotations_by_thread[(event.get("pid"), event.get("tid"))].append(event)
        elif category == "xpu_runtime":
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                runtime_by_correlation[correlation] = event
        elif "kernel" in category or "gpu" in category:
            kernels.append(event)

    for annotations in annotations_by_thread.values():
        annotations.sort(key=lambda event: (event["ts"], -event.get("dur", 0.0)))

    direct_gpu_us: collections.Counter[str] = collections.Counter()
    inclusive_gpu_us: collections.Counter[str] = collections.Counter()
    kernel_counts: collections.Counter[str] = collections.Counter()
    unattributed_names: collections.Counter[str] = collections.Counter()
    attributed_gpu_us = 0.0

    for kernel in kernels:
        duration = float(kernel.get("dur", 0.0))
        correlation = kernel.get("args", {}).get("correlation")
        runtime = runtime_by_correlation.get(correlation)
        matches: list[dict[str, Any]] = []
        if runtime is not None:
            thread = (runtime.get("pid"), runtime.get("tid"))
            matches = [
                annotation
                for annotation in annotations_by_thread.get(thread, [])
                if interval_contains(annotation, float(runtime["ts"]))
            ]
        if matches:
            # The narrowest duration is the immediate semantic owner.  All
            # enclosing ranges also receive an inclusive roll-up.
            matches.sort(key=lambda event: event.get("dur", 0.0))
            direct = str(matches[0]["name"])
            direct_gpu_us[direct] += duration
            kernel_counts[direct] += 1
            attributed_gpu_us += duration
            for annotation in {str(match["name"]): match for match in matches}.values():
                inclusive_gpu_us[str(annotation["name"])] += duration
        else:
            unattributed_names[str(kernel.get("name", "unknown"))] += duration

    annotation_cpu_us: collections.Counter[str] = collections.Counter()
    annotation_counts: collections.Counter[str] = collections.Counter()
    for annotations in annotations_by_thread.values():
        for annotation in annotations:
            label = str(annotation["name"])
            annotation_cpu_us[label] += float(annotation.get("dur", 0.0))
            annotation_counts[label] += 1

    total_gpu_us = sum(float(kernel.get("dur", 0.0)) for kernel in kernels)
    return {
        "trace": str(path),
        "kernel_count": len(kernels),
        "device_busy_sum_ms": total_gpu_us / 1000.0,
        "attributed_device_ms": attributed_gpu_us / 1000.0,
        "attributed_device_pct": (
            100.0 * attributed_gpu_us / total_gpu_us if total_gpu_us else 0.0
        ),
        "stages": [
            {
                "stage": label,
                "calls": annotation_counts[label],
                "cpu_range_total_ms": annotation_cpu_us[label] / 1000.0,
                "direct_device_ms": direct_gpu_us[label] / 1000.0,
                "inclusive_device_ms": inclusive_gpu_us[label] / 1000.0,
                "direct_kernel_count": kernel_counts[label],
            }
            for label in sorted(
                annotation_cpu_us,
                key=lambda item: inclusive_gpu_us[item],
                reverse=True,
            )
        ],
        "top_unattributed_kernels": [
            {"kernel": name, "device_ms": duration / 1000.0}
            for name, duration in unattributed_names.most_common(30)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    reports = [analyze(path) for path in args.traces]
    for report in reports:
        print(f"TRACE -> {report['trace']}")
        print(
            "DEVICE -> "
            f"busy_sum={report['device_busy_sum_ms']:.3f} ms "
            f"attributed={report['attributed_device_ms']:.3f} ms "
            f"({report['attributed_device_pct']:.1f}%)"
        )
        for row in report["stages"]:
            print(
                "STAGE -> "
                f"{row['stage']:<36} calls={row['calls']:5d} "
                f"cpu={row['cpu_range_total_ms']:10.3f} ms "
                f"gpu_direct={row['direct_device_ms']:10.3f} ms "
                f"gpu_inclusive={row['inclusive_device_ms']:10.3f} ms "
                f"kernels={row['direct_kernel_count']:6d}"
            )
        print("UNATTRIBUTED TOP ->")
        for row in report["top_unattributed_kernels"][:15]:
            print(f"  {row['device_ms']:10.3f} ms  {row['kernel'][:140]}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
