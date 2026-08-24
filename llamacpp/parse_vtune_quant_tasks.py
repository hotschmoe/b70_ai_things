#!/usr/bin/env python3
"""Classify VTune GPU-offload computing tasks by GGML quant family.

The input is the CSV emitted by:

    vtune -report hotspots -group-by gpu-adapter,computing-task-offload \
        -format csv -csv-delimiter comma

This parser deliberately does not estimate per-shape time or distribute unknown
task time. A report that cannot prove its columns or requested quant-family
coverage fails closed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TASK_COLUMNS = ("Computing Task", "GPU Computing Task")
TIME_COLUMNS = ("Total Time", "GPU Time", "GPU Computing Task Time")
ADAPTER_COLUMNS = ("GPU Adapter", "Device", "GPU")

# These values are part of ggml_type's serialized compatibility contract. The
# candidate image has the same values; preserving the numeric mapping lets us
# classify template names such as "ggml_type12" when VTune cannot demangle the
# enum value to Q4_K.
GGML_TYPE_NAMES = {
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_K",
    11: "q3_K",
    12: "q4_K",
    13: "q5_K",
    14: "q6_K",
    15: "q8_K",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    29: "iq1_m",
    34: "tq1_0",
    35: "tq2_0",
    39: "mxfp4",
    40: "nvfp4",
    41: "q1_0",
}

# Prefer explicit names, longest first. q8_1 is normally the transient MMVQ
# activation type and is reported only if no weight family occurs in the name.
QUANT_NAMES = tuple(sorted(set(GGML_TYPE_NAMES.values()), key=len, reverse=True))


def _first_present(fieldnames: list[str], choices: Iterable[str]) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for choice in choices:
        found = normalized.get(choice.lower())
        if found is not None:
            return found
    for choice in choices:
        prefix = choice.lower()
        for normalized_name, original in normalized.items():
            if normalized_name.startswith(prefix + " ") or normalized_name.startswith(
                prefix + "("
            ):
                return original
    return None


def _read_report(path: Path) -> tuple[list[dict[str, str]], dict[str, str | None]]:
    """Find and parse the tabular header in a VTune CSV report."""

    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    for index, line in enumerate(lines):
        fields = next(csv.reader([line]))
        task_column = _first_present(fields, TASK_COLUMNS)
        time_column = _first_present(fields, TIME_COLUMNS)
        if task_column is None or time_column is None:
            continue
        adapter_column = _first_present(fields, ADAPTER_COLUMNS)
        reader = csv.DictReader(lines[index:])
        rows = []
        for row in reader:
            task = (row.get(task_column) or "").strip()
            raw_time = (row.get(time_column) or "").strip()
            if task and raw_time:
                rows.append(row)
        if not rows:
            raise ValueError("VTune task table has no data rows")
        return rows, {
            "task": task_column,
            "time": time_column,
            "adapter": adapter_column,
        }
    raise ValueError(
        "cannot find VTune task table with Computing Task and Total Time columns"
    )


def _seconds(raw: str) -> float:
    text = raw.strip().replace(" ", "")
    match = re.fullmatch(
        r"([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"(ns|us|ms|s)?",
        text,
    )
    if not match:
        raise ValueError(f"invalid VTune time value: {raw!r}")
    value = float(match.group(1))
    scales = {None: 1.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    return value * scales[match.group(2)]


def classify_family(task: str) -> str | None:
    lowered = task.lower()
    explicit = []
    for family in QUANT_NAMES:
        # SYCL names can use underscores, namespace punctuation, or the C++
        # identifier embedded in a mangled HostKernel spelling.
        pattern = rf"(?<![a-z0-9]){re.escape(family.lower())}(?![a-z0-9])"
        if re.search(pattern, lowered):
            explicit.append(family)

    # A weight family wins over the q8_1 activation used by quantized matvec.
    non_activation = [family for family in explicit if family != "q8_1"]
    if non_activation:
        return non_activation[0]
    if explicit:
        return explicit[0]

    numeric_values = re.findall(r"ggml_type(?:il)?([0-9]+)e", lowered)
    numeric_values.extend(
        re.findall(r"ggml_type\s*\)?\s*([0-9]+)", lowered)
    )
    numeric = [
        GGML_TYPE_NAMES[int(value)]
        for value in numeric_values
        if int(value) in GGML_TYPE_NAMES
    ]
    non_activation = [family for family in numeric if family != "q8_1"]
    if non_activation:
        return non_activation[0]
    return numeric[0] if numeric else None


def classify_route(task: str) -> str:
    lowered = task.lower()
    if "mmvq" in lowered or "mul_mat_vec" in lowered or "vec_dot" in lowered:
        return "mmvq"
    if "mmq" in lowered or "mul_mat_q" in lowered:
        return "mmq"
    if "dequant" in lowered:
        return "dequantize"
    if "quantize" in lowered:
        return "quantize"
    if "get_rows" in lowered:
        return "get_rows"
    if "cpy_" in lowered or "copy" in lowered or "memcpy" in lowered:
        return "copy"
    return "other"


def parse_report(path: Path, required_families: set[str]) -> dict:
    rows, columns = _read_report(path)
    aggregate: defaultdict[tuple[str, str, str, str], dict[str, float | int]] = (
        defaultdict(lambda: {"task_rows": 0, "total_time_s": 0.0})
    )
    total_time_s = 0.0
    classified_time_s = 0.0
    adapters = set()
    families = set()
    unknown = []

    for row in rows:
        task = (row[columns["task"]] or "").strip()
        duration = _seconds(row[columns["time"]] or "")
        if duration < 0:
            raise ValueError(f"negative VTune task duration for {task!r}")
        adapter_column = columns["adapter"]
        adapter = (
            (row.get(adapter_column) or "").strip()
            if adapter_column is not None
            else "unspecified"
        )
        adapter = adapter or "unspecified"
        family = classify_family(task)
        route = classify_route(task)
        total_time_s += duration
        adapters.add(adapter)
        if family is None:
            unknown.append(
                {"adapter": adapter, "task": task, "total_time_s": duration}
            )
            continue
        classified_time_s += duration
        families.add(family)
        key = (adapter, family, route, task)
        aggregate[key]["task_rows"] += 1
        aggregate[key]["total_time_s"] += duration

    records = []
    for (adapter, family, route, task), values in sorted(aggregate.items()):
        records.append(
            {
                "adapter": adapter,
                "family": family,
                "route": route,
                "task": task,
                "task_rows": values["task_rows"],
                "total_time_s": values["total_time_s"],
            }
        )

    by_adapter_family: defaultdict[tuple[str, str], float] = defaultdict(float)
    for record in records:
        by_adapter_family[(record["adapter"], record["family"])] += record[
            "total_time_s"
        ]
    summary = [
        {"adapter": adapter, "family": family, "total_time_s": duration}
        for (adapter, family), duration in sorted(by_adapter_family.items())
    ]
    missing = sorted(required_families - families)
    checks = {
        "positive_total_time": total_time_s > 0,
        "adapter_column_present": columns["adapter"] is not None,
        "two_or_more_adapters": len(adapters) >= 2,
        "required_families_present": not missing,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "columns": columns,
        "adapters": sorted(adapters),
        "families": sorted(families),
        "missing_required_families": missing,
        "total_task_time_s": total_time_s,
        "classified_quant_task_time_s": classified_time_s,
        "records": records,
        "by_adapter_family": summary,
        "unknown_tasks": sorted(
            unknown, key=lambda item: item["total_time_s"], reverse=True
        ),
        "semantics": {
            "time": "sum of VTune GPU task interval durations",
            "overlap": "task durations may overlap and are not critical-path time",
            "shape_attribution": "not inferred when absent from the task symbol",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--require-family", action="append", default=[])
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    try:
        result = parse_report(args.report, set(args.require_family))
    except (OSError, ValueError, csv.Error) as exc:
        result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    output = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write:
        args.write.write_text(output, encoding="ascii")
    print(output, end="")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
