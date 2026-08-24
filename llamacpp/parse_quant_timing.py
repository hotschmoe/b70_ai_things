#!/usr/bin/env python3
"""Parse sampled GGML_SYCL_QUANT_TIMING rows and project per-shape time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFIX = "[QUANT-TIMING] "
KEY_FIELDS = ("algo", "device", "type", "reordered", "split", "width", "K", "rows")
SUM_FIELDS = (
    "calls_seen",
    "samples",
    "device_ns",
    "barrier_ns",
    "incomplete",
    "invalid",
)
INTEGER_FIELDS = {
    "device",
    "reordered",
    "split",
    "width",
    "K",
    "rows",
    "mean_ns",
    "min_ns",
    "max_ns",
} | set(SUM_FIELDS)


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in text.split():
        if "=" in field:
            key, value = field.split("=", 1)
            values[key] = value
    return values


def parse_lines(lines: list[str]) -> dict:
    aggregate: dict[tuple, dict] = {}
    headers = []
    declared_ends = []

    for raw in lines:
        if PREFIX not in raw:
            continue
        values = parse_key_values(raw.split(PREFIX, 1)[1].strip())
        if "version" in values:
            headers.append(
                {
                    key: int(value) if key not in {"version", "scope"} else value
                    for key, value in values.items()
                }
            )
        if values.get("kind") == "device":
            required = KEY_FIELDS + SUM_FIELDS + ("min_ns", "max_ns")
            missing = [field for field in required if field not in values]
            if missing:
                raise ValueError(f"timing row missing {','.join(missing)}: {raw.rstrip()}")
            converted = {
                key: int(value) if key in INTEGER_FIELDS else value
                for key, value in values.items()
            }
            key = tuple(converted[field] for field in KEY_FIELDS)
            row = aggregate.setdefault(
                key,
                {
                    **{field: converted[field] for field in KEY_FIELDS},
                    **{field: 0 for field in SUM_FIELDS},
                    "min_ns": None,
                    "max_ns": 0,
                },
            )
            for field in SUM_FIELDS:
                row[field] += converted[field]
            if converted["samples"]:
                row["min_ns"] = (
                    converted["min_ns"]
                    if row["min_ns"] is None
                    else min(row["min_ns"], converted["min_ns"])
                )
                row["max_ns"] = max(row["max_ns"], converted["max_ns"])
        if "samples" in values and "device_ns" in values and "kind" not in values:
            declared_ends.append(
                {"samples": int(values["samples"]), "device_ns": int(values["device_ns"])}
            )

    records = []
    for row in aggregate.values():
        row["mean_ns"] = row["device_ns"] / row["samples"] if row["samples"] else None
        row["projected_device_ns"] = (
            row["mean_ns"] * row["calls_seen"] if row["mean_ns"] is not None else None
        )
        records.append(row)
    records.sort(
        key=lambda row: row["projected_device_ns"] if row["projected_device_ns"] is not None else -1,
        reverse=True,
    )
    projected_total = sum(row["projected_device_ns"] or 0 for row in records)
    for row in records:
        row["projected_share"] = (
            (row["projected_device_ns"] or 0) / projected_total if projected_total else None
        )

    return {
        "headers": headers,
        "records": records,
        "summary": {
            "calls_seen": sum(row["calls_seen"] for row in records),
            "samples": sum(row["samples"] for row in records),
            "sampled_device_ns": sum(row["device_ns"] for row in records),
            "barrier_ns": sum(row["barrier_ns"] for row in records),
            "incomplete": sum(row["incomplete"] for row in records),
            "invalid": sum(row["invalid"] for row in records),
            "projected_device_ns": projected_total,
        },
        "declared_ends": declared_ends,
        "projection_note": "mean sampled callback time multiplied by calls_seen; evidence estimate, not wall time",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    lines = []
    for path in args.logs:
        lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
    result = parse_lines(lines)
    output = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write:
        args.write.write_text(output, encoding="ascii")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
