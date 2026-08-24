#!/usr/bin/env python3
"""Parse GGML_SYCL_QUANT_CENSUS key-value rows into JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


PREFIX = "[QUANT-CENSUS] "
INTEGER_FIELDS = {"device", "reordered", "split", "width", "K", "N", "rows", "calls"}
KEY_FIELDS = ("kind", "algo", "device", "type", "reordered", "split", "width", "K", "N", "rows")


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in text.split():
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        values[key] = value
    return values


def parse_lines(lines: list[str]) -> dict:
    aggregate: defaultdict[tuple, int] = defaultdict(int)
    declared_ends = []
    versions = set()

    for raw in lines:
        if PREFIX not in raw:
            continue
        values = parse_key_values(raw.split(PREFIX, 1)[1].strip())
        if "version" in values:
            versions.add(values["version"])
        if values.get("kind") in {"logical", "actual"}:
            missing = [field for field in KEY_FIELDS + ("calls",) if field not in values]
            if missing:
                raise ValueError(f"census row missing {','.join(missing)}: {raw.rstrip()}")
            converted = {
                key: int(value) if key in INTEGER_FIELDS else value
                for key, value in values.items()
            }
            key = tuple(converted[field] for field in KEY_FIELDS)
            aggregate[key] += converted["calls"]
        if "logical_total" in values and "actual_total" in values:
            declared_ends.append(
                {"logical_total": int(values["logical_total"]), "actual_total": int(values["actual_total"])}
            )

    records = []
    for key, calls in sorted(aggregate.items()):
        record = dict(zip(KEY_FIELDS, key, strict=True))
        record["calls"] = calls
        records.append(record)
    computed = {
        "logical_total": sum(record["calls"] for record in records if record["kind"] == "logical"),
        "actual_total": sum(record["calls"] for record in records if record["kind"] == "actual"),
    }
    return {
        "versions": sorted(versions),
        "records": records,
        "computed_totals": computed,
        "declared_ends": declared_ends,
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
