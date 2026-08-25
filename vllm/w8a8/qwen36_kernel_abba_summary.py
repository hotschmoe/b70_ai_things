#!/usr/bin/env python3
"""Summarize one isolated June/August Qwen3.6 kernel A-B-B-A run."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def output_fingerprint(case: dict[str, Any]) -> tuple[str, ...]:
    if case["suite"] == "quant":
        return (case["q_sha256"], case["scale_sha256"])
    return (case["output_sha256"], case["checksum_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--b2", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    order = ("a1", "b1", "b2", "a2")
    paths = {name: getattr(args, name) for name in order}
    artifacts = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    protocols = {item["protocol"] for item in artifacts.values()}
    suites = {item["suite"] for item in artifacts.values()}
    if protocols != {"qwen36-june-august-w8a8-kernel-arm-v1"}:
        raise RuntimeError(f"protocol mismatch: {protocols}")
    if len(suites) != 1:
        raise RuntimeError(f"suite mismatch: {suites}")
    cases_by_arm = {
        arm: {case["id"]: case for case in item["cases"]}
        for arm, item in artifacts.items()
    }
    case_sets = {tuple(sorted(cases)) for cases in cases_by_arm.values()}
    if len(case_sets) != 1:
        raise RuntimeError("case identity mismatch across A-B-B-A arms")

    rows = []
    for case_id in next(iter(case_sets)):
        cases = {arm: cases_by_arm[arm][case_id] for arm in order}
        a_times = [cases[arm]["timing"]["median_ms"] for arm in ("a1", "a2")]
        b_times = [cases[arm]["timing"]["median_ms"] for arm in ("b1", "b2")]
        a_median = statistics.median(a_times)
        b_median = statistics.median(b_times)
        fingerprints = {
            arm: output_fingerprint(case) for arm, case in cases.items()
        }
        rows.append(
            {
                "id": case_id,
                "all_arms_pass": all(case["pass"] for case in cases.values()),
                "exact_output_match_all_arms": len(set(fingerprints.values())) == 1,
                "fingerprints": fingerprints,
                "a_median_ms": a_median,
                "b_median_ms": b_median,
                "b_over_a": b_median / a_median,
                "a2_over_a1": a_times[1] / a_times[0],
                "b2_over_b1": b_times[1] / b_times[0],
            }
        )

    identity_roots = {
        arm: item["identity"]["package_root"] for arm, item in artifacts.items()
    }
    failures = {
        arm: item["failures"] for arm, item in artifacts.items() if item["failures"]
    }
    summary = {
        "protocol": "qwen36-june-august-w8a8-kernel-abba-v1",
        "suite": next(iter(suites)),
        "order": list(order),
        "identity_roots": identity_roots,
        "artifacts": {arm: str(path) for arm, path in paths.items()},
        "cases": rows,
        "failures": failures,
        "pass": not failures and all(row["all_arms_pass"] for row in rows),
        "exact_output_match_all_cases": all(
            row["exact_output_match_all_arms"] for row in rows
        ),
        "geomean_b_over_a": statistics.geometric_mean(
            row["b_over_a"] for row in rows
        ),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
