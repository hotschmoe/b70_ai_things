#!/usr/bin/env python3
"""Compare pure-decode label timing from current and legacy summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(document: dict, rank: str) -> dict[str, dict]:
    rows = document.get("pure_decode_labels_by_mean_total_ms")
    if rows is None:
        rows = document.get("step_summary_by_rank_label", [])
    return {
        str(row["label"]): row
        for row in rows
        if str(row.get("rank", "")) == rank
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rank", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_document = json.loads(args.candidate.read_text())
    reference_document = json.loads(args.reference.read_text())
    candidate_rows = _rows(candidate_document, args.rank)
    reference_rows = _rows(reference_document, args.rank)

    comparisons = []
    for label in sorted(candidate_rows.keys() & reference_rows.keys()):
        candidate_ms = float(candidate_rows[label]["mean_total_ms_per_step"])
        reference_ms = float(reference_rows[label]["mean_total_ms_per_step"])
        comparisons.append(
            {
                "label": label,
                "candidate_mean_total_ms_per_step": candidate_ms,
                "reference_mean_total_ms_per_step": reference_ms,
                "candidate_minus_reference_ms": candidate_ms - reference_ms,
                "candidate_over_reference": (
                    candidate_ms / reference_ms if reference_ms else None
                ),
            }
        )
    comparisons.sort(
        key=lambda row: row["candidate_minus_reference_ms"], reverse=True
    )

    output = {
        "protocol": "xpu-pure-decode-timing-comparison-v1",
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "rank": args.rank,
        "candidate_timing_semantics": candidate_document.get(
            "timing_semantics"
        ),
        "comparison_requirement": (
            "Both inputs must use the same synchronization and sampling "
            "protocol. Nested labels are nonexclusive."
        ),
        "common_label_count": len(comparisons),
        "comparisons_by_added_ms": comparisons,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))
    return 0 if comparisons else 1


if __name__ == "__main__":
    raise SystemExit(main())
