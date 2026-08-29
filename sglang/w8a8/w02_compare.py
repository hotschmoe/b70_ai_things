#!/usr/bin/env python3
"""Summarize matched W02 eager, breakable, and reclaim measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ARMS = ("eager", "breakable", "reclaim500")


def finish_type(value: Any) -> Any:
    return value.get("type") if isinstance(value, dict) else value


def summarize(root: Path, repeats: int) -> dict[str, Any]:
    if repeats < 2:
        raise RuntimeError("at least two measured repeats are required")

    common_contract: dict[str, Any] | None = None
    eager_output_ids: list[Any] | None = None
    arms: dict[str, Any] = {}
    for arm in ARMS:
        results = []
        for repeat in range(1, repeats + 1):
            path = root / arm / f"measured_{repeat}.json"
            result = json.loads(path.read_text(encoding="ascii"))
            contract = {
                "model": result.get("model"),
                "prompt_sha256": result.get("prompt_sha256"),
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "sampling_contract": result.get("sampling_contract"),
                "stream_interval": result.get("stream_interval"),
            }
            if common_contract is None:
                common_contract = contract
            if contract != common_contract:
                raise RuntimeError(f"{arm} repeat {repeat}: contract mismatch")
            if finish_type(result.get("finish_reason")) != "length":
                raise RuntimeError(f"{arm} repeat {repeat}: non-length finish")
            if result.get("passed") is not True:
                raise RuntimeError(f"{arm} repeat {repeat}: client gate failed")
            output_ids = result.get("output_ids")
            if not isinstance(output_ids, list):
                raise RuntimeError(f"{arm} repeat {repeat}: output_ids missing")
            if len(output_ids) != result.get("completion_tokens"):
                raise RuntimeError(f"{arm} repeat {repeat}: output_ids length mismatch")
            text_hash = result.get("text_sha256")
            token_hash = result.get("output_ids_sha256")
            results.append(result)

        arm_output_ids = results[0]["output_ids"]
        if any(item["output_ids"] != arm_output_ids for item in results[1:]):
            raise RuntimeError(f"{arm}: measured repeats are not exact")
        if eager_output_ids is None:
            eager_output_ids = arm_output_ids
        mismatch_indices = [
            index
            for index, (expected, actual) in enumerate(
                zip(eager_output_ids, arm_output_ids, strict=True)
            )
            if expected != actual
        ]

        rates = [float(item["post_first_tok_s"]) for item in results]
        ttfts = [float(item["ttft_ms"]) for item in results]
        flatness = [float(item["stability"]["final_over_first"]) for item in results]
        arms[arm] = {
            "repeats": repeats,
            "median_post_first_tok_s": statistics.median(rates),
            "minimum_post_first_tok_s": min(rates),
            "maximum_post_first_tok_s": max(rates),
            "median_ttft_ms": statistics.median(ttfts),
            "minimum_final_initial_ratio": min(flatness),
            "post_first_tok_s": rates,
            "ttft_ms": ttfts,
            "final_initial_ratios": flatness,
            "text_sha256": results[0]["text_sha256"],
            "output_ids_sha256": results[0]["output_ids_sha256"],
            "target_exact_to_eager": not mismatch_indices,
            "first_mismatch_index": mismatch_indices[0] if mismatch_indices else None,
            "mismatch_count": len(mismatch_indices),
        }

    eager = float(arms["eager"]["median_post_first_tok_s"])
    breakable = float(arms["breakable"]["median_post_first_tok_s"])
    reclaim = float(arms["reclaim500"]["median_post_first_tok_s"])
    graph_ratio = breakable / eager
    reclaim_ratio = reclaim / breakable
    cross_arm_exact = all(
        bool(arms[arm]["target_exact_to_eager"]) for arm in ARMS
    )
    return {
        "protocol": "b70-w02-graph-reclaim-ab-v1",
        "contract": common_contract,
        "cross_arm_exact": cross_arm_exact,
        "arms": arms,
        "comparisons": {
            "breakable_over_eager": graph_ratio,
            "breakable_gain_percent": (graph_ratio - 1.0) * 100.0,
            "reclaim500_over_breakable": reclaim_ratio,
            "reclaim500_delta_percent": (reclaim_ratio - 1.0) * 100.0,
            "graph_gain_at_least_3_percent": graph_ratio >= 1.03,
            "reclaim_within_3_percent": reclaim_ratio >= 0.97,
            "performance_attribution_qualified": cross_arm_exact,
        },
        "passed": cross_arm_exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.repeats)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print("RESULT -> " + json.dumps(result["comparisons"], sort_keys=True))
    if not result["passed"]:
        print("VERDICT -> FAIL cross-arm output mismatch")
        raise SystemExit(1)
    print("VERDICT -> PASS exact matched W02 short comparison")


if __name__ == "__main__":
    main()
