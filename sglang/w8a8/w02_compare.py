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
    common_text_hash: str | None = None
    common_token_hash: str | None = None
    common_output_ids: list[Any] | None = None
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
            if common_output_ids is None:
                common_output_ids = output_ids
            if output_ids != common_output_ids:
                raise RuntimeError(f"{arm} repeat {repeat}: output array mismatch")
            text_hash = result.get("text_sha256")
            token_hash = result.get("output_ids_sha256")
            if common_text_hash is None:
                common_text_hash = text_hash
                common_token_hash = token_hash
            if text_hash != common_text_hash or token_hash != common_token_hash:
                raise RuntimeError(f"{arm} repeat {repeat}: cross-arm output mismatch")
            results.append(result)

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
        }

    eager = float(arms["eager"]["median_post_first_tok_s"])
    breakable = float(arms["breakable"]["median_post_first_tok_s"])
    reclaim = float(arms["reclaim500"]["median_post_first_tok_s"])
    graph_ratio = breakable / eager
    reclaim_ratio = reclaim / breakable
    return {
        "protocol": "b70-w02-graph-reclaim-ab-v1",
        "contract": common_contract,
        "cross_arm_exact": True,
        "text_sha256": common_text_hash,
        "output_ids_sha256": common_token_hash,
        "arms": arms,
        "comparisons": {
            "breakable_over_eager": graph_ratio,
            "breakable_gain_percent": (graph_ratio - 1.0) * 100.0,
            "reclaim500_over_breakable": reclaim_ratio,
            "reclaim500_delta_percent": (reclaim_ratio - 1.0) * 100.0,
            "graph_gain_at_least_3_percent": graph_ratio >= 1.03,
            "reclaim_within_3_percent": reclaim_ratio >= 0.97,
        },
        "passed": True,
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
    print("VERDICT -> PASS exact matched W02 short comparison")


if __name__ == "__main__":
    main()
