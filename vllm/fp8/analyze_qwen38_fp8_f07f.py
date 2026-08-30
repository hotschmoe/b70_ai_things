#!/usr/bin/env python3
"""Validate two F07f strict lifetimes and emit a compact qualification JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_log(path: Path, marker: str) -> None:
    if marker not in path.read_text(encoding="ascii", errors="replace"):
        raise RuntimeError(f"missing marker {marker!r} in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt_1", type=Path)
    parser.add_argument("attempt_2", type=Path)
    parser.add_argument("--publisher-a", type=Path, required=True)
    parser.add_argument("--publisher-b", type=Path, required=True)
    args = parser.parse_args()

    attempts = [args.attempt_1, args.attempt_2]
    performance = [load(path / "performance-screen.json") for path in attempts]
    primary = [
        item["summary"][
            "class_balanced_tok_s_1_100_intervals_after_ttft"
        ]["median"]
        for item in performance
    ]
    for path, item, rate in zip(attempts, performance, primary):
        if item["realistic_final_gate"]["passed"] is not True:
            raise RuntimeError(f"strict performance gate failed: {path}")
        if item["realistic_final_gate"]["cached_tokens_all_zero"] is not True:
            raise RuntimeError(f"cached token gate failed: {path}")
        if rate < 45.0:
            raise RuntimeError(f"primary rate below 45 tok/s: {path}: {rate}")
        require_log(path / "post-card.log", "xpu-health: HEALTHY")
        require_log(path / "post-p2p0.log", "xpu-collective-health: HEALTHY")
        require_log(path / "server.log", "CCL_TOPO_P2P_ACCESS changed to be 1")
        require_log(path / "verdict.txt", "VERDICT -> F07f PASS")

    rows = [{row["prompt_id"]: row for row in item["rows"]} for item in performance]
    prompt_ids = list(rows[0])
    if set(prompt_ids) != set(rows[1]) or len(prompt_ids) != 12:
        raise RuntimeError("strict prompt sets differ or are incomplete")
    exact_local = [
        prompt_id
        for prompt_id in prompt_ids
        if rows[0][prompt_id]["token_ids"] == rows[1][prompt_id]["token_ids"]
    ]
    if len(exact_local) != 12:
        raise RuntimeError(f"cross-lifetime token mismatch: {len(exact_local)}/12")

    concurrent = load(args.attempt_2 / "concurrent-quality.json")
    if concurrent["pass_all"] is not True or concurrent["total_requests"] != 32:
        raise RuntimeError("concurrent quality gate failed")

    publisher_counts = []
    for publisher_path in (args.publisher_a, args.publisher_b):
        publisher = {row["prompt_id"]: row for row in load(publisher_path)["rows"]}
        publisher_counts.append(
            sum(
                rows[1][prompt_id]["token_ids"] == publisher[prompt_id]["token_ids"]
                for prompt_id in prompt_ids
            )
        )

    output = {
        "schema": "b70.qwen38-fp8-f07f-qualification.v1",
        "verdict": "qualified_local_full_graph_route",
        "primary_tok_s_attempts": primary,
        "primary_tok_s_mean": statistics.mean(primary),
        "primary_tok_s_min": min(primary),
        "strict_performance_gates_passed": True,
        "cached_tokens_all_zero": True,
        "cross_lifetime_token_arrays_exact": True,
        "cross_lifetime_exact_prompts": len(exact_local),
        "concurrent_quality_passed": True,
        "concurrent_quality_requests": concurrent["total_requests"],
        "post_card_health_passed": True,
        "post_collective_health_passed": True,
        "publisher_exact_prompts": publisher_counts,
        "publisher_identity_exact": all(count == 12 for count in publisher_counts),
        "performance_sha256": [
            sha256(path / "performance-screen.json") for path in attempts
        ],
        "concurrent_sha256": sha256(args.attempt_2 / "concurrent-quality.json"),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
