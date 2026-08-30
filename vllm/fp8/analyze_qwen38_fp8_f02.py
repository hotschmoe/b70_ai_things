#!/usr/bin/env python3
"""Persist the F02 two-lifetime exactness and diagnostic speed verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def token_hash(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def compare_rows(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids = left["token_ids"]
    right_ids = right["token_ids"]
    first_mismatch = next(
        (index for index, pair in enumerate(zip(left_ids, right_ids)) if pair[0] != pair[1]),
        None,
    )
    if first_mismatch is None and len(left_ids) != len(right_ids):
        first_mismatch = min(len(left_ids), len(right_ids))
    mismatches = sum(a != b for a, b in zip(left_ids, right_ids))
    mismatches += abs(len(left_ids) - len(right_ids))
    return {
        "prompt_id": left["prompt_id"],
        "prompt_class": left["prompt_class"],
        "exact": left_ids == right_ids,
        "first_mismatch_zero_based": first_mismatch,
        "mismatch_count": mismatches,
        "left_token_count": len(left_ids),
        "right_token_count": len(right_ids),
        "left_token_sha256": token_hash(left_ids),
        "right_token_sha256": token_hash(right_ids),
        "left_text_sha256": left["sha256"],
        "right_text_sha256": right["sha256"],
    }


def load_attempt(root: Path, index: int, served: str) -> dict[str, Any]:
    attempt = root / f"attempt-{index}"
    performance = json.loads((attempt / "performance.json").read_text(encoding="ascii"))
    canary = json.loads((attempt / "canaries.json").read_text(encoding="ascii"))
    models = json.loads((attempt / "models.json").read_text(encoding="ascii"))
    model_ids = [item["id"] for item in models["data"]]
    if model_ids != [served]:
        raise ValueError(f"attempt {index} model identity mismatch: {model_ids}")
    gate = performance["realistic_final_gate"]
    fresh = performance["fresh_response_validity"]
    if not gate["passed"] or not fresh["performance_gate_eligible"]:
        raise ValueError(f"attempt {index} performance workload gate failed")
    if not gate["cached_tokens_all_zero"]:
        raise ValueError(f"attempt {index} used cached prompt tokens")
    if len(performance["rows"]) != 12:
        raise ValueError(f"attempt {index} did not complete 12 prompts")
    if not canary["pass_all"]:
        raise ValueError(f"attempt {index} independent canary gate failed")
    if not all(row["token_ids"] for row in performance["rows"]):
        raise ValueError(f"attempt {index} omitted native output token arrays")
    return performance


def publisher_comparisons(
    attempts: list[dict[str, Any]], publisher_paths: list[Path]
) -> list[dict[str, Any]]:
    results = []
    for publisher_path in publisher_paths:
        publisher = json.loads(publisher_path.read_text(encoding="ascii"))
        if len(publisher["rows"]) != 12:
            raise ValueError(f"publisher reference does not have 12 rows: {publisher_path}")
        for index, attempt in enumerate(attempts, start=1):
            comparisons = [
                compare_rows(left, right)
                for left, right in zip(attempt["rows"], publisher["rows"])
            ]
            results.append(
                {
                    "attempt": index,
                    "publisher_path": str(publisher_path),
                    "exact_prompts": sum(item["exact"] for item in comparisons),
                    "total_prompts": len(comparisons),
                    "prompt_comparisons": comparisons,
                }
            )
    return results


def analyze(
    root: Path,
    attempt_count: int,
    served: str,
    publisher_paths: list[Path],
    schema: str = "b70.qwen38-fp8-neural-f02.v2",
    completion_route: str = "explicit-work-wait",
    require_reference_exact: bool = False,
    mtp: int = 0,
    inductor_combo_kernels: bool = True,
    inductor_benchmark_combo_kernel: bool = True,
    inductor_max_autotune: bool = True,
    inductor_coordinate_descent_tuning: bool = True,
    inductor_autotune_pointwise: bool = True,
    inductor_deterministic_config: bool = False,
    p2p: int = 0,
    concurrent_qualified: bool = False,
    long_agent_qualified: bool = False,
    xpu_graph: bool = False,
) -> dict[str, Any]:
    attempts = [load_attempt(root, index, served) for index in range(1, attempt_count + 1)]
    reference = attempts[0]
    pair_comparisons = []
    all_exact = True
    minimum_exact = 12
    for index, candidate in enumerate(attempts[1:], start=2):
        prompts = [
            compare_rows(left, right)
            for left, right in zip(reference["rows"], candidate["rows"])
        ]
        exact = sum(item["exact"] for item in prompts)
        all_exact = all_exact and exact == len(prompts)
        minimum_exact = min(minimum_exact, exact)
        pair_comparisons.append(
            {
                "left_attempt": 1,
                "right_attempt": index,
                "exact_prompts": exact,
                "total_prompts": len(prompts),
                "prompt_comparisons": prompts,
            }
        )

    rates = [
        attempt["summary"]["class_balanced_tok_s_1_100_intervals_after_ttft"]["median"]
        for attempt in attempts
    ]
    reference_results = publisher_comparisons(attempts, publisher_paths)
    reference_exact = all(
        item["exact_prompts"] == item["total_prompts"] for item in reference_results
    )
    qualified = all_exact and (reference_exact or not require_reference_exact)
    if not all_exact:
        verdict = "failed_cross_server_token_exactness"
    elif require_reference_exact and not reference_exact:
        verdict = "failed_reference_token_exactness"
    else:
        verdict = "passed"
    blockers = []
    if not concurrent_qualified:
        blockers.append("concurrent qualification not yet run")
    if not long_agent_qualified:
        blockers.append("long-agent qualification not yet run")
    if not all_exact:
        blockers.insert(
            0, f"fresh P2P-{p2p} server lifetimes changed raw output token arrays"
        )
    if require_reference_exact and not reference_exact:
        blockers.insert(0, "output arrays changed from the required target reference")
    if p2p == 0:
        blockers.append("local P2P-off safety port is not the publisher P2P-on profile")
    return {
        "schema": schema,
        "verdict": verdict,
        "served_model": served,
        "attempts": attempt_count,
        "tp": 2,
        "p2p": p2p,
        "mtp": mtp,
        "xpu_graph": xpu_graph,
        "inductor": True,
        "inductor_combo_kernels": inductor_combo_kernels,
        "inductor_benchmark_combo_kernel": inductor_benchmark_combo_kernel,
        "inductor_max_autotune": inductor_max_autotune,
        "inductor_coordinate_descent_tuning": inductor_coordinate_descent_tuning,
        "inductor_autotune_pointwise": inductor_autotune_pointwise,
        "inductor_deterministic_config": inductor_deterministic_config,
        "completion_route": completion_route,
        "quantization": "fp8-block-weights-w8a16-runtime",
        "dtype": "float16",
        "kv_cache_dtype": "auto-observed-float16-target",
        "complete_token_arrays_exact": all_exact,
        "reference_token_arrays_exact": reference_exact,
        "reference_exactness_required": require_reference_exact,
        "exact_prompts_minimum_pair": minimum_exact,
        "total_prompts": 12,
        "cached_tokens_all_zero": True,
        "independent_canaries_passed": True,
        "concurrent_qualified": concurrent_qualified,
        "long_agent_qualified": long_agent_qualified,
        "class_balanced_tok_s_attempts": rates,
        "class_balanced_tok_s_median_diagnostic": statistics.median(rates),
        "performance_attribution_qualified": qualified,
        "pair_comparisons": pair_comparisons,
        "publisher_comparisons": reference_results,
        "promotion_authorized": qualified
        and concurrent_qualified
        and long_agent_qualified
        and not blockers,
        "promotion_blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--publisher-attempt", action="append", type=Path, default=[])
    parser.add_argument("--schema", default="b70.qwen38-fp8-neural-f02.v2")
    parser.add_argument("--completion-route", default="explicit-work-wait")
    parser.add_argument("--require-reference-exact", action="store_true")
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--p2p", type=int, choices=(0, 1), default=0)
    parser.add_argument("--concurrent-qualified", action="store_true")
    parser.add_argument("--long-agent-qualified", action="store_true")
    parser.add_argument("--xpu-graph", action="store_true")
    parser.add_argument("--inductor-combo-kernels", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--inductor-benchmark-combo-kernel", type=int, choices=(0, 1), default=1
    )
    parser.add_argument("--inductor-max-autotune", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--inductor-coordinate-descent-tuning", type=int, choices=(0, 1), default=1
    )
    parser.add_argument(
        "--inductor-autotune-pointwise", type=int, choices=(0, 1), default=1
    )
    parser.add_argument(
        "--inductor-deterministic-config", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.attempts < 2:
        parser.error("--attempts must be at least 2")
    summary = analyze(
        args.result_dir,
        args.attempts,
        args.served_model,
        args.publisher_attempt,
        args.schema,
        args.completion_route,
        args.require_reference_exact,
        args.mtp,
        bool(args.inductor_combo_kernels),
        bool(args.inductor_benchmark_combo_kernel),
        bool(args.inductor_max_autotune),
        bool(args.inductor_coordinate_descent_tuning),
        bool(args.inductor_autotune_pointwise),
        bool(args.inductor_deterministic_config),
        args.p2p,
        args.concurrent_qualified,
        args.long_agent_qualified,
        args.xpu_graph,
    )
    output = args.output or args.result_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["verdict"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
