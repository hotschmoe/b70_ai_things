#!/usr/bin/env python3
"""Compare compiler-selected Triton configs across fresh server processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NONSEMANTIC_KEYS = {"time_taken_ms", "triton_cache_hash"}


def semantic_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="ascii"))
    return {key: value for key, value in data.items() if key not in NONSEMANTIC_KEYS}


def cache_data(cache_root: Path, attempt: int) -> dict[str, Any]:
    root = cache_root / f"attempt-{attempt}" / "torch_compile_cache" / "torch_aot_compile"
    aot_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(aot_dirs) != 1:
        raise ValueError(f"attempt {attempt} has {len(aot_dirs)} AOT directories")
    aot = aot_dirs[0]
    configs = {path.relative_to(aot): path for path in aot.rglob("*.best_config")}
    return {"aot_key": aot.name, "root": aot, "configs": configs}


def analyze(
    result_root: Path,
    cache_root: Path,
    attempts: int,
    served_model: str,
    schema: str,
    completion_route: str,
    inductor_autotune_pointwise: bool = True,
    inductor_deterministic_config: bool = False,
) -> dict[str, Any]:
    caches = [cache_data(cache_root, index) for index in range(1, attempts + 1)]
    aot_keys = [item["aot_key"] for item in caches]
    reference = caches[0]["configs"]
    comparisons = []
    all_exact = len(set(aot_keys)) == 1
    for index, cache in enumerate(caches[1:], start=2):
        candidate = cache["configs"]
        common = sorted(set(reference) & set(candidate))
        only_reference = sorted(str(path) for path in set(reference) - set(candidate))
        only_candidate = sorted(str(path) for path in set(candidate) - set(reference))
        semantic_differences = []
        nonsemantic_differences = []
        byte_exact = 0
        for relative in common:
            left = reference[relative]
            right = candidate[relative]
            if semantic_config(left) != semantic_config(right):
                semantic_differences.append(str(relative))
            elif left.read_bytes() != right.read_bytes():
                nonsemantic_differences.append(str(relative))
            else:
                byte_exact += 1
        exact = not only_reference and not only_candidate and not semantic_differences
        all_exact = all_exact and exact
        comparisons.append(
            {
                "left_attempt": 1,
                "right_attempt": index,
                "common_best_config_paths": len(common),
                "left_only_paths": only_reference,
                "right_only_paths": only_candidate,
                "semantic_difference_paths": semantic_differences,
                "semantic_differences": len(semantic_differences),
                "nonsemantic_differences": len(nonsemantic_differences),
                "byte_exact": byte_exact,
                "semantic_exact": exact,
            }
        )

    smoke_texts = []
    for index in range(1, attempts + 1):
        models = json.loads(
            (result_root / f"attempt-{index}" / "models.json").read_text(
                encoding="ascii"
            )
        )
        ids = [item["id"] for item in models["data"]]
        if ids != [served_model]:
            raise ValueError(f"attempt {index} model identity mismatch: {ids}")
        smoke = json.loads(
            (result_root / f"attempt-{index}" / "smoke.json").read_text(
                encoding="ascii"
            )
        )
        smoke_texts.append(smoke["choices"][0]["text"])
    smoke_exact = len(set(smoke_texts)) == 1
    all_exact = all_exact and smoke_exact
    return {
        "schema": schema,
        "verdict": "passed" if all_exact else "failed_compile_selection_exactness",
        "served_model": served_model,
        "attempts": attempts,
        "tp": 2,
        "p2p": 0,
        "mtp": 0,
        "xpu_graph": False,
        "compile_oracle": True,
        "completion_route": completion_route,
        "inductor_autotune_pointwise": inductor_autotune_pointwise,
        "inductor_deterministic_config": inductor_deterministic_config,
        "aot_keys": aot_keys,
        "aot_key_exact": len(set(aot_keys)) == 1,
        "best_config_counts": [len(item["configs"]) for item in caches],
        "semantic_config_exact": all(item["semantic_exact"] for item in comparisons),
        "smoke_text_exact": smoke_exact,
        "comparisons": comparisons,
        "promotion_authorized": False,
        "promotion_blockers": [
            "compile oracle is not a full token, speed, long-context, or concurrency qualification"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--completion-route", required=True)
    parser.add_argument(
        "--inductor-autotune-pointwise", type=int, choices=(0, 1), required=True
    )
    parser.add_argument(
        "--inductor-deterministic-config", type=int, choices=(0, 1), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(
        args.result_dir,
        args.cache_root,
        args.attempts,
        args.served_model,
        args.schema,
        args.completion_route,
        bool(args.inductor_autotune_pointwise),
        bool(args.inductor_deterministic_config),
    )
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["verdict"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
