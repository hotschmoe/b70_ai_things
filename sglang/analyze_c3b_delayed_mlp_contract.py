#!/usr/bin/env python3
"""Gate a C3b delayed-MLP-all-reduce contract-only A/B result.

This intentionally makes no performance claim.  It checks that the candidate
routes all 63 target-layer edges through SGLang's delayed marker contract,
consumes every marker through the generic correctness fallback, and preserves
the baseline's deterministic bytes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROUTE_RE = re.compile(
    r"\[c3b-delayed-mlp\] ROUTES "
    r"rank=(\d+) eligible=(\d+) consumed=(\d+) generic=(\d+)"
)
FATAL_RE = re.compile(
    r"device_lost|out_of_resources|ur_result_error|enginedead|!!!!|"
    r"(^|[^a-z])nan([^a-z]|$)",
    re.IGNORECASE | re.MULTILINE,
)


def read_text(path: Path) -> str:
    return path.read_text(errors="replace")


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_id(path: Path) -> str:
    return str(load_json(path)["data"][0]["id"])


def inspect_contract(path: Path, expected_env: str, require_mount: bool) -> bool:
    data = load_json(path)
    if not isinstance(data, list) or len(data) != 1:
        return False
    item = data[0]
    env = set(item.get("Config", {}).get("Env", []))
    if expected_env not in env:
        return False
    if not require_mount:
        return True
    return any(
        mount.get("Destination")
        == "/opt/venv/lib/python3.12/site-packages/xpu_delayed_mlp_ar.py"
        and str(mount.get("Source", "")).endswith(
            "/sglang/patches/xpu_delayed_mlp_ar.py"
        )
        for mount in item.get("Mounts", [])
    )


def container_running(path: Path, expected_name: str) -> bool:
    data = load_json(path)
    return (
        isinstance(data, list)
        and len(data) == 1
        and data[0].get("Name") == f"/{expected_name}"
        and data[0].get("State", {}).get("Running") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--served", required=True)
    parser.add_argument("--eligible", type=int, default=63)
    parser.add_argument("--production-id", default="hotschmoe-dd")
    parser.add_argument(
        "--production-name", default="qwen38_unsloth_ud_q4k_xl_tp2"
    )
    args = parser.parse_args()

    root = Path(args.result_dir).resolve()
    baseline = root / "baseline"
    candidate = root / "candidate"
    base_log = read_text(baseline / "server.log")
    cand_log = read_text(candidate / "server.log")
    route_rows = [tuple(map(int, row)) for row in ROUTE_RE.findall(cand_log)]
    exact_ranks = {
        rank
        for rank, eligible, consumed, generic in route_rows
        if eligible == args.eligible
        and consumed == args.eligible
        and generic == args.eligible
    }

    base_deterministic = load_json(baseline / "deterministic.json")
    cand_deterministic = load_json(candidate / "deterministic.json")
    deterministic_nonempty = (
        len(base_deterministic) == 8
        and all(
            ((row.get("reasoning_content") or "") + (row.get("content") or "")).strip()
            and int(row.get("completion_tokens") or 0) > 0
            for row in base_deterministic
        )
    )

    health_files = [
        root / "health_pre_campaign.log",
        baseline / "health_pre.log",
        baseline / "health_post.log",
        candidate / "health_pre.log",
        candidate / "health_post.log",
        root / "health_before_restore.log",
    ]
    health_green = all(
        path.is_file() and "xpu-health: HEALTHY (cards 0 1)" in read_text(path)
        for path in health_files
    )

    production_expected = (
        read_text(root / "production_expected.txt").strip() == "1"
    )
    production_ok = True
    if production_expected:
        after = root / "production_models_after.json"
        health_after = root / "production_health_after.txt"
        inspect_after = root / "production_inspect_after.json"
        production_ok = (
            after.is_file()
            and model_id(after) == args.production_id
            and health_after.is_file()
            and read_text(health_after).strip() == "200"
            and inspect_after.is_file()
            and container_running(inspect_after, args.production_name)
        )

    checks = {
        "baseline_identity": model_id(baseline / "models.json") == args.served,
        "candidate_identity": model_id(candidate / "models.json") == args.served,
        "baseline_env_off": inspect_contract(
            baseline / "container_inspect.json", "B70_XPU_DELAY_MLP_AR=0", True
        ),
        "candidate_env_on_and_mount": inspect_contract(
            candidate / "container_inspect.json", "B70_XPU_DELAY_MLP_AR=1", True
        ),
        "baseline_has_no_c3b_route": "[c3b-delayed-mlp]" not in base_log,
        "candidate_has_both_rank_contracts": exact_ranks == {0, 1},
        "candidate_routes_stay_consistent": bool(route_rows)
        and {rank for rank, _eligible, _consumed, _generic in route_rows}
        == {0, 1}
        and all(
            eligible >= args.eligible
            and eligible == consumed
            and consumed == generic
            for _rank, eligible, consumed, generic in route_rows
        ),
        "deterministic_nonempty_8": deterministic_nonempty,
        "deterministic_byte_identity": base_deterministic == cand_deterministic,
        "baseline_no_fatal": FATAL_RE.search(base_log) is None,
        "candidate_no_fatal": FATAL_RE.search(cand_log) is None,
        "all_card_health_green": health_green,
        "production_restored_if_present": production_ok,
    }
    passed = all(checks.values())
    summary = {
        "claim": "contract_only_no_performance_promotion",
        "expected_eligible_edges": args.eligible,
        "candidate_route_rows": [
            {
                "rank": rank,
                "eligible": eligible,
                "consumed": consumed,
                "generic": generic,
            }
            for rank, eligible, consumed, generic in route_rows
        ],
        "checks": checks,
        "pass": passed,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    verdict = [
        f"VERDICT -> {'PASS' if passed else 'FAIL'}",
        "CLAIM -> contract-only; no performance or shelf-promotion claim",
    ]
    verdict.extend(
        f"CHECK {name}={'PASS' if value else 'FAIL'}"
        for name, value in checks.items()
    )
    verdict_text = "\n".join(verdict) + "\n"
    (root / "verdict.txt").write_text(verdict_text, encoding="ascii")
    print(verdict_text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
