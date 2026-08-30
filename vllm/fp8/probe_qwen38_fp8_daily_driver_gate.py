#!/usr/bin/env python3
"""Qualify c2/c4 serving and a 30K agent request for the FP8 daily driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


NEEDLE = "B70-ORANGE-4917"
FILLER_TOKENS = 30000
LONG_OUTPUT_TOKENS = 32


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def run(command: list[str], output: Path, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    output.write_text(completed.stdout, encoding="ascii", errors="backslashreplace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"subordinate gate failed rc={completed.returncode}: {' '.join(command)}"
        )


def token_hash(values: list[int]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def long_prompt() -> str:
    return (
        f"Remember this verification code exactly: {NEEDLE}."
        + " memory" * FILLER_TOKENS
        + f"\nReply with only the verification code {NEEDLE}."
    )


def run_concurrency(
    args: argparse.Namespace,
    script: Path,
    concurrency: int,
) -> dict[str, Any]:
    target = args.attempt_dir / f"concurrency-c{concurrency}"
    target.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "F05B_CONCURRENCY": str(concurrency),
            "F05B_OUTPUT_TOKENS": "512",
            "F05B_BATCHES": "2",
            "F05B_REQUIRE_SERIAL_EXACT": "0",
            "F05B_REQUIRE_RESTART_EXACT": "0",
        }
    )
    run(
        [
            sys.executable,
            str(script),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--bench",
            str(args.bench),
            "--attempt-dir",
            str(target),
        ],
        target / "concurrent-probe.stdout",
        env,
    )
    result = load(target / "concurrent.json")
    if result.get("verdict") != "passed" or result.get("concurrency") != concurrency:
        raise RuntimeError(f"c{concurrency} completion probe did not pass")
    return result


def run_long_context(args: argparse.Namespace) -> dict[str, Any]:
    suite = args.attempt_dir / "long-agent-suite.json"
    artifact = args.attempt_dir / "long-agent-performance.json"
    suite.write_text(
        json.dumps(
            {
                "suite_id": "qwen38-fp8-long-agent-30k-v1",
                "version": 1,
                "prompts": [
                    {
                        "id": "long-agent-30k",
                        "prompt_class": "long-context",
                        "prompt": long_prompt(),
                    }
                ],
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="ascii",
    )
    run(
        [
            sys.executable,
            str(args.bench),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--api-mode",
            "completions",
            "--suite",
            str(suite),
            "--max-tokens",
            str(LONG_OUTPUT_TOKENS),
            "--metric-tokens",
            "16",
            "--seed",
            "42",
            "--timeout",
            "1800",
            "--return-token-ids",
            "--allow-screening",
            "--request-extra-json",
            '{"temperature":0,"top_p":1,"ignore_eos":true}',
            "--out",
            str(artifact),
        ],
        args.attempt_dir / "long-agent-performance.stdout",
    )
    data = load(artifact)
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("long-agent probe row count changed")
    row = rows[0]
    if not 29900 <= row.get("prompt_tokens", 0) <= 31000:
        raise RuntimeError(f"long-agent prompt size invalid: {row.get('prompt_tokens')}")
    if row.get("completion_tokens") != LONG_OUTPUT_TOKENS:
        raise RuntimeError("long-agent output was incomplete")
    if len(row.get("token_ids") or []) != LONG_OUTPUT_TOKENS:
        raise RuntimeError("long-agent output token array missing")
    if row.get("cached_tokens") != 0:
        raise RuntimeError("long-agent request reused cached prompt tokens")
    if NEEDLE not in row.get("text", ""):
        raise RuntimeError("long-agent request did not recover the verification code")
    return {
        "artifact": str(artifact),
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "token_sha256": token_hash(row["token_ids"]),
        "text_sha256": row["sha256"],
        "ttft_ms": row["ttft_s"] * 1000.0,
        "tok_s_after_ttft_full": row["tok_s_after_ttft_full"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    args.attempt_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    concurrent_script = script_dir / "probe_qwen38_fp8_concurrent.py"
    source = Path(
        os.environ.get(
            "SOURCE",
            "/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260829/source",
        )
    )
    quality_script = (
        source
        / "experiments/qwen38-27b-b70/scripts/qwen38-concurrent-quality-canary.py"
    )
    if not quality_script.is_file():
        raise RuntimeError(f"missing publisher quality gate: {quality_script}")

    c2 = run_concurrency(args, concurrent_script, 2)
    c4 = run_concurrency(args, concurrent_script, 4)
    quality_path = args.attempt_dir / "concurrent-quality.json"
    run(
        [
            sys.executable,
            str(quality_script),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--concurrency",
            "4",
            "--rounds",
            "8",
            "--timeout",
            "600",
            "--seed",
            "42",
            "--output-json",
            str(quality_path),
        ],
        args.attempt_dir / "concurrent-quality.stdout",
    )
    quality = load(quality_path)
    if quality.get("pass_all") is not True or quality.get("total_requests") != 32:
        raise RuntimeError("c4 exact-answer quality gate did not pass 32 requests")

    prior_pass = None
    long_agent = run_long_context(args)
    restart_long_agent_exact = None
    if args.reference:
        prior = load(args.reference)
        prior_pass = prior.get("verdict") == "passed"
        if not prior_pass:
            raise RuntimeError("prior daily-driver gate was not a pass")
        prior_long = prior["long_agent"]
        restart_long_agent_exact = (
            prior_long["prompt_tokens"] == long_agent["prompt_tokens"]
            and prior_long["token_sha256"] == long_agent["token_sha256"]
        )
        if not restart_long_agent_exact:
            raise RuntimeError("long-agent output changed across server lifetimes")

    summary = {
        "schema": "b70.qwen38-fp8-daily-driver-gate.v1",
        "model": args.model,
        "c2": {
            "median_aggregate_post_first_tok_s": c2[
                "median_aggregate_post_first_tok_s"
            ],
            "summary": str(args.attempt_dir / "concurrency-c2/concurrent.json"),
        },
        "c4": {
            "median_aggregate_post_first_tok_s": c4[
                "median_aggregate_post_first_tok_s"
            ],
            "summary": str(args.attempt_dir / "concurrency-c4/concurrent.json"),
        },
        "concurrent_quality_requests": quality["total_requests"],
        "concurrent_quality_passed": True,
        "long_agent": long_agent,
        "prior_attempt_passed": prior_pass,
        "restart_long_agent_exact": restart_long_agent_exact,
        "verdict": "passed",
    }
    output = args.attempt_dir / "daily-driver-gate.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
