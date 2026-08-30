#!/usr/bin/env python3
"""Run raw completion and concurrent exact-answer gates for Qwen3.8."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


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
    source = Path(
        os.environ.get(
            "SOURCE",
            "/mnt/vm_8tb/b70/steve-repro/"
            "qwen38-fp8-neural-20260829/source",
        )
    )
    quality_script = (
        source
        / "experiments/qwen38-27b-b70/scripts/"
        "qwen38-concurrent-quality-canary.py"
    )
    if not quality_script.is_file():
        raise RuntimeError(f"missing publisher quality gate: {quality_script}")

    raw_env = os.environ.copy()
    raw_env["F05B_REQUIRE_SERIAL_EXACT"] = "0"
    raw_env["F05B_REQUIRE_RESTART_EXACT"] = "0"
    run(
        [
            sys.executable,
            str(script_dir / "probe_qwen38_fp8_concurrent.py"),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--bench",
            str(args.bench),
            "--attempt-dir",
            str(args.attempt_dir),
        ],
        args.attempt_dir / "concurrent-raw-probe.stdout",
        env=raw_env,
    )
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

    raw = load(args.attempt_dir / "concurrent.json")
    quality = load(quality_path)
    if raw.get("verdict") != "passed":
        raise RuntimeError("raw concurrent completion gate did not pass")
    if quality.get("pass_all") is not True:
        raise RuntimeError("concurrent exact-answer quality gate did not pass")
    if quality.get("total_requests") != 32:
        raise RuntimeError("concurrent quality gate request count changed")

    prior_pass = None
    if args.reference:
        prior = load(args.reference)
        prior_pass = prior.get("verdict") == "passed"
        if not prior_pass:
            raise RuntimeError("prior concurrent gate was not a pass")

    summary = {
        "schema": "b70.qwen38-fp8-concurrent-gate.v1",
        "model": args.model,
        "raw_summary": str(args.attempt_dir / "concurrent.json"),
        "quality_artifact": str(quality_path),
        "complete_c4_batches": len(raw["batches"]),
        "output_tokens_each": raw["output_tokens_each"],
        "serial_concurrent_token_arrays_exact": raw[
            "serial_concurrent_token_arrays_exact"
        ],
        "concurrent_quality_requests": quality["total_requests"],
        "concurrent_quality_passed": True,
        "prior_attempt_passed": prior_pass,
        "verdict": "passed",
    }
    output = args.attempt_dir / "concurrent-gate.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
