#!/usr/bin/env python3
"""Run restart-comparable long-context and forced-output vLLM probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CONTEXT_TARGETS = (2048, 8192, 16384, 30000)
CONTEXT_OUTPUT = 128
LONG_OUTPUT = 4096


def write_suite(path: Path, prompts: list[dict[str, str]], suite_id: str) -> None:
    data = {"suite_id": suite_id, "version": 1, "prompts": prompts}
    path.write_text(json.dumps(data, ensure_ascii=True) + "\n", encoding="ascii")


def prompt_for(target: int) -> str:
    prefix = f"Long context probe target {target}. Preserve the history."
    body = " memory" * target
    return f"{prefix}{body}\nReply by analyzing the final instruction."


def run_bench(
    *,
    bench: Path,
    base_url: str,
    model: str,
    suite: Path,
    max_tokens: int,
    output: Path,
    stdout: Path,
) -> None:
    command = [
        sys.executable,
        str(bench),
        "--base-url",
        base_url,
        "--model",
        model,
        "--api-mode",
        "completions",
        "--suite",
        str(suite),
        "--max-tokens",
        str(max_tokens),
        "--metric-tokens",
        "100",
        "--seed",
        "42",
        "--timeout",
        "1800",
        "--return-token-ids",
        "--allow-screening",
        "--request-extra-json",
        '{"temperature":0,"top_p":1,"ignore_eos":true}',
        "--out",
        str(output),
    ]
    with stdout.open("w", encoding="ascii") as handle:
        subprocess.run(command, check=True, stdout=handle)


def validate_result(path: Path, expected_rows: int, expected_output: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="ascii"))
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise RuntimeError(f"unexpected row count in {path}: {len(rows or [])}")
    for row in rows:
        completion = row.get("completion_tokens")
        token_ids = row.get("token_ids")
        if completion != expected_output:
            raise RuntimeError(f"incomplete forced output in {path}: {completion}")
        if not isinstance(token_ids, list) or len(token_ids) != expected_output:
            raise RuntimeError(f"missing raw token array in {path}")
        if row.get("cached_tokens") != 0:
            raise RuntimeError(f"prompt cache reuse in {path}")
        if row.get("finish_reasons") != ["length"]:
            raise RuntimeError(f"unexpected finish reason in {path}")
    return data


def row_identity(row: dict[str, Any]) -> dict[str, Any]:
    token_ids = row["token_ids"]
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return {
        "prompt_id": row["prompt_id"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "token_sha256": hashlib.sha256(encoded).hexdigest(),
        "text_sha256": row["sha256"],
        "ttft_ms": row["ttft_s"] * 1000.0,
        "tok_s_after_ttft_full": row["tok_s_after_ttft_full"],
    }


def compare(reference: dict[str, Any], observed: dict[str, Any], label: str) -> None:
    left = reference["rows"]
    right = observed["rows"]
    if len(left) != len(right):
        raise RuntimeError(f"{label} row-count mismatch")
    for old, new in zip(left, right, strict=True):
        if old["prompt_id"] != new["prompt_id"]:
            raise RuntimeError(f"{label} prompt-order mismatch")
        if old["prompt_tokens"] != new["prompt_tokens"]:
            raise RuntimeError(f"{label} prompt-token mismatch: {old['prompt_id']}")
        if old["token_ids"] != new["token_ids"]:
            raise RuntimeError(f"{label} output-token mismatch: {old['prompt_id']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    args.attempt_dir.mkdir(parents=True, exist_ok=True)
    context_suite = args.attempt_dir / "long-context-suite.json"
    long_suite = args.attempt_dir / "long-output-suite.json"
    context_output = args.attempt_dir / "long-context-performance.json"
    long_output = args.attempt_dir / "long-output-performance.json"

    write_suite(
        context_suite,
        [
            {
                "id": f"context-{target}",
                "prompt_class": "long-context",
                "prompt": prompt_for(target),
            }
            for target in CONTEXT_TARGETS
        ],
        "qwen38-fp8-long-context-v1",
    )
    write_suite(
        long_suite,
        [
            {
                "id": "forced-output-4096",
                "prompt_class": "long-output",
                "prompt": prompt_for(2048),
            }
        ],
        "qwen38-fp8-long-output-v1",
    )

    run_bench(
        bench=args.bench,
        base_url=args.base_url,
        model=args.model,
        suite=context_suite,
        max_tokens=CONTEXT_OUTPUT,
        output=context_output,
        stdout=args.attempt_dir / "long-context-performance.stdout",
    )
    run_bench(
        bench=args.bench,
        base_url=args.base_url,
        model=args.model,
        suite=long_suite,
        max_tokens=LONG_OUTPUT,
        output=long_output,
        stdout=args.attempt_dir / "long-output-performance.stdout",
    )

    context = validate_result(context_output, len(CONTEXT_TARGETS), CONTEXT_OUTPUT)
    long_run = validate_result(long_output, 1, LONG_OUTPUT)
    prompt_counts = [row["prompt_tokens"] for row in context["rows"]]
    if prompt_counts != sorted(prompt_counts) or prompt_counts[-1] < 28000:
        raise RuntimeError(f"long-context prompt sizes are invalid: {prompt_counts}")

    reference_exact = None
    if args.reference:
        prior = json.loads(args.reference.read_text(encoding="ascii"))
        prior_context = json.loads(Path(prior["context_artifact"]).read_text(encoding="ascii"))
        prior_long = json.loads(Path(prior["long_output_artifact"]).read_text(encoding="ascii"))
        compare(prior_context, context, "long-context")
        compare(prior_long, long_run, "long-output")
        reference_exact = True

    summary = {
        "schema": "b70.qwen38-fp8-long-context.v1",
        "model": args.model,
        "context_artifact": str(context_output),
        "long_output_artifact": str(long_output),
        "context_rows": [row_identity(row) for row in context["rows"]],
        "long_output_row": row_identity(long_run["rows"][0]),
        "reference": str(args.reference) if args.reference else None,
        "reference_token_arrays_exact": reference_exact,
        "verdict": "passed",
    }
    summary_path = args.attempt_dir / "long-context.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
