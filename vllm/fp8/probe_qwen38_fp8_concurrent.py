#!/usr/bin/env python3
"""Compare serial and concurrent greedy raw-token arrays across restarts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any, Callable


CONCURRENCY = 4
OUTPUT_TOKENS = 512
BATCHES = 2


def load_post_stream(bench: Path) -> Callable[..., dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("b70_realistic_bench", bench)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module: {bench}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.post_stream


def make_prompts() -> list[str]:
    return [
        (
            f"Concurrent batch probe stream {index}. Preserve all context."
            + " memory" * (2048 + index * 17)
            + "\nWrite a detailed deterministic analysis."
        )
        for index in range(CONCURRENCY)
    ]


def validate(row: dict[str, Any], label: str) -> None:
    token_ids = row.get("token_ids")
    if row.get("completion_tokens") != OUTPUT_TOKENS:
        raise RuntimeError(f"{label} did not return {OUTPUT_TOKENS} tokens")
    if not isinstance(token_ids, list) or len(token_ids) != OUTPUT_TOKENS:
        raise RuntimeError(f"{label} omitted its complete raw-token array")
    if row.get("finish_reasons") != ["length"]:
        raise RuntimeError(f"{label} did not finish at the declared cap")
    usage = row.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    if details.get("cached_tokens") != 0:
        raise RuntimeError(f"{label} reused cached prompt tokens")


def identity(row: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(row["token_ids"], separators=(",", ":")).encode("ascii")
    return {
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "token_sha256": hashlib.sha256(encoded).hexdigest(),
        "text_sha256": row["sha256"],
        "ttft_ms": row["ttft_s"] * 1000.0,
        "tok_s_after_ttft_full": row["tok_s_after_ttft_full"],
    }


def call_one(
    post_stream: Callable[..., dict[str, Any]],
    base_url: str,
    model: str,
    prompt: str,
    request_id: str,
) -> dict[str, Any]:
    row = post_stream(
        base_url=base_url,
        model=model,
        prompt=prompt,
        max_tokens=OUTPUT_TOKENS,
        timeout=1200,
        api_mode="completions",
        seed=42,
        request_extra={"temperature": 0, "top_p": 1, "ignore_eos": True},
        return_token_ids=True,
        system_prompt=None,
        request_id=request_id,
    )
    validate(row, request_id)
    return row


def concurrent_batch(
    post_stream: Callable[..., dict[str, Any]],
    base_url: str,
    model: str,
    prompts: list[str],
    batch: int,
) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [
            pool.submit(
                call_one,
                post_stream,
                base_url,
                model,
                prompt,
                f"f05b-batch-{batch}-stream-{index}",
            )
            for index, prompt in enumerate(prompts)
        ]
        return [future.result() for future in futures]


def compare_rows(
    expected: list[dict[str, Any]], observed: list[dict[str, Any]], label: str
) -> None:
    if len(expected) != len(observed):
        raise RuntimeError(f"{label} stream-count mismatch")
    for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
        if left["prompt_tokens"] != right["prompt_tokens"]:
            raise RuntimeError(f"{label} prompt-token mismatch at stream {index}")
        if left["token_ids"] != right["token_ids"]:
            raise RuntimeError(f"{label} output-token mismatch at stream {index}")


def batch_metric(rows: list[dict[str, Any]], batch: int) -> dict[str, Any]:
    first = [row["first_text_epoch_s"] for row in rows]
    ended = [row["request_ended_epoch_s"] for row in rows]
    aggregate = sum(row["completion_tokens"] - 1 for row in rows) / (
        max(ended) - min(first)
    )
    return {
        "batch": batch,
        "streams": len(rows),
        "aggregate_post_first_tok_s": aggregate,
        "median_stream_post_first_tok_s": statistics.median(
            row["tok_s_after_ttft_full"] for row in rows
        ),
        "rows": [identity(row) for row in rows],
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
    post_stream = load_post_stream(args.bench)
    prompts = make_prompts()
    serial = [
        call_one(
            post_stream,
            args.base_url,
            args.model,
            prompt,
            f"f05b-serial-stream-{index}",
        )
        for index, prompt in enumerate(prompts)
    ]
    batches = []
    raw_batches = []
    for batch in range(BATCHES):
        rows = concurrent_batch(
            post_stream, args.base_url, args.model, prompts, batch
        )
        compare_rows(serial, rows, f"serial-concurrent-batch-{batch}")
        batches.append(batch_metric(rows, batch))
        raw_batches.append(rows)

    raw = {"serial": serial, "batches": raw_batches}
    raw_path = args.attempt_dir / "concurrent-raw.json"
    raw_path.write_text(
        json.dumps(raw, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )

    reference_exact = None
    if args.reference:
        prior_summary = json.loads(args.reference.read_text(encoding="ascii"))
        prior_raw = json.loads(
            Path(prior_summary["raw_artifact"]).read_text(encoding="ascii")
        )
        compare_rows(prior_raw["serial"], serial, "restart-serial")
        for batch, rows in enumerate(raw_batches):
            compare_rows(prior_raw["batches"][batch], rows, f"restart-batch-{batch}")
        reference_exact = True

    summary = {
        "schema": "b70.qwen38-fp8-concurrent.v1",
        "model": args.model,
        "concurrency": CONCURRENCY,
        "output_tokens_each": OUTPUT_TOKENS,
        "batches": batches,
        "median_aggregate_post_first_tok_s": statistics.median(
            batch["aggregate_post_first_tok_s"] for batch in batches
        ),
        "serial_rows": [identity(row) for row in serial],
        "raw_artifact": str(raw_path),
        "reference": str(args.reference) if args.reference else None,
        "reference_token_arrays_exact": reference_exact,
        "serial_concurrent_token_arrays_exact": True,
        "verdict": "passed",
    }
    summary_path = args.attempt_dir / "concurrent.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
