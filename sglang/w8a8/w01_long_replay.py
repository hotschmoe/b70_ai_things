#!/usr/bin/env python3
"""Run one exact-length SGLang stream with token-window stability evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPT = (
    "Continue producing a detailed technical discussion of accelerator runtime "
    "design, graph replay, collective completion, recurrent state ownership, and "
    "long-running inference stability. Do not stop early."
)


def get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def window_summary(
    milestones: list[dict[str, float | int]], minimum_final_initial_ratio: float
) -> dict[str, Any]:
    if len(milestones) < 3:
        raise RuntimeError("at least three throughput milestones are required")
    windows = []
    for previous, current in zip(milestones, milestones[1:], strict=False):
        token_delta = int(current["completion_tokens"]) - int(
            previous["completion_tokens"]
        )
        time_delta = float(current["elapsed_s"]) - float(previous["elapsed_s"])
        if token_delta <= 0 or time_delta <= 0:
            raise RuntimeError("milestones are not strictly increasing")
        windows.append(
            {
                "start_completion_tokens": int(previous["completion_tokens"]),
                "end_completion_tokens": int(current["completion_tokens"]),
                "elapsed_s": time_delta,
                "tok_s": token_delta / time_delta,
            }
        )
    first_rate = float(windows[0]["tok_s"])
    final_rate = float(windows[-1]["tok_s"])
    ratio = final_rate / first_rate
    rates = [float(item["tok_s"]) for item in windows]
    return {
        "windows": windows,
        "first_window_tok_s": first_rate,
        "final_window_tok_s": final_rate,
        "final_over_first": ratio,
        "minimum_final_initial_ratio": minimum_final_initial_ratio,
        "median_window_tok_s": statistics.median(rates),
        "minimum_window_tok_s": min(rates),
        "maximum_window_tok_s": max(rates),
        "passed": ratio >= minimum_final_initial_ratio,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=50_000)
    parser.add_argument("--window-tokens", type=int, default=5_000)
    parser.add_argument("--stream-interval", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=5_400)
    parser.add_argument("--minimum-final-initial-ratio", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_tokens < args.window_tokens * 3:
        raise RuntimeError("output length must contain at least three windows")
    if args.window_tokens <= args.stream_interval:
        raise RuntimeError("window size must exceed the stream interval")
    if not 0 < args.minimum_final_initial_ratio <= 1:
        raise RuntimeError("minimum ratio must be in (0, 1]")

    base = args.base.rstrip("/")
    model_ids = [item["id"] for item in get_json(base + "/v1/models")["data"]]
    if model_ids != [args.model]:
        raise RuntimeError(f"served model identity mismatch: {model_ids}")

    body = {
        "text": PROMPT,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.output_tokens,
            "ignore_eos": True,
            "stream_interval": args.stream_interval,
        },
        "stream": True,
    }
    request = urllib.request.Request(
        base + "/generate",
        data=json.dumps(body).encode("ascii"),
        headers={"content-type": "application/json"},
    )

    start = time.perf_counter()
    first = None
    last_completion_tokens = 0
    next_milestone = args.window_tokens
    milestones: list[dict[str, float | int]] = []
    final_event: dict[str, Any] | None = None
    saw_done = False
    try:
        response = urllib.request.urlopen(request, timeout=args.timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    with response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"stream error: {event['error']}")
            meta = event.get("meta_info") or {}
            completion_tokens = meta.get("completion_tokens")
            if not isinstance(completion_tokens, int):
                raise RuntimeError("stream event omitted integer completion_tokens")
            if completion_tokens <= last_completion_tokens:
                raise RuntimeError("completion token count did not increase")
            now = time.perf_counter()
            if first is None:
                first = now
                milestones.append(
                    {"completion_tokens": completion_tokens, "elapsed_s": 0.0}
                )
            while completion_tokens >= next_milestone:
                milestones.append(
                    {
                        "completion_tokens": completion_tokens,
                        "elapsed_s": now - first,
                        "target_completion_tokens": next_milestone,
                    }
                )
                next_milestone += args.window_tokens
            last_completion_tokens = completion_tokens
            final_event = event

    end = time.perf_counter()
    if first is None or final_event is None:
        raise RuntimeError("stream returned no token event")
    meta = final_event.get("meta_info") or {}
    finish_reason = meta.get("finish_reason")
    finish_type = (
        finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
    )
    output_ids = final_event.get("output_ids")
    if last_completion_tokens != args.output_tokens:
        raise RuntimeError(
            f"forced length mismatch: {last_completion_tokens} != {args.output_tokens}"
        )
    if finish_type != "length":
        raise RuntimeError(f"finish reason was not length: {finish_reason}")
    if not isinstance(output_ids, list) or len(output_ids) != args.output_tokens:
        raise RuntimeError("final event omitted the exact output token array")
    if int(milestones[-1]["completion_tokens"]) != args.output_tokens:
        milestones.append(
            {
                "completion_tokens": args.output_tokens,
                "elapsed_s": end - first,
                "target_completion_tokens": args.output_tokens,
            }
        )

    stability = window_summary(milestones, args.minimum_final_initial_ratio)
    text = final_event.get("text")
    if not isinstance(text, str) or not text:
        raise RuntimeError("final event omitted generated text")
    result = {
        "protocol": "b70-w01-long-replay-v1",
        "model": args.model,
        "model_ids": model_ids,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("ascii")).hexdigest(),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": last_completion_tokens,
        "finish_reason": finish_reason,
        "saw_done": saw_done,
        "seed": None,
        "sampling_contract": "native greedy temperature=0; seed unsupported",
        "stream_interval": args.stream_interval,
        "ttft_ms": (first - start) * 1000,
        "total_s": end - start,
        "post_first_tok_s": (last_completion_tokens - 1) / (end - first),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_ids_sha256": hashlib.sha256(
            json.dumps(output_ids, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "milestones": milestones,
        "stability": stability,
        "passed": stability["passed"],
    }
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(
        "RESULT -> tokens={} post_first_tok_s={:.4f} final_over_first={:.6f}".format(
            last_completion_tokens,
            result["post_first_tok_s"],
            stability["final_over_first"],
        )
    )
    if not stability["passed"]:
        raise RuntimeError("late-throughput flatness gate failed")
    print("VERDICT -> PASS")


if __name__ == "__main__":
    main()
