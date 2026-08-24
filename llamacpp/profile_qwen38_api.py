#!/usr/bin/env python3
"""Matched fixed-prompt llama.cpp decode, coding, and prefill profiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path


CODE_PROMPTS = [
    (
        "Implement a complete LRU cache in Python with O(1) get/put via a "
        "doubly linked list and dict. Include type hints, docstrings, and usage "
        "examples. Write the full code."
    ),
    (
        "Write a thread-safe bounded blocking queue in Python using "
        "threading.Condition. Full type hints, docstrings, and a "
        "producer/consumer demo."
    ),
    (
        "Implement Dijkstra's shortest path in Python with a binary heap and a "
        "Graph class with add_edge. Include type hints, docstrings, and a worked example."
    ),
]

DECODE_PROMPT = (
    "Write a detailed essay about the history of the printing press, its major "
    "technical transitions, and its social consequences."
)

PREFILL_WORDS = (
    "memory hierarchy systolic matrix quantization throughput latency kernel tensor "
    "collective allreduce prefill decode bandwidth register cache pipeline scheduler "
    "attention hybrid checkpoint partition overlap fabric battlemage xe roofline"
).split()


def fixed_prefill_prompt(target_tokens: int, seed: int) -> str:
    words = []
    state = seed
    for index in range(int(target_tokens * 0.9)):
        state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        words.append(PREFILL_WORDS[state % len(PREFILL_WORDS)])
        if index % 32 == 31:
            words.append(f"marker{index}")
    return "Analyze these fixed technical notes, then answer OK. " + " ".join(words)


def finite_median(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.median(clean) if clean else None


def stream_chat(
    base: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool,
    timeout: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 1234,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    headers = {"content-type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    sent = time.monotonic()
    first = None
    ended = None
    text_parts = []
    usage = {}
    finish_reason = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            piece = (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("content")
                or ""
            )
            if piece:
                if first is None:
                    first = time.monotonic()
                text_parts.append(piece)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    ended = time.monotonic()
    if first is None:
        first = ended
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    post_first_s = ended - first
    ttft_s = first - sent
    text = "".join(text_parts)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft_s,
        "post_first_s": post_first_s,
        "e2e_s": ended - sent,
        "post_first_tok_s": (
            (completion_tokens - 1) / post_first_s
            if completion_tokens > 1 and post_first_s > 0
            else None
        ),
        "prefill_proxy_tok_s": (
            prompt_tokens / ttft_s if prompt_tokens > 0 and ttft_s > 0 else None
        ),
        "finish_reason": finish_reason,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_preview": text[:160].replace("\n", "\\n"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="hotschmoe-dd")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--gen-tokens", type=int, default=256)
    parser.add_argument("--prefill-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY", "")
    runs = []

    warmup = stream_chat(
        args.base, args.model, api_key, DECODE_PROMPT, 32, True, args.timeout
    )
    print(
        f"WARMUP -> completion={warmup['completion_tokens']} "
        f"ttft_s={warmup['ttft_s']:.4f}"
    )

    cases = []
    for repetition in range(args.reps):
        cases.append(("decode", repetition, DECODE_PROMPT, args.gen_tokens, True))
    for repetition in range(args.reps):
        prompt = CODE_PROMPTS[repetition % len(CODE_PROMPTS)]
        cases.append(("coding", repetition, prompt, args.gen_tokens, True))
    for repetition in range(args.reps):
        prompt = fixed_prefill_prompt(args.prefill_tokens, 81031 + repetition)
        cases.append(("prefill", repetition, prompt, 8, True))

    errors = []
    for kind, repetition, prompt, max_tokens, ignore_eos in cases:
        try:
            result = stream_chat(
                args.base,
                args.model,
                api_key,
                prompt,
                max_tokens,
                ignore_eos,
                args.timeout,
            )
            result.update({"kind": kind, "repetition": repetition})
            runs.append(result)
            print(
                f"RUN -> kind={kind} rep={repetition} "
                f"prompt={result['prompt_tokens']} completion={result['completion_tokens']} "
                f"ttft_s={result['ttft_s']:.4f} "
                f"post_first_tok_s={result['post_first_tok_s']}"
            )
        except Exception as exc:  # noqa: BLE001 - persist exact campaign failure
            error = {
                "kind": kind,
                "repetition": repetition,
                "error": f"{type(exc).__name__}: {exc}",
            }
            runs.append(error)
            errors.append(error)
            print(f"ERROR -> kind={kind} rep={repetition} error={error['error']}")

    summary = {}
    for kind in ("decode", "coding", "prefill"):
        selected = [run for run in runs if run.get("kind") == kind and "error" not in run]
        summary[kind] = {
            "n_ok": len(selected),
            "median_post_first_tok_s": finite_median(
                [run["post_first_tok_s"] for run in selected if run["post_first_tok_s"] is not None]
            ),
            "median_ttft_s": finite_median([run["ttft_s"] for run in selected]),
            "median_prefill_proxy_tok_s": finite_median(
                [
                    run["prefill_proxy_tok_s"]
                    for run in selected
                    if run["prefill_proxy_tok_s"] is not None
                ]
            ),
            "text_sha256": [run["text_sha256"] for run in selected],
        }

    output = {
        "tag": args.tag,
        "base": args.base,
        "model": args.model,
        "methodology": {
            "fixed_prompts_across_arms": True,
            "post_first_tok_s": "(completion_tokens - 1) / (end - first_content)",
            "prefill_proxy_tok_s": "prompt_tokens / TTFT",
            "temperature": 0,
            "top_p": 1,
            "seed": 1234,
            "ignore_eos": True,
        },
        "warmup": warmup,
        "runs": runs,
        "summary": summary,
        "passed": not errors and all(summary[kind]["n_ok"] == args.reps for kind in summary),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(f"OUTPUT -> {args.out} passed={output['passed']}")
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
