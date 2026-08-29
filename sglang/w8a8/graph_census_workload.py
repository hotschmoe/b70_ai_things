#!/usr/bin/env python3
"""Run one fixed-shape streaming request for graph-census overhead controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.output_tokens <= 1:
        parser.error("--output-tokens must exceed one")

    base = args.base.rstrip("/")
    models = get_json(base + "/v1/models")["data"]
    identifiers = [item["id"] for item in models]
    if identifiers != [args.model]:
        raise RuntimeError(f"served model identity mismatch: {identifiers}")

    paragraph = (
        "Graph replay, collective communication, memory hierarchy, and low "
        "precision kernels determine accelerator inference latency. "
    )
    prompt = (
        f"Profile nonce {args.nonce}. Analyze these notes, then continue with "
        f"technical prose until the output limit.\n{paragraph * 80}"
    )
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode("ascii"),
        headers={"content-type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    end = None
    prompt_tokens = None
    completion_tokens = None
    finish_reason = None
    text = []
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            piece = (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("content")
                or ""
            )
            if piece:
                if first is None:
                    first = time.perf_counter()
                text.append(piece)
    end = time.perf_counter()
    if first is None or prompt_tokens is None or completion_tokens is None:
        raise RuntimeError("stream omitted timing or usage evidence")
    if completion_tokens != args.output_tokens or finish_reason != "length":
        raise RuntimeError(
            f"forced output contract failed: tokens={completion_tokens} "
            f"finish_reason={finish_reason}"
        )
    result = {
        "model": args.model,
        "nonce": args.nonce,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_ms": (first - start) * 1000.0,
        "post_first_tok_s": (completion_tokens - 1) / (end - first),
        "total_s": end - start,
        "text_sha256": hashlib.sha256("".join(text).encode("utf-8")).hexdigest(),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print("RESULT -> " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
