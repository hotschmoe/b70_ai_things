#!/usr/bin/env python3
"""Evict conversation A from GPU and restore it from host RAM.

Sequence, one request at a time, temperature 0:

1. A cold. Record greedy continuation. This is the never-evicted control.
2. A again while it is still resident. GPU prefix hit, same continuation.
3. B and C, each long enough to fill the capped GPU block pool.
4. A again. Host hit must cover the checkpoint boundary, the prefill
   tail must be at most one block, and the greedy continuation must
   match the control.

Reads the attention block size from the serve log. Prompt length is
28 blocks when that fits in --max-model-len, else as many as fit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from kvoffload.layout import checkpoint_boundary
from kvoffload.metrics import counter_delta


def block_size_from_log(text: str) -> int | None:
    needles = (
        "Setting attention block size to ",
        "kv lcm block sizes ",
    )
    found: int | None = None
    for line in text.splitlines():
        for needle in needles:
            if needle in line:
                tail = line.split(needle, 1)[1]
                digits = []
                for ch in tail:
                    if ch.isdigit():
                        digits.append(ch)
                    elif digits:
                        break
                if digits:
                    found = int("".join(digits))
    return found


def build_ids(tokenizer, marker: str, n_tokens: int) -> list[int]:
    piece = tokenizer.encode(
        f"Note {marker}. The quick brown fox keeps talking about topic {marker}. ",
        add_special_tokens=False,
    )
    if not piece:
        raise RuntimeError(f"tokenizer produced no ids for {marker}")
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids.extend(piece)
    return ids[:n_tokens]


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("ascii")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def wait_ready(base: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last = "not started"
    while time.monotonic() < deadline:
        try:
            body = get_text(base + "/v1/models", timeout=10)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(exc)
            time.sleep(5)
            continue
        if "hotschmoe-dev" in body:
            return
        last = body[:200]
        time.sleep(5)
    raise RuntimeError(f"server not ready after {timeout_s:.0f}s: {last}")


def complete(
    base: str,
    model: str,
    token_ids: list[int],
    max_tokens: int,
    timeout: float,
    cache_salt: str | None = None,
    return_token_ids: bool = False,
) -> dict:
    """Stream a greedy completion. Return text, usage, and client TTFT."""
    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    if return_token_ids:
        payload["return_token_ids"] = True
    data = json.dumps(payload).encode("ascii")
    req = urllib.request.Request(
        base + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    ttft = None
    text_parts: list[str] = []
    sampled: list[int] = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            blob = line[5:].strip()
            if blob == "[DONE]":
                break
            event = json.loads(blob)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                piece = choice.get("text") or ""
                if piece and ttft is None:
                    ttft = time.monotonic() - started
                text_parts.append(piece)
                for token in choice.get("token_ids") or []:
                    sampled.append(int(token))
    if ttft is None:
        ttft = time.monotonic() - started
    return {
        "text": "".join(text_parts),
        "token_ids": sampled,
        "ttft_s": ttft,
        "usage": usage,
        "wall_s": time.monotonic() - started,
    }


def cached_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    details = usage.get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    if value is None:
        return None
    return int(value)


def interesting_metrics(text: str) -> list[str]:
    rows: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        if "_created" in line.split()[0]:
            continue
        if (
            "prefix_cache" in line
            or "prompt_tokens_cached" in line
            or "external_prefix" in line
        ):
            rows.append(line)
    return rows[:40]


def prompt_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    value = usage.get("prompt_tokens")
    if value is None:
        return None
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18111")
    parser.add_argument("--model", default="hotschmoe-dev")
    parser.add_argument("--serve-log", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--ready-timeout", type=float, default=3600)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    log_text = Path(args.serve_log).read_text(errors="replace")
    block = block_size_from_log(log_text)
    if block is None:
        print("evict_test: block size not in serve log yet", file=sys.stderr)
        return 2
    # Leave a few tokens of room past the prompt for the decode we request.
    room = args.max_model_len - args.decode_tokens - 8
    n_blocks = min(28, room // block)
    if n_blocks < 4:
        print(f"evict_test: block {block} does not fit 4 blocks in {room}", file=sys.stderr)
        return 2
    n_tokens = n_blocks * block
    boundary = checkpoint_boundary(n_tokens, block)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompts = {
        name: build_ids(tokenizer, name, n_tokens) for name in ("A", "B", "C")
    }
    shared = common_prefix_len(prompts["A"], prompts["B"])
    if shared >= block:
        print(f"evict_test: A and B share {shared} tokens", file=sys.stderr)
        return 2

    print(
        f"evict_test: block={block} tokens={n_tokens} boundary={boundary} "
        f"shared_ab={shared}",
        flush=True,
    )
    wait_ready(args.base_url, args.ready_timeout)
    print("evict_test: server ready", flush=True)

    def run(name: str, ids: list[int]) -> dict:
        print(f"evict_test: start {name} n={len(ids)}", flush=True)
        row = complete(
            args.base_url,
            args.model,
            ids,
            args.decode_tokens,
            args.request_timeout,
        )
        row["name"] = name
        row["cached_tokens"] = cached_tokens(row["usage"])
        row["prompt_tokens"] = prompt_tokens(row["usage"])
        print(
            f"evict_test: done {name} ttft={row['ttft_s']:.3f}s "
            f"cached={row['cached_tokens']} prompt={row['prompt_tokens']} "
            f"text={row['text']!r}",
            flush=True,
        )
        return row

    def metrics() -> str:
        return get_text(args.base_url + "/metrics", timeout=30)

    cold = run("A-cold", prompts["A"])
    # A second step submits the deferred store from the cold prefill.
    flush_b = run("flush-after-cold", build_ids(tokenizer, "flush-cold", 32))
    time.sleep(3)
    resident = run("A-resident", prompts["A"])
    before_evict = metrics()
    run("B", prompts["B"])
    run("flush-after-B", build_ids(tokenizer, "flush-b", 32))
    time.sleep(3)
    run("C", prompts["C"])
    run("flush-after-C", build_ids(tokenizer, "flush-c", 32))
    time.sleep(3)
    before = metrics()
    restored = run("A-restored", prompts["A"])
    after = metrics()

    host_delta = counter_delta(
        before, after, "vllm:prompt_tokens_cached_by_source", 'source="host"'
    )
    device_delta = counter_delta(
        before, after, "vllm:prompt_tokens_cached_by_source", 'source="device"'
    )
    external_delta = counter_delta(before, after, "vllm:external_prefix_cache_hits")
    gpu_delta = counter_delta(before, after, "vllm:prefix_cache_hits")

    cached = restored["cached_tokens"]
    computed = None
    if cached is not None and restored["prompt_tokens"] is not None:
        computed = restored["prompt_tokens"] - cached

    same_control = restored["text"] == cold["text"]
    same_resident = resident["text"] == cold["text"]
    # One block of tail is the align-mode checkpoint. Allow one extra
    # block for hash alignment before calling it a recompute.
    tail_ok = computed is not None and computed <= 2 * block
    host_ok = host_delta >= boundary - block
    # The restore must not be a pure GPU hit. Device tokens may be a
    # short leftover; the host counter is the eviction proof.
    evicted_ok = device_delta < host_delta

    passed = bool(same_control and same_resident and tail_ok and host_ok and evicted_ok)
    report = {
        "block_size": block,
        "prompt_tokens_sent": n_tokens,
        "checkpoint_boundary": boundary,
        "cold_ttft_s": cold["ttft_s"],
        "resident_ttft_s": resident["ttft_s"],
        "restored_ttft_s": restored["ttft_s"],
        "cold_cached": cold["cached_tokens"],
        "resident_cached": resident["cached_tokens"],
        "restored_cached": cached,
        "restored_computed": computed,
        "host_tokens": host_delta,
        "device_tokens": device_delta,
        "external_hits": external_delta,
        "gpu_hits": gpu_delta,
        "cold_text": cold["text"],
        "resident_text": resident["text"],
        "restored_text": restored["text"],
        "same_control": same_control,
        "same_resident": same_resident,
        "tail_ok": tail_ok,
        "host_ok": host_ok,
        "evicted_ok": evicted_ok,
        "passed": passed,
        "flush_after_cold_cached": flush_b["cached_tokens"],
        "metrics_after": interesting_metrics(after),
        "gpu_hits_before_evict_sample": before_evict[:200],
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("evict_test: " + json.dumps({k: report[k] for k in report if k != "gpu_hits_before_evict_sample"}))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"evict_test: HTTP {exc.code} {body[:2000]}", file=sys.stderr)
        sys.exit(1)
