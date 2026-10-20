#!/usr/bin/env python3
"""T2: a real second turn, then the same turn after GPU eviction.

Turn 1 is prompt P1 and greedy reply R1. Turn 2 is P1 + R1 + a new user
message U2. The control sends that turn 2 with a different cache salt, so
it does not see the parked tail. B and C then push A's GPU blocks out.
The restored turn 2 must compute only len(U2)+1 tokens (the reply token
that was sampled and never fed, plus U2). The host counter covers E =
len(P1)+len(R1)-1, minus whatever the device still holds.

Pass when replay is 0, the host counter matches, greedy token ids match
the control, and restore TTFT is below the cold control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kvoffload.evict_test import (
    block_size_from_log,
    build_ids,
    cached_tokens,
    complete,
    get_text,
    prompt_tokens,
    wait_ready,
)
from kvoffload.index import turn2_accounting
from kvoffload.metrics import counter_delta


def _ids(start: int, n: int) -> list[int]:
    return list(range(start, start + n))


def _row(name: str, result: dict) -> dict:
    cached = cached_tokens(result["usage"])
    prompt = prompt_tokens(result["usage"])
    computed = None if cached is None or prompt is None else prompt - cached
    result = dict(result)
    result["name"] = name
    result["cached_tokens"] = cached
    result["prompt_tokens"] = prompt
    result["computed"] = computed
    print(
        f"turn_test: done {name} ttft={result['ttft_s']:.3f}s "
        f"cached={cached} prompt={prompt} computed={computed} "
        f"out={len(result['token_ids'])} text={result['text'][:80]!r}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18111")
    parser.add_argument("--model", default="hotschmoe-dev")
    parser.add_argument("--serve-log", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--reply-tokens", type=int, default=64)
    parser.add_argument("--u2-tokens", type=int, default=17)
    parser.add_argument("--p1-tokens", type=int, default=3000)
    parser.add_argument("--ready-timeout", type=float, default=3600)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    log_text = Path(args.serve_log).read_text(errors="replace")
    block = block_size_from_log(log_text)
    if block is None:
        print("turn_test: block size not in the serve log yet", file=sys.stderr)
        return 2
    if args.p1_tokens % block == 0:
        print("turn_test: P1 must not be a whole number of blocks", file=sys.stderr)
        return 2
    room = args.max_model_len - args.reply_tokens - args.u2_tokens - 8
    if args.p1_tokens > room:
        print(f"turn_test: P1 {args.p1_tokens} does not fit in {room}", file=sys.stderr)
        return 2

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    filler_blocks = min(28, (args.max_model_len - 32) // block)
    if filler_blocks < 4:
        print(f"turn_test: block {block} leaves no filler room", file=sys.stderr)
        return 2
    filler_n = filler_blocks * block

    p1 = build_ids(tokenizer, "park-A", args.p1_tokens)
    u2 = _ids(8000, args.u2_tokens)
    fillers = {
        name: build_ids(tokenizer, name, filler_n) for name in ("B", "C", "cold")
    }
    print(
        f"turn_test: block={block} p1={len(p1)} u2={len(u2)} "
        f"filler={filler_n} reply={args.reply_tokens}",
        flush=True,
    )
    wait_ready(args.base_url, args.ready_timeout)
    models = json.loads(get_text(args.base_url + "/v1/models", timeout=30))
    ids = [row.get("id") for row in models.get("data") or []]
    if not ids or ids[0] != "hotschmoe-dev":
        print(f"turn_test: identity {ids}", file=sys.stderr)
        return 2
    print(f"turn_test: identity {ids}", flush=True)

    def run(name: str, token_ids: list[int], salt: str, n: int) -> dict:
        print(f"turn_test: start {name} n={len(token_ids)} salt={salt}", flush=True)
        return _row(
            name,
            complete(
                args.base_url,
                args.model,
                token_ids,
                n,
                args.request_timeout,
                cache_salt=salt,
                return_token_ids=True,
            ),
        )

    def metrics() -> str:
        return get_text(args.base_url + "/metrics", timeout=30)

    turn1 = run("turn1", p1, "park", args.reply_tokens)
    r1 = turn1["token_ids"]
    if len(r1) != args.reply_tokens:
        print(f"turn_test: reply length {len(r1)} != {args.reply_tokens}", file=sys.stderr)
        _write(args.out, {"pass": False, "reason": "reply length", "turn1": turn1})
        return 1
    expect = turn2_accounting(p1=len(p1), r1=len(r1), u2=len(u2))
    turn2 = p1 + r1 + u2
    # The host park has four slots. A, the salted control, B, and C fill
    # them. Extra flush prompts would be a fifth conversation and would
    # evict A before the restore. The next request's engine step submits
    # the deferred tail copy (has_pending_push_work keeps the engine
    # stepping while that copy is queued).
    control = run("control", turn2, "control", args.reply_tokens)
    run("B", fillers["B"], "fill-b", 1)
    run("C", fillers["C"], "fill-c", 1)
    time.sleep(1)

    before = metrics()
    restored = run("restored", turn2, "park", args.reply_tokens)
    after = metrics()
    cold = run("cold-ttft", fillers["cold"], "cold", 1)

    host = counter_delta(
        before, after, "vllm:prompt_tokens_cached_by_source", 'source="host"'
    )
    device = counter_delta(
        before, after, "vllm:prompt_tokens_cached_by_source", 'source="device"'
    )
    load_bytes = counter_delta(before, after, "vllm:kv_offload_load_bytes")
    load_time = counter_delta(before, after, "vllm:kv_offload_load_time")
    computed = restored["computed"]
    cached = restored["cached_tokens"]
    greedy_match = restored["token_ids"] == control["token_ids"]
    text_match = restored["text"] == control["text"]
    computed_ok = computed == expect["computed"]
    # cached_tokens is the whole hit (device prefix plus host). The host
    # counter is only the part that was not already on the card.
    cached_ok = cached == expect["end"]
    host_ok = int(host) == expect["host"] - int(device) and host > 0
    replay_ok = computed_ok
    ttft_ok = restored["ttft_s"] < control["ttft_s"]
    passed = bool(
        greedy_match and text_match and computed_ok and cached_ok and host_ok and ttft_ok
    )
    report = {
        "pass": passed,
        "block_size": block,
        "expect": expect,
        "identity": ids,
        "host_delta": host,
        "device_delta": device,
        "load_bytes": load_bytes,
        "load_time_s": load_time,
        "computed_ok": computed_ok,
        "cached_ok": cached_ok,
        "host_ok": host_ok,
        "replay_ok": replay_ok,
        "greedy_match": greedy_match,
        "text_match": text_match,
        "ttft_ok": ttft_ok,
        "control_ttft_s": control["ttft_s"],
        "restored_ttft_s": restored["ttft_s"],
        "cold_ttft_s": cold["ttft_s"],
        "turn1": _public(turn1),
        "control": _public(control),
        "restored": _public(restored),
    }
    _write(args.out, report)
    print(
        f"turn_test: pass={passed} host={host} device={device} "
        f"computed={computed} expect={expect['computed']} "
        f"load_bytes={load_bytes} load_s={load_time}",
        flush=True,
    )
    return 0 if passed else 1


def _public(row: dict) -> dict:
    return {
        key: row.get(key)
        for key in (
            "name",
            "text",
            "token_ids",
            "ttft_s",
            "cached_tokens",
            "prompt_tokens",
            "computed",
            "wall_s",
            "usage",
        )
    }


def _write(path: str, report: dict) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
