#!/usr/bin/env python3
"""T3: two decodes running while a parked conversation is restored.

Phase itl parks one conversation the same way T2 does, then decodes two
other streams. The restore is submitted only after each stream has 16
tokens. Baseline ITL is the gaps before that submit, excluding the first
gap. Window ITL is every later gap that overlaps the restore. p50 and p99
of the window must stay within 10 percent of that stream's baseline.
The restore itself must pass the T2 token gate.

Phase cycle parks four conversations, pushes them off the GPU with two
long prefills, then restores eight turns round-robin. Each restore must
match its earlier greedy control. Host slots stay at four because the
controls use other salts and are the entries the later parks evict.

This file does not start a server.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
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


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


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
        f"t3: done {name} ttft={result['ttft_s']:.3f}s "
        f"cached={cached} prompt={prompt} computed={computed} "
        f"out={len(result['token_ids'])}",
        flush=True,
    )
    return result


class Decode:
    """One streaming greedy completion. Token times are client receive times."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.times: list[float] = []
        self.tokens: list[int] = []
        self.error: str | None = None
        self.ready = threading.Event()
        self.done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(
        self,
        base: str,
        model: str,
        token_ids: list[int],
        max_tokens: int,
        salt: str,
        timeout: float,
        warmup: int,
    ) -> None:
        def run() -> None:
            payload = {
                "model": model,
                "prompt": token_ids,
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": 1,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "cache_salt": salt,
                "return_token_ids": True,
            }
            data = json.dumps(payload).encode("ascii")
            req = urllib.request.Request(
                base + "/v1/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
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
                        for choice in event.get("choices") or []:
                            ids = choice.get("token_ids") or []
                            if not ids and not (choice.get("text") or ""):
                                continue
                            now = time.monotonic()
                            for token in ids:
                                self.tokens.append(int(token))
                                self.times.append(now)
                            if ids and len(self.tokens) >= warmup:
                                self.ready.set()
            except Exception as exc:
                self.error = str(exc)
                print(f"t3: stream {self.name} error {exc}", flush=True)
            finally:
                self.ready.set()
                self.done.set()

        self._thread = threading.Thread(target=run, name=self.name, daemon=True)
        self._thread.start()

    def itls(self, t0: float, t1: float) -> tuple[list[float], list[float]]:
        base: list[float] = []
        window: list[float] = []
        for i in range(1, len(self.times)):
            gap = self.times[i] - self.times[i - 1]
            if self.times[i] < t0:
                base.append(gap)
            elif self.times[i - 1] < t1:
                window.append(gap)
        return base[1:], window


def _itl_ok(base: list[float], window: list[float]) -> tuple[bool, dict]:
    p50_b = _pct(base, 0.50)
    p99_b = _pct(base, 0.99)
    p50_w = _pct(window, 0.50)
    p99_w = _pct(window, 0.99)
    ok = (
        p50_b is not None
        and p99_b is not None
        and p50_w is not None
        and p99_w is not None
        and p50_b > 0
        and p99_b > 0
        and p50_w <= 1.10 * p50_b
        and p99_w <= 1.10 * p99_b
        and len(base) >= 8
        and len(window) >= 1
    )
    return ok, {
        "n_base": len(base),
        "n_window": len(window),
        "p50_base": p50_b,
        "p99_base": p99_b,
        "p50_window": p50_w,
        "p99_window": p99_w,
        "ok": ok,
    }


def _public(result: dict) -> dict:
    return {
        "cached_tokens": result.get("cached_tokens"),
        "prompt_tokens": result.get("prompt_tokens"),
        "computed": result.get("computed"),
        "ttft_s": result.get("ttft_s"),
        "n_out": len(result.get("token_ids") or []),
    }


def _metrics(base: str) -> str:
    return get_text(base + "/metrics", timeout=30)


def _restore_ok(restored: dict, expect: dict, gold_ids: list[int], before: str, after: str) -> dict:
    host = counter_delta(
        before, after, "vllm:prompt_tokens_cached_by_source", 'source="host"'
    )
    device = counter_delta(
        before, after, "vllm:prompt_tokens_cached_by_source", 'source="device"'
    )
    computed_ok = restored["computed"] == expect["computed"]
    cached_ok = restored["cached_tokens"] == expect["end"]
    host_ok = int(host) == expect["host"] - int(device) and host > 0
    greedy = restored["token_ids"] == gold_ids
    return {
        "computed_ok": computed_ok,
        "cached_ok": cached_ok,
        "host_ok": host_ok,
        "greedy_match": greedy,
        "host_delta": host,
        "device_delta": device,
        "ok": bool(computed_ok and cached_ok and host_ok and greedy),
        "restored": _public(restored),
    }


def _identity(base: str) -> list[str]:
    models = json.loads(get_text(base + "/v1/models", timeout=30))
    return [row.get("id") for row in models.get("data") or []]


def phase_itl(args, tokenizer, block: int) -> dict:
    room = args.max_model_len - args.reply_tokens - args.u2_tokens - 8
    if args.p1_tokens > room or args.p1_tokens % block == 0:
        raise RuntimeError(f"P1 {args.p1_tokens} does not fit block {block}")
    filler_blocks = min(28, (args.max_model_len - 32) // block)
    filler_n = filler_blocks * block
    p1 = build_ids(tokenizer, "t3-park-A", args.p1_tokens)
    u2 = _ids(9000, args.u2_tokens)
    fillers = {
        name: build_ids(tokenizer, "t3-" + name, filler_n) for name in ("B", "C")
    }

    def run(name: str, token_ids: list[int], salt: str, n: int) -> dict:
        print(f"t3: start {name} n={len(token_ids)} salt={salt}", flush=True)
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

    turn1 = run("turn1", p1, "t3-park", args.reply_tokens)
    if len(turn1["token_ids"]) != args.reply_tokens:
        return {"pass": False, "reason": "reply length", "turn1": _public(turn1)}
    expect = turn2_accounting(p1=len(p1), r1=len(turn1["token_ids"]), u2=len(u2))
    turn2 = p1 + turn1["token_ids"] + u2
    control = run("control", turn2, "t3-control", args.reply_tokens)
    run("B", fillers["B"], "t3-fill-b", 1)
    run("C", fillers["C"], "t3-fill-c", 1)

    streams = []
    for i in range(2):
        prompt = build_ids(tokenizer, f"t3-active-{i}", 128)
        dec = Decode(f"active-{i}")
        dec.start(
            args.base_url,
            args.model,
            prompt,
            args.decode_tokens,
            f"t3-active-{i}",
            args.request_timeout,
            warmup=16,
        )
        streams.append(dec)
    if not all(dec.ready.wait(args.request_timeout) for dec in streams):
        return {"pass": False, "reason": "decode warmup timeout"}
    for dec in streams:
        if dec.error:
            return {"pass": False, "reason": dec.error}

    before = _metrics(args.base_url)
    t0 = time.monotonic()
    restored = run("restored", turn2, "t3-park", args.reply_tokens)
    t1 = time.monotonic()
    after = _metrics(args.base_url)
    gate = _restore_ok(restored, expect, control["token_ids"], before, after)
    itl = {}
    itl_pass = True
    for dec in streams:
        base, window = dec.itls(t0, t1)
        ok, detail = _itl_ok(base, window)
        itl[dec.name] = detail
        itl_pass = itl_pass and ok
        print(
            f"t3: itl {dec.name} base_n={detail['n_base']} "
            f"win_n={detail['n_window']} p50 {detail['p50_base']} -> "
            f"{detail['p50_window']} p99 {detail['p99_base']} -> "
            f"{detail['p99_window']} ok={ok}",
            flush=True,
        )
    alive = all(
        dec.error is None and any(ts >= t0 for ts in dec.times) for dec in streams
    )
    passed = bool(gate["ok"] and itl_pass and alive)
    print(f"t3: ITL_RESULT pass={passed}", flush=True)
    return {
        "pass": passed,
        "expect": expect,
        "gate": gate,
        "itl": itl,
        "streams_alive": alive,
        "control_ttft_s": control["ttft_s"],
        "restored_ttft_s": restored["ttft_s"],
    }


def phase_cycle(args, tokenizer, block: int) -> dict:
    p1_n = 300
    if p1_n % block == 0:
        p1_n += 1
    reply_n = 32
    u2_n = 9
    prompts = [build_ids(tokenizer, f"t3-cycle-{i}", p1_n) for i in range(4)]
    replies: list[list[int]] = []
    for i, prompt in enumerate(prompts):
        row = _row(
            f"gold-turn1-{i}",
            complete(
                args.base_url,
                args.model,
                prompt,
                reply_n,
                args.request_timeout,
                cache_salt=f"t3-gold-{i}",
                return_token_ids=True,
            ),
        )
        if len(row["token_ids"]) != reply_n:
            return {"pass": False, "reason": f"gold reply {i}"}
        replies.append(row["token_ids"])

    golds: list[list[int]] = []
    turns: list[tuple[int, list[int], dict]] = []
    for t in range(8):
        conv = t % 4
        u2 = _ids(12000 + t * 100, u2_n)
        turn2 = prompts[conv] + replies[conv] + u2
        expect = turn2_accounting(p1=p1_n, r1=reply_n, u2=u2_n)
        gold = _row(
            f"gold-{t}",
            complete(
                args.base_url,
                args.model,
                turn2,
                reply_n,
                args.request_timeout,
                cache_salt=f"t3-gold-c-{t}",
                return_token_ids=True,
            ),
        )
        golds.append(gold["token_ids"])
        turns.append((conv, turn2, expect))

    for i, prompt in enumerate(prompts):
        row = _row(
            f"park-turn1-{i}",
            complete(
                args.base_url,
                args.model,
                prompt,
                reply_n,
                args.request_timeout,
                cache_salt=f"t3-cpark-{i}",
                return_token_ids=True,
            ),
        )
        if row["token_ids"] != replies[i]:
            return {"pass": False, "reason": f"park reply drifted {i}"}

    active_n = min(24 * block, args.max_model_len - 64)
    streams = []
    for i in range(2):
        dec = Decode(f"cycle-active-{i}")
        dec.start(
            args.base_url,
            args.model,
            build_ids(tokenizer, f"t3-long-{i}", active_n),
            400,
            f"t3-long-{i}",
            args.request_timeout,
            warmup=1,
        )
        streams.append(dec)
    if not all(dec.ready.wait(args.request_timeout) for dec in streams):
        return {"pass": False, "reason": "cycle decode did not start"}

    results = []
    all_ok = True
    for t, (conv, turn2, expect) in enumerate(turns):
        before = _metrics(args.base_url)
        restored = _row(
            f"cycle-{t}",
            complete(
                args.base_url,
                args.model,
                turn2,
                reply_n,
                args.request_timeout,
                cache_salt=f"t3-cpark-{conv}",
                return_token_ids=True,
            ),
        )
        after = _metrics(args.base_url)
        gate = _restore_ok(restored, expect, golds[t], before, after)
        gate["turn"] = t
        gate["conv"] = conv
        results.append(gate)
        all_ok = all_ok and gate["ok"]
        print(f"t3: cycle turn={t} conv={conv} ok={gate['ok']}", flush=True)
    alive = all(len(dec.tokens) > 1 and dec.error is None for dec in streams)
    passed = bool(all_ok and alive)
    print(f"t3: CYCLE_RESULT pass={passed}", flush=True)
    return {"pass": passed, "turns": results, "streams_alive": alive, "active_prompt": active_n}


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
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--ready-timeout", type=float, default=3600)
    parser.add_argument("--phase", choices=("itl", "cycle", "both"), default="itl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    log_text = Path(args.serve_log).read_text(errors="replace")
    block = block_size_from_log(log_text)
    if block is None:
        print("t3: block size not in the serve log yet", file=sys.stderr)
        return 2
    wait_ready(args.base_url, args.ready_timeout)
    ids = _identity(args.base_url)
    if not ids or ids[0] != "hotschmoe-dev":
        print(f"t3: identity {ids}", file=sys.stderr)
        return 2
    print(f"t3: identity {ids} block={block} phase={args.phase}", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    report: dict = {"identity": ids, "block_size": block, "phase": args.phase}
    try:
        if args.phase in ("itl", "both"):
            report["itl"] = phase_itl(args, tokenizer, block)
        if args.phase in ("cycle", "both"):
            report["cycle"] = phase_cycle(args, tokenizer, block)
    except Exception as exc:
        report["pass"] = False
        report["error"] = str(exc)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"t3: error {exc}", file=sys.stderr)
        return 1

    parts = []
    if "itl" in report:
        parts.append(bool(report["itl"].get("pass")))
    if "cycle" in report:
        parts.append(bool(report["cycle"].get("pass")))
    report["pass"] = bool(parts) and all(parts)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"t3: RESULT pass={report['pass']}", flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
