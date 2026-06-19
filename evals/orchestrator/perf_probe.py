#!/usr/bin/env python3
"""Single-stream performance probe: prefill tok/s (pp), TTFT, and decode tok/s (tg).

Quality is what the tiers measure; this is the SPEED companion. Streams from the OpenAI-compatible
endpoint and times it client-side (concurrency 1, greedy). Warms up first (the oneDNN int8 GEMM
JIT-compiles on the first request -> a cold TTFT that would skew the numbers; see RESULTS.md caveat).

  pp  = prompt_tokens / TTFT_long   (long prompt, max_tokens=1) -> prefill throughput
  tg  = (out_tokens-1) / (t_end - t_first)   (short prompt, max_tokens=N) -> decode throughput
  ttft= t_first - t_send             (short prompt) -> latency to first token

Usage:  perf_probe.py <base_url> <served_model_id> [quant_label]
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import make_client  # noqa: E402


def tok_count(base_url: str, model: str, text: str) -> int:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    req = urllib.request.Request(
        root + "/tokenize",
        data=json.dumps({"model": model, "prompt": text, "add_special_tokens": True}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["count"]


def _stream_first_and_count(client, model, prompt, max_tokens):
    t0 = time.monotonic()
    first = None
    n = 0
    s = client.completions.create(model=model, prompt=prompt, max_tokens=max_tokens,
                                  temperature=0, stream=True)
    for ch in s:
        if ch.choices and ch.choices[0].text:
            now = time.monotonic()
            if first is None:
                first = now
            n += 1
    return t0, first, n, time.monotonic()


def measure_decode(client, model, n_out=128, reps=3):
    ttfts, tgs = [], []
    for _ in range(reps):
        t0, first, n, t_end = _stream_first_and_count(
            client, model, "Write a detailed essay about the history of the printing press.", n_out)
        if first:
            ttfts.append((first - t0) * 1000)
            if n > 1:
                tgs.append((n - 1) / (t_end - first))
    return statistics.median(ttfts), statistics.median(tgs)


def measure_prefill(client, base_url, model, reps=3):
    prompt = "The history of computing is long and full of surprising turns. " * 360
    pt = tok_count(base_url, model, prompt)
    ttfts = []
    for _ in range(reps):
        t0, first, _, _ = _stream_first_and_count(client, model, prompt, 1)
        if first:
            ttfts.append(first - t0)
    ttft = statistics.median(ttfts)
    return pt, ttft * 1000, pt / ttft


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: perf_probe.py <base_url> <model> [label]", file=sys.stderr)
        return 2
    base_url, model = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else model
    client = make_client(base_url)
    # warmup (JIT-cold first request)
    _stream_first_and_count(client, model, "Hello, world.", 8)
    ttft_ms, tg = measure_decode(client, model)
    pt, pf_ttft_ms, pp = measure_prefill(client, base_url, model)
    row = {"label": label, "model": model, "decode_tg_tok_s": round(tg, 2),
           "ttft_ms": round(ttft_ms, 1), "prefill_prompt_tokens": pt,
           "prefill_ttft_ms": round(pf_ttft_ms, 1), "prefill_pp_tok_s": round(pp, 1)}
    print(json.dumps(row))
    print(f"[perf] {label:8s}  decode {row['decode_tg_tok_s']:>6} t/s | ttft {row['ttft_ms']:>7} ms | "
          f"prefill {row['prefill_pp_tok_s']:>7} t/s ({pt} tok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
