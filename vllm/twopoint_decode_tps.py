#!/usr/bin/env python3
"""Reliable decode-only t/s at a deep context via the TWO-POINT method: at a fixed context,
time a short generation (N1 tokens) and a long one (N2 tokens); decode t/s = (N2-N1)/(t2-t1).
The identical prefix means prefill (and TTFT) cancels in the subtraction, so the result is
pure decode throughput -- robust to the multi-second deep prefill that pollutes an end-to-end
tok/s. A warmup request primes the prefix cache so BOTH timed runs see the same (warm) prefill.

Usage: twopoint_decode_tps.py <base_url> <model> [depth_ktok=40] [N1=64] [N2=384]
"""
import sys, os, json, time, glob, urllib.request

BASE = sys.argv[1].rstrip("/"); MODEL = sys.argv[2]
DEPTH = (int(sys.argv[3]) if len(sys.argv) > 3 else 40) * 1000
N1 = int(sys.argv[4]) if len(sys.argv) > 4 else 64
N2 = int(sys.argv[5]) if len(sys.argv) > 5 else 384
API_KEY = os.environ.get("API_KEY", "") or os.environ.get("B70_API_KEY", "")
CHARS_PER_TOK = 3.6


def ctx_prompt(depth):
    files = []
    for pat in ("vllm/*.py", "sglang/patches/*.py", "bin/*.sh"):
        files += sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", pat)))
    c = "".join(open(f, errors="ignore").read() for f in files)
    while len(c) < 500_000: c += c
    return ("Study this codebase excerpt then write and thoroughly explain a Python function "
            "summarize_kv_cache(logs) parsing vLLM KV-cache log lines.\n\n```\n"
            + c[: int(depth * CHARS_PER_TOK)] + "\n```\n")


def timed(prompt, maxtok):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": maxtok, "temperature": 0, "stream": False,
                       "ignore_eos": True}).encode()
    hdr = {"content-type": "application/json"}
    if API_KEY: hdr["Authorization"] = f"Bearer {API_KEY}"
    t0 = time.perf_counter()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=body, headers=hdr)
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.perf_counter() - t0
    return dt, r.get("usage", {}).get("completion_tokens", 0)


def main():
    p = ctx_prompt(DEPTH)
    timed(p, 8)                       # warmup: prime the prefix cache
    t1, c1 = timed(p, N1)
    t2, c2 = timed(p, N2)
    dtok = c2 - c1; dtime = t2 - t1
    tps = dtok / dtime if dtime > 0 else float("nan")
    print(f"model={MODEL} depth~{DEPTH}  N1={c1}({t1:.2f}s) N2={c2}({t2:.2f}s)  "
          f"-> DECODE {tps:.1f} tok/s  ({dtok} tok / {dtime:.2f}s)")


main()
