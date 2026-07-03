#!/usr/bin/env python3
"""Clean DECODE-only t/s at a deep context: streams the response and clocks from the FIRST
generated token to the last (excludes prefill/TTFT), so it measures true decode throughput
at depth -- the metric that decides a spec-decode config, unlike an end-to-end tok/s that a
multi-second deep prefill dominates. Also reports TTFT. Usage:
  deep_decode_tps.py <base_url> <model> [depth_ktok=40] [max_tokens=400]
"""
import sys, os, json, time, glob, urllib.request

BASE = sys.argv[1].rstrip("/"); MODEL = sys.argv[2]
DEPTH = int(sys.argv[3]) * 1000 if len(sys.argv) > 3 else 40000
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 400
API_KEY = os.environ.get("API_KEY", "") or os.environ.get("B70_API_KEY", "")
CHARS_PER_TOK = 3.6


def corpus():
    files = []
    for pat in ("vllm/*.py", "sglang/patches/*.py", "bin/*.sh"):
        files += sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", pat)))
    c = "".join(open(f, errors="ignore").read() for f in files)
    while len(c) < 500_000: c += c
    return c


def run(depth):
    ctx = corpus()[: int(depth * CHARS_PER_TOK)]
    user = ("Study this codebase excerpt, then write a new Python function "
            "`summarize_kv_cache(logs: list[str]) -> dict` that parses vLLM KV-cache init "
            "log lines into {available_gib, kv_tokens, max_concurrency}. Explain your approach "
            "in detail first, then give the function.\n\n```\n" + ctx + "\n```\n")
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": user}],
                       "max_tokens": MAXTOK, "temperature": 0, "stream": True}).encode()
    hdr = {"content-type": "application/json"}
    if API_KEY: hdr["Authorization"] = f"Bearer {API_KEY}"
    t0 = time.perf_counter(); first = None; last = None; n = 0
    req = urllib.request.Request(f"{BASE}/chat/completions", data=body, headers=hdr)
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: d = json.loads(data)
            except Exception: continue
            dl = (d.get("choices", [{}])[0].get("delta", {}) or {})
            piece = dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")
            if piece:
                now = time.perf_counter()
                if first is None: first = now
                last = now; n += 1
    ttft = (first - t0) if first else float("nan")
    dec = (last - first) if (first and last and last > first) else float("nan")
    tps = (n - 1) / dec if dec and dec > 0 else float("nan")
    return ttft, tps, n


def main():
    ttft, tps, n = run(DEPTH)
    print(f"model={MODEL} depth~{DEPTH} tok  |  TTFT={ttft*1000:.0f} ms  DECODE={tps:.1f} tok/s  (gen {n} tok)")


main()
