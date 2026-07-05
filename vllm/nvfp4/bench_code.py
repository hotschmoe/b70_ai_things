#!/usr/bin/env python3
"""Real-code decode A/B (ROBUST, non-streaming, usage-based).

Sends genuine coding prompts (long, predictable output = high MTP acceptance = the real coding
workload) NON-streaming and computes decode t/s = usage.completion_tokens / wall. The prompt is short
(~60 tok, prefill ~0.03s at 2000 t/s) so wall is ~all decode -> tps error < 1%. Counts ALL generated
tokens (reasoning + content) via usage, so it is immune to the streaming/reasoning-parser delta split.

Usage: bench_code.py <base_url> <model> [conc=1] [out=256] [reps=3]
"""
import sys, json, os, time, threading, urllib.request

BASE = sys.argv[1].rstrip("/"); MODEL = sys.argv[2]
NC   = int(sys.argv[3]) if len(sys.argv) > 3 else 1
OUT  = int(sys.argv[4]) if len(sys.argv) > 4 else 256
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
API_KEY = os.environ.get("API_KEY", "")
URL = f"{BASE}/chat/completions"

PROMPTS = [
    "Implement a complete LRU cache in Python with O(1) get/put via a doubly linked list and dict. "
    "Include type hints, docstrings, and usage examples. Write the full code.",
    "Write a thread-safe bounded blocking queue in Python using threading.Condition (put/get block "
    "when full/empty). Full type hints, docstrings, and a producer/consumer demo.",
    "Implement Dijkstra's shortest path in Python with a binary heap and a Graph class with add_edge. "
    "Type hints, docstrings, and a worked example.",
]

def one(idx, res):
    prompt = PROMPTS[idx % len(PROMPTS)]
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": OUT, "temperature": 0.0}).encode()
    hdr = {"content-type": "application/json"}
    if API_KEY: hdr["Authorization"] = f"Bearer {API_KEY}"
    t0 = time.time()
    try:
        req = urllib.request.Request(URL, data=body, headers=hdr)
        with urllib.request.urlopen(req, timeout=600) as r:
            obj = json.load(r)
    except Exception as e:
        res[idx] = dict(err=str(e)[:80]); return
    wall = time.time() - t0
    u = obj.get("usage") or {}
    ct = u.get("completion_tokens")
    res[idx] = dict(wall=wall, ct=ct, tps=(ct/wall if ct and wall else float("nan")))

def run(N):
    res = [None]*N
    ths = [threading.Thread(target=one, args=(i, res)) for i in range(N)]
    for t in ths: t.start()
    for t in ths: t.join()
    ok = [r for r in res if r and r.get("tps") == r.get("tps") and r.get("tps")]
    errs = [r["err"] for r in res if r and r.get("err")]
    if errs: print("   errs:", errs[:2])
    if not ok: return None
    return dict(tps=sum(r["tps"] for r in ok)/len(ok), agg=sum(r["tps"] for r in ok),
                ct=sum(r["ct"] for r in ok)//len(ok), wall=sum(r["wall"] for r in ok)/len(ok))

if __name__ == "__main__":
    print(f"model={MODEL} conc={NC} out={OUT} reps={REPS} (real coding prompts, non-streaming usage-based)")
    _ = run(1)
    rows = [r for r in (run(NC) for _ in range(REPS)) if r]
    if rows:
        best = max(rows, key=lambda x: x["tps"])
        print(f"c{NC}: decode TG/stream avg={sum(x['tps'] for x in rows)/len(rows):.1f} best={best['tps']:.1f} t/s"
              f" | agg={sum(x['agg'] for x in rows)/len(rows):.1f} | out~{rows[0]['ct']} tok | wall~{rows[0]['wall']:.1f}s")
    else:
        print(f"c{NC}: ALL FAILED")
