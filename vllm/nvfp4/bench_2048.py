#!/usr/bin/env python3
"""Warm IN~=2048 / OUT=128 bench matching the README perf table.
Streams chat/completions, measures TTFT (first-token latency) and per-stream decode t/s.
PP = prompt_tokens * 1000 / TTFT_ms (prompt-processing throughput). Reports c1 and cN.

Usage: bench_2048.py <base_url> <model> [N_concurrent=4] [out_tokens=128]
  e.g. bench_2048.py http://localhost:8078/v1 qwen3.6-27b-NVFP4-modelopt-fused 4
"""
import sys, json, os, time, threading, urllib.request

BASE = sys.argv[1].rstrip("/"); MODEL = sys.argv[2]
NC = int(sys.argv[3]) if len(sys.argv) > 3 else 4
OUT = int(sys.argv[4]) if len(sys.argv) > 4 else 128
API_KEY = os.environ.get("API_KEY", "")
URL = f"{BASE}/chat/completions"

# ~2048-token prompt: repeat a dense paragraph until long enough (~4 chars/token).
_PARA = ("In modern GPU architecture the memory hierarchy, systolic matrix arrays, and low-precision "
         "quantization interact in subtle ways that determine end-to-end inference throughput. ")
PROMPT = ("Summarize and critique the following technical notes in detail.\n\n" + _PARA * 65).strip()

def one(idx, res):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": OUT, "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    hdr = {"content-type": "application/json"}
    if API_KEY: hdr["Authorization"] = f"Bearer {API_KEY}"
    t0 = time.time(); first = None; n = 0; ptoks = None
    req = urllib.request.Request(URL, data=body, headers=hdr)
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: obj = json.loads(data)
            except Exception: continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                if first is None: first = time.time()
                n += 1
            if obj.get("usage"): ptoks = obj["usage"].get("prompt_tokens")
    end = time.time()
    ttft = (first - t0) * 1000 if first else float("nan")
    dec = n / (end - first) if first and end > first else float("nan")
    res[idx] = dict(ttft=ttft, dec=dec, n=n, ptoks=ptoks)

def run(N):
    res = [None] * N
    ths = [threading.Thread(target=one, args=(i, res)) for i in range(N)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    ok = [r for r in res if r]
    ttft = sum(r["ttft"] for r in ok) / len(ok)
    dec = sum(r["dec"] for r in ok) / len(ok)
    agg = sum(r["dec"] for r in ok)
    ptoks = ok[0]["ptoks"]
    pp = ptoks * 1000 / ttft if ptoks and ttft == ttft else float("nan")
    return dict(N=N, ttft=ttft, pp=pp, dec=dec, agg=agg, ptoks=ptoks, wall=wall)

if __name__ == "__main__":
    print(f"model={MODEL} OUT={OUT}")
    # warmup
    _w = [None]; one(0, _w)
    print(f"warmup: prompt_tokens={_w[0]['ptoks']} ttft={_w[0]['ttft']:.0f}ms dec={_w[0]['dec']:.2f} t/s")
    for N in (1, NC):
        r = run(N)
        print(f"c{N}: prompt_tokens={r['ptoks']} TTFT={r['ttft']:.0f}ms "
              f"PP={r['pp']:.0f} tok/s  TG/stream={r['dec']:.2f} t/s  agg={r['agg']:.2f} t/s")
