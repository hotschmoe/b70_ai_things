#!/usr/bin/env python3
"""Dependency-free streaming decode t/s probe (urllib + threads). Measures single-stream decode
throughput and, for N>1, aggregate concurrent decode t/s. Usage:
  perf_probe.py <base_url> <model> <N> [max_tokens=128]
"""
import sys, json, os, time, threading, urllib.request

BASE = sys.argv[1].rstrip("/"); MODEL = sys.argv[2]
N = int(sys.argv[3]); M = int(sys.argv[4]) if len(sys.argv) > 4 else 128
API_KEY = os.environ.get("API_KEY", "")
URL = f"{BASE}/chat/completions"
PROMPT = ("Write a thorough, detailed technical essay about the history and architecture of modern "
          "GPUs, covering memory hierarchies, systolic arrays, and quantization. Be comprehensive.")

def one(idx, res):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": M, "temperature": 0.7, "stream": True}).encode()
    hdr = {"content-type": "application/json"}
    if API_KEY: hdr["Authorization"] = f"Bearer {API_KEY}"
    first = None; n = 0
    try:
        req = urllib.request.Request(URL, data=body, headers=hdr)
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                try: d = json.loads(data)
                except Exception: continue
                dl = d.get("choices", [{}])[0].get("delta", {}) or {}
                delta = dl.get("content") or dl.get("reasoning_content")  # count thinking tokens too
                if delta:
                    if first is None: first = time.perf_counter()
                    n += 1
    except Exception as e:
        res[idx] = ("err", str(e)); return
    res[idx] = (first, time.perf_counter(), n)

def main():
    res = {}; ths = []
    t0 = time.perf_counter()
    for i in range(N):
        th = threading.Thread(target=one, args=(i, res)); th.start(); ths.append(th)
    for th in ths: th.join()
    ok = [v for v in res.values() if isinstance(v[0], float) and v[2] > 1]
    if not ok:
        print(f"N={N} FAILED:", list(res.values())[:2]); return
    total = sum(v[2]-1 for v in ok)
    window = max(v[1] for v in ok) - min(v[0] for v in ok)
    agg = total/window if window > 0 else 0
    per = [ (v[2]-1)/(v[1]-v[0]) for v in ok if v[1] > v[0] ]
    mean_per = sum(per)/len(per)
    print(f"N={N:<2} ok={len(ok):<2} | AGG decode={agg:6.1f} t/s | mean_per_stream={mean_per:5.1f} t/s | tok={total} window={window:.2f}s")

main()
