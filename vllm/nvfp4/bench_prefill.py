#!/usr/bin/env python3
"""Cold-prefill A/B bench for the NVFP4 TP=2 prefill campaign (Track 11g).

Focuses on PREFILL cost: each request uses a UNIQUE prompt (fresh random token
prefix) so prefix cache always MISSES -> every TTFT is a true cold prefill.
OUT is tiny (default 8) so decode barely contributes.

Reports, per input length and concurrency:
  TTFT (ms, cold prefill latency) and PP = prompt_tokens*1000/TTFT (tok/s).

Usage: bench_prefill.py <base_url> <model> [conc=1] [out=8] [lens=512,2048,8192] [reps=3]
"""
import sys, json, os, time, threading, urllib.request

BASE = sys.argv[1].rstrip("/"); MODEL = sys.argv[2]
NC   = int(sys.argv[3]) if len(sys.argv) > 3 else 1
OUT  = int(sys.argv[4]) if len(sys.argv) > 4 else 8
LENS = [int(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5 else ["512","2048","8192"])]
REPS = int(sys.argv[6]) if len(sys.argv) > 6 else 3
API_KEY = os.environ.get("API_KEY", "")
URL = f"{BASE}/chat/completions"

# A word pool; a unique prompt = a fresh pseudo-random draw (seeded per request id).
_WORDS = ("memory hierarchy systolic matrix quantization throughput latency kernel tensor "
          "collective allreduce prefill decode bandwidth register cache pipeline scheduler "
          "attention mamba hybrid checkpoint gradient partition overlap posted write fabric "
          "battlemage xe systolic dpas roofline microbench coherent needle context window").split()

def make_prompt(seed, approx_tokens):
    # ~1.3 words/token-ish for these short words; oversample then the server tokenizes exactly.
    import random
    rng = random.Random(seed)
    n_words = int(approx_tokens * 0.85)
    body = " ".join(rng.choice(_WORDS) for _ in range(n_words))
    return f"Analyze technical notes (id {seed}). {body}. Reply with the single word OK."

_seed_ctr = [0]
_seed_lock = threading.Lock()
def next_seed():
    with _seed_lock:
        _seed_ctr[0] += 1
        return _seed_ctr[0] * 100003 + int((time.time() * 1000) % 100000)

def one(approx_tokens, res, idx):
    prompt = make_prompt(next_seed(), approx_tokens)
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
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
    ttft = (first - t0) * 1000 if first else float("nan")
    res[idx] = dict(ttft=ttft, ptoks=ptoks, n=n)

def run(approx_tokens, N):
    res = [None] * N
    ths = [threading.Thread(target=one, args=(approx_tokens, res, i)) for i in range(N)]
    for t in ths: t.start()
    for t in ths: t.join()
    ok = [r for r in res if r and r["ttft"] == r["ttft"]]
    if not ok: return None
    ttft = sum(r["ttft"] for r in ok) / len(ok)
    ptoks = ok[0]["ptoks"]
    pp = ptoks * 1000 / ttft if ptoks and ttft else float("nan")
    return dict(ttft=ttft, pp=pp, ptoks=ptoks, nok=len(ok))

if __name__ == "__main__":
    print(f"model={MODEL} conc={NC} out={OUT} reps={REPS} (unique prompt/call = cold prefill)")
    for L in LENS:
        # one warm-nothing throwaway to reach steady clocks, then REPS measured
        _ = run(L, 1)
        best = None; rows = []
        for _r in range(REPS):
            r = run(L, NC)
            if r is None: continue
            rows.append(r)
            if best is None or r["ttft"] < best["ttft"]: best = r
        if not rows:
            print(f"IN~{L}: ALL FAILED"); continue
        avg_ttft = sum(x["ttft"] for x in rows) / len(rows)
        avg_pp   = sum(x["pp"] for x in rows) / len(rows)
        pt = rows[0]["ptoks"]
        print(f"IN~{L} (real_ptoks={pt}) c{NC}: TTFT avg={avg_ttft:.0f}ms best={best['ttft']:.0f}ms "
              f"| PP avg={avg_pp:.0f} best={best['pp']:.0f} tok/s")
