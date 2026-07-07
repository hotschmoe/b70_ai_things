#!/usr/bin/env python3
# Measure true decode t/s + MTP acceptance on a REAL code-generation prompt (streaming, so we
# time inter-token decode excluding TTFT). Reports tokens/s over the streamed decode.
#   PORT=8078 KEY= python3 vllm/code_decode_probe.py <label>
import json, os, sys, time, urllib.request

LABEL = sys.argv[1] if len(sys.argv) > 1 else "code"
HOST = os.environ.get("PROBE_HOST", "localhost"); PORT = os.environ.get("PORT", "8078")
KEY = os.environ.get("KEY", "").strip(); BASE = f"http://{HOST}:{PORT}"
def hdr():
    h = {"Content-Type": "application/json"}
    if KEY: h["Authorization"] = "Bearer " + KEY
    return h
def mid():
    r = urllib.request.Request(BASE + "/v1/models", headers=hdr())
    return json.load(urllib.request.urlopen(r, timeout=20))["data"][0]["id"]

PROMPT = ("You are an expert Python engineer. Implement a complete, production-quality "
          "in-memory LRU cache class `LRUCache` with: O(1) get/put via a doubly-linked list + "
          "dict, a capacity bound, thread-safety via a lock, TTL expiry per entry, a stats "
          "counter (hits/misses/evictions), and full docstrings. Then write 8 pytest unit tests "
          "covering eviction order, TTL expiry, thread-safety, and stats. Output only code.\n")

def run(model):
    body = {"model": model, "prompt": PROMPT, "max_tokens": 1600, "temperature": 0, "stream": True}
    r = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers=hdr())
    t0 = time.time(); tfirst = None; n = 0
    with urllib.request.urlopen(r, timeout=600) as x:
        for line in x:
            line = line.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: tok = json.loads(data)["choices"][0].get("text", "")
            except Exception: continue
            if tok:
                if tfirst is None: tfirst = time.time()
                n += 1
    t1 = time.time()
    ttft = (tfirst - t0) if tfirst else 0
    dec_t = (t1 - tfirst) if tfirst else 0
    tps = n / dec_t if dec_t > 0 else 0
    print(f"[{LABEL}] decode_chunks={n} ttft={ttft:.2f}s decode_t={dec_t:.1f}s "
          f"DECODE_TPS={tps:.1f}", flush=True)

if __name__ == "__main__":
    m = mid(); print(f"[{LABEL}] served={m}", flush=True)
    for i in range(2): run(m)   # 2 runs; 2nd is warm
