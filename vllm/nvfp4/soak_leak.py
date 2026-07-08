#!/usr/bin/env python3
# soak_leak.py -- self-contained forced-decode soak to trigger/rule-out the NEO linear_stream leak.
# Fires /v1/completions with ignore_eos + big max_tokens so each request forces N decode tokens
# regardless of prompt (no tokenizer dependency). Single-stream by default; CONC>1 for concurrent.
# Baseline (unfixed) NVFP4 TP=2 captured+MTP crashes at ~8-12k tokens single-stream (~6 min).
# Env: BASE, KEY, TOKENS_PER (3000), TARGET (30000), CONC (1), LABEL.
import json, os, sys, time, threading, urllib.request, urllib.error

BASE = os.environ.get("BASE", "http://192.168.10.5:18080")
KEY = os.environ.get("KEY", "")
TOKENS_PER = int(os.environ.get("TOKENS_PER", "3000"))
TARGET = int(os.environ.get("TARGET", "30000"))
CONC = int(os.environ.get("CONC", "1"))
LABEL = os.environ.get("LABEL", "soak")
PROMPT = ("Write an extremely long, exhaustive, step-by-step technical essay on the complete history of "
          "computing hardware and software, decade by decade, sparing no detail. Enumerate architectures, "
          "instruction sets, operating systems, languages, and networking. Keep going with ever more detail.")
H = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}

def _mid():
    r = urllib.request.Request(BASE + "/v1/models", headers=H)
    return json.load(urllib.request.urlopen(r, timeout=15))["data"][0]["id"]

MID = _mid()
print(f"[{LABEL}] served={MID} TOKENS_PER={TOKENS_PER} TARGET={TARGET} CONC={CONC}", flush=True)

total = {"tok": 0}
lock = threading.Lock()
t0 = time.time()
crashed = threading.Event()

def worker(wid):
    i = 0
    while not crashed.is_set():
        with lock:
            if total["tok"] >= TARGET:
                return
        body = {"model": MID, "prompt": PROMPT + f" (worker {wid} request {i})",
                "max_tokens": TOKENS_PER, "temperature": 0, "ignore_eos": True}
        try:
            r = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers=H)
            with urllib.request.urlopen(r, timeout=600) as x:
                d = json.load(x)
            n = d.get("usage", {}).get("completion_tokens", 0)
            txt = d["choices"][0]["text"]
            with lock:
                total["tok"] += n
                cur = total["tok"]
            dt = time.time() - t0
            # crude coherence check: not all one repeated char / not empty
            garbage = (len(set(txt[-200:])) < 5) if len(txt) > 200 else False
            print(f"[{LABEL}] w{wid} req{i}: +{n} tok -> {cur} total  {cur/dt:.1f} tok/s  "
                  f"{'GARBAGE?' if garbage else 'ok'}", flush=True)
            i += 1
        except urllib.error.HTTPError as e:
            print(f"[{LABEL}] w{wid} req{i}: *** HTTP {e.code} (engine likely dead) at {total['tok']} tok, "
                  f"{time.time()-t0:.0f}s", flush=True)
            crashed.set(); return
        except Exception as e:
            print(f"[{LABEL}] w{wid} req{i}: *** EXC {type(e).__name__}: {e} at {total['tok']} tok", flush=True)
            crashed.set(); return

ths = [threading.Thread(target=worker, args=(w,)) for w in range(CONC)]
[t.start() for t in ths]
[t.join() for t in ths]
dt = time.time() - t0
if crashed.is_set():
    print(f"=== [{LABEL}] CRASH at {total['tok']} tok / {dt:.0f}s ===", flush=True)
    sys.exit(1)
print(f"=== [{LABEL}] SURVIVED {total['tok']} tok / {dt:.0f}s -- NO crash ===", flush=True)
