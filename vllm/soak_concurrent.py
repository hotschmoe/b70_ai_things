#!/usr/bin/env python3
# Concurrent forced-decode soak to force the MTP-graph NEO command-list overflow fast.
# Fires WORKERS parallel streams of ignore_eos forced decode, loops until the engine dies
# (HTTP 500 / connection refused) or a token/time ceiling. Reports cumulative decode tokens
# and wall time at crash -- the accumulation threshold.
#
#   PORT=18091 KEY=... WORKERS=6 CTX_CHARS=28000 MAXTOK=4000 CEIL_TOK=600000 CEIL_SEC=2400 \
#     python3 vllm/soak_concurrent.py <label>
import json, os, sys, time, threading, urllib.request, urllib.error

LABEL     = sys.argv[1] if len(sys.argv) > 1 else "soak"
HOST      = os.environ.get("PROBE_HOST", "192.168.10.5")
PORT      = os.environ.get("PORT", "18091")
KEY       = os.environ.get("KEY", "").strip()
WORKERS   = int(os.environ.get("WORKERS", "6"))
CTX_CHARS = int(os.environ.get("CTX_CHARS", "28000"))     # ~8K tok: fast decode -> max replay rate
MAXTOK    = int(os.environ.get("MAXTOK", "4000"))
CEIL_TOK  = int(os.environ.get("CEIL_TOK", "800000"))
CEIL_SEC  = int(os.environ.get("CEIL_SEC", "2400"))
TIMEOUT   = int(os.environ.get("PROBE_TIMEOUT", "600"))
BASE      = f"http://{HOST}:{PORT}"

def hdr():
    h = {"Content-Type": "application/json"}
    if KEY: h["Authorization"] = "Bearer " + KEY
    return h

def models():
    r = urllib.request.Request(BASE + "/v1/models", headers=hdr())
    return json.load(urllib.request.urlopen(r, timeout=20))["data"][0]["id"]

CHUNK = ("def f_{i}(s, c):\n    t = 0\n    for j, b in enumerate(s.blocks):\n"
         "        if b.ref > 0 and not b.free:\n            t += b.n * c.hd\n    return t, s.L + {i}\n\n")
def prompt(nonce):
    buf = [f"# soak {nonce}\n"]; n = 0; i = 0
    while n < CTX_CHARS:
        x = CHUNK.format(i=i); buf.append(x); n += len(x); i += 1
    buf.append("\n# TASK: exhaustively explain every function above, one long paragraph each. Do not stop.\n")
    return "".join(buf)

STATE = {"tok": 0, "reqs": 0, "crash": None, "stop": False, "t0": time.time()}
LOCK = threading.Lock()
MID = None

def worker(wid):
    k = 0
    while not STATE["stop"]:
        with LOCK:
            if STATE["tok"] >= CEIL_TOK or (time.time() - STATE["t0"]) > CEIL_SEC:
                STATE["stop"] = True; break
        body = {"model": MID, "prompt": prompt(f"{LABEL}-w{wid}-{k}-{int(time.time())}"),
                "max_tokens": MAXTOK, "temperature": 0, "ignore_eos": True}
        try:
            r = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers=hdr())
            with urllib.request.urlopen(r, timeout=TIMEOUT) as x:
                d = json.load(x)
            c = d["usage"]["completion_tokens"]
            with LOCK:
                STATE["tok"] += c; STATE["reqs"] += 1
        except urllib.error.HTTPError as e:
            with LOCK:
                if STATE["crash"] is None:
                    STATE["crash"] = (f"HTTP{e.code}", STATE["tok"], round(time.time()-STATE["t0"]))
                STATE["stop"] = True
            return
        except Exception as e:
            # connection refused / reset => engine died
            with LOCK:
                if STATE["crash"] is None:
                    STATE["crash"] = (f"{type(e).__name__}", STATE["tok"], round(time.time()-STATE["t0"]))
                STATE["stop"] = True
            return
        k += 1

def reporter():
    while not STATE["stop"]:
        time.sleep(20)
        with LOCK:
            print(f"  [{LABEL}] t={round(time.time()-STATE['t0'])}s tok={STATE['tok']} reqs={STATE['reqs']}", flush=True)

def main():
    global MID
    MID = models()
    print(f"[{LABEL}] served={MID} workers={WORKERS} ctx_chars={CTX_CHARS} maxtok={MAXTOK} "
          f"ceil_tok={CEIL_TOK} ceil_sec={CEIL_SEC}", flush=True)
    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(WORKERS)]
    tr = threading.Thread(target=reporter, daemon=True); tr.start()
    for t in ts: t.start()
    for t in ts: t.join()
    STATE["stop"] = True
    if STATE["crash"]:
        kind, tok, sec = STATE["crash"]
        print(f"[{LABEL}] *** CRASH ({kind}) after tok={tok} t={sec}s reqs={STATE['reqs']}", flush=True)
        sys.exit(3)
    print(f"[{LABEL}] SURVIVED tok={STATE['tok']} reqs={STATE['reqs']} "
          f"t={round(time.time()-STATE['t0'])}s (no crash within ceiling)", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
