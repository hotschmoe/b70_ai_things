#!/usr/bin/env python3
"""Stage-5 headline gate for the vLLM v0.24.0 rebase: does concurrent MIXED prefill+decode stay
coherent (NO "!!!!"/garbage)? This is the exact failure that paused vLLM. The v0.24.0 target PRs
(#44700 split mixed prefill+decode -> recurrent GDN kernel, #43990, #42430, #43961, #43556) fix it.

Fires WAVES of long-prefill requests staggered against in-flight decoders so new prefills land in the
SAME batch as ongoing decodes (the mixed-batch corruption trigger). Collects full text per request and
flags garbage. Dependency-free (urllib + threads); non-streaming (garbage still shows in the full text).

Usage: gate_concurrent_coherence.py <base_url> <model> [waves=3] [per_wave=6] [max_tokens=200]
Exit 0 = PASS (all coherent), 1 = FAIL (any garbage).
"""
import sys, json, re, os, time, threading, urllib.request, collections

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
WAVES = int(sys.argv[3]) if len(sys.argv) > 3 else 3
PER_WAVE = int(sys.argv[4]) if len(sys.argv) > 4 else 6
MAXTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 200
API_KEY = os.environ.get("API_KEY", "")
URL = f"{BASE}/chat/completions"

LONG = ("Read the following and then answer. " + ("The quick brown fox jumps over the lazy dog. " * 120) +
        "\n\nNow: write a detailed technical essay about GPU memory hierarchies and INT8 quantization.")
SHORT = "Explain in three sentences why the sky is blue."
PROMPTS = [LONG, SHORT, LONG, SHORT, LONG, SHORT, LONG, SHORT]

GARBAGE_RE = re.compile(r"(.)\1{9,}")   # any char repeated 10+ times in a row

def classify(txt):
    if not txt or txt.startswith("<ERROR"):
        return "ERROR", (txt or "")[:90]
    t = txt.strip()
    if len(t) < 5:
        return "EMPTY", repr(t)
    m = GARBAGE_RE.search(t)
    if m:
        return "GARBAGE", f"run of {m.group(1)!r}: ...{t[max(0,m.start()-10):m.start()+30]!r}..."
    alpha = sum(c.isalpha() or c.isspace() for c in t) / len(t)
    if alpha < 0.6:
        return "GARBAGE", f"alpha_frac={alpha:.2f}: {t[:60]!r}"
    return "OK", t[:50]

def ask(idx, prompt, out):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAXTOK, "temperature": 0.7}).encode()
    hdr = {"content-type": "application/json"}
    if API_KEY:
        hdr["Authorization"] = f"Bearer {API_KEY}"
    try:
        req = urllib.request.Request(URL, data=body, headers=hdr)
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        out[idx] = d["choices"][0]["message"]["content"]
    except Exception as e:
        out[idx] = f"<ERROR: {e}>"

def main():
    out = {}
    threads = []
    for w in range(WAVES):
        for k in range(PER_WAVE):
            idx = f"w{w}k{k}"
            p = PROMPTS[(w * PER_WAVE + k) % len(PROMPTS)]
            th = threading.Thread(target=ask, args=(idx, p, out))
            th.start(); threads.append(th)
        time.sleep(1.2)   # stagger: next wave's prefills hit mid-decode of this wave
    for th in threads:
        th.join()
    verdicts = collections.Counter()
    for idx in sorted(out):
        cls, detail = classify(out[idx])
        verdicts[cls] += 1
        print(f"  [{'ok ' if cls=='OK' else 'BAD'}] {idx:6} {cls:8} {detail}")
    n = sum(verdicts.values())
    print(f"\n=== {n} streams: " + ", ".join(f"{k}={v}" for k, v in verdicts.items()) + " ===")
    if verdicts["OK"] == n:
        print("GATE PASS: all streams coherent under concurrent mixed prefill+decode"); sys.exit(0)
    print(f"GATE FAIL: {n - verdicts['OK']}/{n} streams garbage/error"); sys.exit(1)

main()
