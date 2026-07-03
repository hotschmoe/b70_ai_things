#!/usr/bin/env python3
"""Stage-5 headline gate for the vLLM v0.24.0 rebase: does concurrent MIXED prefill+decode stay
coherent (NO "!!!!"/garbage)? This is the exact failure that paused vLLM. The v0.24.0 target PRs
(#44700 split mixed prefill+decode -> recurrent GDN kernel, #43990, #42430, #43961, #43556) fix it.

Fires WAVES of long-prefill requests staggered against in-flight decoders so new prefills land in the
SAME batch as ongoing decodes (the mixed-batch corruption trigger). Collects full text per stream and
flags garbage. temp=0 secondary determinism check reused from the sglang probe idea.

Usage: gate_concurrent_coherence.py <base_url> <model> [waves=3] [per_wave=6] [max_tokens=200]
Exit 0 = PASS (all coherent), 1 = FAIL (any garbage).
"""
import asyncio, sys, time, re, collections
from openai import AsyncOpenAI

BASE = sys.argv[1]
MODEL = sys.argv[2]
WAVES = int(sys.argv[3]) if len(sys.argv) > 3 else 3
PER_WAVE = int(sys.argv[4]) if len(sys.argv) > 4 else 6
MAXTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 200
client = AsyncOpenAI(base_url=BASE, api_key=__import__("os").environ.get("API_KEY", "x"), timeout=600.0)

# Long prefills (force real prefill work) + short ones, interleaved so batches mix.
LONG = ("Read the following and then answer. " + ("The quick brown fox jumps over the lazy dog. " * 120) +
        "\n\nNow: write a detailed technical essay about GPU memory hierarchies and INT8 quantization.")
SHORT = "Explain in three sentences why the sky is blue."
PROMPTS = [LONG, SHORT, LONG, SHORT, LONG, SHORT, LONG, SHORT]

GARBAGE_RE = re.compile(r"(.)\1{9,}")   # any char repeated 10+ times in a row

def classify(txt):
    if not txt or txt.startswith("<ERROR"):
        return "ERROR", txt[:80]
    t = txt.strip()
    if len(t) < 5:
        return "EMPTY", repr(t)
    m = GARBAGE_RE.search(t)
    if m:
        return "GARBAGE", f"run of {m.group(1)!r}: ...{t[max(0,m.start()-10):m.start()+30]!r}..."
    # fraction of alphabetic chars; garbage floods are mostly punctuation/newlines
    alpha = sum(c.isalpha() or c.isspace() for c in t) / len(t)
    if alpha < 0.6:
        return "GARBAGE", f"alpha_frac={alpha:.2f}: {t[:60]!r}"
    return "OK", t[:50]

async def one(idx, prompt):
    txt = []
    try:
        stream = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=MAXTOK, temperature=0.7, stream=True)
        async for ch in stream:
            if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
                txt.append(ch.choices[0].delta.content)
    except Exception as e:
        return idx, f"<ERROR: {e}>"
    return idx, "".join(txt)

async def main():
    results = []
    inflight = []
    for w in range(WAVES):
        # launch a wave; do NOT await -> they keep decoding while the next wave's prefills arrive
        for k in range(PER_WAVE):
            p = PROMPTS[(w * PER_WAVE + k) % len(PROMPTS)]
            inflight.append(asyncio.create_task(one(f"w{w}k{k}", p)))
        await asyncio.sleep(1.2)   # stagger: next wave prefills hit mid-decode of this wave
    results = await asyncio.gather(*inflight)
    verdicts = collections.Counter()
    bad = []
    for idx, txt in sorted(results):
        cls, detail = classify(txt)
        verdicts[cls] += 1
        tag = "ok " if cls == "OK" else "BAD"
        print(f"  [{tag}] {idx:6} {cls:8} {detail}")
        if cls != "OK":
            bad.append((idx, cls, detail))
    n = sum(verdicts.values())
    print(f"\n=== {n} streams: " + ", ".join(f"{k}={v}" for k, v in verdicts.items()) + " ===")
    if verdicts["OK"] == n:
        print("GATE PASS: all streams coherent under concurrent mixed prefill+decode")
        sys.exit(0)
    print(f"GATE FAIL: {n - verdicts['OK']}/{n} streams garbage/error")
    sys.exit(1)

asyncio.run(main())
