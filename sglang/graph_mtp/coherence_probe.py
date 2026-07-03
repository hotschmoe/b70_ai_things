#!/usr/bin/env python3
# coherence_probe.py -- Bug C metric for the captured W8A8 MTP verify graph.
# temp=0 greedy decode MUST be deterministic: N identical requests -> N byte-identical outputs.
# The run-27c cross-device VISIBILITY race made ~67-75% of requests diverge/garble. This probe sends
# N identical greedy requests and reports: distinct-output count, coherent count, and a repeatability
# verdict (all identical == race gone). Also runs a few DIFFERENT prompts to confirm real coherence.
#
# Usage: python coherence_probe.py [PORT] [N]   (defaults 31004 20)
import sys, json, urllib.request, collections

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 31004
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SERVED = "qwen36-27b-w8a8-mtp"
URL = f"http://localhost:{PORT}/v1/chat/completions"

PROMPTS = [
    "Why is the sky blue? Answer in two sentences.",
    "Write a haiku about the ocean.",
    "What is 17 times 23? Show your reasoning briefly.",
    "List three primary colors.",
]

def ask(prompt, max_tokens=96):
    body = json.dumps({
        "model": SERVED,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"]["content"]
    except Exception as e:
        return f"<ERROR: {e}>"

def looks_coherent(txt):
    if txt.startswith("<ERROR"):
        return False
    # garble signature: floods of '!' / newlines / periods, or mostly non-alpha
    alpha = sum(c.isalpha() or c.isspace() for c in txt)
    if len(txt) < 5:
        return False
    if txt.count("!") > 8 or txt.count("\n") > 12:
        return False
    return alpha / max(1, len(txt)) > 0.75

print(f"=== DETERMINISM: {N} identical greedy requests -> '{PROMPTS[0]}' ===")
outs = [ask(PROMPTS[0]) for _ in range(N)]
counter = collections.Counter(outs)
coherent = sum(looks_coherent(o) for o in outs)
distinct = len(counter)
top, topn = counter.most_common(1)[0]
print(f"distinct_outputs={distinct}/{N}  coherent={coherent}/{N}  modal_output_freq={topn}/{N}")
print("--- unique outputs (truncated) ---")
for i, (o, c) in enumerate(counter.most_common()):
    tag = "OK " if looks_coherent(o) else "BAD"
    print(f"[{tag} x{c}] {o[:140]!r}")
    if i >= 6:
        print(f"... (+{distinct-7} more distinct)")
        break

verdict = "RACE GONE (deterministic)" if distinct == 1 and coherent == N else \
          ("PARTIAL" if coherent > N * 0.9 else "RACE PRESENT")
print(f"\nVERDICT: {verdict}  (want distinct=1, coherent={N}/{N})")

print(f"\n=== COHERENCE across {len(PROMPTS)} different prompts (x2 each) ===")
for p in PROMPTS:
    a = ask(p); b = ask(p)
    ident = "identical" if a == b else "DIVERGED"
    print(f"[{'OK ' if looks_coherent(a) else 'BAD'}|{ident}] {p[:40]!r} -> {a[:90]!r}")
