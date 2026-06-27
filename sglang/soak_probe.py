#!/usr/bin/env python3
# soak_probe.py -- decode-rate SOAK + coherence probe for the testing regime. Streams ONE long
# single-stream decode and reports WINDOWED decode t/s, so we catch rate DEGRADATION over a soak
# (the failure mode that killed graph replay: 26->7 t/s over a soak) that an aggregate-TPOT bench hides.
# Also classifies the output for the degenerate "!!!!" signature -> coherence gate in one shot.
#   usage: soak_probe.py [port] [served] [max_tokens] [window] [host]
import json, sys, time, urllib.request

PORT   = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
SERVED = sys.argv[2] if len(sys.argv) > 2 else "qwen36-27b-w8a8-sqgptq"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
WINDOW = int(sys.argv[4]) if len(sys.argv) > 4 else 400
HOST   = sys.argv[5] if len(sys.argv) > 5 else "127.0.0.1"

# a prompt that elicits a long, structured (easy-to-eyeball-coherent) answer; ignore_eos forces MAXTOK.
req = {
    "model": SERVED,
    "messages": [{"role": "user", "content":
        "Write a detailed numbered explanation of how a CPU executes an instruction, "
        "from fetch to retire, with at least 40 steps. Be thorough."}],
    "max_tokens": MAXTOK, "temperature": 0.3, "stream": True, "ignore_eos": True,
}
r = urllib.request.Request("http://%s:%d/v1/chat/completions" % (HOST, PORT),
    data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})

t0 = time.time()
times = []          # arrival time of each token (relative to first token)
chars = []          # content pieces (for coherence)
first = None
with urllib.request.urlopen(r, timeout=600) as resp:
    for line in resp:
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            d = json.loads(data)["choices"][0].get("delta", {})
        except Exception:
            continue
        piece = d.get("content") or d.get("reasoning_content") or ""
        if piece:
            now = time.time()
            if first is None:
                first = now
            times.append(now - first)
            chars.append(piece)

n = len(times)
text = "".join(chars)
# windowed decode t/s: tokens [i, i+WINDOW) over their wall-clock span
print("SOAK port=%d served=%s tokens=%d ttft=%.0fms" % (PORT, SERVED, n, (first - t0) * 1000 if first else -1))
print("window        tok/s")
rates = []
for s in range(0, n - 1, WINDOW):
    e = min(s + WINDOW, n - 1)
    span = times[e] - times[s]
    if span > 0:
        tps = (e - s) / span
        rates.append(tps)
        print("  [%5d-%5d]  %6.2f" % (s, e, tps))
overall = (n - 1) / times[-1] if n > 1 and times[-1] > 0 else 0
# coherence: degenerate single-char flood?
verdict = "OK"
st = text.strip()
if len(st) >= 16:
    from collections import Counter
    ch, c = Counter(st).most_common(1)[0]
    if c / len(st) >= 0.6:
        verdict = "GARBAGE(%r x%d%%)" % (ch, int(100 * c / len(st)))
if not st:
    verdict = "EMPTY"
degr = (rates[0] / rates[-1]) if len(rates) >= 2 and rates[-1] > 0 else 1.0
print("OVERALL decode %.2f t/s | first/last window ratio %.2fx %s | coherence %s | first 100 chars: %r"
      % (overall, degr, "(DEGRADES!)" if degr > 1.25 else "(stable)", verdict, st[:100]))
