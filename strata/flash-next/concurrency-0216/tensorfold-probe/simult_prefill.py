#!/usr/bin/env python3
"""N distinct ~L-token prompts (real text, token ids) arrive at the same time; per-request TTFT and aggregate prefill tok/s.
usage: simult_prefill.py URL MODEL TOKENIZER_JSON CORPUS OUT.json N L"""
import json, sys, threading, time, urllib.request
from tokenizers import Tokenizer
url, model, tokj, corpus, out, n, L = sys.argv[1:8]; n = int(n); L = int(L)
ids = Tokenizer.from_file(tokj).encode(open(corpus, errors="ignore").read()).ids
assert len(ids) >= n * L + 1000
res = [None] * n
def one(i):
    p = ids[i * (len(ids) // n): i * (len(ids) // n) + L]
    body = {"model": model, "prompt": p, "max_tokens": 8, "temperature": 0, "stream": True, "ignore_eos": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                if d.get("usage"): usage = d["usage"]
                if first is None and d.get("choices") and d["choices"][0].get("text"): first = time.perf_counter() - t0
        res[i] = dict(i=i, ttft_s=first, usage=usage, done_s=time.perf_counter() - t0)
    except Exception as e:
        res[i] = dict(i=i, error=repr(e))
ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
T0 = time.perf_counter()
for t in ths: t.start()
for t in ths: t.join()
ok = [r for r in res if r.get("ttft_s")]
tot = sum((r["usage"] or {}).get("prompt_tokens", L) for r in ok)
summ = dict(n=n, prompt_len=L, ttft_s=sorted(r["ttft_s"] for r in ok), last_ttft_s=max(r["ttft_s"] for r in ok) if ok else None,
            aggregate_prefill_tps=tot / max(r["ttft_s"] for r in ok) if ok else None, errors=[r for r in res if r.get("error")])
json.dump(dict(summary=summ, results=res), open(out, "w"), indent=1); print(json.dumps(summ, indent=1))
