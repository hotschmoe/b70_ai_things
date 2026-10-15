#!/usr/bin/env python3
"""E4 client: per-stream decode tok/s and inter-token latency at C1 and C4 against SGLang /generate."""
import json, random, sys, threading, time, urllib.request
URL, OUT, NEW = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 256
def gen(seed):
    random.seed(seed); ids = [random.randint(1000, 200000) for _ in range(512)]
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": NEW, "temperature": 0, "ignore_eos": True}, "stream": True}
    req = urllib.request.Request(URL + "/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); stamps = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if line.startswith(b"data: {"): stamps.append(time.time())
    gaps = sorted(b - a for a, b in zip(stamps[1:], stamps[2:]))
    return {"ttft_s": round(stamps[0] - t0, 3), "chunks": len(stamps),
            "decode_tps": round((len(stamps) - 2) / (stamps[-1] - stamps[1]), 2),
            "itl_p50_ms": round(1000 * gaps[len(gaps) // 2], 2), "itl_p95_ms": round(1000 * gaps[int(.95 * len(gaps))], 2)}
res = {}
gen(0)  # warm
res["c1"] = [gen(s) for s in (1, 2)]
for c in (2, 4):
    out = [None] * c
    ts = [threading.Thread(target=lambda i=i: out.__setitem__(i, gen(10 * c + i))) for i in range(c)]
    [t.start() for t in ts]; [t.join() for t in ts]; res[f"c{c}"] = out
json.dump(res, open(OUT, "w"), indent=1)
for k, v in res.items(): print(k, [(x["decode_tps"], x["itl_p50_ms"], x["itl_p95_ms"]) for x in v], flush=True)
