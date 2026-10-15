#!/usr/bin/env python3
"""E5b: llama-server short-prompt coherence: serial then concurrent arithmetic, repeated."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/strata/flash-next")
from bench_serving import send  # noqa: E402
BASE, OUT = sys.argv[1], sys.argv[2]
Q = [(17, 19, "323"), (23, 29, "667")]
def ask(q):
    r = send(BASE, "hotschmoe-dd", [{"role": "user", "content": f"Return only the integer: {q[0]} * {q[1]}."}], 32)
    return dict(want=q[2], got=r["text"].strip(), ok=r["text"].strip() == q[2], draft=(r["timings"] or {}).get("draft_n"))
res = []
for rnd in range(3):
    res.append(dict(round=rnd, mode="serial", rows=[ask(q) for q in Q]))
    with ThreadPoolExecutor(max_workers=2) as p:
        res.append(dict(round=rnd, mode="concurrent", rows=list(p.map(ask, Q))))
for r in res:
    print(r["round"], r["mode"], [(x["ok"], x["got"][:40]) for x in r["rows"]], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
