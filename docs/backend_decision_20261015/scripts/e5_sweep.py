#!/usr/bin/env python3
"""E5c: arithmetic correctness vs prompt length (neutral padding before the question)."""
import json, sys
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/strata/flash-next")
from bench_serving import send  # noqa: E402
BASE, OUT = sys.argv[1], sys.argv[2]
PAD = "Background note: this is a routine arithmetic check with no tricks. "
res = []
for reps in [0, 1, 2, 4, 8, 16, 32]:
    for a, b, want in [(17, 19, "323"), (23, 29, "667")]:
        r = send(BASE, "hotschmoe-dd", [{"role": "user", "content": PAD * reps + f"Return only the integer: {a} * {b}."}], 24)
        row = dict(pad=reps, prompt_tokens=r["usage"]["prompt_tokens"], want=want, got=r["text"].strip()[:40], ok=r["text"].strip() == want)
        res.append(row); print(json.dumps(row, ensure_ascii=True), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
