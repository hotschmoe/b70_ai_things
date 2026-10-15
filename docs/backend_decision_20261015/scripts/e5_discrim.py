#!/usr/bin/env python3
"""E5d: same server as E2b; E2b request format vs Strata send() on coding and arithmetic prompts."""
import json, sys
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/strata/flash-next")
import importlib.util
from bench_serving import send  # noqa: E402
spec = importlib.util.spec_from_file_location("e2b", "/mnt/vm_8tb/b70/build/backend-decision/e2b_serve.py")
BASE, OUT = sys.argv[1], sys.argv[2]
sys.argv = ["e2b", BASE, "/dev/null", "/mnt/vm_8tb/b70/build/backend-decision/e2-coding-template.json", "64"]
e2b = importlib.util.module_from_spec(spec); spec.loader.exec_module(e2b)
arith = "Return only the integer: 17 * 19."
res = {}
res["e2b_fmt_coding"] = e2b.stream(e2b.coding)["text"][:200]
res["e2b_fmt_arith"] = e2b.stream(arith)["text"][:200]
res["strata_fmt_arith"] = send(BASE, "x", [{"role": "user", "content": arith}], 32)["text"][:200]
res["strata_fmt_coding"] = send(BASE, "x", [{"role": "user", "content": e2b.coding}], 64)["text"][:200]
for k, v in res.items():
    print(k, repr(v[:120]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
