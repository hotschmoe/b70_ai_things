#!/usr/bin/env python3
"""HTTP-only N1 128-token repeat/batch parity gate; run in owner's window."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import promo_gate as base

base.REPEAT_PROMPTS = [
    "Write a long, detailed essay about the history of the city of Paris.",
    "Explain step by step how a hash map works, then implement one in Python with open addressing.",
    "List the first forty prime numbers and explain the sieve of Eratosthenes.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
]


class Gate(base.Gate):
    def complete(self, prompt, count, forced=True):
        return super().complete(prompt, 128, forced)


parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:18147")
parser.add_argument("--out", required=True)
parser.add_argument("--expect-alias", default="qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4")
parser.add_argument("--reference")
parser.add_argument("--timeout", default=600, type=float)
args = parser.parse_args()
args.profile = "flashnext"
gate = Gate(args)
gate.result["agent"] = "N1"
gate.check("identity", gate.identity)
gate.check("greedy_17x19", lambda: gate.greedy(17, 19))
gate.check("greedy_23x29", lambda: gate.greedy(23, 29))
gate.check("exact_repeat", gate.repeat)
rounds = gate.result["repeat_rounds"]
checks = gate.result["checks"]
checks["exact_repeat"]["forced_tokens"] = 128
for mode in ("c1", "c2"):
    same = [a["token_ids"] == b["token_ids"] for a, b in zip(rounds[mode+"_1"], rounds[mode+"_2"], strict=True)]
    checks[mode+"_same_load_repeat"] = {"pass": all(same), "streams": same}
if args.reference:
    reference = json.loads(Path(args.reference).read_text())["repeat_rounds"]
    same = {key: [a["token_ids"] == b["token_ids"] for a, b in zip(reference[key], rows, strict=True)] for key, rows in rounds.items()}
    checks["reference_parity"] = {"pass": all(all(row) for row in same.values()), "streams": same, "source": args.reference}
gate.result["pass"] = all(check.get("pass", check.get("pass_", False)) for check in checks.values())
gate.save()
print(json.dumps(dict(checks={key: {k: v for k,v in value.items() if k in ("pass", "pass_", "streams")} for key,value in checks.items()}, pass_=gate.result["pass"])), flush=True)
raise SystemExit(0 if gate.result["pass"] else 3)
