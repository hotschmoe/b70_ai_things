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


def summarize_rounds(result, reference=None):
    """Keep gate HTTP errors and mark unavailable streams incomplete."""
    rounds = result.get("repeat_rounds")
    rounds = rounds if isinstance(rounds, dict) else {}
    checks = result["checks"]
    expected = len(base.REPEAT_PROMPTS)
    labels = ("c1_1", "c1_2", "c2_1", "c2_2")

    def tokens(rows, index):
        if not isinstance(rows, list) or index >= len(rows):
            return None
        row = rows[index]
        ids = row.get("token_ids") if isinstance(row, dict) else None
        return ids if isinstance(ids, list) and len(ids) == 128 else None

    def compare(left, right):
        complete = (isinstance(left, list) and isinstance(right, list)
                    and len(left) == len(right) == expected)
        same = []
        for index in range(expected):
            a, b = tokens(left, index), tokens(right, index)
            same.append(a == b if a is not None and b is not None else None)
        complete = complete and None not in same
        return {"pass": complete and all(same), "complete": complete,
                "streams": same}

    exact = checks.setdefault("exact_repeat", {"pass": False})
    exact["forced_tokens"] = 128
    incomplete = [key for key in labels
                  if not compare(rounds.get(key), rounds.get(key))["complete"]]
    exact["complete"] = not incomplete
    if incomplete:
        exact["pass"] = False
        exact["incomplete_rounds"] = incomplete
    for mode in ("c1", "c2"):
        checks[mode + "_same_load_repeat"] = compare(
            rounds.get(mode + "_1"), rounds.get(mode + "_2"))
    if reference is not None:
        prior = reference.get("repeat_rounds")
        prior = prior if isinstance(prior, dict) else {}
        compared = {key: compare(prior.get(key), rounds.get(key)) for key in labels}
        row = checks.setdefault("reference_parity", {})
        row.update({"pass": all(value["pass"] for value in compared.values()),
                    "complete": all(value["complete"] for value in compared.values()),
                    "streams": {key: value["streams"] for key, value in compared.items()}})
    result["pass"] = all(row.get("pass") is True for row in checks.values())


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
checks = gate.result["checks"]
reference = None
if args.reference:
    try:
        reference = json.loads(Path(args.reference).read_text())
        if not isinstance(reference, dict):
            raise ValueError("reference result must be an object")
    except Exception as exc:
        reference = {}
        checks["reference_parity"] = {"pass": False, "error": ascii(str(exc))}
summarize_rounds(gate.result, reference)
if args.reference:
    checks["reference_parity"]["source"] = args.reference
gate.save()
print(json.dumps(dict(checks={key: {k: v for k,v in value.items() if k in ("pass", "pass_", "complete", "streams", "error")} for key,value in checks.items()}, pass_=gate.result["pass"])), flush=True)
raise SystemExit(0 if gate.result["pass"] else 3)
