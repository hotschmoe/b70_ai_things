#!/usr/bin/env python3
"""Exact initial identity and arithmetic calls of N1's former callback."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness")
from promo_gate import Gate

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", required=True)
parser.add_argument("--expect-alias", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
args.timeout = 120
args.profile = "flashnext"
gate = Gate(args)
gate.result["agent"] = "D1-N1-loaded-liveness"
gate.result["scope"] = "exact N1 initial callback calls only; diagnostic"
if not gate.check("identity", gate.identity):
    raise SystemExit(1)
ok = gate.check("greedy_17x19", lambda: gate.greedy(17, 19))
gate.result["pass"] = ok
gate.save()
raise SystemExit(0 if ok else 1)
