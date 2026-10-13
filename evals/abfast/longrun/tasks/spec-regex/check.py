#!/usr/bin/env python3
"""Hidden checker for spec-regex (verifier container). One process per case, wall timeout per case."""
import json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from mscore import Milestones

SUB, HID = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
BIN = f"{SUB}/target/release/rx"
MS = [("r01", "literals, dot, escapes"), ("r02", "anchors and word boundaries"), ("r03", "character classes"), ("r04", "alternation and groups, capture spans"),
      ("r05", "greedy quantifiers and counted repeats"), ("r06", "lazy quantifiers"), ("r07", "back references"), ("r08", "flags i m s x and scoped flags"),
      ("r09", "lookahead and lookbehind"), ("r10", "possessive quantifiers and atomic groups"), ("r11", "error classification"),
      ("r12", "findall with empty-match rules"), ("r13", "sub with templates"), ("r14", "backtracking priority and capture semantics"),
      ("r15", "performance: catastrophic backtracking patterns")]
cases = json.load(open(os.path.join(HID, "cases.json")))
groups = {}
for c in cases:
    groups.setdefault(c["ms"], []).append(c)


def one(c):
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", newline="") as f:
        f.write(c["input"])
    try:
        p = subprocess.run([BIN, c["cmd"], f.name], capture_output=True, timeout=c.get("timeout", 10))
        out = p.stdout.decode(errors="replace").split("\n")
        return out[0] == c["expect"]
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        os.unlink(f.name)


M = Milestones()
with ThreadPoolExecutor(4) as ex:
    for mid, name in MS:
        M.add(mid, name, list(ex.map(one, groups.get(mid, []))))
M.emit()
