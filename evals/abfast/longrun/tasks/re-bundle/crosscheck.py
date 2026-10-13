#!/usr/bin/env python3
"""Build-time sanity (in container): the READABLE tool.js must behave identically to the recorded obfuscated bundle."""
import json, os, sys
from mscore import run_case
HID, TOOL = sys.argv[1], sys.argv[2]
cases = json.load(open(os.path.join(HID, "cases.json")))
bad = 0
for c in cases:
    ok, why = run_case(["node", TOOL], HID, c)
    if not ok:
        bad += 1; print("DIFF", c["id"], c["args"][:3], why)
print("crosscheck cases", len(cases), "diffs", bad)
sys.exit(1 if bad else 0)
