#!/usr/bin/env python3
"""Run the public example cases against ../minisql.py.   python3 tests/run_public.py [-v] [case-id-prefix]"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.dirname(HERE)
cases = json.load(open(os.path.join(HERE, "public_cases.json")))
verbose = "-v" in sys.argv
pref = [a for a in sys.argv[1:] if not a.startswith("-")]
ok = n = 0
for c in cases:
    if pref and not c["id"].startswith(pref[0]):
        continue
    n += 1
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, "sqlcase.py"), SUB], input=json.dumps(c).encode(), capture_output=True, timeout=30)
        out = p.stdout.decode(errors="replace").strip() or ("FAIL " + p.stderr.decode(errors="replace")[-300:])
    except subprocess.TimeoutExpired:
        out = "FAIL timeout"
    ok += out.startswith("OK")
    if verbose or not out.startswith("OK"):
        print(c["id"], out[:300])
print(f"{ok}/{n} public cases passed")
