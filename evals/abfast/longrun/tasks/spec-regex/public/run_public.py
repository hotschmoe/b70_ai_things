#!/usr/bin/env python3
"""Run the public example cases against ../target/release/rx.   python3 tests/run_public.py [-v] [case-id-prefix]"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(os.path.dirname(HERE), "target", "release", "rx")
cases = json.load(open(os.path.join(HERE, "public_cases.json")))
verbose = "-v" in sys.argv
pref = [a for a in sys.argv[1:] if not a.startswith("-")]
ok = n = 0
for c in cases:
    if pref and not c["id"].startswith(pref[0]):
        continue
    n += 1
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", newline="") as f:
        f.write(c["input"])
    try:
        p = subprocess.run([BIN, c["cmd"], f.name], capture_output=True, timeout=c.get("timeout", 10))
        got = p.stdout.decode(errors="replace").split("\n")[0]
    except (subprocess.TimeoutExpired, OSError) as e:
        got = f"<{type(e).__name__}>"
    finally:
        os.unlink(f.name)
    good = got == c["expect"]
    ok += good
    if verbose or not good:
        print(("ok   " if good else "FAIL ") + c["id"], c["cmd"], repr(c["input"]), "expected:", c["expect"], "got:", got)
print(f"{ok}/{n} public cases passed")
