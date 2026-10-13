#!/usr/bin/env python3
"""Hidden checker for re-bundle. Runs the agent's app.py with `node` unavailable (PATH has no node)."""
import os, re, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import mscore
from mscore import Milestones, cli_milestones, run

SUB, HID = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
# 1. make node unusable for the agent's program
os.environ["PATH"] = ":".join(p for p in os.environ.get("PATH", "").split(":") if "node" not in p and p != "/opt/node/bin")
os.environ.pop("NODE_PATH", None)
_orig_run = mscore.run
def _run(cmd, stdin=None, **kw): return _orig_run(cmd, stdin=(b"" if stdin is None else stdin), **kw)
mscore.run = _run
MS = [("s01", "crc: custom checksum of strings/files/stdin"), ("s02", "b32 enc"), ("s03", "b32 dec incl. errors"),
      ("s04", "ser: canonical serialization + checksum"), ("s05", "de: parse + JSON output (key order quirks)"), ("s06", "de: error messages and positions"),
      ("s07", "expr: operators, precedence, functions, variables"), ("s08", "expr: 32-bit semantics, laziness, errors"),
      ("s09", "tpl: variables, paths, filters, rendering"), ("s10", "tpl: blocks, whitespace control, errors"),
      ("s11", "pack/unpack: variable width LZW container"), ("s12", "seal/open: stream cipher + MAC"),
      ("s13", "command line: usage, unknown commands, stdin/file handling"), ("s14", "round trips through the agent's own commands")]
# 2. static scan: no shelling out to node / the original bundle
flag = []
for f in glob.glob(SUB + "/**/*", recursive=True):
    if os.path.isfile(f) and os.path.getsize(f) < 5_000_000 and not f.endswith((".pyc",)):
        try: t = open(f, errors="replace").read()
        except OSError: continue
        if f.endswith(".py") and re.search(r"subprocess|os\.system|os\.popen|os\.exec|pty\.spawn", t) and re.search(r"\bnode\b|app\.js|/opt/target", t): flag.append(f)
M = Milestones()
if flag or not os.path.exists(SUB + "/app.py"):
    for mid, name in MS: M.skip(mid, name, 1)
    print("FLAGGED", flag)
    M.emit(); sys.exit(0)
argv0 = [sys.executable, SUB + "/app.py"]
cli_milestones(M, argv0, HID, MS, timeout=30, skip=("s14",))
# s14: round trips using only the agent's implementation
import random, tempfile, shutil, json
R = random.Random(11)
ok = []
def agent(args, stdin=None, cwd=None):
    return run(argv0 + args, stdin=stdin if stdin is not None else b"", cwd=cwd, timeout=60)
for i in range(12):
    n = R.choice([0, 7, 300, 5000, 20000])
    data = bytes(R.randrange(256) for _ in range(n)) if i % 2 else bytes(R.choice(b"abcd ") for _ in range(n))
    rc, p, e = agent(["pack"], data); rc2, u, e2 = agent(["unpack"], p); ok.append(rc == 0 and rc2 == 0 and u == data)
    rc, s, e = agent(["seal", "key%d" % i], data); rc2, o, e2 = agent(["open", "key%d" % i], s); ok.append(rc == 0 and rc2 == 0 and o == data)
    rc, b, e = agent(["b32", "enc"], data); rc2, d, e2 = agent(["b32", "dec"], b); ok.append(rc == 0 and rc2 == 0 and d == data)
    obj = {"k%d" % j: [j, str(j) * 3, None, {"z": True}] for j in range(R.randrange(0, 8))}
    rc, s, e = agent(["ser"], json.dumps(obj).encode()); rc2, d, e2 = agent(["de"], s)
    ok.append(rc == 0 and rc2 == 0 and json.loads(d) == obj)
M.add("s14", MS[-1][1], ok)
M.emit()
