#!/usr/bin/env python3
"""Hidden checker for re-vault (runs in the verifier container; no reference binary on the agent's side)."""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from mscore import Milestones, cli_milestones, run, sha
import tempfile, shutil

SUB, HID = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
BIN = [f"{SUB}/target/release/vault"]
REF = [f"{HID}/ref/vault"]
MS = [("m01", "digest: custom crc32 + fnv64"), ("m02", "info/list/list -l on stored archives"),
      ("m03", "read method 0 (store)"), ("m04", "read method 1 (rle)"), ("m05", "read method 2 (lz) and auto archives"),
      ("m06", "extract + verify incl. corruption"), ("m07", "encrypted archives: cat/extract/verify, wrong/missing key"),
      ("m08", "pack -m 0 byte-exact"), ("m09", "pack -m 1 byte-exact"), ("m10", "pack -m 2 byte-exact (lz encoder rules)"),
      ("m11", "pack auto byte-exact"), ("m12", "pack -k encrypted byte-exact"),
      ("m13", "error handling: exit codes + stdout"), ("m14", "error messages on stderr, extract subsets"),
      ("m15", "interop: agent archives accepted by the original, large files")]
M = Milestones()
cli_milestones(M, BIN, HID, MS, skip=("m15",))


def m15():
    R = random.Random(7)
    out = []
    for i in range(10):
        d = tempfile.mkdtemp()
        try:
            files = {}
            for j in range(R.randrange(1, 4)):
                n = R.choice([20000, 150000, 400000])
                kind = R.choice(["text", "rep", "rand", "mixed"])
                if kind == "text": b = (" ".join(R.choice(["alpha", "beta", "gamma", "delta", "x", "yy"]) for _ in range(n // 4))).encode()[:n]
                elif kind == "rep": b = (bytes(R.randrange(256) for _ in range(13)) * (n // 13 + 1))[:n]
                elif kind == "rand": b = bytes(R.randrange(256) for _ in range(n))
                else: b = b"".join(bytes([R.randrange(256)]) * R.randrange(1, 90) + b"abcabcabc" * R.randrange(1, 9) for _ in range(n // 100))[:n]
                files[f"f{j}.bin"] = b
                open(os.path.join(d, f"f{j}.bin"), "wb").write(b)
            key = ["-k", "pw"] if i % 2 else []
            meth = [["-m", "auto"], ["-m", "2"], ["-m", "1"]][i % 3]
            rc, o, e = run(BIN + ["pack"] + key + meth + ["a.vlt"] + list(files), cwd=d, timeout=120)
            ok = rc == 0 and os.path.exists(os.path.join(d, "a.vlt"))
            if ok:
                rc2, o2, e2 = run(REF + ["verify"] + key + ["a.vlt"], cwd=d, timeout=60)
                ok = rc2 == 0
                for n, b in files.items():
                    rc3, o3, e3 = run(REF + ["cat"] + key + ["a.vlt", n], cwd=d, timeout=60)
                    ok = ok and rc3 == 0 and o3 == b
                    rc4, o4, e4 = run(BIN + ["cat"] + key + ["a.vlt", n], cwd=d, timeout=60)
                    ok = ok and rc4 == 0 and o4 == b
            out.append(ok)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return out


M.add("m15", MS[-1][1], m15())
M.emit()
