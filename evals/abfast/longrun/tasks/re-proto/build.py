#!/usr/bin/env python3
"""Build all artifacts for re-proto into LDATA/re-proto/{agent,target,hidden,ref_sub,ref_partial}."""
import os, shutil, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", ".."))
import lr_common as C

TID = "re-proto"
base = C.task_data(TID)
for s in ("agent", "target", "hidden", "ref_sub", "ref_partial"):
    shutil.rmtree(f"{base}/{s}"); os.makedirs(f"{base}/{s}")
script = """set -e
gcc -O2 -o /tmp/kvd_unstripped /src/src/kvd.c
strip -s -o /out/target/kvd /tmp/kvd_unstripped
PYTHONPATH=/mlib python3 /src/gen_cases.py /out/target/kvd /out/hidden /out/agent/samples
"""
rc, o = C.docker_job(script, binds=[(H, "/src", "ro"), (os.path.join(H, "..", "..", "lib"), "/mlib", "ro"), (base, "/out", "rw")], timeout=3000)
print(o[-1500:]); assert rc == 0, "build failed"
shutil.copy(f"{H}/check.py", f"{base}/hidden/check.py")
shutil.copy(f"{H}/INSTRUCTION.md", f"{base}/agent/TASK.md")
open(f"{base}/agent/kvd.py", "w").write('import sys\nsys.stderr.write("kvd.py: not implemented\\n")\nsys.exit(2)\n')
ref = open(f"{H}/src/ref_kvd.py").read()
open(f"{base}/ref_sub/kvd.py", "w").write(ref)
marker = "        S = c.work if c.intx else self.G\n"
assert marker in ref
part = ref.replace(marker, "        if op not in (1, 2, 3, 4, 5, 6):\n            return self.err(c, rid, E_UNKNOWN)\n" + marker, 1)
open(f"{base}/ref_partial/kvd.py", "w").write(part)
print("re-proto built")
