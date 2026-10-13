#!/usr/bin/env python3
"""Build all artifacts for re-vm into LDATA/re-vm/{agent,target,hidden,ref_sub,ref_partial}."""
import os, shutil, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", ".."))
import lr_common as C

TID = "re-vm"
base = C.task_data(TID)
for s in ("agent", "target", "hidden", "ref_sub", "ref_partial"):
    shutil.rmtree(f"{base}/{s}"); os.makedirs(f"{base}/{s}")
script = """set -e
gcc -O2 -o /out/vmrun_unstripped /src/src/vmrun.c
strip -s -o /out/target/vmrun /out/vmrun_unstripped
mkdir -p /out/agent/samples
PYTHONPATH=/mlib python3 /src/gen_cases.py /out/target/vmrun /out/hidden /out/agent/samples
cd /out/agent/samples
for f in *.bc; do b=${f%.bc}; if [ $b = echo ]; then printf 'one\\ntwo\\n' | /out/target/vmrun $f > $b.out; else /out/target/vmrun $f > $b.out; fi; done
"""
rc, o = C.docker_job(script, binds=[(H, "/src", "ro"), (os.path.join(H, "..", "..", "lib"), "/mlib", "ro"), (base, "/out", "rw")], timeout=1800)
print(o[-1500:]); assert rc == 0, "build failed"
os.replace(f"{base}/vmrun_unstripped", f"{base}/vmrun_unstripped.keep")
shutil.copy(f"{H}/check.py", f"{base}/hidden/check.py")
# agent tree
open(f"{base}/agent/go.mod", "w").write("module vmrun\n\ngo 1.22\n")
open(f"{base}/agent/main.go", "w").write('package main\n\nimport (\n\t"fmt"\n\t"os"\n)\n\nfunc main() {\n\tfmt.Fprintln(os.Stderr, "vmrun: not implemented")\n\tos.Exit(1)\n}\n')
shutil.copy(f"{H}/INSTRUCTION.md", f"{base}/agent/TASK.md")
# references
src = open(f"{H}/src/ref_go/main.go").read()
for d in ("ref_sub", "ref_partial"):
    open(f"{base}/{d}/go.mod", "w").write("module vmrun\n\ngo 1.22\n")
    s = src
    if d == "ref_partial":
        s = s.replace("var partial = false", "var partial = true")
    open(f"{base}/{d}/main.go", "w").write(s)
print("re-vm built")
