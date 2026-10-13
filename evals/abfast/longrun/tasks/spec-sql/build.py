#!/usr/bin/env python3
"""Build all artifacts for spec-sql into LDATA/spec-sql/{agent,hidden,ref_sub,ref_partial} (host python3, no docker needed)."""
import os, shutil, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", ".."))
import lr_common as C

TID = "spec-sql"
base = C.task_data(TID)
for s in ("agent", "target", "hidden", "ref_sub", "ref_partial"):
    shutil.rmtree(f"{base}/{s}"); os.makedirs(f"{base}/{s}")
os.makedirs(f"{base}/agent/tests")
subprocess.run([sys.executable, f"{H}/gen_cases.py", f"{base}/hidden", f"{base}/agent/tests"], check=True)
shutil.copy(f"{H}/check.py", f"{base}/hidden/check.py")
shutil.copy(f"{H}/../../lib/sqlcase.py", f"{base}/agent/tests/sqlcase.py")
shutil.copy(f"{H}/public/run_public.py", f"{base}/agent/tests/run_public.py")
shutil.copy(f"{H}/SPEC.md", f"{base}/agent/SPEC.md")
shutil.copy(f"{H}/INSTRUCTION.md", f"{base}/agent/TASK.md")
shutil.copy(f"{H}/public/minisql_stub.py", f"{base}/agent/minisql.py")
full = open(f"{H}/src/minisql.py").read()
os.makedirs(f"{base}/ref_sub", exist_ok=True)
open(f"{base}/ref_sub/minisql.py", "w").write(full)
trailer = open(f"{H}/public/partial_trailer.py").read()
marker = 'if __name__ == "__main__":'
assert marker in full
open(f"{base}/ref_partial/minisql.py", "w").write(full.replace(marker, trailer + "\n\n" + marker))
print("spec-sql built")
