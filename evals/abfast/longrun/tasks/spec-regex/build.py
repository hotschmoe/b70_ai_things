#!/usr/bin/env python3
"""Build all artifacts for spec-regex into LDATA/spec-regex/{agent,hidden,ref_sub,ref_partial}."""
import os, shutil, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", ".."))
import lr_common as C

TID = "spec-regex"
base = C.task_data(TID)
for s in ("agent", "target", "hidden", "ref_sub", "ref_partial"):
    shutil.rmtree(f"{base}/{s}"); os.makedirs(f"{base}/{s}")
os.makedirs(f"{base}/agent/tests"); os.makedirs(f"{base}/agent/src"); os.makedirs(f"{base}/ref_bin", exist_ok=True)
# 1. compile the reference (used to cross-check the Python oracle and to record the performance expectations)
rc, o = C.docker_job("set -e; rm -rf /tmp/b; cp -r /src/src/ref_rs /tmp/b; cd /tmp/b; cargo build --release --offline 2>&1 | tail -3; cp target/release/rx /out/ref_bin/rx",
                     binds=[(H, "/src", "ro"), (base, "/out", "rw")], timeout=900)
print(o[-500:]); assert rc == 0
subprocess.run([sys.executable, f"{H}/gen_cases.py", f"{base}/hidden", f"{base}/agent/tests", f"{base}/ref_bin/rx"], check=True,
               env=dict(os.environ, PYTHONWARNINGS="ignore"))
shutil.copy(f"{H}/check.py", f"{base}/hidden/check.py")
shutil.copy(f"{H}/public/run_public.py", f"{base}/agent/tests/run_public.py")
shutil.copy(f"{H}/SPEC.md", f"{base}/agent/SPEC.md")
shutil.copy(f"{H}/INSTRUCTION.md", f"{base}/agent/TASK.md")
shutil.copy(f"{H}/src/ref_rs/Cargo.toml", f"{base}/agent/Cargo.toml")
shutil.copy(f"{H}/public/main_stub.rs", f"{base}/agent/src/main.rs")
# 2. references
full = open(f"{H}/src/ref_rs/src/main.rs").read()
for d in ("ref_sub", "ref_partial"):
    os.makedirs(f"{base}/{d}/src")
    shutil.copy(f"{H}/src/ref_rs/Cargo.toml", f"{base}/{d}/Cargo.toml")
src = full
open(f"{base}/ref_sub/src/main.rs", "w").write(full)
ns = {}
exec(open(f"{H}/public/partial_patch.py").read(), ns)
assert ns["ANCHOR"] in full
patched = full.replace(ns["ANCHOR"], ns["INSERT"], 1) + ns["FUNC"]
open(f"{base}/ref_partial/src/main.rs", "w").write(patched)
print("spec-regex built")
