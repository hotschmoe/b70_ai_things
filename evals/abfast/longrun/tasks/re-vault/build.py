#!/usr/bin/env python3
"""Build all artifacts for re-vault into LDATA/re-vault/{agent,target,hidden,ref_sub,ref_partial}."""
import os, shutil, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..")); sys.path.insert(0, os.path.join(H, "..", "..", "lib"))
import lr_common as C

TID = "re-vault"
base = C.task_data(TID)
for s in ("agent", "target", "hidden", "ref_sub", "ref_partial"):
    shutil.rmtree(f"{base}/{s}"); os.makedirs(f"{base}/{s}")
# 1. compile reference + record cases (inside the container; source dir read-only)
script = f"""set -e
gcc -O2 -o /out/target/vault_unstripped /src/src/vault.c
cp /out/target/vault_unstripped /out/ref_unstripped_vault
strip -s -o /out/target/vault /out/target/vault_unstripped && rm /out/target/vault_unstripped
mkdir -p /out/hidden/ref && cp /out/target/vault /out/hidden/ref/vault
PYTHONPATH=/mlib python3 /src/gen_cases.py /out/target/vault /out/hidden /out/agent/samples
"""
rc, o = C.docker_job(script, binds=[(H, "/src", "ro"), (os.path.join(H, "..", "..", "lib"), "/mlib", "ro"), (base, "/out", "rw")], timeout=1800)
print(o[-1500:]);  assert rc == 0, "build failed"
# the unstripped reference binary is kept out of agent/ and hidden/
os.replace(f"{base}/ref_unstripped_vault", f"{base}/ref_unstripped_vault.keep") if os.path.exists(f"{base}/ref_unstripped_vault") else None
shutil.copy(f"{H}/check.py", f"{base}/hidden/check.py")
# 2. agent starting tree
cargo = open(f"{H}/src/ref_rs/Cargo.toml").read()
os.makedirs(f"{base}/agent/src", exist_ok=True)
open(f"{base}/agent/Cargo.toml", "w").write(cargo)
open(f"{base}/agent/src/main.rs", "w").write('fn main() {\n    eprintln!("vault: not implemented");\n    std::process::exit(2);\n}\n')
shutil.copy(f"{H}/INSTRUCTION.md", f"{base}/agent/TASK.md")
# 3. references
for d in ("ref_sub", "ref_partial"):
    shutil.copy(cargo and f"{H}/src/ref_rs/Cargo.toml", f"{base}/{d}/Cargo.toml")
    os.makedirs(f"{base}/{d}/src")
    src = open(f"{H}/src/ref_rs/src/main.rs").read()
    if d == "ref_partial":
        for arm in ('"pack"', '"extract"', '"verify"'):
            assert f"{arm} => {{" in src
            src = src.replace(f"{arm} => {{", f"{arm} if false => {{", 1)
    open(f"{base}/{d}/src/main.rs", "w").write(src)
print("re-vault built")
