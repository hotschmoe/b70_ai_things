#!/usr/bin/env python3
"""Build all artifacts for auto-opt into LDATA/auto-opt/{agent,hidden,ref_sub,ref_partial}."""
import os, shutil, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", ".."))
import lr_common as C

TID = "auto-opt"
base = C.task_data(TID)
for s in ("agent", "target", "hidden", "ref_sub", "ref_partial"):
    shutil.rmtree(f"{base}/{s}"); os.makedirs(f"{base}/{s}")
script = r"""set -e
export HOME=/tmp CARGO_HOME=/tmp/cargo
cp -r /src/src/naive /tmp/nv && (cd /tmp/nv && cargo build --release --offline -q)
python3 /src/gen_hidden.py /out/hidden /out/agent
# cross-check the independent oracle against the original program on every non-large input
for f in /out/hidden/inputs/small*.in /out/hidden/inputs/e_*.in /out/hidden/inputs/medium*.in; do
  /tmp/nv/target/release/nnjoin $f | cmp - ${f%.in}.out || { echo "ORACLE MISMATCH $f"; exit 1; }
done
echo oracle-ok
# public small tests for the agent
mkdir -p /out/agent/tests
for i in 1 2 3 4 5 6; do
  python3 /src/public/tools/gen.py $((i*7)) $((1000+i)) 40 0 12 > /out/agent/tests/case$i.in
  python3 /src/public/tools/expected.py /out/agent/tests/case$i.in > /out/agent/tests/case$i.out
  /tmp/nv/target/release/nnjoin /out/agent/tests/case$i.in | cmp - /out/agent/tests/case$i.out
done
printf '3\n1 2 3\n3 2\n\n' > /out/agent/tests/tiny.in; python3 /src/public/tools/expected.py /out/agent/tests/tiny.in > /out/agent/tests/tiny.out
"""
rc, o = C.docker_job(script, binds=[(H, "/src", "ro"), (base, "/out", "rw")], timeout=1800)
print(o[-1500:]); assert rc == 0, "build failed"
shutil.copytree(f"{H}/src/naive", f"{base}/hidden/baseline"); shutil.copy(f"{H}/check.py", f"{base}/hidden/check.py")
shutil.copytree(f"{H}/src/naive", f"{base}/agent", dirs_exist_ok=True)
shutil.copytree(f"{H}/public/tools", f"{base}/agent/tools")
shutil.copy(f"{H}/INSTRUCTION.md", f"{base}/agent/TASK.md")
open(f"{base}/agent/.gitignore", "w").write("target/\n")
shutil.copytree(f"{H}/src/ref_opt", f"{base}/ref_sub", dirs_exist_ok=True)
shutil.copytree(f"{H}/src/ref_part", f"{base}/ref_partial", dirs_exist_ok=True)
print("auto-opt built")
