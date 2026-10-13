#!/usr/bin/env python3
"""Build all artifacts for re-bundle into LDATA/re-bundle/{agent,target,hidden,ref_sub,ref_partial}."""
import os, shutil, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", ".."))
import lr_common as C

TID = "re-bundle"
base = C.task_data(TID)
for s in ("agent", "target", "hidden", "ref_sub", "ref_partial"):
    shutil.rmtree(f"{base}/{s}"); os.makedirs(f"{base}/{s}")
# 1. obfuscate on the host (javascript-obfuscator 5.9.0 in a scratch dir; needs network once)
obf = f"{C.LDATA}/_obf"
os.makedirs(obf, exist_ok=True)
if not os.path.isdir(f"{obf}/node_modules/javascript-obfuscator"):
    subprocess.run(["npm", "install", "--silent", "javascript-obfuscator@5.9.0"], cwd=obf, check=True)
env = dict(os.environ, OBF_NODE_MODULES=f"{obf}/node_modules")
subprocess.run(["node", f"{H}/src/obfuscate.js", f"{H}/src/tool.js", f"{base}/target/app.js"], check=True, env=env)
# 2. record cases against the bundle, and check that the readable source agrees with the bundle on all of them
script = """set -e
mkdir -p /out/agent/samples /out/hidden
PYTHONPATH=/mlib python3 /src/gen_cases.py /out/target/app.js /out/hidden /out/agent/samples
PYTHONPATH=/mlib python3 /src/crosscheck.py /out/hidden /src/src/tool.js
"""
rc, o = C.docker_job(script, binds=[(H, "/src", "ro"), (os.path.join(H, "..", "..", "lib"), "/mlib", "ro"), (base, "/out", "rw")], timeout=3000)
print(o[-2500:]); assert rc == 0, "build failed"
shutil.copy(f"{H}/check.py", f"{base}/hidden/check.py")
open(f"{base}/agent/app.py", "w").write('#!/usr/bin/env python3\nimport sys\nsys.stderr.write("tool: not implemented\\n")\nsys.exit(2)\n')
shutil.copy(f"{H}/INSTRUCTION.md", f"{base}/agent/TASK.md")
src = open(f"{H}/src/ref_py/app.py").read()
open(f"{base}/ref_sub/app.py", "w").write(src)
cut = '"expr": cmd_expr, "tpl": cmd_tpl,\n        "pack": cmd_pack, "unpack": cmd_unpack, "seal": cmd_seal, "open": cmd_open}'
assert cut in src
open(f"{base}/ref_partial/app.py", "w").write(src.replace(cut, "}").replace('"b32": cmd_b32, "ser": cmd_ser, "de": cmd_de, }', '"b32": cmd_b32, "ser": cmd_ser, "de": cmd_de}'))
print("re-bundle built")
