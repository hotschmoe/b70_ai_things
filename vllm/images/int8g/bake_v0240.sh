#!/usr/bin/env bash
# Stage 3/4 of the vLLM v0.24.0 rebase: bake vllm-xpu-env:int8g-v0240 FROM vllm-xpu-env:v0240.
# Pure-Python layer (the int8 _xpu_C ops come from the RUNTIME-mounted .so, Stage 2):
#   1) copy contrib/vllm_int8_xpu/xpu_int8.py (XPUInt8ScaledMMLinearKernel, auto-registers fakes at import)
#      into vllm/model_executor/kernels/linear/scaled_mm/xpu_int8.py
#   2) patch vllm/model_executor/kernels/linear/__init__.py: import the class + register it in
#      _POSSIBLE_INT8_KERNELS[PlatformEnum.XPU] + harden the chooser (.get with a clear error).
# The three text anchors were verified present in v0.24.0 (JOURNAL 2026-07-03, lines 173/282/512).
# xpu_int8.py auto-registers fakes at import -> :int8 and :int8g collapse to ONE image here.
set -uo pipefail
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
BASE="${BASE:-vllm-xpu-env:v0240}"
SRC="$REPO/vllm/contrib/vllm_int8_xpu/xpu_int8.py"
DATE="${DATE:-$(date +%Y%m%d 2>/dev/null || echo manual)}"
TAG="vllm-xpu-env:int8g-v0240-$DATE"

[ -f "$SRC" ] || { echo "MISSING $SRC"; exit 1; }
grep -q register_fake "$SRC" || { echo "FAIL: $SRC has no register_fake"; exit 1; }
docker image inspect "$BASE" >/dev/null 2>&1 || { echo "FAIL: base $BASE not present"; exit 1; }

docker rm -f int8g_v0240_build 2>/dev/null || true
docker run --name int8g_v0240_build -v "$SRC:/tmp/xpu_int8.py:ro" --entrypoint bash "$BASE" -c '
set -e
python - <<'"'"'PY'"'"'
import os, shutil, sys, vllm
BASE = os.path.dirname(vllm.__file__)
# copy class into EVERY resolvable scaled_mm dir (editable /workspace + venv site-packages)
import glob
dsts = set()
for root in ("/workspace", "/opt/venv", BASE):
    for d in glob.glob(root + "/**/model_executor/kernels/linear/scaled_mm", recursive=True):
        dsts.add(os.path.join(d, "xpu_int8.py"))
dsts.add(os.path.join(BASE, "model_executor/kernels/linear/scaled_mm/xpu_int8.py"))
for dst in sorted(dsts):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile("/tmp/xpu_int8.py", dst); print("copied class ->", dst)

# patch each resolvable linear/__init__.py registry
inits = set()
for root in ("/workspace", "/opt/venv", BASE):
    for f in glob.glob(root + "/**/model_executor/kernels/linear/__init__.py", recursive=True):
        inits.add(f)
inits.add(os.path.join(BASE, "model_executor/kernels/linear/__init__.py"))
patched = 0
for V in sorted(inits):
    if not os.path.exists(V): continue
    src = open(V).read()
    if "XPUInt8ScaledMMLinearKernel" in src:
        print("already patched:", V); patched += 1; continue
    anchor = ("from vllm.model_executor.kernels.linear.scaled_mm.xpu import (\n"
              "    XPUFp8BlockScaledMMKernel,\n"
              "    XPUFP8ScaledMMLinearKernel,\n)")
    if anchor not in src: sys.exit("FAIL anchor(import) not found in "+V)
    src = src.replace(anchor, anchor +
          "\nfrom vllm.model_executor.kernels.linear.scaled_mm.xpu_int8 import (\n"
          "    XPUInt8ScaledMMLinearKernel,\n)", 1)
    key = "_POSSIBLE_INT8_KERNELS"; i = src.find(key)
    if i < 0: sys.exit("FAIL _POSSIBLE_INT8_KERNELS not found in "+V)
    j = src.find("\n}", i)
    if j < 0: sys.exit("FAIL end of INT8 dict not found in "+V)
    if "PlatformEnum.XPU" not in src[i:j]:
        src = src[:j] + "\n    PlatformEnum.XPU: [XPUInt8ScaledMMLinearKernel]," + src[j:]
    sub = "    platform_kernels = possible_kernels[current_platform._enum]"
    if sub in src:
        src = src.replace(sub,
            "    platform_kernels = possible_kernels.get(current_platform._enum)\n"
            "    if not platform_kernels:\n"
            "        raise ValueError(\n"
            "            \"No ScaledMM linear kernels registered for platform \"\n"
            "            f\"{current_platform._enum}.\")", 1)
    open(V, "w").write(src); print("PATCHED registry ->", V); patched += 1
if patched < 1: sys.exit("FAIL: patched no __init__.py")
# import smoke: class must be importable + in the registry (ops come from runtime .so, not checked here)
from vllm.model_executor.kernels.linear import _POSSIBLE_INT8_KERNELS
from vllm.platforms import PlatformEnum
assert PlatformEnum.XPU in _POSSIBLE_INT8_KERNELS, "XPU not registered"
from vllm.model_executor.kernels.linear.scaled_mm.xpu_int8 import XPUInt8ScaledMMLinearKernel
print("REGISTRY OK: XPU ->", _POSSIBLE_INT8_KERNELS[PlatformEnum.XPU])
print("BAKE_OK")
PY
'
[ "$(docker inspect -f '{{.State.ExitCode}}' int8g_v0240_build)" = 0 ] || { echo "BAKE FAILED"; docker logs int8g_v0240_build 2>&1 | tail -30; docker rm -f int8g_v0240_build; exit 1; }
docker commit --change 'ENTRYPOINT []' \
  -m "vLLM 0.24.0 + XPUInt8ScaledMMLinearKernel registry + register_fake (int8 W8A8 XPU graph capture)" \
  int8g_v0240_build "$TAG"
docker rm -f int8g_v0240_build
docker tag "$TAG" vllm-xpu-env:int8g-v0240
echo "=== built ==="; docker images "$TAG" --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
echo "digest (record in README): $(docker image inspect --format '{{.Id}}' "$TAG")"
