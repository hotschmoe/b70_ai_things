#!/usr/bin/env bash
# Stage 3/4 of the vLLM v0.25.1 rebase: bake vllm-xpu-env:int8g-v0251 FROM vllm-xpu-env:v0251.
# Pure-Python layer (the int8 _xpu_C ops come from the RUNTIME-mounted .so, Stage 2):
#   1) copy contrib/vllm_int8_xpu/xpu_int8.py (XPUInt8ScaledMMLinearKernel, auto-registers fakes at import)
#      into vllm/model_executor/kernels/linear/scaled_mm/xpu_int8.py
#   2) patch vllm/model_executor/kernels/linear/__init__.py: import the class + register it in
#      _POSSIBLE_INT8_KERNELS[PlatformEnum.XPU] + harden the chooser (.get with a clear error).
# The three text anchors (import block, _POSSIBLE_INT8_KERNELS, platform_kernels chooser) are
# CHECKED AT RUNTIME below; if v0.24.0->v0.25.1 drifted them the script fails loudly (fix + rerun).
# xpu_int8.py auto-registers fakes at import -> :int8 and :int8g collapse to ONE image here.
set -uo pipefail
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
BASE="${BASE:-vllm-xpu-env:v0251}"
SRC="$REPO/vllm/contrib/vllm_int8_xpu/xpu_int8.py"
DATE="${DATE:-$(date +%Y%m%d 2>/dev/null || echo manual)}"
TAG="vllm-xpu-env:int8g-v0251-$DATE"

[ -f "$SRC" ] || { echo "MISSING $SRC"; exit 1; }
grep -q register_fake "$SRC" || { echo "FAIL: $SRC has no register_fake"; exit 1; }
docker image inspect "$BASE" >/dev/null 2>&1 || { echo "FAIL: base $BASE not present"; exit 1; }

docker rm -f int8g_v0251_build 2>/dev/null || true
docker run --name int8g_v0251_build -v "$SRC:/tmp/xpu_int8.py:ro" --entrypoint bash "$BASE" -c '
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
    # v0.25.1: the XPU scaled_mm import block renamed XPUFP8ScaledMMLinearKernel ->
    # XPUW8A8FP8LinearKernel + XPUW8A16FP8LinearKernel (verified in-image 2026-07-16).
    anchor = ("from vllm.model_executor.kernels.linear.scaled_mm.xpu import (\n"
              "    XPUFp8BlockScaledMMKernel,\n"
              "    XPUW8A8FP8LinearKernel,\n"
              "    XPUW8A16FP8LinearKernel,\n)")
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
[ "$(docker inspect -f '{{.State.ExitCode}}' int8g_v0251_build)" = 0 ] || { echo "BAKE FAILED"; docker logs int8g_v0251_build 2>&1 | tail -30; docker rm -f int8g_v0251_build; exit 1; }
# v0.25.1 FIX (2026-07-16): the upstream Dockerfile.xpu hardcoded a MINIMAL env (LD_LIBRARY_PATH =
# ccl/mpi/compiler only; CCL_ROOT/CCL_CONFIGURATION/FI_PROVIDER_PATH/OCL_ICD_FILENAMES/... all UNSET)
# instead of the full oneAPI setvars env that v0.24.0 baked. Two failures result:
#   1. torch.xpu sees 0 devices (level-zero/UR misses umf/tcm/tbb/pti/dnnl) -> vLLM "Failed to infer
#      device type".
#   2. TP=2 oneCCL init dies: "ze_handle_manager mem_to_ipc_handle: device_fd != invalid_fd failed"
#      (no CCL_CONFIGURATION=cpu_gpu_dpcpp / CCL_ROOT).
# FIX: bake the COMPLETE setvars env delta (the vars setvars sets/changes vs the image default) so the
# image is XPU + oneCCL ready with no in-container `source setvars`. Matches how v0.24.0's env was built.
DELTA="$(docker run --rm --entrypoint bash "$BASE" -c '
env | sort > /tmp/b.env
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
env | sort > /tmp/a.env
comm -13 /tmp/b.env /tmp/a.env | grep -vE "^(PWD|SHLVL|_|OLDPWD|SETVARS_|BASH)"')"
[ -n "$DELTA" ] || { echo "FAIL: could not capture setvars env delta"; docker rm -f int8g_v0251_build; exit 1; }
CHANGES=( --change 'ENTRYPOINT []' )
while IFS= read -r _l; do [ -z "$_l" ] && continue; CHANGES+=( --change "ENV $_l" ); done <<< "$DELTA"
docker commit "${CHANGES[@]}" \
  -m "vLLM 0.25.1 + XPUInt8ScaledMMLinearKernel registry + register_fake + FULL oneAPI setvars env (XPU device + oneCCL TP=2 fix)" \
  int8g_v0251_build "$TAG"
docker rm -f int8g_v0251_build
# v0.25.1 FIX (2026-07-16): the upstream Dockerfile bundled oneCCL 2021.15, which FAILS TP=2 worker
# init ("ze_handle_manager mem_to_ipc_handle: device_fd is invalid") for pidfd/sockets/drmfd alike.
# v0.24.0's oneCCL 2021.17 works. Swap the 2021.17 tree in over the 2021.15 path (so the baked
# CCL_ROOT/LD_LIBRARY_PATH resolve unchanged). Source = extracted from vllm-xpu-env:int8g-v0240.
CCL217="${CCL217:-/mnt/vm_8tb/b70/ccl_2021.17/2021.17}"
if [ -d "$CCL217" ]; then
  docker rm -f int8g_cclfix >/dev/null 2>&1
  docker run -d --name int8g_cclfix --entrypoint sleep "$TAG" 600 >/dev/null
  docker exec int8g_cclfix rm -rf /opt/intel/oneapi/ccl/2021.15
  docker cp "$CCL217" int8g_cclfix:/opt/intel/oneapi/ccl/2021.15
  docker commit -m "oneCCL 2021.17 (replaces bundled 2021.15 that fails ze mem_to_ipc_handle at TP=2)" int8g_cclfix "$TAG" >/dev/null
  docker rm -f int8g_cclfix >/dev/null
  echo "=== oneCCL 2021.17 swapped in (TP=2 fix) ==="
else
  echo "[!] WARN: $CCL217 not found -- TP=2 will fail on bundled oneCCL 2021.15. Extract from int8g-v0240:"
  echo "    cid=\$(docker create vllm-xpu-env:int8g-v0240); docker cp \$cid:/opt/intel/oneapi/ccl/2021.17 /mnt/vm_8tb/b70/ccl_2021.17/; docker rm \$cid"
fi
docker tag "$TAG" vllm-xpu-env:int8g-v0251
echo "=== built ==="; docker images "$TAG" --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
echo "digest (record in README): $(docker image inspect --format '{{.Id}}' "$TAG")"
