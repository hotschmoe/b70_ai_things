#!/usr/bin/env bash
# AOT-build the O4e fused layerlet proto for BMG-G31. Compile-only (no GPU).
set -o pipefail
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
SRC="$REPO/vllm/nvfp4/proto_moe_m1"
OUT="${OUT:-/mnt/vm_8tb/b70/lmx_overnight/o4e_layerlet}"
IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
mkdir -p "$OUT"
docker run --rm \
  -v "$SRC":/src:ro \
  -v "$OUT":/work \
  --entrypoint bash "$IMG" -c '
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
which icpx >/dev/null || { echo NO_icpx; exit 2; }
icpx --version | head -1
cd /work
rm -rf dump_nvfp4_m1_layerlet
mkdir -p dump_nvfp4_m1_layerlet
echo "=================== BUILD nvfp4_m1_layerlet ==================="
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/work/dump_nvfp4_m1_layerlet \
icpx -fsycl -std=c++17 -O2 \
  -fsycl-targets=intel_gpu_bmg_g31 \
  /src/nvfp4_m1_layerlet.cpp -o /work/nvfp4_m1_layerlet \
  2> /work/build_layerlet.err
rc=$?
echo "compile rc=$rc"
if [ $rc -ne 0 ]; then
  echo "--- errors (tail) ---"
  tail -80 /work/build_layerlet.err
  exit $rc
fi
ls -l /work/nvfp4_m1_layerlet
echo "--- dpas (expect none) ---"
grep -rIhoE "dpas[.a-z0-9_]*" dump_nvfp4_m1_layerlet 2>/dev/null | sort | uniq -c | head || true
echo "--- block_load / lsc_load ---"
grep -rIhoE "block_load|lsc_load[_a-z0-9]*" dump_nvfp4_m1_layerlet 2>/dev/null | sort | uniq -c | head -20 || true
echo "=== BUILD OK ==="
'
