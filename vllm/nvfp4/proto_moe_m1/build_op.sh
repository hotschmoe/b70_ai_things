#!/usr/bin/env bash
# Build the O4c torch XPU op as a sidecar .so. Compile-only (no GPU).
# Does not touch the live serve _xpu_C.
set -o pipefail
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
SRC="$REPO/vllm/nvfp4/proto_moe_m1"
OUT="${OUT:-/mnt/vm_8tb/b70/lmx_overnight/o4_m1}"
IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
mkdir -p "$OUT"
docker run --rm \
  -v "$SRC":/src:ro \
  -v "$OUT":/work \
  --entrypoint bash "$IMG" -c '
set -o pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
which icpx >/dev/null || { echo NO_icpx; exit 2; }
TORCH=/opt/venv/lib/python3.12/site-packages/torch
test -f "$TORCH/include/c10/xpu/XPUStream.h" || { echo NO_torch_headers; exit 2; }
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path(\"include\"))")
icpx --version | head -1
echo "=================== BUILD b70_nvfp4_m1_gemv.so ==================="
icpx -fsycl -fPIC -shared -std=c++17 -O2 \
  -fsycl-targets=intel_gpu_bmg_g31 \
  -I"$TORCH/include" \
  -I"$TORCH/include/torch/csrc/api/include" \
  -I"$PYINC" \
  /src/nvfp4_m1_gemv_op.cpp \
  -L"$TORCH/lib" \
  -Wl,-rpath,"$TORCH/lib" \
  -Wl,--no-as-needed \
  -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu \
  -o /work/b70_nvfp4_m1_gemv.so \
  2> /work/build_m1_op.err
rc=$?
echo "compile rc=$rc"
if [ $rc -ne 0 ]; then
  echo "--- errors (tail) ---"
  tail -80 /work/build_m1_op.err
  exit $rc
fi
ls -l /work/b70_nvfp4_m1_gemv.so
echo "=== BUILD OP OK ==="
'
