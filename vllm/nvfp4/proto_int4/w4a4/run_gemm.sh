#!/usr/bin/env bash
# GPU run of the built s4s4 W4A4 GEMM microbench. Needs the GPU lease (card 1).
# Build first with build_gemm.sh. Coordinator runs this and reports TOPS + PASS.
#   gpu-run --card 1 bash run_gemm.sh
set -euo pipefail
IMG=${IMG:-vllm-xpu-env:int8g-v0240}
BIN=/mnt/vm_8tb/b70/int4_dpas_build/w4a4/s4s4_gemm
[ -x "$BIN" ] || { echo "build first: bash build_gemm.sh"; exit 2; }
docker run --rm --device /dev/dri \
  -e ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-1}" \
  -e ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}" \
  -v /mnt/vm_8tb/b70/int4_dpas_build:/build \
  --entrypoint bash "$IMG" -c '
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
/build/w4a4/s4s4_gemm'
