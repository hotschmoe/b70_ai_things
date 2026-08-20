#!/usr/bin/env bash
# Run the prebuilt O4 proto on CARD (default 1). Do not use card 0 if ornith_o3 is up.
# Invoke: ./bin/gpu-run --card 1 bash vllm/nvfp4/proto_moe_m1/run.sh
set -o pipefail
IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
DIR="${OUT:-/mnt/vm_8tb/b70/lmx_overnight/o4_m1}"
CARD="${CARD:-1}"
test -x "$DIR/nvfp4_m1_gemv" || { echo "missing $DIR/nvfp4_m1_gemv"; exit 2; }
docker run --rm --device /dev/dri \
  -v "$DIR":/work \
  -e ZE_AFFINITY_MASK="$CARD" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  --entrypoint bash "$IMG" -c '
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
cd /work
echo "======== RUN nvfp4_m1_gemv (affinity '"$CARD"') ========"
./nvfp4_m1_gemv
echo "exit=$?"
'
