#!/usr/bin/env bash
# Run O4e fused layerlet on CARD (default 1). Do not use card 0 if ornith_o1 is up.
set -o pipefail
IMG="${IMG:-vllm-xpu-env:int8g-v0260}"
DIR="${OUT:-/mnt/vm_8tb/b70/lmx_overnight/o4e_layerlet}"
CARD="${CARD:-1}"
test -x "$DIR/nvfp4_m1_layerlet" || { echo "missing $DIR/nvfp4_m1_layerlet"; exit 2; }
docker run --rm --device /dev/dri \
  -v "$DIR":/work \
  -e ZE_AFFINITY_MASK="$CARD" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  --entrypoint bash "$IMG" -c '
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
cd /work
echo "======== RUN nvfp4_m1_layerlet (affinity '"$CARD"') ========"
./nvfp4_m1_layerlet
echo "exit=$?"
'
