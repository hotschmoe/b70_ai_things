#!/usr/bin/env bash
# Run a prebuilt block-scaled DPAS binary on CARD 0.
# Invoke via: cd repo && ./bin/gpu-run --card 0 bash /mnt/vm_8tb/b70/nvfp4_blockscale/run.sh <binname> [envs...]
set -o pipefail
IMG=vllm-xpu-env:int8g-v0240
DIR=/mnt/vm_8tb/b70/nvfp4_blockscale
BIN=${1:-bs_dpas_m1}
MODELS=/mnt/vm_8tb/github/b70_ai_things/models/files
docker run --rm --device /dev/dri \
  -v "$DIR":/work -v "$MODELS":/models \
  -e ZE_AFFINITY_MASK=0 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e BS_TILE_BIN="${BS_TILE_BIN:-}" \
  --entrypoint bash "$IMG" -c '
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
cd /work
echo "======== RUN '"$BIN"' (card 0) ========"
./'"$BIN"'
echo "exit=$?"
'
