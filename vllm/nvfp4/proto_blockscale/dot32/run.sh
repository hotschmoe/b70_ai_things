#!/usr/bin/env bash
# Run a prebuilt dot32 block-scaled DPAS binary on CARD 0. GPU touch -- coordinator only.
# Invoke: cd repo && ./bin/gpu-run --card 0 bash vllm/nvfp4/proto_blockscale/dot32/run.sh <bin> [MM=..]
#   <bin> in {dot32, bs, pc}.  Sweep: for M in 512 1024 2048; do MM=$M ... run.sh dot32; done
set -o pipefail
IMG=vllm-xpu-env:int8g-v0240
DIR=/mnt/vm_8tb/b70/nvfp4_blockscale/dot32
BIN=${1:-dot32}
docker run --rm --device /dev/dri \
  -v "$DIR":/work \
  -e ZE_AFFINITY_MASK=0 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e MM="${MM:-512}" -e ITERS="${ITERS:-30}" \
  --entrypoint bash "$IMG" -c '
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
cd /work
echo "======== RUN '"$BIN"' (card 0) MM=$MM ITERS=$ITERS ========"
./'"$BIN"'
echo "exit=$?"
'
