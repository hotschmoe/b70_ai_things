#!/usr/bin/env bash
# Runs the three prebuilt ESIMD DPAS probes on CARD 1 inside one container.
# Invoke via: cd repo && ./bin/gpu-run --card 1 bash vllm/nvfp4/proto_int4/run.sh
set -o pipefail
IMG=vllm-xpu-env:int8g-v0240
OUT=/mnt/vm_8tb/b70/int4_dpas_build
docker run --rm --device /dev/dri -v "$OUT":/out \
  -e ZE_AFFINITY_MASK=1 \
  --entrypoint bash "$IMG" -c '
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
for p in s8 s4 s2; do
  echo "======== RUN $p ========"
  /out/int4_dpas_$p
  echo "exit=$?"
done
'
