#!/usr/bin/env bash
set -o pipefail
IMG=vllm-xpu-env:int8g-v0240
OUT=/mnt/vm_8tb/b70/int4_dpas_build
docker run --rm --device /dev/dri -v "$OUT":/out -e ZE_AFFINITY_MASK=1 \
  --entrypoint bash "$IMG" -c '
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
for n in s8 s4 s2; do echo "==== $n ===="; /out/bench_$n; done '
