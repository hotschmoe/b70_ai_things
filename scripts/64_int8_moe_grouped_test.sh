#!/usr/bin/env bash
# Grouped INT8 MoE GEMM test -- reuses our dense int8 oneDNN op (int8_gemm_w8a8) as a per-expert grouped
# GEMM. Validates correctness + measures int8-vs-bf16 speedup + per-expert launch overhead (decode+prefill).
# Foundation for an XPUExpertsInt8. Single card. Route via gpu-run so it QUEUES behind any current GPU user:
#   ./gpu-run bash 64_int8_moe_grouped_test.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8g}"   # the image carrying our int8_gemm_w8a8 op
docker rm -f int8moe 2>/dev/null || true
docker run --rm --name int8moe --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -e ZE_AFFINITY_MASK=0 \
  -v "$ROOT/tmp_ssd:/tmp_ssd" -v "$ROOT/int8_moe_grouped_test.py:/int8_moe_grouped_test.py:ro" \
  -e TMPDIR=/tmp_ssd \
  --entrypoint bash "$IMG" -lc 'python /int8_moe_grouped_test.py'
