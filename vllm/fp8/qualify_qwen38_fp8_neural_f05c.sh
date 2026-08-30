#!/usr/bin/env bash
# F05c bounded oracle for the corrected mixed spec/non-spec XPU GDN kernel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

exec env \
  IMAGE=neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-mixed-gdn-f05c-local \
  EXPECTED_IMAGE_ID=sha256:8e0e3deb0dbddfc7b2ca24cc06f5077a61d3b00c059d3bd0b963685acbd91b81 \
  LAYERNORM_SHA256=50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8 \
  SPECULATIVE_TOKENS=1 \
  RMS_PACKED_SERIAL_EXACT=1 \
  GDN_PERSISTENT_SCRATCH=1 \
  MAX_MODEL_LEN=32768 \
  MAX_NUM_SEQS=4 \
  MAX_NUM_BATCHED_TOKENS=32768 \
  F05B_OUTPUT_TOKENS=64 \
  F05B_BATCHES=1 \
  F05B_REQUIRE_SERIAL_EXACT=0 \
  EXTRA_WORKLOAD="$SCRIPT_DIR/probe_qwen38_fp8_concurrent.py" \
  EXTRA_WORKLOAD_RESULT=concurrent.json \
  CAMPAIGN_ID=f05c \
  CAMPAIGN_LABEL=F05c \
  CONTAINER_PREFIX=qwen38-fp8-neural-f05c-mixed-gdn-oracle \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f05c-mixed-gdn-oracle.v1 \
  COMPLETION_ROUTE=mtp1-packed-rms-deterministic-mixed-gdn-c4-oracle \
  COMPILE_ORACLE=1 \
  SHARED_CACHE=0 \
  INDUCTOR_COMBO_KERNELS=0 \
  INDUCTOR_BENCHMARK_COMBO_KERNEL=0 \
  INDUCTOR_MAX_AUTOTUNE=0 \
  INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
  INDUCTOR_AUTOTUNE_POINTWISE=0 \
  INDUCTOR_DETERMINISTIC_CONFIG=1 \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p0-fp16kv-f05c-mixed-gdn-oracle \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f05c_qwen38_fp8_neural/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f05c_qwen38_fp8_neural/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
