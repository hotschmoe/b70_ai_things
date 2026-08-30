#!/usr/bin/env bash
# F05e target-only control for the corrected mixed-path GDN kernel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
F05D_ROOT="${F05D_ROOT:-$ROOT/results/f05d_qwen38_fp8_neural/20260830T075200Z}"
F05D_CACHE="${F05D_CACHE:-$ROOT/cache/f05d_qwen38_fp8_neural/20260830T075200Z/attempt-1}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

for required in \
  "$F05D_ROOT/attempt-1/performance.json" \
  "$F05D_ROOT/attempt-2/performance.json" \
  "$F05D_CACHE"; do
  [ -e "$required" ] || {
    echo "missing frozen F05d corrected-kernel evidence: $required" >&2
    exit 1
  }
done

exec env \
  IMAGE=neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-mixed-gdn-f05c-local \
  EXPECTED_IMAGE_ID=sha256:8e0e3deb0dbddfc7b2ca24cc06f5077a61d3b00c059d3bd0b963685acbd91b81 \
  LAYERNORM_SHA256=50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8 \
  SPECULATIVE_TOKENS=0 \
  RMS_PACKED_SERIAL_EXACT=1 \
  GDN_PERSISTENT_SCRATCH=1 \
  MAX_MODEL_LEN=32768 \
  MAX_NUM_SEQS=1 \
  MAX_NUM_BATCHED_TOKENS=32768 \
  CAMPAIGN_ID=f05e \
  CAMPAIGN_LABEL=F05e \
  CONTAINER_PREFIX=qwen38-fp8-neural-f05e-mixed-gdn-mtp0 \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f05e-mixed-gdn-mtp0.v1 \
  COMPLETION_ROUTE=target-only-deterministic-mixed-gdn \
  COMPILE_ORACLE=0 \
  SHARED_CACHE=1 \
  SEED_CACHE_FROM="$F05D_CACHE" \
  REQUIRE_REFERENCE_EXACT=1 \
  INDUCTOR_COMBO_KERNELS=0 \
  INDUCTOR_BENCHMARK_COMBO_KERNEL=0 \
  INDUCTOR_MAX_AUTOTUNE=0 \
  INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
  INDUCTOR_AUTOTUNE_POINTWISE=0 \
  INDUCTOR_DETERMINISTIC_CONFIG=1 \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp0-p2p0-fp16kv-f05e-mixed-gdn \
  PUBLISHER_A="$F05D_ROOT/attempt-1/performance.json" \
  PUBLISHER_B="$F05D_ROOT/attempt-2/performance.json" \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f05e_qwen38_fp8_neural/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f05e_qwen38_fp8_neural/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
