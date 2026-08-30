#!/usr/bin/env bash
# F02d compile oracle: disable PyTorch Triton pointwise autotuning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

exec env \
  CAMPAIGN_ID=f02d \
  CAMPAIGN_LABEL=F02d \
  CONTAINER_PREFIX=qwen38-fp8-neural-f02d-no-pointwise-tune \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f02d-compile-oracle.v1 \
  COMPLETION_ROUTE=compile-oracle-no-pointwise-autotune \
  COMPILE_ORACLE=1 \
  SHARED_CACHE=0 \
  REQUIRE_REFERENCE_EXACT=0 \
  INDUCTOR_COMBO_KERNELS=0 \
  INDUCTOR_BENCHMARK_COMBO_KERNEL=0 \
  INDUCTOR_MAX_AUTOTUNE=0 \
  INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
  INDUCTOR_AUTOTUNE_POINTWISE=0 \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp0-p2p0-fp16kv-f02d-compile-oracle \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f02d_qwen38_fp8_neural/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f02d_qwen38_fp8_neural/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
