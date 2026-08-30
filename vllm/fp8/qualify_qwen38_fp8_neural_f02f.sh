#!/usr/bin/env bash
# F02f full target gate with explicit deterministic reduction filtering.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

exec env \
  CAMPAIGN_ID=f02f \
  CAMPAIGN_LABEL=F02f \
  CONTAINER_PREFIX=qwen38-fp8-neural-f02f-deterministic-reduction \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f02f-full-target.v1 \
  COMPLETION_ROUTE=explicit-work-wait-deterministic-reduction \
  COMPILE_ORACLE=0 \
  SHARED_CACHE=0 \
  REQUIRE_REFERENCE_EXACT=1 \
  INDUCTOR_COMBO_KERNELS=0 \
  INDUCTOR_BENCHMARK_COMBO_KERNEL=0 \
  INDUCTOR_MAX_AUTOTUNE=0 \
  INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
  INDUCTOR_AUTOTUNE_POINTWISE=0 \
  INDUCTOR_DETERMINISTIC_CONFIG=1 \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp0-p2p0-fp16kv-f02f-deterministic \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f02f_qwen38_fp8_neural/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f02f_qwen38_fp8_neural/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
