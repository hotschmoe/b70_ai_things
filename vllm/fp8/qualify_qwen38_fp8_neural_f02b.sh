#!/usr/bin/env bash
# F02b: repeat F02 with XPU combo kernels and their runtime benchmark disabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

exec env \
  CAMPAIGN_ID=f02b \
  CAMPAIGN_LABEL=F02b \
  CONTAINER_PREFIX=qwen38-fp8-neural-f02b-no-combo \
  ANALYZER_SCHEMA=b70.qwen38-fp8-neural-f02b-no-combo.v1 \
  COMPLETION_ROUTE=explicit-work-wait-no-combo-kernels \
  SHARED_CACHE=0 \
  REQUIRE_REFERENCE_EXACT=1 \
  INDUCTOR_COMBO_KERNELS=0 \
  INDUCTOR_BENCHMARK_COMBO_KERNEL=0 \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp0-p2p0-fp16kv-f02b-no-combo \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f02b_qwen38_fp8_neural/$STAMP}" \
  CACHE_ROOT="${CACHE_ROOT:-$ROOT/cache/f02b_qwen38_fp8_neural/$STAMP}" \
  "$SCRIPT_DIR/qualify_qwen38_fp8_neural_f02.sh" "$@"
