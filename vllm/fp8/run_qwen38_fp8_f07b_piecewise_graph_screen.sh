#!/usr/bin/env bash
# F07b: bounded MTP1 PIECEWISE XPU-graph natural-prompt speed screen.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SEED_CACHE="${SEED_CACHE:-$ROOT/cache/f07a_qwen38_fp8_neural_publisher_exact/20260830T152100Z/attempt-1}"

exec env \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f07b_qwen38_fp8_piecewise_graph/$STAMP}" \
  CACHE_DIR="${CACHE_DIR:-$ROOT/cache/f07b_qwen38_fp8_piecewise_graph/$STAMP}" \
  NAME="${NAME:-qwen38-fp8-f07b-piecewise-graph-$STAMP}" \
  SEED_CACHE="$SEED_CACHE" \
  SPECULATIVE_TOKENS=1 \
  CAMPAIGN_LABEL=F07b \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p1-fp16kv-f07b-piecewise-graph \
  COMPILATION_PROFILE=publisher \
  GPU_MEMORY_UTILIZATION=0.96 \
  XPU_GRAPH=1 \
  CUDAGRAPH_MODE=PIECEWISE \
  FORCE_GRAPH_WITH_COMM=1 \
  MAX_MODEL_LEN=1024 \
  MAX_NUM_SEQS=1 \
  MAX_NUM_BATCHED_TOKENS=1024 \
  PERFORMANCE_SCREEN_PROMPT=benchmark-analysis \
  "$SCRIPT_DIR/run_qwen38_fp8_p2p_f06c_model_smoke.sh" "$@"
