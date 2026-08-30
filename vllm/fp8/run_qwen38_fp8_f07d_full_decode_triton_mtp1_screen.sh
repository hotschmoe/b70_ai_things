#!/usr/bin/env bash
# F07d: bounded MTP1 FULL_DECODE_ONLY XPU-graph plus Triton-attention screen.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SEED_CACHE="${SEED_CACHE:-$ROOT/cache/f07c_qwen38_fp8_full_decode_triton_mtp0/20260830T160000Z}"

exec env \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f07d_qwen38_fp8_full_decode_triton_mtp1/$STAMP}" \
  CACHE_DIR="${CACHE_DIR:-$ROOT/cache/f07d_qwen38_fp8_full_decode_triton_mtp1/$STAMP}" \
  NAME="${NAME:-qwen38-fp8-f07d-full-decode-triton-mtp1-$STAMP}" \
  PORT="${PORT:-18193}" \
  SEED_CACHE="$SEED_CACHE" \
  SPECULATIVE_TOKENS=1 \
  CAMPAIGN_LABEL=F07d \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p1-fp16kv-f07d-full-decode-triton \
  COMPILATION_PROFILE=publisher \
  GPU_MEMORY_UTILIZATION=0.96 \
  XPU_GRAPH=1 \
  CUDAGRAPH_MODE=FULL_DECODE_ONLY \
  ATTENTION_BACKEND=TRITON_ATTN \
  FORCE_GRAPH_WITH_COMM=1 \
  MAX_MODEL_LEN=1024 \
  MAX_NUM_SEQS=1 \
  MAX_NUM_BATCHED_TOKENS=1024 \
  PERFORMANCE_SCREEN_PROMPT=benchmark-analysis \
  "$SCRIPT_DIR/run_qwen38_fp8_p2p_f06c_model_smoke.sh" "$@"
