#!/usr/bin/env bash
# F07f: full strict varied-prompt suite for the F07e FULL graph candidate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SEED_CACHE="${SEED_CACHE:-$ROOT/cache/f07e_qwen38_fp8_full_decode_triton_mtp1_draft_triton/20260830T162100Z}"

exec env \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f07f_qwen38_fp8_full_decode_triton_mtp1_strict/$STAMP}" \
  CACHE_DIR="${CACHE_DIR:-$ROOT/cache/f07f_qwen38_fp8_full_decode_triton_mtp1_strict/$STAMP}" \
  NAME="${NAME:-qwen38-fp8-f07f-full-mtp1-strict-$STAMP}" \
  PORT="${PORT:-18195}" \
  SEED_CACHE="$SEED_CACHE" \
  SPECULATIVE_TOKENS=1 \
  CAMPAIGN_LABEL=F07f \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p1-fp16kv-f07f-full-triton-draft-triton \
  COMPILATION_PROFILE=publisher \
  GPU_MEMORY_UTILIZATION=0.96 \
  XPU_GRAPH=1 \
  CUDAGRAPH_MODE=FULL_DECODE_ONLY \
  ATTENTION_BACKEND=TRITON_ATTN \
  DRAFT_ATTENTION_BACKEND=TRITON_ATTN \
  FORCE_GRAPH_WITH_COMM=1 \
  MAX_MODEL_LEN=1024 \
  MAX_NUM_SEQS=1 \
  MAX_NUM_BATCHED_TOKENS=1024 \
  PERFORMANCE_SCREEN_PROMPT=ALL \
  "$SCRIPT_DIR/run_qwen38_fp8_p2p_f06c_model_smoke.sh" "$@"
