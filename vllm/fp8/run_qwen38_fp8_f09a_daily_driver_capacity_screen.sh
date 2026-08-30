#!/usr/bin/env bash
# F09a: bounded 262K/c4 startup and singleton screen for dynamic MTP8/MTP1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SEED_CACHE="${SEED_CACHE:-$ROOT/cache/f08b_qwen38_fp8_full_decode_triton_mtp8_strict/20260830T172000Z}"

exec env \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f09a_qwen38_fp8_daily_driver_capacity/$STAMP}" \
  CACHE_DIR="${CACHE_DIR:-$ROOT/cache/f09a_qwen38_fp8_daily_driver_capacity/$STAMP}" \
  NAME="${NAME:-qwen38-fp8-f09a-daily-driver-capacity-$STAMP}" \
  PORT="${PORT:-18198}" \
  IMAGE=b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a \
  EXPECTED_IMAGE_ID=sha256:9ae697d4bbe64338518e8b139ec69e1d101d26bb6766c501c6ef83b022a9d5df \
  EXPECTED_LAYERNORM_SHA256=d911627c6c8f16fc11e02846286c378220120d24bd75898a5337fadf459318bd \
  SEED_CACHE="$SEED_CACHE" \
  SPECULATIVE_TOKENS=8 \
  SPECULATIVE_TOKENS_PER_BATCH_SIZE='[[1,1,8],[2,128,1]]' \
  CAMPAIGN_LABEL=F09a \
  SERVED=qwen3.8-27b-FP8-official-W8A16-dynamic-mtp8-mtp1-p2p1-fp16kv-f09a \
  COMPILATION_PROFILE=publisher \
  GPU_MEMORY_UTILIZATION=0.96 \
  XPU_GRAPH=1 \
  CUDAGRAPH_MODE=FULL_DECODE_ONLY \
  ATTENTION_BACKEND=TRITON_ATTN \
  DRAFT_ATTENTION_BACKEND=TRITON_ATTN \
  FORCE_GRAPH_WITH_COMM=1 \
  MAX_MODEL_LEN=262144 \
  MAX_NUM_SEQS=4 \
  MAX_NUM_BATCHED_TOKENS=32768 \
  PERFORMANCE_SCREEN_PROMPT=benchmark-analysis \
  "$SCRIPT_DIR/run_qwen38_fp8_p2p_f06c_model_smoke.sh" "$@"
