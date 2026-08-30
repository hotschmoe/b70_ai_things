#!/usr/bin/env bash
# F08b: strict varied-prompt suite for singleton MTP8 FULL graph.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SEED_CACHE="${SEED_CACHE:-$ROOT/cache/f08a_qwen38_fp8_full_decode_triton_mtp8/20260830T165500Z}"

exec env \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f08b_qwen38_fp8_full_decode_triton_mtp8_strict/$STAMP}" \
  CACHE_DIR="${CACHE_DIR:-$ROOT/cache/f08b_qwen38_fp8_full_decode_triton_mtp8_strict/$STAMP}" \
  NAME="${NAME:-qwen38-fp8-f08b-full-mtp8-strict-$STAMP}" \
  PORT="${PORT:-18197}" \
  IMAGE=b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a \
  EXPECTED_IMAGE_ID=sha256:9ae697d4bbe64338518e8b139ec69e1d101d26bb6766c501c6ef83b022a9d5df \
  EXPECTED_LAYERNORM_SHA256=d911627c6c8f16fc11e02846286c378220120d24bd75898a5337fadf459318bd \
  SEED_CACHE="$SEED_CACHE" \
  SPECULATIVE_TOKENS=8 \
  CAMPAIGN_LABEL=F08b \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp8-p2p1-fp16kv-f08b-full-triton \
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
