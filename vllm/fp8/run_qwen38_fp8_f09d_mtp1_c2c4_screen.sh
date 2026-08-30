#!/usr/bin/env bash
# F09d: static MTP1 FULL graph at 262K with matched c2/c4 throughput probes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SEED_CACHE="${SEED_CACHE:-$ROOT/cache/f09a_qwen38_fp8_daily_driver_capacity/20260830T180000Z}"
QUALITY="$ROOT/steve-repro/qwen38-fp8-neural-20260829/source/experiments/qwen38-27b-b70/scripts/qwen38-concurrent-quality-canary.py"

exec env \
  STAMP="$STAMP" \
  RESULT_DIR="${RESULT_DIR:-$ROOT/results/f09d_qwen38_fp8_mtp1_c2c4/$STAMP}" \
  CACHE_DIR="${CACHE_DIR:-$ROOT/cache/f09d_qwen38_fp8_mtp1_c2c4/$STAMP}" \
  NAME="${NAME:-qwen38-fp8-f09d-mtp1-c2c4-$STAMP}" \
  PORT="${PORT:-18201}" \
  IMAGE=b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a \
  EXPECTED_IMAGE_ID=sha256:9ae697d4bbe64338518e8b139ec69e1d101d26bb6766c501c6ef83b022a9d5df \
  EXPECTED_LAYERNORM_SHA256=d911627c6c8f16fc11e02846286c378220120d24bd75898a5337fadf459318bd \
  SEED_CACHE="$SEED_CACHE" \
  SPECULATIVE_TOKENS=1 \
  CAMPAIGN_LABEL=F09d \
  SERVED=qwen3.8-27b-FP8-official-W8A16-mtp1-p2p1-fp16kv-f09d-full-c4-262k \
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
  CONCURRENT_PROBE_LEVELS='2 4' \
  CONCURRENT_PROBE_BATCHES=2 \
  CONCURRENT_PROBE_OUTPUT_TOKENS=512 \
  EXTRA_SMOKE="$QUALITY" \
  "$SCRIPT_DIR/run_qwen38_fp8_p2p_f06c_model_smoke.sh" "$@"
