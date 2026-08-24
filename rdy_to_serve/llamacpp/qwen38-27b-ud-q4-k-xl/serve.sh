#!/usr/bin/env bash
# Unsloth Qwen3.8-27B UD-Q4_K_XL quality-first daily-driver candidate.
# Exact path: TP=2, F16 KV, 262144 context, MTP off, lab doors off.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export NAME="${NAME:-qwen38_unsloth_ud_q4k_xl_tp2}"
export PORT="${PORT:-18080}"
export SERVED="${SERVED:-hotschmoe-dd}"
export HOST_MODELS="${HOST_MODELS:-$REPO/models/files/qwen3.8-27b/ud-q4-k-xl-unsloth}"
export MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-UD-Q4_K_XL.gguf}"
export MODEL_SIZE="${MODEL_SIZE:-17559178144}"
export MODEL_SHA256="${MODEL_SHA256:-3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e}"
export MODEL_LABEL="${MODEL_LABEL:-Unsloth UD-Q4_K_XL}"
export CTX_SIZE="${CTX_SIZE:-262144}"
export BATCH="${BATCH:-1024}"
export UBATCH="${UBATCH:-256}"
export LAB_DOORS="${LAB_DOORS:-0}"
export ENABLE_MTP="${ENABLE_MTP:-0}"

exec bash "$REPO/llamacpp/serve_qwen38_stock_q4km_tp2.sh" "${1:-start}"
