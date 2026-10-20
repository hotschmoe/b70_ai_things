#!/usr/bin/env bash
# One-card eager serve of the Qwen3.6-35B-A3B UD-IQ3_XXS GGUF.
# --model is the HF config/tokenizer directory. The GGUF is B70_GGUF_PATH.
# The fork is put on PYTHONPATH ahead of the installed wheel. B70_GGUF=1
# makes sitecustomize register the loader and load b70_gguf_xpu.
#
#   CARD=0 PORT=18104 vllm/flashnext/harness/serve_qwen36_gguf.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CARD="${CARD:-0}"
PORT="${PORT:-18104}"
MAX_LEN="${MAX_LEN:-2048}"
GPU_MEM="${GPU_MEM:-0.90}"
NAME="${NAME:-vllm-fn-qwen36-gguf-c${CARD}}"

FORK=/mnt/vm_8tb/b70/build/vllm-flashnext/vllm
GGUF_PY=/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py
PLUGIN=/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_gguf_xpu
HF=/mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B
WEIGHTS=/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
ALIAS=qwen3.6-35b-a3b-ud-iq3-xxs

if [[ ! -f "$HF/config.json" ]]; then
  echo "serve_qwen36_gguf: missing $HF/config.json" >&2
  exit 1
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "serve_qwen36_gguf: missing $WEIGHTS" >&2
  exit 1
fi

ARGS=(
  vllm serve "$HF"
  --host 0.0.0.0
  --port "$PORT"
  --served-model-name hotschmoe-dev "$ALIAS"
  --tensor-parallel-size 1
  --dtype bfloat16
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM"
  --max-num-seqs 1
  --language-model-only
  --enforce-eager
  --load-format b70_gguf
  --quantization b70_gguf
  --trust-remote-code
)

echo "serve_qwen36_gguf: card=$CARD port=$PORT max_len=$MAX_LEN name=$NAME"
echo "serve_qwen36_gguf: ${ARGS[*]}"

exec "$HERE/xpu-run.sh" --card "$CARD" --name "$NAME" --port "$PORT" \
  --env B70_GGUF=1 \
  --env "B70_GGUF_PATH=${WEIGHTS}" \
  --env "PYTHONPATH=${FORK}:${GGUF_PY}:${PLUGIN}" \
  --env PYTHONUNBUFFERED=1 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_NO_USAGE_STATS=1 \
  --env "B70_GGUF_REF=${B70_GGUF_REF:-0}" \
  --env "B70_EAGER_QK=${B70_EAGER_QK:-0}" \
  -- "${ARGS[@]}"
