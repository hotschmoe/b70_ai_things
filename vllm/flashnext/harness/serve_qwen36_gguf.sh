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
MAX_SEQS="${MAX_SEQS:-1}"
GPU_MEM="${GPU_MEM:-0.90}"
# GRAPH=1 drops --enforce-eager. GRAPH_MODE FULL_DECODE_ONLY matches the
# D1 recipe (compilation mode NONE, capture sizes 1,2,3,4,8).
GRAPH="${GRAPH:-0}"
GRAPH_MODE="${GRAPH_MODE:-FULL_DECODE_ONLY}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,8}"
ATTN="${ATTN:-}"
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
  --max-num-seqs "$MAX_SEQS"
  --language-model-only
  --load-format b70_gguf
  --quantization b70_gguf
  --trust-remote-code
)
if [[ "$GRAPH" == "0" ]]; then
  ARGS+=(--enforce-eager)
else
  SIZES_JSON="[${CAPTURE_SIZES}]"
  case "$GRAPH_MODE" in
    FULL|FULL_DECODE_ONLY)
      CFG="{\"mode\":0,\"cudagraph_mode\":\"${GRAPH_MODE}\",\"cudagraph_capture_sizes\":${SIZES_JSON}}"
      ;;
    *)
      CFG="{\"cudagraph_mode\":\"${GRAPH_MODE}\",\"cudagraph_capture_sizes\":${SIZES_JSON}}"
      ;;
  esac
  ARGS+=(--compilation-config "$CFG")
fi
if [[ -n "$ATTN" ]]; then
  ARGS+=(--attention-backend "$ATTN")
fi
if [[ -n "${GDN_BACKEND:-}" ]]; then
  ARGS+=(--gdn-prefill-backend "$GDN_BACKEND")
fi

echo "serve_qwen36_gguf: card=$CARD port=$PORT max_len=$MAX_LEN seqs=$MAX_SEQS graph=$GRAPH mode=$GRAPH_MODE attn=${ATTN:-default} name=$NAME"
echo "serve_qwen36_gguf: ${ARGS[*]}"

ENV_ARGS=()
for key in B70_XPU_CG_RECLAIM B70_GGUF_SO B70_REPLAY_LOG B70_STEP_LOG B70_COLL_LOG; do
  if [[ -n "${!key:-}" ]]; then
    ENV_ARGS+=(--env "${key}=${!key}")
  fi
done

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
  "${ENV_ARGS[@]}" \
  -- "${ARGS[@]}"
