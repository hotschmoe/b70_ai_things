#!/usr/bin/env bash
# Serve a GPTQ checkpoint through the b70_lowbit XPU kernels on one B70.
# Modeled on harness/serve_ornith_gptq.sh. This script does not edit
# harness/. xpu-run.sh takes the gpu-run lease; do not wrap it again.
#
#   CARD=1 PORT=18084 \
#     vllm/flashnext/plugins/b70_lowbit_xpu/serve_lowbit.sh
#
#   GRAPH=1 GRAPH_MODE=FULL_DECODE_ONLY ATTN=FLASH_ATTN \
#   B70_XPU_CG_RECLAIM=500 B70_LOWBIT_MOE_LAUNCHES=2 \
#   CARD=1 PORT=18084 \
#     vllm/flashnext/plugins/b70_lowbit_xpu/serve_lowbit.sh
#
# GRAPH=1 drops --enforce-eager. FULL and FULL_DECODE_ONLY set compilation
# mode NONE (no inductor). Capture sizes default to 1,2,3,4,8.
# The kernel reads bf16 activations and bf16 scales, so the default dtype
# is bfloat16. A3's published int4 baseline was float16 + oneDNN.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
XPU_RUN="$REPO/vllm/flashnext/harness/xpu-run.sh"

CARD="${CARD:-0}"
PORT="${PORT:-18083}"
MAX_LEN="${MAX_LEN:-33792}"
GPU_MEM="${GPU_MEM:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
GRAPH="${GRAPH:-0}"
GRAPH_MODE="${GRAPH_MODE:-FULL_DECODE_ONLY}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,8}"
ATTN="${ATTN:-}"
GDN_BACKEND="${GDN_BACKEND:-}"
ASYNC="${ASYNC:-1}"
NAME="${NAME:-vllm-fn-lowbit-c${CARD}}"
MODEL="${MODEL:-$REPO/models/files/ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2}"
ALIAS="${ALIAS:-ornith-1.5-35b-a3b-b70-lowbit-int4}"

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "serve_lowbit: missing $MODEL/config.json" >&2
  exit 1
fi
if [[ ! -x "$XPU_RUN" ]]; then
  echo "serve_lowbit: missing $XPU_RUN" >&2
  exit 1
fi

ARGS=(
  vllm serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --served-model-name hotschmoe-dev "$ALIAS"
  --quantization b70_lowbit
  --tensor-parallel-size 1
  --dtype "$DTYPE"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM"
  --language-model-only
  --max-num-seqs 4
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
      if [[ "${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}" == "1" ]]; then
        CFG="{\"mode\":0,\"cudagraph_mode\":\"${GRAPH_MODE}\",\"cudagraph_capture_sizes\":${SIZES_JSON}}"
      else
        CFG="{\"cudagraph_mode\":\"${GRAPH_MODE}\",\"cudagraph_capture_sizes\":${SIZES_JSON}}"
      fi
      ;;
  esac
  ARGS+=(--compilation-config "$CFG")
fi

if [[ -n "$ATTN" ]]; then
  ARGS+=(--attention-backend "$ATTN")
fi
if [[ -n "$GDN_BACKEND" ]]; then
  ARGS+=(--gdn-prefill-backend "$GDN_BACKEND")
fi
if [[ "$ASYNC" == "0" ]]; then
  ARGS+=(--no-async-scheduling)
fi

echo "serve_lowbit: card=$CARD port=$PORT dtype=$DTYPE max_len=$MAX_LEN graph=$GRAPH mode=$GRAPH_MODE attn=${ATTN:-default} name=$NAME"
echo "serve_lowbit: model=$MODEL"
echo "serve_lowbit: alias=$ALIAS"
echo "serve_lowbit: launches=${B70_LOWBIT_MOE_LAUNCHES:-4}"
echo "serve_lowbit: ${ARGS[*]}"

ENV_ARGS=()
if [[ -n "${B70_LOWBIT_MOE_LAUNCHES:-}" ]]; then
  ENV_ARGS+=(--env "B70_LOWBIT_MOE_LAUNCHES=${B70_LOWBIT_MOE_LAUNCHES}")
fi
if [[ -n "${B70_LOWBIT_SO:-}" ]]; then
  ENV_ARGS+=(--env "B70_LOWBIT_SO=${B70_LOWBIT_SO}")
fi
for key in B70_XPU_CG_RECLAIM B70_REPLAY_LOG B70_STEP_TIMER B70_STEP_LOG \
           VLLM_USE_BREAKABLE_CUDAGRAPH B70_LOWBIT_PREFILL; do
  if [[ -n "${!key:-}" ]]; then
    ENV_ARGS+=(--env "${key}=${!key}")
  fi
done

exec "$XPU_RUN" --card "$CARD" --name "$NAME" --port "$PORT" \
  "${ENV_ARGS[@]}" -- "${ARGS[@]}"
