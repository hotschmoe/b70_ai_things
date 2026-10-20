#!/usr/bin/env bash
# Serve a GPTQ checkpoint through the b70_lowbit XPU kernels on one B70.
# Modeled on harness/serve_ornith_gptq.sh. This script does not edit
# harness/. xpu-run.sh takes the gpu-run lease; do not wrap it again.
#
#   CARD=0 PORT=18083 \
#     vllm/flashnext/plugins/b70_lowbit_xpu/serve_lowbit.sh
#
#   MODEL=/mnt/cache/b70/devmodel/quant/ornith-rtn-int3-from-gptq4 \
#   ALIAS=ornith-1.5-35b-a3b-b70-lowbit-rtn-int3 \
#   CARD=0 PORT=18083 \
#     vllm/flashnext/plugins/b70_lowbit_xpu/serve_lowbit.sh
#
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
  --enforce-eager
)

echo "serve_lowbit: card=$CARD port=$PORT dtype=$DTYPE max_len=$MAX_LEN name=$NAME"
echo "serve_lowbit: model=$MODEL"
echo "serve_lowbit: alias=$ALIAS"
echo "serve_lowbit: launches=${B70_LOWBIT_MOE_LAUNCHES:-4}"
echo "serve_lowbit: ${ARGS[*]}"

ENV_ARGS=()
if [[ -n "${B70_LOWBIT_MOE_LAUNCHES:-}" ]]; then
  ENV_ARGS+=(--env "B70_LOWBIT_MOE_LAUNCHES=${B70_LOWBIT_MOE_LAUNCHES}")
fi

exec "$XPU_RUN" --card "$CARD" --name "$NAME" --port "$PORT" \
  "${ENV_ARGS[@]}" -- "${ARGS[@]}"
