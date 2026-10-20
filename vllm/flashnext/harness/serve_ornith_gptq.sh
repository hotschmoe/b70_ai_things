#!/usr/bin/env bash
# Serve Ornith-1.5-35B-A3B GPTQ int4 mixedcal-v2 on one B70.
# The process holds the card lease (via xpu-run.sh) until vLLM exits.
#
#   vllm/flashnext/harness/serve_ornith_gptq.sh
#   CARD=1 PORT=18081 MAX_LEN=32768 GRAPH=0 MTP=0 \
#     vllm/flashnext/harness/serve_ornith_gptq.sh
#
# GRAPH=1  drop --enforce-eager so vLLM may capture an XPU graph.
# MTP=1    speculative MTP, MTP_TOKENS draft tokens (default 2).
# NAME     docker container name (default vllm-fn-ornith-c<CARD>).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CARD="${CARD:-1}"
PORT="${PORT:-18081}"
# 32K prompt plus one generated token must fit. 33792 leaves 1K of
# room past the 32768-token TTFT point without growing KV much.
MAX_LEN="${MAX_LEN:-33792}"
GRAPH="${GRAPH:-0}"
MTP="${MTP:-0}"
MTP_TOKENS="${MTP_TOKENS:-2}"
GPU_MEM="${GPU_MEM:-0.90}"
NAME="${NAME:-vllm-fn-ornith-c${CARD}}"

MODEL=/mnt/vm_8tb/github/b70_ai_things/models/files/ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2
ALIAS=ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "serve_ornith_gptq: missing $MODEL/config.json" >&2
  exit 1
fi

ARGS=(
  vllm serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --served-model-name hotschmoe-dev "$ALIAS"
  --tensor-parallel-size 1
  --dtype float16
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM"
  --language-model-only
  --max-num-seqs 4
)

if [[ "$GRAPH" == "0" ]]; then
  ARGS+=(--enforce-eager)
else
  # Platform code turns off the CUDA-only fusion passes on XPU.
  # PIECEWISE is the mode this host has actually captured before.
  ARGS+=(--compilation-config '{"cudagraph_mode":"PIECEWISE"}')
fi

if [[ "$MTP" == "1" ]]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi

echo "serve_ornith_gptq: card=$CARD port=$PORT max_len=$MAX_LEN graph=$GRAPH mtp=$MTP name=$NAME"
echo "serve_ornith_gptq: ${ARGS[*]}"

exec "$HERE/xpu-run.sh" --card "$CARD" --name "$NAME" --port "$PORT" -- "${ARGS[@]}"
