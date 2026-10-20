#!/usr/bin/env bash
# Serve Ornith-1.5-35B-A3B GPTQ int4 mixedcal-v2 on one B70.
# The process holds the card lease (via xpu-run.sh) until vLLM exits.
#
#   vllm/flashnext/harness/serve_ornith_gptq.sh
#   CARD=1 PORT=18081 MAX_LEN=32768 GRAPH=0 MTP=0 \
#     vllm/flashnext/harness/serve_ornith_gptq.sh
#
# GRAPH=1  drop --enforce-eager so vLLM may capture an XPU graph.
# GRAPH_MODE  PIECEWISE (default), FULL, FULL_DECODE_ONLY, FULL_AND_PIECEWISE.
#             FULL and FULL_DECODE_ONLY also set compilation mode NONE so
#             inductor does not run. Capture sizes default to 1,2,3,4,8.
# ATTN     optional --attention-backend value (TRITON_ATTN, FLASH_ATTN).
# GDN_BACKEND  optional --gdn-prefill-backend (triton skips the SYCL GDN op).
# MTP=1    speculative MTP, MTP_TOKENS draft tokens (default 2).
# ASYNC=0  pass --no-async-scheduling (default on). Async overlaps host
#          with the next step and can add a step of decode latency.
# NAME     docker container name (default vllm-fn-ornith-c<CARD>).
# Forwards into the container when set in the environment:
#   B70_STEP_TIMER B70_DISPATCH_STEP B70_CPROFILE_STEP B70_PROFILE_STEP
#   B70_STEP_LOG B70_XPU_CG_RECLAIM B70_REPLAY_LOG
#   B70_D2_PROFILE B70_D2_LOG B70_D2_DONE
#   B70_D2_SHISA_MTP B70_D2_SHISA_PATH B70_D2_SHISA_LOG
#   B70_D2_INT4_LINEAR B70_D2_INT4_LOG
#   VLLM_USE_BREAKABLE_CUDAGRAPH
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CARD="${CARD:-1}"
PORT="${PORT:-18081}"
# 32K prompt plus one generated token must fit. 33792 leaves 1K of
# room past the 32768-token TTFT point without growing KV much.
MAX_LEN="${MAX_LEN:-33792}"
GRAPH="${GRAPH:-0}"
GRAPH_MODE="${GRAPH_MODE:-PIECEWISE}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,8}"
ATTN="${ATTN:-}"
GDN_BACKEND="${GDN_BACKEND:-}"
MTP="${MTP:-0}"
MTP_TOKENS="${MTP_TOKENS:-2}"
ASYNC="${ASYNC:-1}"
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
  # FULL cudagraphs do not need inductor. PIECEWISE does.
  SIZES_JSON="[${CAPTURE_SIZES}]"
  case "$GRAPH_MODE" in
    FULL|FULL_DECODE_ONLY)
      CFG="{\"mode\":0,\"cudagraph_mode\":\"${GRAPH_MODE}\",\"cudagraph_capture_sizes\":${SIZES_JSON}}"
      ;;
    *)
      # Breakable piecewise captures without inductor. Leaving mode unset
      # otherwise lets vLLM compile, which PIECEWISE needs on CUDA.
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

if [[ "$MTP" == "1" ]]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi

if [[ "$ASYNC" == "0" ]]; then
  ARGS+=(--no-async-scheduling)
fi

ENV_ARGS=()
for key in B70_STEP_TIMER B70_DISPATCH_STEP B70_CPROFILE_STEP B70_PROFILE_STEP \
           B70_STEP_LOG B70_XPU_CG_RECLAIM B70_REPLAY_LOG \
           B70_D2_PROFILE B70_D2_LOG B70_D2_DONE \
           B70_D2_SHISA_MTP B70_D2_SHISA_PATH B70_D2_SHISA_LOG \
           B70_D2_INT4_LINEAR B70_D2_INT4_LOG \
           VLLM_USE_BREAKABLE_CUDAGRAPH; do
  if [[ -n "${!key:-}" ]]; then
    ENV_ARGS+=(--env "${key}=${!key}")
  fi
done

echo "serve_ornith_gptq: card=$CARD port=$PORT max_len=$MAX_LEN graph=$GRAPH mode=$GRAPH_MODE attn=${ATTN:-default} gdn=${GDN_BACKEND:-default} mtp=$MTP async=$ASYNC name=$NAME"
echo "serve_ornith_gptq: ${ARGS[*]}"

exec "$HERE/xpu-run.sh" --card "$CARD" --name "$NAME" --port "$PORT" "${ENV_ARGS[@]}" -- "${ARGS[@]}"
