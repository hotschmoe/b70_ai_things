#!/usr/bin/env bash
# Flash-Next IQ3_XXS sidecar. World size 1 on card 0. The plugin owns
# card 1 and the remote expert set (Strata snake, default 138/374).
# No data parallel, no vLLM expert parallel, no oneCCL.
#
# The PLE table is one shared memmap (B70_PLE_NBYTES). This script does
# not exec xpu-run: a watchdog has to outlive the launcher and docker-stop
# the container if MemAvailable falls under 8 GiB.
#
# Do not launch while another agent holds the cards. Phase B waits for
# reports/1C7b_flashnext_serve.md, or for both gpu lock owner files to
# have been empty for 10 minutes.
#
#   PORT=18116 vllm/flashnext/harness/serve_flashnext_ep.sh
#   GRAPH=1 GRAPH_MODE=FULL_DECODE_ONLY ATTN=FLASH_ATTN MAX_SEQS=8 \
#     vllm/flashnext/harness/serve_flashnext_ep.sh
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PORT="${PORT:-18116}"
MAX_LEN="${MAX_LEN:-2048}"
MAX_SEQS="${MAX_SEQS:-2}"
GPU_MEM="${GPU_MEM:-0.90}"
GRAPH="${GRAPH:-0}"
GRAPH_MODE="${GRAPH_MODE:-FULL_DECODE_ONLY}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,8}"
ATTN="${ATTN:-}"
NAME="${NAME:-vllm-fn-h1-fn}"
MEMORY="${MEMORY:-100g}"

FORK=/mnt/vm_8tb/b70/build/vllm-flashnext/vllm
GGUF_PY=/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py
PLUGIN="$REPO/vllm/flashnext/plugins/b70_gguf_xpu"
EP_PLUGIN="$REPO/vllm/flashnext/plugins/b70_ep_xpu"
QSA_PLUGIN="$REPO/vllm/flashnext/plugins/b70_qsa_xpu"
HF=/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext
WEIGHTS=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
ALIAS=qwen3.8-flash-next-iq3-xxs
PROFILE="${B70_EP_PROFILE:-/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin}"
SO="${B70_GGUF_SO:-$PLUGIN/b70_gguf_xpu/_C.graph.so}"
PLE_NBYTES="${B70_PLE_NBYTES:-28800138240}"
HARD_KB=8388608
START_KB="${B70_EP_MEM_MIN_KB:-$HARD_KB}"
# About 27 GiB for the PLE map plus the private expert copy. A warning,
# not the abort. Raise B70_EP_MEM_MIN_KB to refuse a short start. That
# same variable is the in-process floor at PLE install, so a high value
# can also refuse a load that already spent the expert copy.
WARN_KB="${B70_EP_MEM_WARN_KB:-75497472}"

if [[ ! -f "$HF/config.json" ]]; then
  echo "serve_flashnext_ep: missing $HF/config.json" >&2
  exit 1
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "serve_flashnext_ep: missing $WEIGHTS" >&2
  exit 1
fi
if [[ ! -f "$PROFILE" ]]; then
  echo "serve_flashnext_ep: missing $PROFILE" >&2
  exit 1
fi
if [[ ! -f "$SO" ]]; then
  echo "serve_flashnext_ep: missing $SO" >&2
  exit 1
fi
if ! nm -D "$SO" | grep -q '_Z13b70_moe_layerPvPKtPKhiS3_iS3_iPKiPKfS3_iS3_iS3_iPfiiiii'; then
  echo "serve_flashnext_ep: $SO is missing b70_moe_layer" >&2
  exit 1
fi
if ! nm -D "$SO" | grep -q '_Z24b70_reserve_decode_queuePv'; then
  echo "serve_flashnext_ep: $SO is missing b70_reserve_decode_queue" >&2
  exit 1
fi

AVAIL_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
echo "serve_flashnext_ep: MemAvailable_kB=${AVAIL_KB:-0} start_floor_kB=$START_KB hard_floor_kB=$HARD_KB"
if [[ "${AVAIL_KB:-0}" -lt "$START_KB" ]]; then
  echo "ABORT MemAvailable ${AVAIL_KB:-0} kB is under start floor ${START_KB} kB" >&2
  exit 4
fi
if [[ "${AVAIL_KB:-0}" -lt "$WARN_KB" ]]; then
  echo "WARN MemAvailable ${AVAIL_KB:-0} kB is under ${WARN_KB} kB. The PLE map is 26.8 GiB and the expert pack is a private copy on top of it." >&2
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

QSA="${B70_QSA:-1}"
echo "serve_flashnext_ep: port=$PORT max_len=$MAX_LEN seqs=$MAX_SEQS graph=$GRAPH mode=$GRAPH_MODE attn=${ATTN:-default} qsa=$QSA name=$NAME mem=$MEMORY"
echo "serve_flashnext_ep: ${ARGS[*]}"

# Foreground xpu-run. The watchdog is a sibling, not a child of exec.
(
  while kill -0 "$$" 2>/dev/null; do
    avail=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [[ "${avail:-0}" -lt "$HARD_KB" ]]; then
      echo "ABORT_LOW_RAM avail_kb=$avail $(date -u +%H:%M:%S)" >&2
      docker stop -t 5 "$NAME" >/dev/null 2>&1 || true
      exit 0
    fi
    sleep 2
  done
) &
WATCH_PID=$!
cleanup() {
  kill "$WATCH_PID" 2>/dev/null || true
  wait "$WATCH_PID" 2>/dev/null || true
}
trap cleanup EXIT

OPT_ENV=()
if [[ -n "${B70_EP_SPIN:-}" ]]; then
  OPT_ENV+=(--env "B70_EP_SPIN=${B70_EP_SPIN}")
fi
if [[ -n "${B70_EP_SHARE0:-}" ]]; then
  OPT_ENV+=(--env "B70_EP_SHARE0=${B70_EP_SHARE0}")
fi

"$HERE/xpu-run.sh" --card both --memory "$MEMORY" --name "$NAME" --port "$PORT" \
  --env B70_GGUF=1 \
  --env B70_EP=1 \
  --env "B70_QSA=${QSA}" \
  --env "B70_GGUF_PATH=${WEIGHTS}" \
  --env "B70_GGUF_SO=${SO}" \
  --env "B70_EP_PROFILE=${PROFILE}" \
  --env "B70_EP_Q0=${B70_EP_Q0:-138}" \
  --env "B70_EP_SPLIT=${B70_EP_SPLIT:-profile}" \
  --env "B70_PLE_NBYTES=${PLE_NBYTES}" \
  --env "B70_EP_MEM_MIN_KB=${B70_EP_MEM_MIN_KB:-$HARD_KB}" \
  --env "B70_EP_MAX_T=${B70_EP_MAX_T:-8192}" \
  --env "B70_XPU_CG_RECLAIM=${B70_XPU_CG_RECLAIM:-500}" \
  --env "PYTHONPATH=${FORK}:${GGUF_PY}:${PLUGIN}:${EP_PLUGIN}:${QSA_PLUGIN}" \
  --env PYTHONUNBUFFERED=1 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_NO_USAGE_STATS=1 \
  --env VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-800}" \
  --env "B70_GGUF_REF=${B70_GGUF_REF:-0}" \
  --env "B70_EAGER_QK=${B70_EAGER_QK:-0}" \
  --env SYCL_CACHE_PERSISTENT=0 \
  --env SYCL_CACHE_DIR=/mnt/vm_8tb/b70/build/vllm-flashnext/cache/sycl-h1-fn \
  "${OPT_ENV[@]}" \
  -- "${ARGS[@]}"
