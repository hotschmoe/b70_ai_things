#!/usr/bin/env bash
# One-card Ornith GPTQ serve with exact-tail KV parking.
#
# A finished request stores GDN state and the partial attention page at
# E = num_computed_tokens. The next turn restores that tail from host
# RAM. GPU blocks are capped so B and C evict A off the card.
#
#   CARD=1 vllm/flashnext/kvoffload/serve_ornith.sh
#
# Holds the card lease until vLLM exits. Does not edit the A3 harness.
# GRAPH=0 falls back to eager. Default is PIECEWISE graphs.
# GRAPH_MODE=FULL_DECODE_ONLY selects the full decode graph.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
HARNESS="$REPO/vllm/flashnext/harness/xpu-run.sh"

CARD="${CARD:-1}"
PORT="${PORT:-18111}"
MAX_LEN="${MAX_LEN:-32768}"
# One ~28-block prompt fits. Two do not, so the second and third
# conversations evict the first from GPU. The host copy remains.
NUM_GPU_BLOCKS="${NUM_GPU_BLOCKS:-45}"
# 32 GiB pinned host pool. Four conversation slots. The container shm
# must be larger than this mapping (G1's 16g shm could not hold it).
CPU_BYTES="${CPU_BYTES:-34359738368}"
SHM_SIZE="${SHM_SIZE:-40g}"
MEMORY="${MEMORY:-120g}"
SLOTS="${SLOTS:-4}"
GRAPH="${GRAPH:-1}"
GRAPH_MODE="${GRAPH_MODE:-PIECEWISE}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,8}"
GPU_MEM="${GPU_MEM:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
# PIECEWISE on this image has no ocloc, so inductor autotune dies.
# D1's working path is compilation mode NONE plus breakable graphs.
BREAKABLE="${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}"
RECLAIM="${B70_XPU_CG_RECLAIM:-500}"
NAME="${NAME:-vllm-fn-g4-kv-c${CARD}}"
FORK="${FORK:-/mnt/vm_8tb/b70/build/vllm-flashnext/vllm}"

MODEL="$REPO/models/files/ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2"
ALIAS=ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "serve_ornith: missing $MODEL/config.json" >&2
  exit 1
fi

# ParkingOffloadConnector forces offload_prompt_only off and the
# whole-conversation policy. HybridCPUOffloadConnector stays importable
# for the G1 resend. The fork must precede the installed wheel so
# --park-finished-tail exists.
KV_JSON=$(printf '%s' \
  '{"kv_connector":"ParkingOffloadConnector","kv_role":"kv_both","kv_connector_module_path":"kvoffload.parking","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES}"',"slots":'"${SLOTS}"',"offload_prompt_only":false}}')

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
  --max-num-seqs "$MAX_NUM_SEQS"
  --enable-prefix-caching
  --mamba-cache-mode align
  --enable-prompt-tokens-details
  --no-async-scheduling
  --park-finished-tail
  --kv-transfer-config "$KV_JSON"
)
# 0 leaves the profiled pool. The default 45 is the T2 and T3 cap.
if [[ "$NUM_GPU_BLOCKS" != "0" ]]; then
  ARGS+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS")
fi

if [[ "$GRAPH" == "0" ]]; then
  ARGS+=(--enforce-eager)
else
  SIZES_JSON="[${CAPTURE_SIZES}]"
  case "$GRAPH_MODE" in
    FULL|FULL_DECODE_ONLY)
      CFG="{\"mode\":0,\"cudagraph_mode\":\"${GRAPH_MODE}\",\"cudagraph_capture_sizes\":${SIZES_JSON}}"
      ;;
    *)
      if [[ "$BREAKABLE" == "1" ]]; then
        CFG="{\"mode\":0,\"cudagraph_mode\":\"${GRAPH_MODE}\",\"cudagraph_capture_sizes\":${SIZES_JSON}}"
      else
        CFG="{\"cudagraph_mode\":\"${GRAPH_MODE}\",\"cudagraph_capture_sizes\":${SIZES_JSON}}"
      fi
      ;;
  esac
  ARGS+=(--compilation-config "$CFG")
fi

echo "g4 serve: card=$CARD port=$PORT max_len=$MAX_LEN blocks=$NUM_GPU_BLOCKS cpu_bytes=$CPU_BYTES shm=$SHM_SIZE memory=$MEMORY slots=$SLOTS seqs=$MAX_NUM_SEQS graph=$GRAPH mode=$GRAPH_MODE breakable=$BREAKABLE reclaim=$RECLAIM copy=${B70_KVPARK_COPY:-0} name=$NAME"
echo "g4 serve: ${ARGS[*]}"

export B70_AGENT="${B70_AGENT:-g4}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="$BREAKABLE"
export B70_XPU_CG_RECLAIM="$RECLAIM"
exec "$HARNESS" --card "$CARD" --name "$NAME" --port "$PORT" \
  --shm-size "$SHM_SIZE" --memory "$MEMORY" \
  --env "PYTHONPATH=${FORK}:$REPO/vllm/flashnext:$REPO/vllm/flashnext/plugins/b70_kvpark" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env "VLLM_USE_BREAKABLE_CUDAGRAPH=$BREAKABLE" \
  --env "B70_XPU_CG_RECLAIM=$RECLAIM" \
  --env "B70_KVPARK_COPY=${B70_KVPARK_COPY:-0}" \
  --env "B70_KVPARK_SO=${B70_KVPARK_SO:-/mnt/vm_8tb/b70/build/vllm-flashnext/b70_kvpark/libb70_kvpark.so}" \
  -- "${ARGS[@]}"
