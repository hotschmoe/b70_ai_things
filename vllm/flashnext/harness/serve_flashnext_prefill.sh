#!/usr/bin/env bash
# X3 stock DP=2/EP prefill candidate. Run only after priority windows and
# health/collective-health. xpu-run owns both GPU leases. No service actions.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
BUILD=/mnt/vm_8tb/b70/build/vllm-flashnext
PLUGIN=$REPO/vllm/flashnext/plugins/b70_gguf_xpu
HF=/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext
GGUF=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
PORT=${PORT:-18135}
NAME=${NAME:-vllm-fn-x3-prefill}
SO=${B70_GGUF_SO:-$PLUGIN/b70_gguf_xpu/_C.c12.so}
WIDE=${B70_PREFILL_WIDE_SO:-$PLUGIN/b70_gguf_xpu/_C.x3wide.so}
PYPATH=$BUILD/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$PLUGIN:$REPO/vllm/flashnext/plugins/b70_memory_xpu:$REPO/vllm/flashnext/plugins/b70_qsa_xpu
if [[ -d "$BUILD/c7b-fa-overlay/vllm_xpu_kernels" ]]; then
  PYPATH=$BUILD/c7b-fa-overlay:$PYPATH
fi
for f in "$SO" "$WIDE" "$HF/config.json" "$GGUF"; do test -f "$f"; done
if (( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) < 8388608 )); then
  echo 'ABORT_LOW_RAM' >&2
  exit 4
fi
OPT=()
if [[ -n "${PREFILL_ALLOC_CONF:-}" ]]; then
  OPT+=(--env "PYTORCH_ALLOC_CONF=$PREFILL_ALLOC_CONF")
fi
PRE=()
if [[ "${PREFILL_PRE_COLLECTIVE:-0}" == "1" ]]; then
  PRE=(bash "$HERE/x3_health_then_serve.sh")
fi
if [[ "${B70_MEMORY_FIX:-0}" == "1" ]]; then
  REF=${B70_EMBED_REF:-$BUILD/x3/embedding-reference.npz}
  test -f "$REF"
  OPT+=(--env "B70_EMBED_REF=$REF")
fi
if [[ -n "${B70_PREFILL_PROFILE:-}" ]]; then
  export B70_PREFILL_PROFILE_CONTROL=${B70_PREFILL_PROFILE_CONTROL:-$BUILD/x3/flashnext-profile-mode}
  export B70_PREFILL_CAPTURE=${B70_PREFILL_CAPTURE:-/mnt/cache/b70/x3-captures/flashnext-first-attention.pt}
  echo 1 > "$B70_PREFILL_PROFILE_CONTROL"
fi
for key in ${!B70_PREFILL@}; do OPT+=(--env "$key=${!key}"); done
for key in ${!B70_MEMORY@}; do OPT+=(--env "$key=${!key}"); done
(
  while kill -0 "$$" 2>/dev/null; do
    if (( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) < 8388608 )); then
      echo 'ABORT_LOW_RAM' >&2
      if docker inspect "$NAME" >/dev/null 2>&1; then
        docker stop -t 5 "$NAME" >/dev/null 2>&1 || true
        break
      fi
    fi
    sleep 2
  done
) &
watchdog=$!
trap 'kill "$watchdog" 2>/dev/null || true; wait "$watchdog" 2>/dev/null || true' EXIT
"$HERE/xpu-run.sh" --card both --memory 100g --name "$NAME" --port "$PORT" \
  --env B70_GGUF=1 --env "B70_GGUF_PATH=$GGUF" --env "B70_GGUF_SO=$SO" \
  --env "PYTHONPATH=$PYPATH" --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 \
  --env "B70_QSA=${B70_QSA:-0}" --env B70_EP=0 --env B70_FUSE_SHARED=0 --env MAX_JOBS=4 \
  --env B70_PREFILL=1 --env "B70_PREFILL_WIDE_SO=$WIDE" \
  --env B70_PREFILL_DQ_OVERLAP=0 --env B70_PREFILL_REDUCE=grouped \
  --env B70_PREFILL_EXPERT_TILE=64 --env B70_PREFILL_BUCKET=1 \
  --env B70_PREFILL_MAX_PAD_ROWS=65536 --env B70_PREFILL_HC=triton \
  --env B70_PREFILL_EP_DRAIN=1 --env B70_PREFILL_EP_SYNC=1 \
  --env SYCL_CACHE_PERSISTENT=0 \
  --env B70_PREFILL_PLE_HASH=numpy --env B70_XPU_CG_RECLAIM=500 \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env VLLM_NO_USAGE_STATS=1 \
  "${OPT[@]}" -- "${PRE[@]}" vllm serve "$HF" --host 0.0.0.0 --port "$PORT" \
  --served-model-name hotschmoe-dd qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-x3-prefill \
  --tensor-parallel-size 1 --data-parallel-size 2 --api-server-count 1 --enable-expert-parallel \
  --dtype bfloat16 --load-format b70_gguf --quantization b70_gguf \
  --language-model-only --trust-remote-code \
  --max-model-len "${MAX_LEN:-16384}" --max-num-seqs "${MAX_SEQS:-2}" \
  --max-num-batched-tokens "${MAX_BATCHED:-12288}" \
  --gpu-memory-utilization "${GPU_MEM:-0.95}" --no-enable-prefix-caching \
  --mamba-cache-mode align \
  --gdn-prefill-backend sycl --attention-backend FLASH_ATTN \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,8]}'
