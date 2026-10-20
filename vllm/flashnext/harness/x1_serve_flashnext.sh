#!/usr/bin/env bash
# Flash-Next packed Strata MTP, vLLM DP=2 + EP. Takes both GPU leases.
# Run xpu-health/collective-health first, and health after teardown.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
BUILD=/mnt/vm_8tb/b70/build/vllm-flashnext
STRATA=/mnt/vm_8tb/b70/build/strata-ab
export B70_AGENT=${B70_AGENT:-X1} B70_GPU_LOCK_TIMEOUT=${B70_GPU_LOCK_TIMEOUT:-7200}
PORT=${PORT:-18121}
MTP_TOKENS=${MTP_TOKENS:-2}
NAME=${NAME:-vllm-fn-x1-flashnext-mtp${MTP_TOKENS}}
if [[ -n "${X1_EVENTLESS_SHIM:-}" && "${X1_PRE_COLLECTIVE:-0}" != 1 ]]; then
  echo "Eventless shim requires X1_PRE_COLLECTIVE=1 for scoped preload"
  exit 2
fi
SPEC_ARGS=()
LAUNCH=(vllm)
if [[ "${X1_PRE_COLLECTIVE:-0}" = 1 ]]; then
  LAUNCH=("$HERE/x1_health_then_serve.sh" vllm)
fi
if [[ "$MTP_TOKENS" != 0 ]]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
fi
PYPATH=$BUILD/vllm:$STRATA/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_gguf_xpu
if [[ -e "$BUILD/c7b-fa-overlay/vllm_xpu_kernels/libattn_kernels_xe_2.so" ]]; then
  PYPATH=$BUILD/c7b-fa-overlay:$PYPATH
fi
available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if ((available < 8388608)); then echo "ABORT MemAvailable=$available"; exit 4; fi
"$HERE/xpu-run.sh" --card both --memory 100g --name "$NAME" --port "$PORT" \
  --env B70_GGUF=1 \
  --env B70_GGUF_PATH=/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --env B70_MTP_RT="$STRATA/mtp-bf16/rt" \
  --env B70_MTP_GGUF="$STRATA/mtp-bf16/mtp-q2_0.gguf" \
  --env B70_GGUF_SO="${B70_GGUF_SO:-$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.o3.so}" \
  --env PYTHONPATH="$PYPATH" \
  --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 --env B70_PREFILL=0 \
  --env B70_FUSE_SHARED=0 \
  --env B70_XPU_CG_RECLAIM="${B70_XPU_CG_RECLAIM:-500}" \
  --env B70_MTP_CANONICAL_MOE="${B70_MTP_CANONICAL_MOE:-0}" \
  --env X1_EVENTLESS_SHIM="${X1_EVENTLESS_SHIM:-}" \
  --env B70_RECLAIM_TRACE=1 \
  --env B70_STEP_LOG="$BUILD/x1/reclaim.log" \
  --env SYCL_CACHE_PERSISTENT=0 \
  --env SYCL_CACHE_DIR="$BUILD/cache/sycl-x1-flashnext" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  -- "${LAUNCH[@]}" serve /mnt/vm_8tb/b70/build/backend-decision/hf-flashnext \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name hotschmoe-dd "qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-vllm-rt-mtp${MTP_TOKENS}" \
  --tensor-parallel-size 1 --data-parallel-size 2 --api-server-count 1 \
  --enable-expert-parallel --dtype bfloat16 --max-model-len 2048 \
  --max-num-batched-tokens 512 --gpu-memory-utilization 0.98 \
  --max-num-seqs 2 --language-model-only --load-format b70_gguf \
  --quantization b70_gguf --trust-remote-code \
  --attention-backend "${ATTN:-FLASH_ATTN}" \
  --mamba-cache-mode align \
  --no-enable-prefix-caching \
  "${SPEC_ARGS[@]}" \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,6,8]}' &
serve_pid=$!
cleanup() {
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
  docker stop -t 15 "$NAME" >/dev/null 2>&1 || true
  wait "$serve_pid" 2>/dev/null || true
}
(
  while kill -0 "$serve_pid" 2>/dev/null; do
    available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if ((available < 8388608)); then
      echo "ABORT MemAvailable=$available"
      docker stop -t 10 "$NAME" >/dev/null 2>&1 || true
    fi
    sleep 2
  done
) &
watchdog_pid=$!
trap cleanup EXIT INT TERM
wait "$serve_pid"
