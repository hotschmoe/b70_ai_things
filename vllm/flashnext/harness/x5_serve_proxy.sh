#!/usr/bin/env bash
# X5 single-card quant/MTP comparison. xpu-run owns and pins the GPU lease.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
QUANT=${QUANT:-UD-IQ3_XXS}
MTP_TOKENS=${MTP_TOKENS:-0}
CARD=${CARD:-1}
PORT=${PORT:-18125}
NAME=${NAME:-vllm-fn-x5-proxy}
ALIAS=qwen3.6-35b-a3b-${QUANT,,}-x5-mtp${MTP_TOKENS}
WEIGHTS=/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-${QUANT}.gguf
SO=${B70_GGUF_SO:-$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.o3g.so}
RECLAIM=${B70_XPU_CG_RECLAIM:-500}
if [[ "${B70_X5_EVENTLESS:-0}" == 1 ]]; then RECLAIM=0; fi
test -f "$WEIGHTS"
sha256sum "$SO"
ENV_ARGS=(--env "B70_D3_GDN=${B70_D3_GDN:-0}" --env "B70_FUSE_SHARED=${B70_FUSE_SHARED:-1}")
ENV_ARGS+=(--env "B70_X5_PACK_LINEAR=${B70_X5_PACK_LINEAR:-0}")
if [[ "${B70_X5_GDN_BF16:-0}" == 1 ]]; then
  ENV_ARGS+=(--env B70_D3_GDN=0 --env B70_X5_GDN_BF16=1
    --env "B70_GDN_SO=$B/x5/gdn-bf16/libb70_gdn_xpu.so")
fi
ARGS=(vllm serve /mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B
  --host 0.0.0.0 --port "$PORT" --served-model-name hotschmoe-dd "$ALIAS"
  --tensor-parallel-size 1 --dtype "${DTYPE:-bfloat16}" --max-model-len "${MAX_LEN:-4096}"
  --max-num-seqs 2 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90
  --language-model-only --load-format b70_gguf --quantization b70_gguf
  --trust-remote-code --attention-backend FLASH_ATTN --no-enable-prefix-caching
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,6,8]}')
if [[ "$MTP_TOKENS" != 0 ]]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
if [[ "${B70_X5_GDN_BF16:-0}" == 1 || "${B70_X5_PACK_LINEAR:-0}" == 1 ]]; then
  ARGS=(/opt/venv/bin/python "$REPO/vllm/flashnext/harness/x5_api.py" "${ARGS[@]:1}")
fi
if [[ "${B70_X5_EVENTLESS:-0}" == 1 ]]; then
  ARGS=("$REPO/vllm/flashnext/harness/x5_eventless_serve.sh" "${ARGS[@]}")
fi
exec "$REPO/vllm/flashnext/harness/xpu-run.sh" --card "$CARD" --name "$NAME" --port "$PORT" \
  --env B70_GGUF=1 --env "B70_GGUF_PATH=$WEIGHTS" --env "B70_GGUF_SO=$SO" \
  --env "PYTHONPATH=$B/vllm:/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py:$REPO/vllm/flashnext/plugins/b70_gguf_xpu" \
  --env "B70_XPU_CG_RECLAIM=$RECLAIM" \
  --env B70_GGUF_REF=0 --env B70_EAGER_QK=0 --env B70_PREFILL=0 \
  --env SYCL_CACHE_PERSISTENT=0 --env "SYCL_CACHE_DIR=$B/cache/sycl-x5" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env VLLM_NO_USAGE_STATS=1 \
  "${ENV_ARGS[@]}" -- "${ARGS[@]}"
