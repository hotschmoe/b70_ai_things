#!/usr/bin/env bash
# FULL_DECODE_ONLY sidecar. Same model and capture sizes as the one-card
# 11.11 ms control, with the slotted GGUF library and B70_EP=1.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
EP_SO="${B70_GGUF_SO:-$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.ep.so}"
PORT="${PORT:-18115}"
LOG="${LOG:-/mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1-m3-serve.log}"
export B70_EP=1
export B70_GGUF_SO="$EP_SO"
export B70_XPU_CG_RECLAIM="${B70_XPU_CG_RECLAIM:-500}"
export B70_GPU_LOCK_TIMEOUT="${B70_GPU_LOCK_TIMEOUT:-1800}"
export B70_AGENT="${B70_AGENT:-H1}"
CARD=both PORT="$PORT" GRAPH=1 GRAPH_MODE=FULL_DECODE_ONLY ATTN=FLASH_ATTN \
  MAX_LEN=33792 MAX_SEQS=4 GPU_MEM=0.90 \
  NAME=vllm-fn-h1-m3 \
  "$REPO/vllm/flashnext/harness/serve_qwen36_gguf.sh" >"$LOG" 2>&1
