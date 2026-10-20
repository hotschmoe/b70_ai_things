#!/usr/bin/env bash
# Eager two-card serve. Identity + the short greedy coherence prompt.
# No 32K prefill. The caller holds no lease; xpu-run takes both cards.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
EP_SO="${B70_GGUF_SO:-$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.ep.so}"
PORT="${PORT:-18114}"
LOG="${LOG:-/mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1-m2-serve.log}"
export B70_EP=1
export B70_GGUF_SO="$EP_SO"
export B70_XPU_CG_RECLAIM=0
export B70_GPU_LOCK_TIMEOUT="${B70_GPU_LOCK_TIMEOUT:-1800}"
export B70_AGENT="${B70_AGENT:-H1}"
# shellcheck disable=SC2097,SC2098
CARD=both PORT="$PORT" GRAPH=0 MAX_LEN=2048 MAX_SEQS=4 GPU_MEM=0.90 \
  NAME=vllm-fn-h1-m2 \
  "$REPO/vllm/flashnext/harness/serve_qwen36_gguf.sh" >"$LOG" 2>&1
