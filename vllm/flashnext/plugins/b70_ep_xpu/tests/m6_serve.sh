#!/usr/bin/env bash
# Flash-Next sidecar serve. Eager by default. GRAPH=1 is the
# FULL_DECODE_ONLY latency gate. Does not start unless this script
# is invoked; Phase B still has to wait for C7b.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
PORT="${PORT:-18116}"
LOG="${LOG:-/mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1d-m6-serve.log}"
export B70_EP=1
export B70_QSA="${B70_QSA:-1}"
export B70_EP_SPLIT="${B70_EP_SPLIT:-profile}"
export B70_EP_Q0="${B70_EP_Q0:-138}"
export B70_PLE_NBYTES="${B70_PLE_NBYTES:-28800138240}"
export B70_GGUF_SO="${B70_GGUF_SO:-$REPO/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.graph.so}"
export B70_XPU_CG_RECLAIM="${B70_XPU_CG_RECLAIM:-500}"
export B70_AGENT="${B70_AGENT:-H1d}"
export PORT
mkdir -p "$(dirname "$LOG")"
echo "m6_serve: port=$PORT graph=${GRAPH:-0} log=$LOG"
exec "$REPO/vllm/flashnext/harness/serve_flashnext_ep.sh" >"$LOG" 2>&1
