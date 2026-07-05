#!/usr/bin/env bash
# Qwen3.6-27B NVFP4 (nvidia/Qwen3.6-27B-NVFP4, ModelOpt MIXED_PRECISION). TWO sweep-gated modes in ONE
# entry (per shelf rules: options as settings, not sibling dirs). Thin wrapper over the maintained recipe
# vllm/nvfp4/serve_nvfp4_27b.sh.
#
#   ./bin/gpu-run --card 0 bash serve.sh start   # [DEFAULT] single-card champion, port 8078
#   TP=2 ./bin/gpu-run bash serve.sh start        # long-context daily driver (both cards), port 8078
#   bash serve.sh stop
#
# [TP=1, DEFAULT] THE single-card champion (2026-07-04, M6-M9): HumanEval+ 0.988/0.945 = box leaderboard
#   #1, decode 40.7-44.1 t/s random / 67 code on ONE card, gate 18/18, vision under capture+MTP. Baked:
#   MODE=fused GRAPH=1 MTPTOK=5 CAPSIZES=1,2,4,8 UTIL=0.85 MAXLEN=8192 (KV ~10.7k). The QUALITY +
#   single-stream pick. Do not fear MAXLEN, fear UTIL (0.88 OOMs the 24.1 GiB-resident serve).
#
# [TP=2] the LONG-CONTEXT daily driver (2026-07-04/05, Track 11d/11g): both cards, MTP5 decode +
#   push-AR PREFILL overlay (3.3x cold prefill, PP now MATCHES single-card 1702; decode-neutral) +
#   prefix cache + full 256K ctx. Auto-enables PUSH_AR=1 PREFIXCACHE=1 and the pi/omp/hermes agentic
#   parsers (tool-call qwen3_coder + reasoning qwen3). KV ~757k tokens @ 256K (2.89x), gate 18/18.
#   Long-prefill stack (optional): MAXBATCH=16384 PUSH_AR_MAXB=268435456 = +11.5% @ 32K cold prefill.
# Kernel .so: /mnt/vm_8tb/b70/nvfp4_fused_kernel_gdn/ (build: vllm/nvfp4/NVFP4_KERNEL_BUILD.md).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

NAME="${NAME:-nvfp4_27b}"
ACTION="${1:-start}"

if [ "$ACTION" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi
# start / run / smoke all bring the serve up (detached); the DD orchestrator's `docker wait` pins the lease.

TP="${TP:-1}"
# shared baked best-config
export MODE=fused GRAPH=1 MTPTOK="${MTPTOK:-5}" CAPSIZES="${CAPSIZES:-1,2,4,8}" \
       UTIL="${UTIL:-0.85}" MAXSEQS="${MAXSEQS:-8}" PORT="${PORT:-8078}" NAME="$NAME"

if [ "$TP" = 1 ]; then
  # [single-card champion] -- unchanged.
  export CARD="${CARD:-0}" MAXLEN="${MAXLEN:-8192}"
else
  # [TP=2 long-context daily driver] -- both cards; push-AR prefill + prefix cache + agentic parsers on.
  export TP MAXLEN="${MAXLEN:-131072}"                 # >=128K; the DD passes DD_MAXLEN (253952/256K).
  export PUSH_AR="${PUSH_AR:-1}" PREFIXCACHE="${PREFIXCACHE:-1}"
  export TOOLCALL="${TOOLCALL:-1}" TOOLPARSER="${TOOLPARSER:-qwen3_coder}" REASONPARSER="${REASONPARSER:-qwen3}"
  # API_KEY / VLLM_API_KEY (DD_API_KEY) pass through the environment to serve_nvfp4_27b.sh unchanged.
fi

exec bash "$REPO/vllm/nvfp4/serve_nvfp4_27b.sh"
