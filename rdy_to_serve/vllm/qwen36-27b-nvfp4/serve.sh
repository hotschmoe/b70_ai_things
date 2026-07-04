#!/usr/bin/env bash
# Qwen3.6-27B NVFP4 (nvidia/Qwen3.6-27B-NVFP4, ModelOpt MIXED_PRECISION) -- THE single-card
# champion (2026-07-04, M6-M9): HumanEval+ 0.988/0.945 = box leaderboard #1, decode 40.7-44.1
# t/s random / 67 t/s code on ONE card, gate_concurrent 18/18, vision verified under capture+MTP.
#
#   ./bin/gpu-run --card 0 bash serve.sh        # start, stays up (container nvfp4_27b, port 8078)
#   bash serve.sh stop                           # stop the container
#
# This is a THIN WRAPPER around the maintained experiment recipe vllm/nvfp4/serve_nvfp4_27b.sh
# with the sweep-gated best config BAKED (per shelf rules: one best config, options as settings):
#   MODE=fused    custom nvfp4_gemm_w4a16 oneDNN op, weights 4-bit f4_e2m1 resident (24.1 GiB)
#   GRAPH=1      PIECEWISE capture (needs the register_fake in vllm/nvfp4/patches/sitecustomize.py)
#   MTPTOK=5     NEXTN MTP, sweep winner (spec3 58 code / spec5 67 / spec7 63)
#   CAPSIZES=1,2,4,8  REQUIRED with MTP (default spec sizes [1..64] OOM at capture)
#   UTIL=0.85    HARD ceiling -- 0.88 loads+captures then OOMs on the first 2048-tok prefill
# CAVEATS: KV only ~8.5k tokens (24.1 GiB weights + drafter + graphs on a 31.9 GiB card) ->
# concurrent long-prefill streams serialize; this entry is the QUALITY + single-stream pick.
# Kernel .so: /mnt/vm_8tb/b70/nvfp4_fused_kernel_gdn/ (build: vllm/nvfp4/NVFP4_KERNEL_BUILD.md).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

NAME="${NAME:-nvfp4_27b}"

if [ "${1:-}" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

MODE=fused GRAPH=1 MTPTOK="${MTPTOK:-5}" CAPSIZES="${CAPSIZES:-1,2,4,8}" \
CARD="${CARD:-0}" PORT="${PORT:-8078}" MAXLEN="${MAXLEN:-4096}" UTIL="${UTIL:-0.85}" \
MAXSEQS="${MAXSEQS:-8}" NAME="$NAME" \
  bash "$REPO/vllm/nvfp4/serve_nvfp4_27b.sh"
