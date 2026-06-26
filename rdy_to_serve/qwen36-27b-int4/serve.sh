#!/usr/bin/env bash
# Qwen3.6-27B int4-AutoRound (W4A16) -- PRIMARY single-card quality pick. PIECEWISE captured.
# Self-contained recipe; shared plumbing in ../_common/lib.sh (engine = proven host 30_serve).
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh        # start (GRAPH capture), wait healthy, gen-probe, stay up
#   bash serve.sh stop                            # stop + release the GPU
#   bash serve.sh bench                           # concurrency sweep vs the running server
#   GRAPH=1 /mnt/vm_8tb/b70/gpu-run bash serve.sh run   # serve + bench + stop in one lease
#
# IMAGE: vllm-xpu-env:v0230 (vLLM 0.23.0). Plain v0230 serves int4 AutoRound -- no runtime patch.
# Decode ~30.8 t/s captured (eager ~7.8). Model ~16.7 GiB + KV. Fits ONE 32 GB B70.
# This is the current DAILY DRIVER model (served 2x data-parallel via ../../daily_driver_serve.sh).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMG="${IMG:-vllm-xpu-env:v0230}"
export CKPT="${CKPT:-/models/Lorbus_Qwen3.6-27B-int4-AutoRound}"
export SERVED="${SERVED:-qwen36-27b-int4}"
export GRAPH="${GRAPH:-1}"                 # PIECEWISE capture = the ~4x decode lever
export DTYPE="${DTYPE:-auto}"
export UTIL="${UTIL:-0.92}"
export MAXLEN="${MAXLEN:-8192}"            # daily driver runs 131072; fp16-KV single-card cap ~133k
export MAXSEQS="${MAXSEQS:-64}"
export CAPSIZES="${CAPSIZES:-1,2,4,8,16,32,64}"
export NOMM="${NOMM:-1}"                   # 27B is a qwen3_5 VLM -> text-only (skip vision profiling crash)
export TOOLCALL="${TOOLCALL:-1}"          # agents (pi etc.): Qwen3.6 emits XML tool calls
export TOOLPARSER="${TOOLPARSER:-qwen3_coder}"
export REASONPARSER="${REASONPARSER:-qwen3}"

# B70_EXTRA_ENV: space-separated NAME=VAL injected as -e docker flags (e.g. the daily-driver's
# VLLM_API_KEY for WAN/Traefik exposure). lib.sh b70_serve passes "${DOCKER_ENV[@]}" through.
DOCKER_ENV=()
if [ -n "${B70_EXTRA_ENV:-}" ]; then
  for kv in ${B70_EXTRA_ENV}; do DOCKER_ENV+=( -e "$kv" ); done
  echo "=== B70_EXTRA_ENV -> injected: ${B70_EXTRA_ENV%%=*}=... ===" >&2
fi

source "$SCRIPT_DIR/../_common/lib.sh"
b70_dispatch "$@"
