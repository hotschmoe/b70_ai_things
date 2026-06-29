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
export CKPT="${CKPT:-/models/qwen3.6-27b/int4-autoround}"
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

# --- GDN kernel overlay (vllm-xpu-kernels v0.1.10 / PR #411) -- EXPERIMENT, default OFF -------------
# v0.1.10 fixes a GDN chunk-kernel shared-local-memory write-after-read race (NaN at >=32K on Xe2). We
# TESTED the overlay (host-built v0.1.10 .so pair, same mechanism as the W4A8 recipe) against our mixed
# prefill+decode NaN repro (JOURNAL 2026-06-26): it does NOT fix that failure, and under GRAPH=1 the
# overlaid _xpu_C OOM-crashes inductor autotuning. OFF by default; set GDN_FIX=1 only to re-test it.
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO="${GDN_SO:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so}"
GDN_LIB="${GDN_LIB:-$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so}"
MOUNTS=()
if [ "${GDN_FIX:-0}" = 1 ]; then
  MOUNTS=( -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
  echo "=== GDN_FIX=1 -> overlay v0.1.10 GDN kernels (PR #411 race fix) over baked v0.1.9 ===" >&2
fi

# --- GDN packed-recurrent-decode toggle (mixed prefill+decode NaN hunt, JOURNAL 2026-06-26) ---------
# VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE defaults ON (vllm envs.py); it selects the PACKED recurrent
# DECODE kernel in qwen_gdn_linear_attn.py -- the prime suspect for the NaN that only appears when
# decodes are in flight alongside prefills. FLA_DECODE=0 disables it (unpacked recurrent decode path).
[ "${FLA_DECODE:-1}" = 0 ] && { DOCKER_ENV+=( -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 ); \
  echo "=== FLA_DECODE=0 -> VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 (unpacked recurrent decode) ===" >&2; }

source "$SCRIPT_DIR/../_common/lib.sh"
b70_dispatch "$@"
