#!/usr/bin/env bash
# Qwen3.6-27B NVFP4 (nvidia/Qwen3.6-27B-NVFP4, ModelOpt MIXED_PRECISION). TWO sweep-gated modes in ONE
# entry (per shelf rules: options as settings, not sibling dirs). Thin wrapper over the maintained recipe
# vllm/nvfp4/serve_nvfp4_27b.sh.
#
#   ./bin/gpu-run --card 0 bash serve.sh start   # [DEFAULT] DP-replica / daily-driver config, port 8078
#   TP=2 ./bin/gpu-run bash serve.sh start        # TP=2 fallback (both cards), port 8078
#   bash serve.sh stop
#
# [TP=1, DEFAULT] THE single-card serving config (gated 2026-07-21, headless DP=2 campaign): captured
#   no-MTP + CALIBRATED fp8 KV + prefix cache + agentic parsers at 100K ctx. Baked: MODE=fused GRAPH=1
#   MTP OFF, MAXSEQS=4 CAPSIZES=1,2,4 UTIL=0.93 MAXBATCH=2048 MAXLEN=102400, KV_FP8=1
#   KV_SCALES=vllm/nvfp4/kv_scales_nvfp4_27b.json. Measured: 25.7 t/s single-stream, 99 t/s aggregate
#   at DP=2 conc=4, gate_concurrent_coherence 18/18 per card, 30k single + 40k concurrent soak clean.
#   This is the DD replica config (b70_daily_0/1 behind bin/dp_nginx.conf on :18080).
#   MEMORY WALLS (2026-07-21, v0.25.1): MTP costs a padded 17th KV layer (43.2 vs 34.6 KB/tok) + 0.8 GiB
#   drafter -> MTP and >64K ctx are MUTUALLY EXCLUSIVE on one card. UTIL>=0.96 CRASHES under concurrent
#   prefill (UR_OUT_OF_RESOURCES in nvfp4_gemm_w4a16; profiling under-reserves ~1 GiB) -- 0.93 is the
#   gated wall. MAXBATCH<1600 rejected (GDN mamba-align block_size 1600).
#   The old short-ctx MTP champion (HumanEval+ 0.988/0.945, 67 t/s code) is reachable via env:
#   MTPTOK=5 MAXLEN=8192 UTIL=0.85 MAXSEQS=8 CAPSIZES=1,2,4,8 bash serve.sh start
#
# [TP=2] the 256K-context fallback (2026-07-04/05, Track 11d/11g): both cards, MTP5 decode + push-AR
#   PREFILL overlay + prefix cache + full 256K ctx. Auto-enables PUSH_AR=1 PREFIXCACHE=1 + the agentic
#   parsers. KV ~757k tokens @ 256K, gate 18/18. Superseded as the DD by DP=2 (wedge-immunity + a free
#   research card) but kept measured + serveable.
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
# shared baked best-config (int8g-v0251 = the vLLM 0.25.1 promotion, 2026-07-17)
export MODE=fused GRAPH=1 PORT="${PORT:-8078}" NAME="$NAME" IMG="${IMG:-vllm-xpu-env:int8g-v0251}"

if [ "$TP" = 1 ]; then
  # [DP-replica / daily-driver config] -- gated 2026-07-21 (see header). MTP intentionally OFF: it is
  # memory-incompatible with >64K ctx on one card (padded drafter KV layer); set MTPTOK=5 explicitly
  # (with MAXLEN<=64K) for the short-ctx champion.
  export CARD="${CARD:-0}" MAXLEN="${MAXLEN:-102400}" UTIL="${UTIL:-0.93}" \
         MAXSEQS="${MAXSEQS:-4}" CAPSIZES="${CAPSIZES:-1,2,4}" MAXBATCH="${MAXBATCH:-2048}" \
         MTPTOK="${MTPTOK:-}"
  export KV_FP8="${KV_FP8:-1}" KV_SCALES="${KV_SCALES:-$REPO/vllm/nvfp4/kv_scales_nvfp4_27b.json}"
  export PREFIXCACHE="${PREFIXCACHE:-1}"
  export TOOLCALL="${TOOLCALL:-1}" TOOLPARSER="${TOOLPARSER:-qwen3_coder}" REASONPARSER="${REASONPARSER:-qwen3}"
  export OVERRIDE_TEMP="${OVERRIDE_TEMP:-0.6}"    # Qwen3 thinking-runaway fix (2026-07-11); THINK_BUDGET
                                                  # defaults 4096 in the recipe itself.
else
  # [TP=2 256K fallback] -- both cards; push-AR prefill + prefix cache + agentic parsers on.
  export TP MAXLEN="${MAXLEN:-131072}" MTPTOK="${MTPTOK:-5}" CAPSIZES="${CAPSIZES:-1,2,4,8}" \
         UTIL="${UTIL:-0.85}" MAXSEQS="${MAXSEQS:-8}"
  export PUSH_AR="${PUSH_AR:-1}" PREFIXCACHE="${PREFIXCACHE:-1}"
  export TOOLCALL="${TOOLCALL:-1}" TOOLPARSER="${TOOLPARSER:-qwen3_coder}" REASONPARSER="${REASONPARSER:-qwen3}"
  # API_KEY / VLLM_API_KEY (DD_API_KEY) pass through the environment to serve_nvfp4_27b.sh unchanged.
fi

exec bash "$REPO/vllm/nvfp4/serve_nvfp4_27b.sh"
