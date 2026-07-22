#!/usr/bin/env bash
# Qwen3.6-27B NVFP4 (nvidia/Qwen3.6-27B-NVFP4, ModelOpt MIXED_PRECISION). TWO sweep-gated modes in ONE
# entry (per shelf rules: options as settings, not sibling dirs). Thin wrapper over the maintained recipe
# vllm/nvfp4/serve_nvfp4_27b.sh.
#
#   ./bin/gpu-run --card 0 bash serve.sh start   # [DEFAULT] DP-replica / daily-driver config, port 8078
#   TP=2 ./bin/gpu-run bash serve.sh start        # TP=2 fallback (both cards), port 8078
#   bash serve.sh stop
#
# [TP=1, DEFAULT] THE single-card serving config (gated 2026-07-21 evening, embed-INT8 lever): captured
#   + MTP5 + CALIBRATED fp8 KV + INT8 embed_tokens (sitecustomize block 13, frees 1.18 GiB) + prefix
#   cache + agentic parsers at 100K ctx. Baked: MODE=fused GRAPH=1 MTPTOK=5 MAXSEQS=8 CAPSIZES=1,2,4,8
#   UTIL=0.95 MAXBATCH=2048 MAXLEN=100352, KV_FP8=1 KV_SCALES=vllm/nvfp4/kv_scales_nvfp4_27b.json,
#   B70_EMBED_INT8=1, native E4M3 NVFP4 scales for M<=8. Measured (card 0): 64.6 t/s single-stream
#   code (1.06x the 60.8-61.1 folded-scale path), 179.1 t/s aggregate conc=4 on one card, gate 18/18
#   plus 36/36 stress, model residency 22.76 GiB, KV 144,408 tokens. HumanEval+ 0.976/0.939
#   (current folded-scale run 0.976/0.945; one-task plus delta amid documented graph nondeterminism),
#   needle@93k-depth 4/4, repscan clean. The prior embed-INT8 quality gate was HumanEval+ 0.976/0.945
#   (plus IDENTICAL to stock 0.988/0.945 -- embed int8 is quality-neutral), 30k single-stream soak
#   flat 41.3 t/s + 40k concurrent soak clean. This is the DD replica config (b70_daily_0/1 behind
#   bin/dp_nginx.conf on :18080).
#   MEMORY WALLS (2026-07-21, v0.25.1): the MTP drafter costs a padded 17th KV layer + 0.8 GiB weights;
#   without the embed-INT8 lever MTP caps at ~48k ctx at UTIL=0.93. UTIL=0.96 CRASHES under concurrent
#   prefill (UR_OUT_OF_RESOURCES; profiling under-reserves ~1 GiB) -- 0.95 is the operator-set MAX, do
#   NOT go over. MAXBATCH<1600 rejected (GDN mamba-align block_size 1600). The freed embed memory only
#   reaches the KV budget because block 13 also adjusts model_memory_usage (vLLM uses the load-time
#   snapshot). Previous gated no-MTP config: MTPTOK= MAXSEQS=4 CAPSIZES=1,2,4 UTIL=0.93 MAXLEN=102400
#   B70_EXTRA_ENV= (25.7 t/s, the conservative fallback).
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
  # [DP-replica / daily-driver config] -- gated 2026-07-21 evening (see header): captured+MTP5 at 100K
  # ctx via the embed-INT8 lever (block 13) at the operator-capped UTIL=0.95.
  export CARD="${CARD:-0}" MAXLEN="${MAXLEN:-100352}" UTIL="${UTIL:-0.95}" \
         MAXSEQS="${MAXSEQS:-8}" CAPSIZES="${CAPSIZES:-1,2,4,8}" MAXBATCH="${MAXBATCH:-2048}" \
         MTPTOK="${MTPTOK:-5}"
  export KV_FP8="${KV_FP8:-1}" KV_SCALES="${KV_SCALES:-$REPO/vllm/nvfp4/kv_scales_nvfp4_27b.json}"
  export PREFIXCACHE="${PREFIXCACHE:-1}"
  # Decode-scale optimization (2026-07-22): the GDN-enabled candidate keeps checkpoint-native
  # E4M3 block scales for M<=8 and retains folded BF16 scales above that threshold. The old
  # nvfp4_fused_kernel_gdn artifact remains untouched for rollback.
  export FUSED_SO="${FUSED_SO:-/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/_xpu_C.abi3.so}"
  export GDN_LIB="${GDN_LIB:-/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/libgdn_attn_kernels_xe_2.so}"
  export TOOLCALL="${TOOLCALL:-1}" TOOLPARSER="${TOOLPARSER:-qwen3_coder}" REASONPARSER="${REASONPARSER:-qwen3}"
  export OVERRIDE_TEMP="${OVERRIDE_TEMP:-0.6}"    # Qwen3 thinking-runaway fix (2026-07-11); THINK_BUDGET
                                                  # defaults 4096 in the recipe itself.
  # Baked B70_EXTRA_ENV flags (folded in idempotently, without clobbering caller-supplied extras):
  #  B70_EMBED_INT8=1   -- block 13, REQUIRED for MTP5@100K (without it MTP caps ~48k).
  #  B70_PC_EAGLE_KEEP=1 + B70_PC_CHUNK_ALIGN=1 -- blocks 14b/14c (2026-07-22, JOURNAL b): make MTP5 x
  #    fp8-KV x prefix-caching actually produce cache hits (was 0 -- the EAGLE last-block drop x the
  #    1664-tok fp8 attention block zeroed every hit). Gated correct: needle@93k 4/4 twice, gate 18/18,
  #    byte-identical cache KV on the deterministic (eager) path, 30k single + 52k concurrent soak clean.
  for _pcf in B70_EMBED_INT8=1 B70_PC_EAGLE_KEEP=1 B70_PC_CHUNK_ALIGN=1 \
      B70_NVFP4_F8_SCALE_M_MAX=8; do
    case " ${B70_EXTRA_ENV:-} " in
      *" ${_pcf%%=*}="*) : ;;
      *) export B70_EXTRA_ENV="${B70_EXTRA_ENV:+$B70_EXTRA_ENV }$_pcf" ;;
    esac
  done
else
  # [TP=2 256K fallback] -- both cards; push-AR prefill + prefix cache + agentic parsers on.
  export TP MAXLEN="${MAXLEN:-131072}" MTPTOK="${MTPTOK:-5}" CAPSIZES="${CAPSIZES:-1,2,4,8}" \
         UTIL="${UTIL:-0.85}" MAXSEQS="${MAXSEQS:-8}"
  export PUSH_AR="${PUSH_AR:-1}" PREFIXCACHE="${PREFIXCACHE:-1}"
  export TOOLCALL="${TOOLCALL:-1}" TOOLPARSER="${TOOLPARSER:-qwen3_coder}" REASONPARSER="${REASONPARSER:-qwen3}"
  # API_KEY / VLLM_API_KEY (DD_API_KEY) pass through the environment to serve_nvfp4_27b.sh unchanged.
fi

exec bash "$REPO/vllm/nvfp4/serve_nvfp4_27b.sh"
