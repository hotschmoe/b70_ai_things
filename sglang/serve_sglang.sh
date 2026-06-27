#!/usr/bin/env bash
# SGLang Intel-XPU serve for Qwen3.6-27B (qwen3_5 GDN) on the dual Arc B70 (Battlemage) box.
# Campaign goal: prove SGLang serves the GDN linear-attention path WITHOUT the vLLM 0.23 mixed
# prefill+decode NaN ("!!!!") bug. See JOURNAL 2026-06-27 + contrib/gdn_nan_repro/.
#
# GPU touch -> MUST hold the lease:  ../bin/gpu-run --card 0 bash sglang/serve_sglang.sh start
#   (TP=2 uses both cards -> ../bin/gpu-run bash ... TP=2)
#
# Subcommands:  start | stop | logs | gen | status
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"

IMG="${IMG:-sglang-xpu:bmg}"
NAME="${NAME:-sglang_test}"
# RECOMMENDED first smoke (research-verified, 2026-06-27): bf16 + triton FLA + fp32 SSM + NO mixed-chunk.
#   bf16 Qwen_Qwen3.6-27B is 55.6GB -> needs TP=2 (both cards). Hybrid GDN KV is light (16/64 full-attn).
# Why bf16 not int4/fp8: int4-AutoRound has NO proven XPU int4 GEMM (Marlin is CUDA-gated) + a GDN
#   in_proj_ba exclusion-propagation gap; FP8 is open-bugged for this model (#23687 / #19603). The
#   GDN-NaN we are testing is quant-INDEPENDENT, so bf16 is a valid (and the safest) test of the path.
CKPT="${CKPT:-/models/Qwen_Qwen3.6-27B}"
SERVED="${SERVED:-qwen36-27b-bf16-sglang}"
PORT="${PORT:-30000}"
DEVICE="${DEVICE:-0}"          # card pin for TP=1 (ZE_AFFINITY_MASK); ignored when TP=2
TP="${TP:-2}"
PP="${PP:-1}"                 # pipeline-parallel size; PP>1 exposes both cards (no affinity pin)
CTX="${CTX:-32768}"           # --context-length
MEMFRAC="${MEMFRAC:-0.85}"    # --mem-fraction-static (weights+KV pool fraction)
PAGE="${PAGE:-64}"            # intel_xpu supports 32/64/128
ATTN="${ATTN:-intel_xpu}"
LINATTN="${LINATTN:-triton}" # GDN/linear-attn kernel backend; triton is the only non-CUDA path
SSMDTYPE="${SSMDTYPE:-float32}" # SSM recurrent state dtype (fp32 = GDN-safe; this is the default too)
QUANT="${QUANT:-}"           # e.g. auto-round / compressed-tensors / fp8; empty = auto-detect
EXTRA="${EXTRA:-}"           # extra launch_server flags
DENV="${DENV:-}"             # extra docker -e env, space-separated KEY=VAL (e.g. DENV="FLA_USE_FAST_OPS=1")
MOUNTS="${MOUNTS:-}"         # extra docker -v specs, space-separated host:container[:ro] (patch overlays)

cmd="${1:-start}"

start() {
  mkdir -p "$ROOT"/{hf_cache,sgl_cache} 2>/dev/null || true
  # GPU passthrough mirrors rdy_to_serve/_common/lib.sh; pin card via ZE_AFFINITY_MASK for TP=1.
  GDOCK=()
  if [ "$TP" = 1 ] && [ "$PP" = 1 ]; then GDOCK=(-e ZE_AFFINITY_MASK="$DEVICE"); fi
  local q=(); [ -n "$QUANT" ] && q=(--quantization "$QUANT")
  local denv=(); for kv in $DENV; do denv+=(-e "$kv"); done
  local mounts=(); for mv in $MOUNTS; do mounts+=(-v "$mv"); done
  echo "=== sglang serve: $SERVED  IMG=$IMG  TP=$TP card=${DEVICE} ctx=$CTX memfrac=$MEMFRAC attn=$ATTN port=$PORT ==="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size "${SHM:-16g}" -p "${PORT}:${PORT}" "${GDOCK[@]}" \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    "${denv[@]}" "${mounts[@]}" \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --attention-backend '$ATTN' --linear-attn-backend '$LINATTN' \
      --mamba-ssm-dtype '$SSMDTYPE' --disable-overlap-schedule --page-size $PAGE \
      --disable-radix-cache \
      --tp $TP --pp-size $PP --context-length $CTX --mem-fraction-static $MEMFRAC \
      --host 0.0.0.0 --port $PORT ${q[*]} $EXTRA"
  echo "container started; tail logs with: bash $0 logs"
}

case "$cmd" in
  start)  start ;;
  stop)   docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" ;;
  logs)   docker logs -f "$NAME" ;;
  status) docker ps --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}'; curl -s "http://localhost:$PORT/health" && echo " <- /health" ;;
  gen)    curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
            -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about the ocean.\"}],\"max_tokens\":64}" | python3 -m json.tool ;;
  *) echo "usage: $0 {start|stop|logs|gen|status}"; exit 2 ;;
esac
