#!/usr/bin/env bash
# serve_nvfp4_27b_sglang.sh -- nvidia/Qwen3.6-27B-NVFP4 (ModelOpt MIXED_PRECISION) on sglang-XPU (B70).
#
# FIRST sglang serve of the NVFP4 quant (all prior NVFP4 work is vLLM-only). The multimodal
# Qwen3_5ForConditionalGeneration arch (GDN hybrid + vision + mtp) is native to sglang; only the
# quant path is shimmed (sglang/patches/nvfp4_shim.py):
#   * W4A16_NVFP4 MLP  -> our oneDNN op torch.ops._xpu_C.nvfp4_gemm_w4a16 (4-bit f4_e2m1 resident,
#                         bf16 acts, [K/16,N] bf16 folded scale) -- built for sglang's torch 2.12 ABI
#                         by sglang/nvfp4/build_nvfp4_kernel_sglang.sh
#   * FP8 attention    -> dequant-at-load to bf16 + F.linear (XPU-safe; native fp8 = OPEN opt)
#   * KV cache         -> bf16 (the ckpt declares fp8 KV but ships no scales, and fp8 KV is
#                         unsupported on sglang-XPU; the fp8 kv_cache_scheme is STRIPPED from a
#                         patched config.json mounted over the read-only ckpt, mirroring vLLM KV_FP8=0)
#   * vision / mtp / norms -> bf16 unquantized (not in quantized_layers -> UnquantizedLinearMethod)
#
# RUNTIME MOUNTS (not a baked image): the built _xpu_C.abi3.so + nvfp4_shim.py + its .pth, over the
# sglang-xpu:mtp image (the proven multimodal Qwen3_5 + XPUGraph + NEXTN-MTP image). LD_LIBRARY_PATH
# must PREPEND the oneAPI compiler libs in-container (or the ctypes-loaded .so resolves but torch
# loses the XPU device -- see sglang/W4A8_BUILD.md).
#
#   /mnt/vm_8tb/b70/gpu-run --card 0 bash serve_nvfp4_27b_sglang.sh start   # serve, wait healthy, coherence gate
#   bash serve_nvfp4_27b_sglang.sh stop | logs | status | gen
#   /mnt/vm_8tb/b70/gpu-run --card 0 bash serve_nvfp4_27b_sglang.sh run     # start + bench + stop in one lease
#
# [!] Pin card 0 (card1 is display-attached -> downclocked). This script does NOT take the GPU lease
#     itself -- wrap with gpu-run. Default config is the CONSERVATIVE bring-up (eager, no MTP, no
#     radix, no graph); flip GRAPH=1 / MTP=1 / RADIX=1 once the eager coherence gate is green.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"

IMG="${IMG:-sglang-xpu:mtp}"
NAME="${NAME:-sglang_nvfp4_27b}"
CKPT="${CKPT:-/models/qwen3.6-27b/nvfp4-modelopt}"      # ModelOpt MIXED_PRECISION NVFP4 (vision retained)
TOK="${TOK:-/models/qwen3.6-27b/bf16}"                  # bench tokenizer (base model)
SERVED="${SERVED:-qwen36-27b-NVFP4-modelopt-sglang}"
PORT="${PORT:-30000}"
DEVICE="${DEVICE:-0}"
CTX="${CTX:-4096}"
MEMFRAC="${MEMFRAC:-0.85}"                              # 24GiB 4-bit-resident weights -> 0.85 headroom (vLLM ceiling)
MAXREQ="${MAXREQ:-4}"

KERNEL_DIR="${KERNEL_DIR:-$ROOT/nvfp4_kernel_sglang}"   # holds _xpu_C.abi3.so (nvfp4_gemm_w4a16; NOT in git)
SHIMS="${SHIMS:-$REPO/sglang/patches}"
SITE=/opt/venv/lib/python3.12/site-packages

GRAPH="${GRAPH:-0}"        # 1 -> XPUGraph decode capture (B70_XPU_CUDAGRAPH=1, triton attn). 0 -> eager.
MTP="${MTP:-0}"           # 1 -> NEXTN MTP spec decode (B70_XPU_MTP=1). ckpt carries bf16 mtp.* natively.
SPEC_STEPS="${SPEC_STEPS:-5}"                           # NVFP4 MTP sweet spot on vLLM was 5; re-sweep on sglang
SPEC_DRAFT="${SPEC_DRAFT:-6}"
RADIX="${RADIX:-0}"       # 1 -> prefix/radix cache (mamba extra-buffer path). 0 -> --disable-radix-cache.
ATTN="${ATTN:-}"          # override attention backend; default triton if GRAPH else intel_xpu
KVFP8="${KVFP8:-0}"       # 0 -> strip fp8 kv_cache_scheme (bf16 KV). 1 -> leave ckpt fp8 KV (UNSUPPORTED on XPU).
DENV="${DENV:-}"
LOG="${LOG:-$ROOT/nvfp4_sglang_serve_card${DEVICE}.log}"

cmd="${1:-start}"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

preflight() {
  [ -f "$KERNEL_DIR/_xpu_C.abi3.so" ] || { say "MISSING kernel: $KERNEL_DIR/_xpu_C.abi3.so (build: sglang/nvfp4/build_nvfp4_kernel_sglang.sh)"; return 1; }
  [ -f "$SHIMS/nvfp4_shim.py" ]  || { say "MISSING shim: $SHIMS/nvfp4_shim.py"; return 1; }
  [ -f "$SHIMS/nvfp4_shim.pth" ] || { say "MISSING shim loader: $SHIMS/nvfp4_shim.pth"; return 1; }
  docker image inspect "$IMG" >/dev/null 2>&1 || { say "MISSING image: $IMG"; return 1; }
}

# Build a config.json with the fp8 kv_cache_scheme stripped (bf16 KV) unless KVFP8=1. Weights + NVFP4
# weight-quant config are untouched. Mounted read-only over the ckpt config (weights stay RO-mounted).
KV_MOUNT=()
prep_kv_config() {
  [ "$KVFP8" = 1 ] && { say "KVFP8=1 -> leaving ckpt fp8 KV (UNSUPPORTED on sglang-XPU; expect failure)"; return 0; }
  local src="$REPO/models/files/qwen3.6-27b/nvfp4-modelopt/config.json"
  local out="${KV_PATCH:-/tmp}/b70_config.nvfp4.sglang.nokvfp8.json"
  python3 -c "import json;d=json.load(open('$src'));d.get('quantization_config',{}).pop('kv_cache_scheme',None);json.dump(d,open('$out','w'))" \
    || { say "failed to generate $out"; return 1; }
  KV_MOUNT=( -v "$out:$CKPT/config.json:ro" )
  say "KVFP8=0 -> fp8 kv_cache_scheme stripped (bf16 KV) via $out"
}

start() {
  preflight || return 1
  prep_kv_config || return 1
  mkdir -p "$ROOT"/{hf_cache,sgl_cache} 2>/dev/null || true
  local denv=(); for kv in $DENV; do denv+=(-e "$kv"); done
  local genv=() gflags=()

  local attn="${ATTN:-}"
  if [ "$GRAPH" = 1 ]; then
    # XPUGraph capture needs pure-triton attention (the FlashAttn/intel_xpu kernel hits the SYCL-Graph
    # work_group_scratch_memory wall at capture -- see rdy_to_serve/sglang/qwen36-27b-w4a8/serve.sh).
    genv+=(-e B70_XPU_CUDAGRAPH=1); gflags+=(--cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1 --max-running-requests 1)
    [ -z "$attn" ] && attn=triton
    say "GRAPH=1 -> XPUGraph decode capture (bs=1, triton attn)"
  else
    gflags+=(--disable-cuda-graph --max-running-requests "$MAXREQ")
    [ -z "$attn" ] && attn=intel_xpu
  fi

  if [ "$MTP" = 1 ]; then
    genv+=(-e B70_XPU_MTP=1)
    gflags+=(--speculative-algorithm NEXTN --speculative-num-steps "$SPEC_STEPS" --speculative-eagle-topk 1 \
             --speculative-num-draft-tokens "$SPEC_DRAFT" --speculative-draft-attention-backend triton)
    say "MTP=1 -> NEXTN spec decode (steps=$SPEC_STEPS, draft=$SPEC_DRAFT)"
  fi

  local radixflag="--disable-radix-cache"
  if [ "$RADIX" = 1 ]; then radixflag=""; genv+=(-e B70_XPU_MAMBA_EXTRA_BUFFER=1); say "RADIX=1 -> prefix/radix cache (mamba extra-buffer)"; fi

  say "=== sglang NVFP4 serve: $SERVED  IMG=$IMG  card=$DEVICE  ctx=$CTX  memfrac=$MEMFRAC  attn=$attn  port=$PORT ==="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size "${SHM:-16g}" -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK="$DEVICE" \
    -v "$REPO/models/files:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -v "$KERNEL_DIR:/work/nvfp4_kernel:ro" \
    -v "$SHIMS/nvfp4_shim.py:$SITE/nvfp4_shim.py:ro" \
    -v "$SHIMS/nvfp4_shim.pth:$SITE/nvfp4_shim.pth:ro" \
    "${KV_MOUNT[@]}" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e B70_XPU_NVFP4=1 -e B70_XPU_C_SO=/work/nvfp4_kernel/_xpu_C.abi3.so -e B70_NVFP4_DEBUG="${DBG:-0}" \
    "${genv[@]}" "${denv[@]}" \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
      exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --dtype bfloat16 --attention-backend $attn --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 $radixflag --skip-server-warmup \
      ${gflags[*]} \
      --tp 1 --context-length $CTX --mem-fraction-static $MEMFRAC \
      --host 0.0.0.0 --port $PORT" >/dev/null

  say "container started; waiting for /health (weight load + repack; logfile: $LOG)..."
  local ok=0
  for i in $(seq 1 180); do
    docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "CONTAINER EXITED"; docker logs "$NAME" > "$LOG" 2>&1; say "see $LOG"; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
    sleep 5
  done
  docker logs "$NAME" > "$LOG" 2>&1
  [ "$ok" = 1 ] || { say "NOT healthy; see $LOG"; return 1; }

  say "=== coherence gate (Rayleigh, not garbage) ==="
  local g
  g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":96,\"temperature\":0}")
  echo "$g" | python3 -c "import sys,json
from collections import Counter
try: c=(json.load(sys.stdin)['choices'][0]['message']['content'] or '').strip()
except Exception: print('GATE FAIL: parse'); sys.exit(1)
if not c: print('GATE FAIL: empty'); sys.exit(1)
if len(c)>=16:
 ch,n=Counter(c).most_common(1)[0]
 if n/len(c)>=0.6: print('GATE FAIL: GARBAGE '+repr(c[:120])); sys.exit(1)
print('GATE OK: '+repr(c[:180]))" || { say "coherence gate FAILED -- see $LOG"; return 1; }
  say "=== NVFP4 layers wired ==="; grep -c "NVFP4 layer ready" "$LOG" 2>/dev/null | sed 's/^/  nvfp4 layers: /'
  grep -c "FP8->bf16 layer ready" "$LOG" 2>/dev/null | sed 's/^/  fp8->bf16 layers: /'
  say "=== healthy + coherent; serving on :$PORT (model=$SERVED) ==="
}

case "$cmd" in
  start) start ;;
  run)   start && bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "nvfp4-sglang"; rc=$?; docker rm -f "$NAME" >/dev/null 2>&1; exit $rc ;;
  stop)  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" ;;
  logs)  docker logs "$NAME" > "$LOG" 2>&1; echo "wrote $LOG" ;;
  status) docker ps --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}'; curl -s "http://localhost:$PORT/health" && echo " <- /health" ;;
  gen)   curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
           -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":96,\"temperature\":0}" | python3 -m json.tool ;;
  bench) bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "nvfp4-sglang" ;;
  smoke) start; rc=$?; docker rm -f "$NAME" >/dev/null 2>&1; exit $rc ;;
  *) echo "usage: $0 {start|run|smoke|stop|logs|status|gen|bench}"; exit 2 ;;
esac
