#!/usr/bin/env bash
# ===========================================================================================
# serve_w8a8_0515.sh -- 0.5.15 BENCH VARIANT of rdy_to_serve/sglang/qwen36-27b-w8a8/serve.sh.
# Same proven settings, pointed at the v0.5.15.post1 re-graft image (sglang-xpu:mtp-0515) with
# prefix/radix caching ON by default (RADIX=1). NOT a shelf entry -- the shelf keeps the proven
# 0.5.6 config until this is GPU-bench-gated faster-or-equal AND coherent. See sglang/SGLANG_0515_UPGRADE.md.
# Deltas vs the shelf serve.sh: IMG=sglang-xpu:mtp-0515, NAME=sglang_w8a8_mtp_0515, RADIX=1 (caching on).
# All runtime mounts (w8a8_shim, qwen3_coder_detector, mtp_tree_xpu, the built _xpu_C.so) are the same
# canonical sglang/patches/ files -- validated to still apply against 0.5.15 source.
# ===========================================================================================
# qwen36-27b-w8a8-mtp -- W8A8 (int8) + NEXTN MTP, the int8 all-rounder that HANDILY beats bf16/fp8 on
# prefill, TTFT, AND decode. Built fused int8 oneDNN ops (int8_gemm_w8a16 decode fp16-act /
# int8_gemm_w8a8 prefill s8-act) + NEXTN chain-MTP (steps=10) on the grafted W8A8 vision+MTP checkpoint.
#
#   c1 (IN2048/OUT128, warm, TP=2): DECODE ~25.2 t/s | PP ~4344 tok/s | TTFT ~471 ms
#   vs bf16 TP=2 (9.03 / 3098 / 661): TG +180% (2.8x), PP +40%, TTFT -29%.  > int4+MTP decode (15.3).
#   ACCURACY: HumanEval+ 0.970 / 0.933 (base/plus) -- HIGHER than int4 same-stack (0.933/0.896).
#   VISION retained (grafted vision+MTP ckpt). GREEDY-only (XPU MTP ignores sampling, like all XPU NEXTN).
#   Full head-to-head + build: ../../../research/w8a8/W8A8_SGLANG_PLAN.md + ../../../research/w8a8/W8A8_BUILD.md.
#
# RUNTIME MOUNTS (not a baked image):
#   - IMG = sglang-xpu:mtp  (baked XPU NEXTN gates + compressed_tensors W8A8 scheme + woqgemm)
#   - the built _xpu_C.abi3.so (B70_XPU_C_SO); B70_XPU_W8A8_FUSED=1 routes int8 linears to the fused ops
#   - the updated w8a8_shim.py (FUSED hybrid) over the baked copy
#   - the patched qwen3_coder_detector.py over the baked copy: streams STRING tool-call args
#     incrementally (the baked parser buffers the whole <parameter> body until </parameter> --
#     minutes of zero bytes on a large file write -> client idle-timeout/"terminated"; see vLLM
#     issue #30439). Faithful copy + localized change, offline-validated byte-exact vs baked +
#     non-streaming. See ../../../sglang/patches/qwen3_coder_detector.py + JOURNAL 2026-06-29.
#   - the complete (materialized) ckpt at $REPO/models/files/qwen3.6-27b/w8a8-sqgptq
#     (vision+MTP grafted, real files -- mounted -> /models/qwen3.6-27b/w8a8-sqgptq)
#   - in-container: source oneAPI setvars + PREPEND the oneAPI compiler lib to LD_LIBRARY_PATH (or the
#     ctypes-loaded .so resolves but torch loses the XPU device -- see ../../../research/w8a8/W8A8_BUILD.md)
#
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh start   # serve TP=2 (both cards), coherence-gated, stay up
#   bash serve.sh stop                            # stop + release
#   /mnt/vm_8tb/b70/gpu-run bash serve.sh run     # start + bench + stop in one lease
#
# [!] TP=2 = BOTH cards. cudagraph DISABLED (--disable-cuda-graph): W8A8 TP=2+MTP is stable that way; capture
#     is a CEILING at TP=2 (decode is all-reduce-bound, not launch-bound). For pure prefill/sampling use the
#     eager sibling (../../scripts/123_w8a8_fused_ab.sh, no MTP -> PP 4570 / TTFT 448 / decode 8.1, samples).
# [!] --max-running-requests 4 (spec mamba cache cap). >4 OOMs KV at ctx 8192; extra requests queue + complete.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"; REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
IMG="${IMG:-sglang-xpu:mtp-0515}"
NAME="${NAME:-sglang_w8a8_mtp_0515}"
CKPT="${CKPT:-/models/qwen3.6-27b/w8a8-sqgptq}"
TOK="${TOK:-/models/qwen3.6-27b/bf16}"
SERVED="${SERVED:-qwen36-27b-w8a8-mtp}"
KDIR="${KDIR:-$ROOT/w8a8_kernel}"
SPEC_STEPS="${SPEC_STEPS:-10}"; SPEC_DRAFT="${SPEC_DRAFT:-11}"; MAXREQ="${MAXREQ:-4}"
PORT="${PORT:-30000}"; TP=2; CTX="${CTX:-${MAXLEN:-8192}}"; MEMFRAC="${MEMFRAC:-0.90}"
# Agentic harness knobs (pi.dev / omp.sh / hermes). CTX honors the backend-agnostic MAXLEN knob so the
# daily_driver's DD_MAXLEN=131072 actually lands (it passes MAXLEN=, which the sglang path ignored before);
# bare shelf use still defaults to 8192. Tool/reason parsers match the vLLM shelf entries.
TOOLCALL="${TOOLCALL:-1}"; TOOLPARSER="${TOOLPARSER:-qwen3_coder}"  # Qwen3.6 emits XML <tool_call> (NOT hermes JSON)
REASONPARSER="${REASONPARSER:-qwen3}"                               # split <think> -> reasoning_content
METRICS="${METRICS:-1}"   # --enable-metrics: expose Prometheus /metrics on the serve PORT (same :$PORT, NOT
                          # api-key-gated). Off-the-shelf dashboard = Prometheus scrape + Grafana (sglang ships
                          # examples/monitoring/ compose + dashboard JSON). Time-series for input/output token
                          # counters, TTFT (prefill), gen throughput (decode), cache_hit_rate, queue depth.
                          # Observability-only -- no effect on model outputs. METRICS=0 to disable.
RADIX="${RADIX:-1}"   # 0515 BENCH VARIANT default = 1 (caching ON, user request). prefix/radix cache.
                      # RADIX=1 = the XPU-WORKING recipe (JOURNAL 2026-07-02, sweep PASS):
                      # mamba EXTRA_BUFFER strategy + --page-size 128, KEEPING the intel_xpu XMX attention
                      # backend. Un-gated on XPU by the mounted mtp_tree_xpu.py DOMINO 5 (drops the CUDA/MUSA/
                      # NPU assert; "FLA" = sglang's vendored Triton, all XPU-capable) via B70_XPU_MAMBA_EXTRA_
                      # BUFFER=1. Measured vs the earlier no_buffer+triton+page1 path: same caching (8x warm)
                      # but NO long-context decode collapse (11.5 t/s @60k vs 1-4) and ~3.5x faster cold
                      # prefill, coherent under concurrent prefill+decode. RADIX=0 = prod (intel_xpu, page 64,
                      # no cache). Superseded no_buffer path + rationale: docs/20260702_mamba_extra_buffer_xpu_plan.md.
THINKCAP="${THINKCAP:-4096}"                                        # int -> SGLANG_MAX_THINK_TOKENS (graceful </think> cap); empty = unlimited
                                                                   # 4096 (was 8192): caps worst-case "thinking" dead-air before the
                                                                   # first tool-call token (~3min at 25t/s) that fronting-proxy idle
                                                                   # timeouts cut on long agentic tool calls. See JOURNAL 2026-06-29.
API_KEY="${API_KEY:-}"   # if set, sglang ENFORCES it on /v1/* (--api-key); /health stays open. Used by the
                         # daily driver (DD_API_KEY) for WAN exposure. Empty = OPEN (LAN-only). Inert if unset.
LOG="${LOG:-$SCRIPT_DIR/serve.log}"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
APIKEY_ARG=""; AUTH_H=(); [ -n "$API_KEY" ] && { APIKEY_ARG="--api-key $API_KEY"; AUTH_H=(-H "Authorization: Bearer $API_KEY"); }

start(){
  say "pre-flight xpu-health"; "$REPO/bin/xpu-health" 2>&1 | tail -2 || { say "UNHEALTHY -- abort"; return 3; }
  docker rm -f "$NAME" >/dev/null 2>&1
  say "serve W8A8 fused+MTP (steps=$SPEC_STEPS) TP=2 -> $SERVED on :$PORT (ctx=$CTX radix=$RADIX tool=$TOOLCALL think=${THINKCAP:-inf} metrics=$METRICS img=$IMG)"
  # agentic args (built from the knobs; empty -> dropped by word-splitting, same pattern as $APIKEY_ARG)
  # RADIX=1 -> cache-on recipe: extra_buffer strategy + int8 mamba checkpoint pool (~2x cached-prefix capacity,
  #            0.6GB from headroom) + page_size=128, KEEP intel_xpu XMX attn (no decode collapse); mount the
  #            un-gate shim + set B70_XPU_MAMBA_EXTRA_BUFFER=1. RADIX=0 -> prod (no cache). Both sweep-gated 2026-07-02.
  local ATTN=intel_xpu PAGE=64 RADIX_ARG="--disable-radix-cache" CACHE_ARG="" EB_MOUNT=() EB_ENV=(-e B70_XPU_MAMBA_EXTRA_BUFFER=0)
  if [ "$RADIX" = 1 ]; then
    PAGE=128; RADIX_ARG="--mamba-radix-cache-strategy extra_buffer --enable-int8-mamba-checkpoint"; CACHE_ARG="--enable-cache-report"
    EB_MOUNT=(-v "$REPO/sglang/patches/mtp_tree_xpu.py:/opt/venv/lib/python3.12/site-packages/mtp_tree_xpu.py:ro")
    EB_ENV=(-e B70_XPU_MAMBA_EXTRA_BUFFER=1)
  fi
  local TOOL_ARG="";   [ "$TOOLCALL" = 1 ]    && TOOL_ARG="--tool-call-parser $TOOLPARSER"
  local REASON_ARG=""; [ -n "$REASONPARSER" ] && REASON_ARG="--reasoning-parser $REASONPARSER"
  local METRICS_ARG=""; [ "$METRICS" = 1 ]    && METRICS_ARG="--enable-metrics"
  local THINK_ENV=();  [ -n "$THINKCAP" ]     && THINK_ENV=(-e "SGLANG_MAX_THINK_TOKENS=$THINKCAP")
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g -p "${PORT}:${PORT}" \
    -v "$REPO/models/files:/models:ro" \
    -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" -v "$KDIR:/work/kernel:ro" \
    -v "$REPO/sglang/patches/w8a8_shim.py:/opt/venv/lib/python3.12/site-packages/w8a8_shim.py:ro" \
    -v "$REPO/sglang/patches/qwen3_coder_detector.py:/opt/venv/lib/python3.12/site-packages/sglang/srt/function_call/qwen3_coder_detector.py:ro" \
    "${EB_MOUNT[@]}" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e B70_XPU_MTP=1 -e B70_XPU_W8A8=1 -e B70_XPU_W8A8_FUSED=1 -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so \
    "${EB_ENV[@]}" "${THINK_ENV[@]}" \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
      exec python -m sglang.launch_server --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --attention-backend $ATTN --linear-attn-backend triton \
      --speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens $SPEC_DRAFT --speculative-draft-attention-backend triton --disable-cuda-graph \
      --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size $PAGE $RADIX_ARG $CACHE_ARG --skip-server-warmup \
      $TOOL_ARG $REASON_ARG $METRICS_ARG \
      --tp $TP --context-length $CTX --mem-fraction-static $MEMFRAC --max-running-requests $MAXREQ $APIKEY_ARG \
      --host 0.0.0.0 --port $PORT" >/dev/null
  say "waiting for /health (load + spec JIT ~3-6min)..."
  for i in $(seq 1 140); do
    docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "EXITED"; docker logs "$NAME" 2>&1|tail -40; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null||echo 000)" = 200 ] && { say "/health 200 (~$((i*5))s)"; break; }
    sleep 5
  done
  local g; g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" "${AUTH_H[@]}" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":256,\"temperature\":0}")
  echo "$g" | python3 -c "import sys,json
m=json.load(sys.stdin)['choices'][0]['message']            # --reasoning-parser puts <think> in reasoning_content
c=(m.get('content') or '') or (m.get('reasoning_content') or '')
print('COHERENCE OK:',repr(c[:160])) if c.strip() and (len(c)<16 or max(c.count(x) for x in set(c))/len(c)<0.6) else (print('GATE FAIL:',repr(c[:120])) or sys.exit(1))" \
    || { say "coherence gate FAILED -- see: docker logs $NAME"; return 1; }
  say "healthy + coherent; serving $SERVED on :$PORT"
}
stop(){ docker rm -f "$NAME" >/dev/null 2>&1; say "stopped $NAME"; "$REPO/bin/xpu-health" 2>&1 | tail -2 || true; }
# c1 + c4 regime bench (same harness as the int4/w4a8 sglang entries -> comparable table rows).
bench(){ bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "w8a8-fused-mtp"; }

case "${1:-start}" in
  start) start ;;
  stop)  stop ;;
  bench) bench ;;
  gen)   curl -s "http://localhost:$PORT/v1/chat/completions" "${AUTH_H[@]}" -H 'content-type: application/json' \
           -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"${2:-Why is the sky blue?}\"}],\"max_tokens\":128,\"temperature\":0}" ;;
  run)   start && bench; rc=$?; stop; exit $rc ;;
  smoke) start; rc=$?; stop; exit $rc ;;
  *) echo "usage: serve.sh {start|stop|bench|gen|run|smoke}"; exit 2 ;;
esac
