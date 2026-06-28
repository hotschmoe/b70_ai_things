#!/usr/bin/env bash
# serve_w4a8_woq.sh -- serve the PROVEN Lorbus Qwen3.6-27B int4-AutoRound checkpoint (multimodal
# Qwen3_5ForConditionalGeneration -- HAS vision, FULL GDN+MLP int4) through sglang's proven
# multimodal model path, but with its int4 linears dispatching to the oneDNN int4_gemm ops
# (decode = int4_gemm_w4a16 fp16-act; prefill = int4_gemm_w4a8 per-token int8-act) instead of
# auto_round woqgemm. This reuses the working vision+arch+logits plumbing and directly upgrades
# the daily driver.
#
# WIRING (no image rebuild; runtime mounts over the baked shim):
#   - image sglang-xpu:mtp (the champion int4-graph image; multimodal Qwen3_5 path proven)
#   - bind-mount the UPDATED woq_shim.py over the baked copy (carries the _XpuW4A8WoqKernel)
#   - bind-mount the built _xpu_C.abi3.so dir; B70_XPU_C_SO points the shim at it
#   - B70_XPU_W4A8_WOQ=1 makes woq_shim route GPTQ int4 linears to the int4_gemm hybrid kernel
#     (the auto_gptq->op layout conversion is numerically gated by sglang/w4a8_from_woq_probe.py)
#   - in-container: source oneAPI setvars, PREPEND the oneAPI compiler lib to LD_LIBRARY_PATH
#     (required or the ctypes-loaded .so resolves but torch loses the XPU device -- W4A8_BUILD.md)
#   - act-quant is EAGER (no B70_W4A8_COMPILE; compile of it HANGS serve startup)
#
#   GPU touch -> hold the lease:  ../bin/gpu-run --card 0 bash sglang/serve_w4a8_woq.sh start
#                                 ../bin/gpu-run --card 0 bash sglang/serve_w4a8_woq.sh run   (start+bench+stop)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"; REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"

IMG="${IMG:-sglang-xpu:mtp}"
NAME="${NAME:-sglang_w4a8woq}"
CKPT="${CKPT:-/models/Lorbus_Qwen3.6-27B-int4-AutoRound}"   # multimodal int4 AutoRound, vision retained
TOK="${TOK:-/models/Qwen_Qwen3.6-27B}"
SERVED="${SERVED:-qwen36-27b-w4a8woq}"
PORT="${PORT:-30000}"; DEVICE="${DEVICE:-0}"
CTX="${CTX:-4096}"; MEMFRAC="${MEMFRAC:-0.85}"
KERNEL_DIR="${KERNEL_DIR:-/mnt/vm_8tb/b70/w4a8_kernel}"
SHIMS="${SHIMS:-$REPO/sglang/patches}"
SITE=/opt/venv/lib/python3.12/site-packages
GRAPH="${GRAPH:-0}"          # 1 -> stack XPUGraph decode capture (only AFTER a clean eager bench)
DENV="${DENV:-}"
LOG="${LOG:-$ROOT/w4a8woq_serve.log}"

cmd="${1:-start}"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

start() {
  mkdir -p "$ROOT"/{hf_cache,sgl_cache} 2>/dev/null || true
  local denv=(); for kv in $DENV; do denv+=(-e "$kv"); done
  local gflags=() genv=()
  if [ "$GRAPH" = 1 ]; then
    genv=(-e B70_XPU_CUDAGRAPH=1)
    gflags=(--cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1 --max-running-requests 1)
    say "GRAPH=1 -> stacking XPUGraph decode capture (bs=1)"
  fi
  say "=== sglang W4A8(woq) serve: $SERVED  IMG=$IMG  card=$DEVICE  ctx=$CTX  memfrac=$MEMFRAC  port=$PORT ==="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size "${SHM:-16g}" -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK="$DEVICE" \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -v "$KERNEL_DIR:/work/w4a8_kernel:ro" \
    -v "$SHIMS/woq_shim.py:$SITE/woq_shim.py:ro" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e B70_XPU_W4A8_WOQ=1 -e B70_XPU_C_SO=/work/w4a8_kernel/_xpu_C.abi3.so \
    "${genv[@]}" "${denv[@]}" \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
      exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --dtype bfloat16 --attention-backend triton --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 \
      --disable-radix-cache --skip-server-warmup \
      ${gflags[*]} \
      --tp 1 --context-length $CTX --mem-fraction-static $MEMFRAC \
      --host 0.0.0.0 --port $PORT" >/dev/null

  say "container started; waiting for /health (logfile: $LOG)..."
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
  say "=== W4A8(woq) layers wired (count) ==="; grep -c "w4a8-woq] layer ready" "$LOG" 2>/dev/null | sed 's/^/  layers: /'
  say "=== healthy + coherent; serving on :$PORT (model=$SERVED) ==="
}

case "$cmd" in
  start) start ;;
  run)   start && bash "$SCRIPT_DIR/serve_w4a8_woq.sh" bench; rc=$?; bash "$SCRIPT_DIR/serve_w4a8_woq.sh" stop; exit $rc ;;
  stop)  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" ;;
  logs)  docker logs "$NAME" > "$LOG" 2>&1; echo "wrote $LOG" ;;
  status) docker ps --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}'; curl -s "http://localhost:$PORT/health" && echo " <- /health" ;;
  gen)   curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
           -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":96,\"temperature\":0}" | python3 -m json.tool ;;
  bench) bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "w4a8woq" ;;
  *) echo "usage: $0 {start|run|stop|logs|status|gen|bench}"; exit 2 ;;
esac
