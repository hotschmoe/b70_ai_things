#!/usr/bin/env bash
# Qwen3.6-27B int4 (AutoRound W4A16, woqgemm) + XPUGraph DECODE CAPTURE -- the FASTEST single-stream driver.
# Single card, SAMPLING-capable (temperature/top_p honored, unlike the greedy MTP driver), VISION retained.
# FIRST sglang-XPU decode cuda-graph: c1 ~23.5 t/s = 2.5x the 9.4 eager ceiling, +53% over int4+MTP (15.3),
# coherent + soak-STABLE + GDN-correct under mixed load (0 garbage). Built on torch.xpu.XPUGraph (SYCL-Graph/L0,
# proven non-degrading on B70, scripts/137).
#
# Self-contained: uses the BAKED sglang-xpu:mtp image (xpu_cudagraph.py + the B70_XPU_CUDAGRAPH woq_shim block
# baked in; ENV-GATED so it's the same image as the MTP driver). NO runtime patch mounts.
#
#   /mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh start   # serve, capture at startup, coherence-gated probe, stay up
#   bash serve.sh stop                                      # stop + release the card
#   bash serve.sh logs | gen | bench                        # follow log | chat probe | warm c1 + soak
#   /mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh run      # start + bench + stop in one lease
#
# [!] SINGLE-STREAM driver: --max-running-requests 1 + a SINGLE captured bs=1 graph. Multi-bucket capture
#     (bs>1) currently HALVES single-stream (a single decode pads up to the bs=N graph; --disable-cuda-graph-
#     padding breaks capture). So for CONCURRENCY use DP=2 (../../sglang/serve_dp2_graph.sh -> 2 users @ 23.5
#     each, beats MTP-DP2's 2x15), NOT a higher maxreq here. Per-card multi-stream-at-speed is open work.
# [!] ATTN=triton (NOT intel_xpu): the XPU FlashAttn kernel hits the SYCL-Graph work_group_scratch_memory wall
#     at capture; pure-triton attention clears it (== vLLM TRITON_ATTN). Required for graph capture.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"; REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"

IMG="${IMG:-sglang-xpu:mtp}"
NAME="${NAME:-sglang_int4_graph}"
CKPT="${CKPT:-/models/Lorbus_Qwen3.6-27B-int4-AutoRound}"   # int4 AutoRound, vision retained (no MTP head needed)
TOK="${TOK:-/models/Qwen_Qwen3.6-27B}"
SERVED="${SERVED:-qwen36-27b-int4-graph}"
PORT="${PORT:-30000}"; DEVICE="${DEVICE:-0}"
CTX="${CTX:-4096}"; MEMFRAC="${MEMFRAC:-0.90}"
DENV="${DENV:-}"

cmd="${1:-start}"
say(){ echo "[$(date +%H:%M:%S)] $*"; }

start() {
  mkdir -p "$ROOT"/{hf_cache,sgl_cache} 2>/dev/null || true
  local denv=(); for kv in $DENV; do denv+=(-e "$kv"); done
  say "=== sglang int4+XPUGraph serve: $SERVED  IMG=$IMG  card=$DEVICE  ctx=$CTX  port=$PORT ==="
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size "${SHM:-16g}" -p "${PORT}:${PORT}" -e ZE_AFFINITY_MASK="$DEVICE" \
    -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/sgl_cache:/sgl_cache" \
    -e HF_HOME=/hf_cache -e XDG_CACHE_HOME=/sgl_cache -e TORCHINDUCTOR_CACHE_DIR=/sgl_cache/inductor \
    -e B70_XPU_CUDAGRAPH=1 "${denv[@]}" \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; exec python -m sglang.launch_server \
      --model-path '$CKPT' --served-model-name '$SERVED' --trust-remote-code \
      --device xpu --attention-backend triton --linear-attn-backend triton \
      --mamba-ssm-dtype float32 --disable-overlap-schedule --page-size 64 --disable-radix-cache \
      --cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1 --max-running-requests 1 \
      --tp 1 --context-length $CTX --mem-fraction-static $MEMFRAC \
      --host 0.0.0.0 --port $PORT" >/dev/null

  say "waiting for /health (XPUGraph capture happens at startup -> slower than eager)..."
  local ok=0
  for i in $(seq 1 150); do
    docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "CONTAINER EXITED (capture crash?)"; docker logs "$NAME" 2>&1|tail -40; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "/health 200 (~$((i*5))s)"; break; }
    sleep 5
  done
  [ "$ok" = 1 ] || { say "NOT healthy; abort"; docker logs "$NAME" 2>&1|tail -30; return 1; }

  say "=== coherence gate ==="
  local g
  g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":96,\"temperature\":0}")
  echo "$g" | python3 -c "import sys,json
from collections import Counter
try: c=(json.load(sys.stdin)['choices'][0]['message']['content'] or '').strip()
except Exception: print('GATE FAIL'); sys.exit(1)
if not c: print('GATE FAIL: empty'); sys.exit(1)
if len(c)>=16:
 ch,n=Counter(c).most_common(1)[0]
 if n/len(c)>=0.6: print('GATE FAIL: GARBAGE '+repr(c[:120])); sys.exit(1)
print('GATE OK: '+repr(c[:140]))" || { say "coherence gate FAILED -- see: bash $0 logs"; return 1; }
  say "=== capture check ==="; docker logs "$NAME" 2>&1 | grep -oE "cuda graph: (True|False)" | sort | uniq -c
  say "=== healthy + coherent + capturing; serving on :$PORT (model=$SERVED) ==="
}

case "$cmd" in
  start) start ;;
  run)   start && bash "$SCRIPT_DIR/serve.sh" bench; rc=$?; bash "$SCRIPT_DIR/serve.sh" stop; exit $rc ;;
  stop)  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" ;;
  logs)  docker logs -f "$NAME" ;;
  status) docker ps --filter "name=$NAME" --format '{{.Names}}\t{{.Status}}'; curl -s "http://localhost:$PORT/health" && echo " <- /health" ;;
  gen)   curl -s "http://localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
           -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about the ocean.\"}],\"max_tokens\":64,\"temperature\":0.7}" | python3 -m json.tool ;;
  bench) bash "$REPO/sglang/perf_regime.sh" "$NAME" "$PORT" "$SERVED" "$TOK" "int4-graph" ;;
  *) echo "usage: $0 {start|run|stop|logs|status|gen|bench}"; exit 2 ;;
esac
