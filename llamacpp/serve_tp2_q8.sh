#!/usr/bin/env bash
# llamacpp/serve_tp2_q8.sh -- "W8A8-like, TP=2 DP=1": qwen3.6-27b Q8_0 sharded across BOTH cards via
# llama.cpp's SYCL tensor-parallel meta-backend (--split-mode tensor). The HIGHER-RISK path.
#
# WHY RISKY (REVIEW_intel_arch.md sec 2/7 -- coherence-gate before trusting):
#  - --split-mode tensor needs flash-attn (auto-enabled); --flash-attn off is rejected with it.
#  - qwen35 is a HYBRID recurrent (GDN) arch. The tensor-split arch gate (llama-arch.cpp:976) does NOT
#    refuse qwen35, but it EXCLUDES every other recurrent/hybrid arch -- so TP across the GDN recurrent
#    state is UNVERIFIED. Watch for "!!!!"-style garbage under concurrent load (like our sglang W8A8
#    warmup-poisoning). The coherence gate below is the guard.
#  - compute-runtime 26.x has a known multi-GPU issue (#21747); the image ships 26.18. If TP=2 misbehaves,
#    the fallback is the DP=2 Q4_K_M path (serve_dp2_q4km.sh), which is the recommended production default.
#  - backend sampling falls back to CPU with tensor split (llama-context.cpp:1195) -- a possible TG cost.
#
# Quant note: Q8_0 = 8-bit WEIGHTS only, fp16 activations (NOT true W8A8; no int8-activation path in
# llama.cpp). On B70 Q8_0 has been ~4x slower than Q4_K_M (#21517) -- benchmark before trusting as "best".
#
# MUST run under the GPU lease holding BOTH cards:
#   cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run bash llamacpp/serve_tp2_q8.sh start
#   bash llamacpp/serve_tp2_q8.sh stop
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"; REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
IMG="${IMG:-sglang-xpu:mtp}"
SRC="${SRC:-$ROOT/llama.cpp}"
GGUF="${GGUF:-$ROOT/llamacpp/gguf}"
TAG="${TAG:-qwen3.6-27b}"
MODEL="${MODEL:-/gguf/${TAG}-Q8_0.gguf}"           # set MODEL=/gguf/${TAG}-Q4_K_M.gguf to TP=2 the 4-bit instead
MMPROJ="${MMPROJ:-/gguf/${TAG}-mmproj-f16.gguf}"
SERVED="${SERVED:-qwen36-27b-q8-tp2}"
NAME="${NAME:-llamacpp_q8_tp2}"
PORT="${PORT:-18080}"
CTX="${CTX:-${MAXLEN:-32768}}"
PAR="${PAR:-4}"
METRICS="${METRICS:-1}"; API_KEY="${API_KEY:-}"
LOG="${LOG:-$SCRIPT_DIR/serve_tp2_q8.log}"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
AUTH_H=(); [ -n "$API_KEY" ] && AUTH_H=(-H "Authorization: Bearer $API_KEY")

start(){
  say "pre-flight xpu-health"; "$REPO/bin/xpu-health" 2>&1 | tail -2 || { say "UNHEALTHY -- abort"; return 3; }
  [ -s "$GGUF/$(basename "$MODEL")" ] || { say "missing $MODEL -- run convert_gguf.sh first"; return 2; }
  docker rm -f "$NAME" >/dev/null 2>&1
  local mm=""; { [ "${NOMM:-0}" != 1 ] && [ -s "$GGUF/${TAG}-mmproj-f16.gguf" ]; } && mm="--mmproj $MMPROJ"
  local key=""; [ -n "$API_KEY" ] && key="--api-key $API_KEY"
  local met=""; [ "$METRICS" = 1 ] && met="--metrics"
  say "TP=2 (--split-mode tensor) $(basename "$MODEL") -> $SERVED on :$PORT (ctx=$CTX par=$PAR mm=$([ -n "$mm" ] && echo on || echo off))"
  docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 32g -p "${PORT}:${PORT}" \
    -v "$SRC:/llama:ro" -v "$GGUF:/gguf:ro" \
    -e ZES_ENABLE_SYSMAN=1 \
    "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
      export LD_LIBRARY_PATH=/llama/build/bin:/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
      exec /llama/build/bin/llama-server -m $MODEL $mm -ngl 999 --ctx-size $CTX \
      --split-mode tensor --main-gpu 0 --flash-attn auto --parallel $PAR --cont-batching \
      --jinja $met $key --alias $SERVED --host 0.0.0.0 --port $PORT" >/dev/null
  say "waiting for /health (model load + TP init)..."
  for i in $(seq 1 160); do
    docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME" || { say "EXITED"; docker logs "$NAME" 2>&1|tail -40; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null||echo 000)" = 200 ] && { say "/health 200 (~$((i*5))s)"; break; }
    sleep 5
  done
  local g; g=$(curl -s --max-time 180 "http://localhost:$PORT/v1/chat/completions" "${AUTH_H[@]}" -H 'content-type: application/json' \
      -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":256,\"temperature\":0}")
  echo "$g" | python3 -c "import sys,json
m=json.load(sys.stdin)['choices'][0]['message']
c=(m.get('content') or '') or (m.get('reasoning_content') or '')
print('COHERENCE OK:',repr(c[:160])) if c.strip() and (len(c)<16 or max(c.count(x) for x in set(c))/len(c)<0.6) else (print('GATE FAIL (TP=2 GDN tensor-split likely incoherent -- use serve_dp2_q4km.sh):',repr(c[:120])) or sys.exit(1))" \
    || { say "coherence gate FAILED -- TP=2 on the GDN recurrent state is the documented risk; fall back to DP=2 Q4_K_M"; return 1; }
  say "healthy + coherent; serving $SERVED (TP=2) on :$PORT"
}
stop(){ docker rm -f "$NAME" >/dev/null 2>&1; say "stopped $NAME"; "$REPO/bin/xpu-health" 2>&1 | tail -1 || true; }

case "${1:-start}" in
  start) start ;;
  stop)  stop ;;
  gen)   curl -s "http://localhost:$PORT/v1/chat/completions" "${AUTH_H[@]}" -H 'content-type: application/json' \
           -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"${2:-Why is the sky blue?}\"}],\"max_tokens\":128,\"temperature\":0}" ;;
  smoke) start; rc=$?; stop; exit $rc ;;
  *) echo "usage: serve_tp2_q8.sh {start|stop|gen|smoke}"; exit 2 ;;
esac
