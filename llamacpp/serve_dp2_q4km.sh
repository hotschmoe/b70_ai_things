#!/usr/bin/env bash
# llamacpp/serve_dp2_q4km.sh -- "W4A16-like, TP=1 DP=2": qwen3.6-27b Q4_K_M, ONE llama-server per card +
# nginx round-robin on the public port. The LOW-RISK path (no cross-card collectives; mirrors the existing
# vLLM/sglang DP=2 daily-driver pattern). Each replica is a full single-GPU serve -> ~2x aggregate capacity.
#
# Quant note: Q4_K_M = 4-bit WEIGHTS, fp16 compute (REVIEW_intel_arch.md sec 4). This is the community
# B70-validated config and the expected production default (Q8_0 has been ~4x slower on B70, #21517).
#
# MUST run under the GPU lease holding BOTH cards (each replica pins one card via ONEAPI_DEVICE_SELECTOR):
#   cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run bash llamacpp/serve_dp2_q4km.sh start
#   bash llamacpp/serve_dp2_q4km.sh stop
#
# Prereqs: build_sycl.sh (binaries at $SRC/build/bin) + convert_gguf.sh (GGUF at $GGUF).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"; REPO="${REPO:-/mnt/vm_8tb/github/b70_ai_things}"
IMG="${IMG:-sglang-xpu:mtp}"                       # ABI-matched runtime (binaries built in this image)
SRC="${SRC:-$ROOT/llama.cpp}"                      # built llama.cpp (build/bin)
GGUF="${GGUF:-$ROOT/llamacpp/gguf}"
TAG="${TAG:-qwen3.6-27b}"
MODEL="${MODEL:-/gguf/${TAG}-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-/gguf/${TAG}-mmproj-f16.gguf}"   # vision tower; auto-included if the file exists (NOMM=1 to skip)
SERVED="${SERVED:-qwen36-27b-q4km}"
NAME="${NAME:-llamacpp_q4km}"                      # base container name (-> ${NAME}_0 / ${NAME}_1 / ${NAME}_proxy)
PORT="${PORT:-18080}"                              # PUBLIC endpoint (proxy)
P0="${P0:-18181}"; P1="${P1:-18182}"               # per-card backend ports
CTX="${CTX:-${MAXLEN:-32768}}"                     # honors the backend-agnostic MAXLEN knob (daily-driver DD_MAXLEN)
PAR="${PAR:-4}"                                    # --parallel (concurrent slots) per replica
METRICS="${METRICS:-1}"; API_KEY="${API_KEY:-}"
LOG="${LOG:-$SCRIPT_DIR/serve_dp2_q4km.log}"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
AUTH_H=(); [ -n "$API_KEY" ] && AUTH_H=(-H "Authorization: Bearer $API_KEY")

# Build the llama-server arg string for a single-card replica pinned to card $1 on backend port $2.
replica_cmd(){ local card="$1" p="$2"
  local mm=""; { [ "${NOMM:-0}" != 1 ] && [ -s "$GGUF/${TAG}-mmproj-f16.gguf" ]; } && mm="--mmproj $MMPROJ"
  local key=""; [ -n "$API_KEY" ] && key="--api-key $API_KEY"
  local met=""; [ "$METRICS" = 1 ] && met="--metrics"
  cat <<EOF
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export LD_LIBRARY_PATH=/llama/build/bin:/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH
exec /llama/build/bin/llama-server -m $MODEL $mm -ngl 999 --ctx-size $CTX \
  --split-mode none --main-gpu 0 --flash-attn auto --parallel $PAR --cont-batching \
  --jinja $met $key --served-model-name $SERVED --host 0.0.0.0 --port $p
EOF
}

run_replica(){ local card="$1" p="$2" cname="$3"
  docker rm -f "$cname" >/dev/null 2>&1
  docker run -d --name "$cname" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g -p "${p}:${p}" \
    -v "$SRC:/llama:ro" -v "$GGUF:/gguf:ro" \
    -e ZES_ENABLE_SYSMAN=1 -e ONEAPI_DEVICE_SELECTOR="level_zero:$card" \
    "$IMG" bash -c "$(replica_cmd "$card" "$p")" >/dev/null
}

write_nginx(){ cat > "$SCRIPT_DIR/.dp_nginx.conf" <<EOF
worker_processes 1;
events { worker_connections 1024; }
http {
  upstream llamacpp_dp { server 127.0.0.1:$P0; server 127.0.0.1:$P1; }
  server {
    listen $PORT;
    location / {
      proxy_pass http://llamacpp_dp;
      proxy_read_timeout 1800s; proxy_send_timeout 1800s;
      proxy_set_header Host \$host;
    }
  }
}
EOF
}

start(){
  say "pre-flight xpu-health"; "$REPO/bin/xpu-health" 2>&1 | tail -2 || { say "UNHEALTHY -- abort"; return 3; }
  [ -s "$GGUF/${TAG}-Q4_K_M.gguf" ] || { say "missing $GGUF/${TAG}-Q4_K_M.gguf -- run convert_gguf.sh first"; return 2; }
  say "DP=2 Q4_K_M: replica0->card0 :$P0  replica1->card1 :$P1  proxy :$PORT  (ctx=$CTX par=$PAR mm=$([ "${NOMM:-0}" = 1 ] && echo off || echo auto))"
  run_replica 0 "$P0" "${NAME}_0"
  run_replica 1 "$P1" "${NAME}_1"
  # wait both replicas healthy (model load on first run can take minutes)
  local r p ok
  for r in 0 1; do p=$([ $r = 0 ] && echo "$P0" || echo "$P1"); ok=0
    say "waiting replica$r (/health :$p)..."
    for i in $(seq 1 120); do
      docker ps --filter "name=${NAME}_$r" --format '{{.Names}}' | grep -q "${NAME}_$r" || { say "replica$r EXITED"; docker logs "${NAME}_$r" 2>&1|tail -30; return 1; }
      [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$p/health 2>/dev/null||echo 000)" = 200 ] && { ok=1; say "replica$r healthy (~$((i*5))s)"; break; }
      sleep 5
    done
    [ "$ok" = 1 ] || { say "replica$r NOT healthy"; docker logs "${NAME}_$r" 2>&1|tail -30; return 1; }
  done
  write_nginx
  docker rm -f "${NAME}_proxy" >/dev/null 2>&1
  docker run -d --name "${NAME}_proxy" --network host \
    -v "$SCRIPT_DIR/.dp_nginx.conf:/etc/nginx/nginx.conf:ro" --restart unless-stopped nginx:alpine >/dev/null
  coherence_gate "$PORT" || { say "coherence gate FAILED"; return 1; }
  say "healthy + coherent; serving $SERVED (DP=2) on :$PORT"
}

coherence_gate(){ local port="$1" g
  g=$(curl -s --max-time 180 "http://localhost:$port/v1/chat/completions" "${AUTH_H[@]}" -H 'content-type: application/json' \
      -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Why is the sky blue? Two sentences.\"}],\"max_tokens\":256,\"temperature\":0}")
  echo "$g" | python3 -c "import sys,json
m=json.load(sys.stdin)['choices'][0]['message']
c=(m.get('content') or '') or (m.get('reasoning_content') or '')
print('COHERENCE OK:',repr(c[:160])) if c.strip() and (len(c)<16 or max(c.count(x) for x in set(c))/len(c)<0.6) else (print('GATE FAIL:',repr(c[:120])) or sys.exit(1))"
}

stop(){ docker rm -f "${NAME}_proxy" "${NAME}_0" "${NAME}_1" >/dev/null 2>&1; say "stopped DP=2"; "$REPO/bin/xpu-health" 2>&1 | tail -1 || true; }

case "${1:-start}" in
  start) start ;;
  stop)  stop ;;
  gen)   curl -s "http://localhost:$PORT/v1/chat/completions" "${AUTH_H[@]}" -H 'content-type: application/json' \
           -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"${2:-Why is the sky blue?}\"}],\"max_tokens\":128,\"temperature\":0}" ;;
  smoke) start; rc=$?; stop; exit $rc ;;
  *) echo "usage: serve_dp2_q4km.sh {start|stop|gen|smoke}"; exit 2 ;;
esac
