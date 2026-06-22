#!/usr/bin/env bash
# daily_driver_serve.sh -- bring up THE daily-driver model on the dual B70 as 2x DATA-PARALLEL replicas
# (one captured replica per card) behind an nginx round-robin proxy at http://192.168.10.5:18080/v1.
# Data-parallel is the dual-GPU win on this no-P2P rig: ~2.1x aggregate throughput, full single-stream
# latency per replica, ZERO inter-GPU comms (measured 2026-06-21; see FINDINGS.md / scripts/64). The public
# endpoint stays :18080, so Open WebUI and every app keep working unchanged. Run from the dev box; SSHes to host.
#
# To change the daily driver: edit the CONFIG block below. Recipes: docs/SERVING.md.
#
# Usage:
#   ./daily_driver_serve.sh [start]   start (default): serve 2 replicas + proxy, hold GPU lease, return when healthy
#   ./daily_driver_serve.sh stop      stop both replicas + proxy (releases the GPU lease for experiments/quant)
#   ./daily_driver_serve.sh restart   stop then start
#   ./daily_driver_serve.sh status    replicas + proxy + GPU lock + served id + web ui + endpoint
#   ./daily_driver_serve.sh logs      follow replica 0's log (use 'logs1' for replica 1)
#   ./daily_driver_serve.sh logs1     follow replica 1's log
#   ./daily_driver_serve.sh webui / webui-down   Open WebUI up/down (does NOT touch the model servers)
#
# Note: the daily driver HOLDS the gpu-run lease (both cards) while up. Run 'stop' before GPU experiments
# or quantization runs.
set -uo pipefail

# ===== daily-driver CONFIG (EDIT THIS to change what we serve) ================================
# CURRENT DAILY DRIVER: Qwen3.6-27B W4A16 (int4 AutoRound), PIECEWISE captured, fp16 KV -- one replica PER CARD.
# (DP requires a model that FITS ONE CARD. For a model too big for one card, use PP=2 instead -- see SERVING.md.)
DD_NAME="Qwen3.6-27B W4A16 (int4 AutoRound), captured, 2x data-parallel"
SERVE_ENV=(
  IMG=vllm-xpu-env:v0230
  MODEL=/models/Lorbus_Qwen3.6-27B-int4-AutoRound   # container path (/models = host /mnt/vm_8tb/b70/models)
  SERVED=qwen36-27b-int4
  GRAPH=1                                            # PIECEWISE capture -> ~30.8 t/s decode/replica (eager ~7.8)
  DTYPE=auto
  UTIL=0.92
  MAXLEN=131072                                      # max single-card ctx at fp16 KV (~133k cap); fp8 KV doubles it
  MAXSEQS=64
  CAPSIZES=1,2,4,8,16,32,64
  NOMM=1                                             # 27B is a VLM -> text-only (skip vision profiling crash)
  TOOLCALL=1                                         # OpenAI tool/function calling -- for pi & coding agents
  TOOLPARSER=qwen3_coder                             # Qwen3.6 emits XML <function=..> (NOT hermes JSON)
  REASONPARSER=qwen3                                 # split <think> reasoning into reasoning_content
)
# OPT-IN MTP (multi-token prediction / speculative decode). DD_MTP=1 ./daily_driver_serve.sh start
# -> spec=4 -> ~1.79x single-stream interactive decode (27B int4: ~30.8 -> ~55 t/s; MEASURED 2026-06-22 MTP
# campaign, beats the 45.2 Lorbus precedent; Half-KV-free). TRADEOFF: MTP is a LOW-CONCURRENCY lever -- the
# spec-verify runs the model x(1+spec) so it goes compute-bound as concurrency rises and can REDUCE aggregate
# throughput past ~C8 (FINDINGS: 27B int4 per-stream decode drops past C8). Enable for INTERACTIVE use (1-few
# users / Open WebUI / single coding agent); leave OFF for high-concurrency batch/agent fan-out. Default OFF.
# (Passes MTPTOK=4 -- a quote-safe integer -- to 30_serve, which builds the spec-config JSON; passing the JSON
# directly through the nested ssh/bash -c strips its quotes. COMPILESZ= omits compile_sizes (spec-decode rejects [1]).)
DD_MTP="${DD_MTP:-0}"
if [ "$DD_MTP" = 1 ]; then
  SERVE_ENV+=( MTPTOK=4 COMPILESZ= )
  DD_NAME="$DD_NAME + MTP spec=4 (~1.79x single-stream)"
fi
# =============================================================================================

HOST_IP=192.168.10.5
HOST=root@$HOST_IP
ROOT=/mnt/vm_8tb/b70
PORT=18080                          # PUBLIC endpoint (nginx proxy)
P0=18091; P1=18092                  # per-replica backend ports
DP0=vllm_daily_dp0                  # replica 0 -> card 0
DP1=vllm_daily_dp1                  # replica 1 -> card 1
PROXY=vllm_daily_proxy             # nginx round-robin proxy on :$PORT
LOG="$ROOT/logs/daily_driver.log"
ENDPOINT="http://$HOST_IP:${PORT}/v1"

# ----- web UI (Open WebUI) -- tied to the daily-driver lifecycle -----------------------------
WEBUI_ENABLE=1
WEBUI_NAME=open-webui
WEBUI_PORT=3000
WEBUI_IMAGE=ghcr.io/open-webui/open-webui:main
# ---------------------------------------------------------------------------------------------

ssh_h() { ssh -o ConnectTimeout=8 -o ServerAliveInterval=60 "$HOST" "$@"; }
served_id() { ssh_h "curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -oE '\"id\":\"[^\"]*\"' | head -1"; }

webui_up() {
  [ "${WEBUI_ENABLE:-0}" = 1 ] || return 0
  if ssh_h "docker inspect $WEBUI_NAME >/dev/null 2>&1"; then
    ssh_h "docker start $WEBUI_NAME >/dev/null 2>&1 && echo '  web ui: up   -> http://$HOST_IP:$WEBUI_PORT'"
  else
    ssh_h "docker run -d --name $WEBUI_NAME -p $WEBUI_PORT:8080 \
      -e OPENAI_API_BASE_URL=http://$HOST_IP:$PORT/v1 -e OPENAI_API_KEY=dummy \
      -e ENABLE_OLLAMA_API=False -e WEBUI_AUTH=False \
      -v open-webui:/app/backend/data --restart unless-stopped $WEBUI_IMAGE >/dev/null 2>&1 \
      && echo '  web ui: created -> http://$HOST_IP:$WEBUI_PORT (first open pulls the image, ~1 min)'"
  fi
}
webui_down() { ssh_h "docker stop $WEBUI_NAME >/dev/null 2>&1 && echo '  web ui: down' || true"; }

start() {
  if ssh_h "curl -sf http://localhost:$PORT/health >/dev/null 2>&1"; then
    echo "daily driver already UP (proxy :$PORT): $(served_id)"; echo "endpoint: $ENDPOINT"; webui_up; return 0
  fi
  echo "starting daily driver (2x data-parallel): $DD_NAME"
  echo "  replica 0 -> card 0 :$P0 ; replica 1 -> card 1 :$P1 ; proxy -> :$PORT  (holds GPU lease until 'stop')"
  # Both replicas keep the SAME served-model-name (proxy is transparent: clients use one model id);
  # only the container NAME + card (DEVICE) + backend PORT differ.
  local serve0="${SERVE_ENV[*]} DEVICE=0 PORT=$P0 NAME=$DP0"
  local serve1="${SERVE_ENV[*]} DEVICE=1 PORT=$P1 NAME=$DP1"
  local proxy_run="docker rm -f $PROXY >/dev/null 2>&1; docker run -d --name $PROXY --network host -v $ROOT/dp_nginx.conf:/etc/nginx/nginx.conf:ro --restart unless-stopped nginx:alpine >/dev/null 2>&1"
  # gpu-run holds the lease; bring up BOTH replicas (sequential), then the proxy, then 'docker wait' both
  # replicas pins the lease for the whole serving lifetime (released when 'stop' -> docker stop -> wait returns).
  local bringup="$serve0 bash ./30_serve_w4a8_graph.sh && $serve1 bash ./30_serve_w4a8_graph.sh && $proxy_run && docker wait $DP0 $DP1"
  ssh_h "cd $ROOT && mkdir -p logs && nohup setsid ./gpu-run bash -c \"$bringup\" > $LOG 2>&1 < /dev/null & echo '  launched (pid '\$!')'"
  echo -n "  waiting for both replicas + proxy (model load + graph capture x2, up to ~18 min) "
  local ok=0
  for _ in $(seq 1 216); do
    case "$(ssh_h "if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then echo OK; elif docker inspect -f '{{.State.Status}}' $DP0 2>/dev/null | grep -q exited || docker inspect -f '{{.State.Status}}' $DP1 2>/dev/null | grep -q exited; then echo EXITED; else echo WAIT; fi")" in
      OK) ok=1; break ;;
      EXITED) echo " A REPLICA EXITED EARLY"; ssh_h "docker logs $DP0 2>&1 | tail -20; echo '--- dp1 ---'; docker logs $DP1 2>&1 | tail -20"; return 1 ;;
      *) echo -n "."; sleep 5 ;;
    esac
  done
  echo
  if [ "$ok" = 1 ]; then
    echo "UP (2x DP). proxy served id: $(served_id)"
    echo "endpoint: $ENDPOINT   (chat: $ENDPOINT/chat/completions)"
    webui_up
  else
    echo "NOT healthy after wait -- check: ./daily_driver_serve.sh logs"; return 1
  fi
}

stop() {
  echo "stopping daily driver (releases GPU lease)..."
  ssh_h "docker stop $PROXY $DP0 $DP1 >/dev/null 2>&1 && echo '  replicas + proxy stopped' || echo '  not running'"
  webui_down
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) echo "restarting..."; ssh_h "docker stop $PROXY $DP0 $DP1 >/dev/null 2>&1 || true"; start ;;
  status)
    echo "=== GPU lock ===";     ssh_h "$ROOT/gpu-run --status"
    echo "=== replicas + proxy ==="; ssh_h "docker ps --filter name=vllm_daily --format '{{.Names}}  {{.Status}}' | grep . || echo '(not running)'"
    echo "=== served model ==="; echo "$(served_id || true)" | grep . || echo "(not serving)"
    echo "=== web ui ===";       ssh_h "docker ps --filter name=$WEBUI_NAME --format '{{.Names}}  {{.Status}}' | grep . || echo '(not running)'"
    echo "endpoint: $ENDPOINT    web ui: http://$HOST_IP:$WEBUI_PORT" ;;
  logs)    ssh_h "docker logs --tail 60 -f $DP0" ;;
  logs1)   ssh_h "docker logs --tail 60 -f $DP1" ;;
  webui)      webui_up ;;
  webui-down) webui_down ;;
  *) echo "usage: $0 [start|stop|restart|status|logs|logs1|webui|webui-down]"; exit 2 ;;
esac
