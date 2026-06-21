#!/usr/bin/env bash
# daily_driver_serve.sh -- bring up THE current daily-driver model on the B70 and keep it serving at
# http://192.168.10.5:18080/v1 so our apps can hit the API. Run from the dev box; it SSHes to the GPU host.
#
# To change the daily driver: edit the CONFIG block below (or swap to a preset). Recipes: docs/SERVING.md.
#
# Usage:
#   ./daily_driver_serve.sh [start]   start (default): serve + hold the GPU lease, return once healthy
#   ./daily_driver_serve.sh stop      stop the server (releases the GPU lease for experiments)
#   ./daily_driver_serve.sh restart   stop then start
#   ./daily_driver_serve.sh status    container + GPU lock + served model id + web ui + endpoint
#   ./daily_driver_serve.sh logs      follow the server log
#   ./daily_driver_serve.sh webui     bring Open WebUI up standalone (does NOT touch the model server)
#   ./daily_driver_serve.sh webui-down  stop Open WebUI only
#
# Open WebUI (a chat frontend, CPU-only, no GPU) is tied to the daily driver: it comes UP on `start`
# and DOWN on `stop`. It hits the API endpoint, so launching/stopping it never restarts the model server.
#
# Note: the daily driver HOLDS the gpu-run lease while up (only one model fits the card), so any
# experiment's `gpu-run` will wait. Run `./daily_driver_serve.sh stop` before doing GPU experiments.
set -uo pipefail

# ===== daily-driver CONFIG (EDIT THIS to change what we serve) ================================
# CURRENT DAILY DRIVER: Qwen3.6-27B W4A16 (int4 AutoRound), PIECEWISE captured, fp16 KV, 128k ctx.
DD_NAME="Qwen3.6-27B W4A16 (int4 AutoRound), captured, 128k ctx, fp16 KV"
SERVE_ENV=(
  IMG=vllm-xpu-env:v0230
  MODEL=/models/Lorbus_Qwen3.6-27B-int4-AutoRound   # container path (/models = host /mnt/vm_8tb/b70/models)
  SERVED=qwen36-27b-int4
  GRAPH=1                                            # PIECEWISE capture -> ~30.8 t/s decode (eager is ~7.8)
  DTYPE=auto
  UTIL=0.92
  MAXLEN=131072                                      # max single-card ctx at fp16 KV (~133k cap); fp8 KV doubles it
  MAXSEQS=64
  CAPSIZES=1,2,4,8,16,32,64                          # capture batch sizes (else batches >8 fall back to eager)
  NOMM=1                                             # 27B is a VLM -> text-only (skip vision profiling crash)
  TOOLCALL=1                                         # OpenAI tool/function calling -- for pi & coding agents
  TOOLPARSER=qwen3_coder                             # Qwen3.6 emits XML <function=..> (NOT hermes JSON) -> qwen3_coder
  REASONPARSER=qwen3                                 # split <think> reasoning into reasoning_content
)
# --- PRESETS (uncomment one block, comment the one above) ------------------------------------
# 35B-A3B MoE (FASTEST decode ~65 t/s, fp8 KV):  image :v0230moe
#   DD_NAME="Qwen3.6-35B-A3B MoE int4, captured, fp8 KV"
#   SERVE_ENV=(IMG=vllm-xpu-env:v0230moe MODEL=/models/Intel_Qwen3.6-35B-A3B-int4-AutoRound \
#     SERVED=qwen36-35b-a3b-int4 GRAPH=1 DTYPE=auto UTIL=0.90 MAXLEN=8192 MAXSEQS=64 \
#     CAPSIZES=1,2,4,8,16,32,64 KVDTYPE=fp8_e5m2)
# 27B W4A8 (quality GDN bf16, ~20.9 t/s; needs prepack + rebuilt GDN .so + fp8 KV -- see docs/SERVING.md)
#   DD_NAME="Qwen3.6-27B W4A8 prepacked, captured, fp8 KV"
#   SERVE_ENV=(IMG=vllm-xpu-env:int8g MODEL=/models/Qwen3.6-27B-W4A8-q-prepacked SERVED=qwen36-27b-w4a8 \
#     GRAPH=1 PREPACK=1 NOMM=1 KVDTYPE=fp8_e5m2 UTIL=0.90 \
#     KERNEL_SO=/mnt/vm_8tb/b70/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so)
# =============================================================================================

HOST_IP=192.168.10.5
HOST=root@$HOST_IP
ROOT=/mnt/vm_8tb/b70
PORT=18080
CONTAINER=vllm_daily        # fixed name for the daily driver (independent of experiment containers)
LOG="$ROOT/logs/daily_driver.log"
ENDPOINT="http://$HOST_IP:${PORT}/v1"

# ----- web UI (Open WebUI) -- tied to the daily-driver lifecycle -----------------------------
WEBUI_ENABLE=1                                # 1 = bring Open WebUI up on start, down on stop
WEBUI_NAME=open-webui
WEBUI_PORT=3000
WEBUI_IMAGE=ghcr.io/open-webui/open-webui:main
# ---------------------------------------------------------------------------------------------

ssh_h() { ssh -o ConnectTimeout=8 "$HOST" "$@"; }
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
    echo "daily driver already UP: $(served_id)"; echo "endpoint: $ENDPOINT"; webui_up; return 0
  fi
  echo "starting daily driver: $DD_NAME"
  echo "  serving on $ENDPOINT  (holds the GPU lease until you 'stop')"
  local serve_kv="${SERVE_ENV[*]} NAME=$CONTAINER"
  # gpu-run holds the lease; 30_serve starts+health-waits and returns; 'docker wait' then pins the
  # lease for the container's whole lifetime (released when you 'stop' -> docker stop -> wait returns).
  ssh_h "cd $ROOT && mkdir -p logs && nohup setsid ./gpu-run bash -c '$serve_kv bash ./30_serve_w4a8_graph.sh && docker wait $CONTAINER' > $LOG 2>&1 < /dev/null & echo '  launched (pid '\$!')'"
  echo -n "  waiting for readiness (model load + graph capture, up to ~14 min) "
  local ok=0
  for _ in $(seq 1 180); do
    case "$(ssh_h "if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then echo OK; elif docker inspect -f '{{.State.Status}}' $CONTAINER 2>/dev/null | grep -q exited; then echo EXITED; else echo WAIT; fi")" in
      OK) ok=1; break ;;
      EXITED) echo " EXITED EARLY"; ssh_h "docker logs $CONTAINER 2>&1 | tail -30"; return 1 ;;
      *) echo -n "."; sleep 5 ;;
    esac
  done
  echo
  if [ "$ok" = 1 ]; then
    echo "UP. served model: $(served_id)"
    echo "endpoint: $ENDPOINT   (chat: $ENDPOINT/chat/completions)"
    webui_up
  else
    echo "NOT healthy after wait -- check: ./daily_driver_serve.sh logs"; return 1
  fi
}

case "${1:-start}" in
  start)   start ;;
  stop)    echo "stopping daily driver (releases GPU lease)..."; ssh_h "docker stop $CONTAINER >/dev/null 2>&1 && echo stopped || echo 'not running'"; webui_down ;;
  restart) echo "restarting..."; ssh_h "docker stop $CONTAINER >/dev/null 2>&1 || true"; start ;;
  status)
    echo "=== GPU lock ===";     ssh_h "$ROOT/gpu-run --status"
    echo "=== container ===";    ssh_h "docker ps --filter name=$CONTAINER --format '{{.Names}}  {{.Status}}' | grep . || echo '(not running)'"
    echo "=== served model ==="; echo "$(served_id || true)" | grep . || echo "(not serving)"
    echo "=== web ui ===";       ssh_h "docker ps --filter name=$WEBUI_NAME --format '{{.Names}}  {{.Status}}' | grep . || echo '(not running)'"
    echo "endpoint: $ENDPOINT    web ui: http://$HOST_IP:$WEBUI_PORT" ;;
  logs)    ssh_h "docker logs --tail 60 -f $CONTAINER" ;;
  webui)      webui_up ;;
  webui-down) webui_down ;;
  *) echo "usage: $0 [start|stop|restart|status|logs|webui|webui-down]"; exit 2 ;;
esac
