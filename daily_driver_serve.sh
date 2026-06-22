#!/usr/bin/env bash
# daily_driver_serve.sh -- bring up THE daily-driver model on the dual B70 behind one public endpoint
# at http://192.168.10.5:18080/v1 (Open WebUI + every app keep working unchanged). Run from the dev box.
#
# This is now a THIN orchestrator over the golden path: it PICKS an rdy_to_serve model and serves it via
# that model's own self-contained serve.sh -- ZERO recipe duplication (the recipe lives in one place,
# rdy_to_serve/<model>/serve.sh). Data-parallel (one captured replica per card, ~2.1x aggregate, zero
# inter-GPU comms; measured 2026-06-21) is the dual-GPU win for any model that fits one card.
#
# To change what we serve: set DD_MODEL to any dir under rdy_to_serve/ (the ONE knob). Examples:
#   ./daily_driver_serve.sh start                              # default model (below), 2x data-parallel
#   DD_MODEL=qwen36-35b-a3b-int4 ./daily_driver_serve.sh start # switch to the fast MoE
#   DD_MODEL=qwen3-14b-w8a8 ./daily_driver_serve.sh start      # switch to the int8 baseline
#   DD_REPLICAS=1 DD_MODEL=qwen36-35b-a3b-quark-w8a8-int8 ./daily_driver_serve.sh start  # a TP=2 model (no DP)
#   DD_MTP=1 ./daily_driver_serve.sh start                     # opt-in MTP spec decode (interactive lever)
#
# Usage: start (default) | stop | restart | status | logs | logs1 | webui | webui-down
# Note: HOLDS the gpu-run lease (both cards) while up. 'stop' before GPU experiments or quantization.
set -uo pipefail

# ===== daily-driver CONFIG (EDIT DD_MODEL to change what we serve) ============================
DD_MODEL="${DD_MODEL:-qwen36-27b-int4}"   # any rdy_to_serve/<dir> -- THE knob. Default: 27B int4 (PRIMARY quality).
DD_REPLICAS="${DD_REPLICAS:-2}"           # 2 = data-parallel (model must fit ONE card). 1 = single serve
                                          #     (for a TP=2 / too-big-for-one-card model -- it uses both cards itself).
DD_MAXLEN="${DD_MAXLEN:-131072}"          # daily-driver context (the model serve.sh default is a modest 8192).
DD_MTP="${DD_MTP:-0}"                      # 1 = MTP spec=4 (~1.79x single-stream interactive; DENSE models only --
                                          #     do NOT use for the MoE; goes compute-bound past ~C8, see SERVING.md).
DD_ENV="${DD_ENV:-}"                       # advanced: extra env passed verbatim to serve.sh (e.g. "GRAPH=0 TOOLCALL=1").
# =============================================================================================

HOST_IP=192.168.10.5
HOST=root@$HOST_IP
ROOT=/mnt/vm_8tb/b70
RTS="$ROOT/rdy_to_serve"
SERVE="$RTS/$DD_MODEL/serve.sh"
PORT=18080                          # PUBLIC endpoint
P0=18091; P1=18092                  # per-replica backend ports (data-parallel)
DP0=vllm_daily_dp0                  # replica 0 -> card 0  (also THE container for DD_REPLICAS=1)
DP1=vllm_daily_dp1                  # replica 1 -> card 1
PROXY=vllm_daily_proxy             # nginx round-robin proxy on :$PORT (DD_REPLICAS=2 only)
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

# env passed to EVERY replica's serve.sh (on top of the model's own defaults).
replica_env() {
  local e="MAXLEN=$DD_MAXLEN"
  [ "$DD_MTP" = 1 ] && e="$e MTPTOK=4 COMPILESZ="    # MTP: integer is quote-safe through nested ssh; COMPILESZ= omits compile_sizes
  [ -n "$DD_ENV" ] && e="$e $DD_ENV"
  printf '%s' "$e"
}

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
    echo "daily driver already UP (:$PORT): $(served_id)"; echo "endpoint: $ENDPOINT"; webui_up; return 0
  fi
  if ! ssh_h "test -f $SERVE"; then
    echo "[!] no such model: $SERVE"; echo "    pick a dir under rdy_to_serve/:"; ssh_h "ls -1 $RTS | grep -v '^_'"; return 1
  fi
  local renv; renv="$(replica_env)"
  echo "starting daily driver: model=$DD_MODEL replicas=$DD_REPLICAS$([ "$DD_MTP" = 1 ] && echo ' +MTP') (holds GPU lease until 'stop')"

  local bringup wait_targets
  if [ "$DD_REPLICAS" = 2 ]; then
    # data-parallel: one captured replica per card (same served-model-name -> proxy is transparent), nginx round-robin.
    local serve0="$renv DEVICE=0 PORT=$P0 NAME=$DP0 bash $SERVE start"
    local serve1="$renv DEVICE=1 PORT=$P1 NAME=$DP1 bash $SERVE start"
    local proxy_run="docker rm -f $PROXY >/dev/null 2>&1; docker run -d --name $PROXY --network host -v $ROOT/bin/dp_nginx.conf:/etc/nginx/nginx.conf:ro --restart unless-stopped nginx:alpine >/dev/null 2>&1"
    bringup="$serve0 && $serve1 && $proxy_run && docker wait $DP0 $DP1"
    wait_targets="$DP0 $DP1"
    echo "  replica 0 -> card 0 :$P0 ; replica 1 -> card 1 :$P1 ; proxy -> :$PORT"
  else
    # single serve directly on the public port (TP=2 / too-big-for-one-card models drive both cards themselves).
    bringup="$renv PORT=$PORT NAME=$DP0 bash $SERVE start && docker wait $DP0"
    wait_targets="$DP0"
    echo "  single replica -> :$PORT (the model's own serve.sh decides card/TP)"
  fi
  # gpu-run holds the lease; 'docker wait' pins it for the whole serving lifetime (released on stop -> docker stop).
  ssh_h "cd $ROOT && mkdir -p logs && nohup setsid ./bin/gpu-run bash -c \"$bringup\" > $LOG 2>&1 < /dev/null & echo '  launched (pid '\$!')'"

  echo -n "  waiting for healthy (model load + capture x$DD_REPLICAS, up to ~18 min) "
  local ok=0 _
  for _ in $(seq 1 216); do
    case "$(ssh_h "if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then echo OK; else for c in $wait_targets; do docker inspect -f '{{.State.Status}}' \$c 2>/dev/null | grep -q exited && { echo EXITED; break; }; done; fi" 2>/dev/null)" in
      *OK*)     ok=1; break ;;
      *EXITED*) echo " A REPLICA EXITED EARLY"; for c in $wait_targets; do echo "--- $c ---"; ssh_h "docker logs $c 2>&1 | tail -20"; done; return 1 ;;
      *)        echo -n "."; sleep 5 ;;
    esac
  done
  echo
  if [ "$ok" = 1 ]; then
    echo "UP. served id: $(served_id)"; echo "endpoint: $ENDPOINT   (chat: $ENDPOINT/chat/completions)"; webui_up
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
    echo "=== model ===";        echo "  DD_MODEL=$DD_MODEL  replicas=$DD_REPLICAS  mtp=$DD_MTP"
    echo "=== GPU lock ===";      ssh_h "$ROOT/bin/gpu-run --status"
    echo "=== replicas+proxy ==="; ssh_h "docker ps --filter name=vllm_daily --format '{{.Names}}  {{.Status}}' | grep . || echo '(not running)'"
    echo "=== served model ===";  echo "$(served_id || true)" | grep . || echo "(not serving)"
    echo "=== web ui ===";        ssh_h "docker ps --filter name=$WEBUI_NAME --format '{{.Names}}  {{.Status}}' | grep . || echo '(not running)'"
    echo "endpoint: $ENDPOINT    web ui: http://$HOST_IP:$WEBUI_PORT" ;;
  logs)    ssh_h "docker logs --tail 60 -f $DP0" ;;
  logs1)   ssh_h "docker logs --tail 60 -f $DP1" ;;
  webui)      webui_up ;;
  webui-down) webui_down ;;
  *) echo "usage: $0 [start|stop|restart|status|logs|logs1|webui|webui-down]   (pick model via DD_MODEL=)"; exit 2 ;;
esac
