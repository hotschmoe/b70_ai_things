#!/usr/bin/env bash
# daily_driver_serve.sh -- bring up THE daily-driver model on the dual B70 behind one public endpoint
# at http://192.168.10.5:18080/v1 (Open WebUI + every app keep working unchanged). Run from the dev box.
#
# This is now a THIN orchestrator over the golden path: it PICKS an rdy_to_serve model and serves it via
# that model's own self-contained serve.sh -- ZERO recipe duplication (the recipe lives in one place,
# rdy_to_serve/<model>/serve.sh). Data-parallel (one captured replica per card, ~2.1x aggregate, zero
# inter-GPU comms; measured 2026-06-21) is the dual-GPU win for any model that fits one card.
#
# THE knobs (set on the command line). The THREE serving modes:
#
#   [DP=2] replicate a fits-one-card model across BOTH cards for ~2.1x aggregate capacity  (DEFAULT)
#     ./daily_driver_serve.sh start                              # default model, 2x data-parallel + proxy
#     DD_MODEL=qwen36-35b-a3b-int4 ./daily_driver_serve.sh start # switch model (the one knob)
#
#   [TP=2] one model too big for one card, sharded (tensor-parallel) across both cards
#     DD_REPLICAS=1 DD_MODEL=qwen36-35b-a3b-quark-w8a8-int8 ./daily_driver_serve.sh start
#     (the model's own serve.sh sets TP=2; uses both cards. For PP=2 instead, add DD_ENV="TP=1 PP=2"
#      once the model serve.sh/_common supports PP -- TP=2 is the supported path today.)
#
#   [1 CARD] serve a small model on ONE card, leave the OTHER card FREE for experiments
#     DD_CARD=0 DD_MODEL=qwen3-14b-w8a8 ./daily_driver_serve.sh start   # daily driver pinned to card 0
#     # then experiment on the free card via the per-card lease, e.g.:
#     ssh root@192.168.10.5 'cd /mnt/vm_8tb/b70 && ./bin/gpu-run --card 1 bash <your-experiment>'
#
#   DD_MTP=1 ... start   # opt-in MTP spec decode (dense models, interactive -- see SERVING.md)
#
# Usage: start (default) | stop | restart | status | logs | logs1 | webui | webui-down
# Lease: DP/TP modes hold BOTH cards; DD_CARD mode holds ONLY that card (leaving the other free).
set -uo pipefail

# ===== daily-driver CONFIG (knobs; defaults below) ===========================================
DD_MODEL="${DD_MODEL:-qwen36-27b-int4}"   # any rdy_to_serve/<dir> -- THE model knob. Default: 27B int4 (PRIMARY).
DD_REPLICAS="${DD_REPLICAS:-2}"           # 2 = data-parallel (model fits ONE card). 1 = single serve (TP=2 / big).
DD_CARD="${DD_CARD:-}"                     # set to 0 or 1 -> ONE-CARD mode: pin to that card + lease ONLY that card
                                          #     (leaves the other free for `gpu-run --card <other>` experiments).
DD_MAXLEN="${DD_MAXLEN:-131072}"          # daily-driver context (the model serve.sh default is a modest 8192).
DD_MTP="${DD_MTP:-0}"                      # 1 = MTP spec=4 (~1.79x single-stream interactive; DENSE models only).
DD_ENV="${DD_ENV:-}"                       # advanced: extra env passed verbatim to serve.sh (e.g. "GRAPH=0").
# =============================================================================================
[ -n "$DD_CARD" ] && DD_REPLICAS=1        # one-card mode is inherently a single replica

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

  local bringup wait_targets gpurun="./bin/gpu-run"
  if [ "$DD_REPLICAS" = 2 ]; then
    # [DP=2] one captured replica per card (same served-model-name -> proxy transparent), nginx round-robin. Both cards.
    local serve0="$renv DEVICE=0 PORT=$P0 NAME=$DP0 bash $SERVE start"
    local serve1="$renv DEVICE=1 PORT=$P1 NAME=$DP1 bash $SERVE start"
    local proxy_run="docker rm -f $PROXY >/dev/null 2>&1; docker run -d --name $PROXY --network host -v $ROOT/bin/dp_nginx.conf:/etc/nginx/nginx.conf:ro --restart unless-stopped nginx:alpine >/dev/null 2>&1"
    bringup="$serve0 && $serve1 && $proxy_run && docker wait $DP0 $DP1"
    wait_targets="$DP0 $DP1"
    echo "  replica 0 -> card 0 :$P0 ; replica 1 -> card 1 :$P1 ; proxy -> :$PORT"
  elif [ -n "$DD_CARD" ]; then
    # [1 CARD] pin to card $DD_CARD, lease ONLY that card (other card stays free for `gpu-run --card <other>`).
    gpurun="./bin/gpu-run --card $DD_CARD"
    bringup="$renv DEVICE=$DD_CARD PORT=$PORT NAME=$DP0 bash $SERVE start && docker wait $DP0"
    wait_targets="$DP0"
    echo "  single replica -> card $DD_CARD :$PORT  (leasing ONLY card $DD_CARD; the other card is FREE)"
  else
    # [TP=2] single serve on the public port; the model's own serve.sh drives both cards (TP=2 / too big for one). Both cards.
    bringup="$renv PORT=$PORT NAME=$DP0 bash $SERVE start && docker wait $DP0"
    wait_targets="$DP0"
    echo "  single replica -> :$PORT (the model's serve.sh decides card/TP; both cards leased)"
  fi
  # gpu-run holds the lease; 'docker wait' pins it for the whole serving lifetime (released on stop -> docker stop).
  ssh_h "cd $ROOT && mkdir -p logs && nohup setsid $gpurun bash -c \"$bringup\" > $LOG 2>&1 < /dev/null & echo '  launched (pid '\$!')'"

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
    echo "=== model ===";        echo "  DD_MODEL=$DD_MODEL  replicas=$DD_REPLICAS  card=${DD_CARD:-both}  mtp=$DD_MTP"
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
