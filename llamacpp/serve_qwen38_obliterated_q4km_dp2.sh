#!/usr/bin/env bash
# OBLITERATUS Qwen3.8-27B V3 Q4_K_M on DP=2: one B70 replica per card,
# one OpenAI-compatible endpoint behind nginx, served as hotschmoe-dd.
#
# GPU discipline:
#   ./bin/gpu-run bash llamacpp/serve_qwen38_obliterated_q4km_dp2.sh start
#   ./bin/gpu-run bash llamacpp/serve_qwen38_obliterated_q4km_dp2.sh stop
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-start}"
NAME="${NAME:-qwen38_oblit_q4km_dp2}"
PORT="${PORT:-18080}"
P0="${P0:-18181}"
P1="${P1:-18182}"
SERVED="${SERVED:-hotschmoe-dd}"
CTX_SIZE="${CTX_SIZE:-245760}"
PARALLEL="${PARALLEL:-1}"
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-256}"
KV_TYPE="${KV_TYPE:-q8_0}"
LAB_DOORS="${LAB_DOORS:-1}"
ENABLE_MTP="${ENABLE_MTP:-1}"
MTP_SIDECAR="${MTP_SIDECAR:-0}"
MTP_DRAFT_MAX="${MTP_DRAFT_MAX:-3}"
HOST_MODELS="${HOST_MODELS:-$REPO/models/files/qwen3.8-27b/obliterated-q4km}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf}"
MODEL_SHA256="${MODEL_SHA256:-c5e4fe705883e244a468c9e445c8d6ba37fd310b0113e25d2b8a7f2d6f1243e8}"
MTP_FILE="${MTP_FILE:-mtp-Qwen3.8-27B-Q4_0.gguf}"
MTP_SHA256="${MTP_SHA256:-051a1764cff8c4f3ee6ae8b00593a0364c7539c67fa50ffc58f3f96509fca38e}"
OVERLAY="${OVERLAY:-$REPO/llamacpp/obliterated_q4km_entrypoint.sh}"
IMG="${IMG:-qwen38-b70:latest}"
IMG_ID="${IMG_ID:-sha256:8c6dc0462011e7d4596882009fc7fb1128fbe656cb17a998999cd1e720a2b4de}"
API_KEY_FILE="${API_KEY_FILE:-/mnt/vm_8tb/b70/secrets/dd_api_key}"
RUNTIME_DIR="${RUNTIME_DIR:-/mnt/vm_8tb/b70/llamacpp}"
NGINX_CONF="$RUNTIME_DIR/${NAME}.nginx.conf"

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

auth_args() {
    AUTH_H=()
    if [ -s "$API_KEY_FILE" ]; then
        AUTH_H=(-H "Authorization: Bearer $(<"$API_KEY_FILE")")
    fi
}

check_artifacts() {
    [ -s "$HOST_MODELS/$MODEL_FILE" ] || { say "missing $HOST_MODELS/$MODEL_FILE"; return 2; }
    local size
    size="$(stat -c %s "$HOST_MODELS/$MODEL_FILE")"
    [ "$size" = 16810714400 ] || { say "wrong model size: $size"; return 2; }
    echo "$MODEL_SHA256  $HOST_MODELS/$MODEL_FILE" | sha256sum -c - || return 2
    [ -x "$OVERLAY" ] || { say "entrypoint is not executable: $OVERLAY"; return 2; }
    docker image inspect "$IMG" >/dev/null 2>&1 || { say "missing image $IMG"; return 2; }
    local actual_id
    actual_id="$(docker image inspect "$IMG" --format '{{.Id}}')"
    [ "$actual_id" = "$IMG_ID" ] || { say "image id mismatch: $actual_id"; return 2; }
    if [ "$ENABLE_MTP" = 1 ] && [ "$MTP_SIDECAR" = 1 ]; then
        [ -s "$HOST_MODELS/$MTP_FILE" ] || { say "missing MTP sidecar $HOST_MODELS/$MTP_FILE"; return 2; }
        echo "$MTP_SHA256  $HOST_MODELS/$MTP_FILE" | sha256sum -c - || return 2
    fi
}

run_replica() {
    local card="$1" host_port="$2" cname="$3"
    local key_mount=()
    [ -s "$API_KEY_FILE" ] && key_mount=(-v "$API_KEY_FILE:/run/secrets/dd_api_key:ro")
    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run -d --name "$cname" --restart unless-stopped \
        --device /dev/dri --ipc=host --shm-size 8g \
        -v /dev/dri/by-path:/dev/dri/by-path:ro \
        -v "$HOST_MODELS:/models:ro" \
        -v "$OVERLAY:/entrypoint.sh:ro" \
        "${key_mount[@]}" \
        -e ONEAPI_DEVICE_SELECTOR="level_zero:$card" \
        -e MODELS_DIR=/models -e MODEL_FILE="$MODEL_FILE" -e MODEL_SHA256="$MODEL_SHA256" \
        -e SERVED="$SERVED" -e PORT=8010 \
        -e CTX_SIZE="$CTX_SIZE" -e PARALLEL="$PARALLEL" \
        -e BATCH="$BATCH" -e UBATCH="$UBATCH" -e KV_TYPE="$KV_TYPE" \
        -e LAB_DOORS="$LAB_DOORS" \
        -e ENABLE_MTP="$ENABLE_MTP" -e MTP_SIDECAR="$MTP_SIDECAR" -e MTP_FILE="$MTP_FILE" \
        -e MTP_SHA256="$MTP_SHA256" -e MTP_DRAFT_MAX="$MTP_DRAFT_MAX" \
        -p "127.0.0.1:${host_port}:8010" \
        --entrypoint bash "$IMG" /entrypoint.sh >/dev/null
}

wait_replicas() {
    local deadline=$((SECONDS + 1200))
    local ok0=0 ok1=0
    while [ "$SECONDS" -lt "$deadline" ]; do
        docker ps --format '{{.Names}}' | grep -qx "${NAME}_0" || {
            say "replica 0 exited"; docker logs --tail 120 "${NAME}_0" 2>&1; return 1;
        }
        docker ps --format '{{.Names}}' | grep -qx "${NAME}_1" || {
            say "replica 1 exited"; docker logs --tail 120 "${NAME}_1" 2>&1; return 1;
        }
        curl -fsS --max-time 3 "http://127.0.0.1:$P0/health" >/dev/null 2>&1 && ok0=1
        curl -fsS --max-time 3 "http://127.0.0.1:$P1/health" >/dev/null 2>&1 && ok1=1
        if [ "$ok0" = 1 ] && [ "$ok1" = 1 ]; then
            say "both replicas healthy"
            return 0
        fi
        sleep 5
    done
    say "replica health wait timed out"
    docker logs --tail 120 "${NAME}_0" 2>&1 || true
    docker logs --tail 120 "${NAME}_1" 2>&1 || true
    return 1
}

write_nginx() {
    mkdir -p "$RUNTIME_DIR"
    cat >"$NGINX_CONF" <<EOF
worker_processes 1;
events { worker_connections 4096; }
http {
  upstream qwen38_obliterated_dp2 {
    server 127.0.0.1:$P0;
    server 127.0.0.1:$P1;
  }
  server {
    listen $PORT;
    client_max_body_size 32m;
    location / {
      proxy_pass http://qwen38_obliterated_dp2;
      proxy_http_version 1.1;
      proxy_buffering off;
      proxy_request_buffering off;
      proxy_read_timeout 3600s;
      proxy_send_timeout 3600s;
      proxy_set_header Host \$host;
      add_header X-B70-Upstream \$upstream_addr always;
    }
  }
}
EOF
}

start_proxy() {
    write_nginx
    docker rm -f "${NAME}_proxy" >/dev/null 2>&1 || true
    docker run -d --name "${NAME}_proxy" --restart unless-stopped --network host \
        -v "$NGINX_CONF:/etc/nginx/nginx.conf:ro" nginx:alpine >/dev/null
}

gate_one() {
    local base="$1" label="$2" payload body content
    payload="{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
    body="$(curl -fsS --max-time 300 "${AUTH_H[@]}" -H 'content-type: application/json' \
        -d "$payload" "$base/v1/chat/completions")" || return 1
    content="$(printf '%s' "$body" | jq -r '.choices[0].message.content // empty')"
    printf '%s' "$content" | grep -qi paris || {
        say "$label coherence FAIL: ${content:0:160}"; return 1;
    }
    say "$label coherence PASS: ${content:0:120}"
}

coherence_gate() {
    auth_args
    gate_one "http://127.0.0.1:$P0" replica0 || return 1
    gate_one "http://127.0.0.1:$P1" replica1 || return 1
    gate_one "http://127.0.0.1:$PORT" proxy0 || return 1
    gate_one "http://127.0.0.1:$PORT" proxy1 || return 1
}

start() {
    say "pre-flight xpu-health"
    "$REPO/bin/xpu-health" 2>&1 | tail -5 || return 3
    check_artifacts || return $?
    if ss -ltnH "sport = :$PORT" | grep -q . && ! docker ps --format '{{.Names}}' | grep -qx "${NAME}_proxy"; then
        say "port $PORT is already owned by another service"
        return 2
    fi
    say "start DP=2 served=$SERVED ctx=$CTX_SIZE kv=$KV_TYPE mtp=$ENABLE_MTP draft_max=$MTP_DRAFT_MAX"
    run_replica 0 "$P0" "${NAME}_0" || return 1
    run_replica 1 "$P1" "${NAME}_1" || return 1
    wait_replicas || return 1
    start_proxy || return 1
    coherence_gate || return 1
    say "UP http://0.0.0.0:$PORT/v1 served=$SERVED DP=2 ctx=$CTX_SIZE"
}

stop() {
    docker rm -f "${NAME}_proxy" "${NAME}_0" "${NAME}_1" >/dev/null 2>&1 || true
    say "stopped $NAME"
    "$REPO/bin/xpu-health" 2>&1 | tail -5 || true
}

status() {
    docker ps --format '{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}' | grep -E "^${NAME}_(0|1|proxy)\\|" || true
    auth_args
    curl -fsS --max-time 10 "${AUTH_H[@]}" "http://127.0.0.1:$PORT/v1/models" | jq .
}

case "$ACTION" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    logs) docker logs --tail "${2:-200}" "${NAME}_${3:-0}" ;;
    gate) coherence_gate ;;
    *) echo "usage: $0 {start|stop|status|gate|logs [lines] [0|1|proxy]}"; exit 2 ;;
esac
