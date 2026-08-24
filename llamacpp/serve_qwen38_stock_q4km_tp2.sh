#!/usr/bin/env bash
# Qwen3.8-27B GGUF, tensor parallel across both B70 cards. The defaults select
# the stock ggml-org Q4_K_M artifact used by the 0.970/0.927 HumanEval+ run;
# pinned shelf wrappers may override the artifact fields for another GGUF.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-start}"
NAME="${NAME:-qwen38_stock_q4km_tp2}"
PORT="${PORT:-18080}"
SERVED="${SERVED:-hotschmoe-dd}"
HOST_MODELS="${HOST_MODELS:-$REPO/models/files/qwen3.8-27b/q4km-ggml-org}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-Q4_K_M.gguf}"
MODEL_SIZE="${MODEL_SIZE:-18973870432}"
MODEL_SHA256="${MODEL_SHA256:-31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34}"
MODEL_LABEL="${MODEL_LABEL:-stock Q4_K_M}"
CTX_SIZE="${CTX_SIZE:-262144}"
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-256}"
LAB_DOORS="${LAB_DOORS:-0}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_SPEC_TYPE="${MTP_SPEC_TYPE:-mtp}"
MTP_DRAFT_MAX_FLAG="${MTP_DRAFT_MAX_FLAG:---draft-max}"
GGML_SYCL_QUANT_CENSUS="${GGML_SYCL_QUANT_CENSUS:-0}"
GGML_SYCL_QUANT_TIMING_SAMPLE="${GGML_SYCL_QUANT_TIMING_SAMPLE:-0}"
GGML_SYCL_QUANT_TIMING_SKIP="${GGML_SYCL_QUANT_TIMING_SKIP:-4}"
GGML_SYCL_QUANT_TIMING_MAX="${GGML_SYCL_QUANT_TIMING_MAX:-65536}"
UR_L0_USE_DRIVER_INORDER_LISTS="${UR_L0_USE_DRIVER_INORDER_LISTS:-}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
OVERLAY="${OVERLAY:-$REPO/llamacpp/qwen38_b70_entrypoint_overlay.sh}"
IMG="${IMG:-qwen38-b70:latest}"
IMG_ID="${IMG_ID:-sha256:8c6dc0462011e7d4596882009fc7fb1128fbe656cb17a998999cd1e720a2b4de}"
API_KEY_FILE="${API_KEY_FILE:-/mnt/vm_8tb/b70/secrets/dd_api_key}"

say() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

auth_args() {
    AUTH_H=()
    if [ -s "$API_KEY_FILE" ]; then
        AUTH_H=(-H "Authorization: Bearer $(<"$API_KEY_FILE")")
    fi
}

check_artifacts() {
    [ -s "$HOST_MODELS/$MODEL_FILE" ] || { say "missing $HOST_MODELS/$MODEL_FILE"; return 2; }
    [ "$(stat -c %s "$HOST_MODELS/$MODEL_FILE")" = "$MODEL_SIZE" ] || {
        say "wrong model size"; return 2;
    }
    echo "$MODEL_SHA256  $HOST_MODELS/$MODEL_FILE" | sha256sum -c - || return 2
    [ -x "$OVERLAY" ] || { say "entrypoint is not executable: $OVERLAY"; return 2; }
    docker image inspect "$IMG" >/dev/null 2>&1 || { say "missing image $IMG"; return 2; }
    [ "$(docker image inspect "$IMG" --format '{{.Id}}')" = "$IMG_ID" ] || {
        say "image id mismatch"; return 2;
    }
}

wait_healthy() {
    local deadline=$((SECONDS + 1200))
    while [ "$SECONDS" -lt "$deadline" ]; do
        docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
            say "container exited"; docker logs --tail 120 "$NAME" 2>&1; return 1;
        }
        if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            say "server healthy"
            return 0
        fi
        sleep 5
    done
    say "health wait timed out"
    docker logs --tail 120 "$NAME" 2>&1 || true
    return 1
}

coherence_gate() {
    local payload body content
    auth_args
    payload="{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one short sentence.\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
    body="$(curl -fsS --max-time 300 "${AUTH_H[@]}" -H 'content-type: application/json' \
        -d "$payload" "http://127.0.0.1:$PORT/v1/chat/completions")" || return 1
    content="$(printf '%s' "$body" | jq -r '.choices[0].message.content // empty')"
    printf '%s' "$content" | grep -qi paris || {
        say "coherence FAIL: ${content:0:160}"; return 1;
    }
    say "coherence PASS: ${content:0:120}"
}

start() {
    say "pre-flight xpu-health"
    # IMG selects the llama.cpp serve image. Do not leak it into xpu-health,
    # whose independent IMG variable selects a PyTorch probe image.
    env -u IMG "$REPO/bin/xpu-health" 2>&1 | tail -5 || return 3
    check_artifacts || return $?
    if ss -ltnH "sport = :$PORT" | grep -q .; then
        say "port $PORT is already in use"
        return 2
    fi
    local key_mount=()
    local runtime_env=()
    [ -s "$API_KEY_FILE" ] && key_mount=(-v "$API_KEY_FILE:/run/secrets/dd_api_key:ro")
    [ -n "$UR_L0_USE_DRIVER_INORDER_LISTS" ] && \
        runtime_env=(-e "UR_L0_USE_DRIVER_INORDER_LISTS=$UR_L0_USE_DRIVER_INORDER_LISTS")
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    say "start $MODEL_LABEL TP=2 ctx=$CTX_SIZE mtp=$ENABLE_MTP lab_doors=$LAB_DOORS"
    docker run -d --name "$NAME" --restart "$RESTART_POLICY" \
        --device /dev/dri --ipc=host --shm-size 8g \
        -v /dev/dri/by-path:/dev/dri/by-path:ro \
        -v "$HOST_MODELS:/models:ro" \
        -v "$OVERLAY:/entrypoint.sh:ro" \
        "${key_mount[@]}" \
        "${runtime_env[@]}" \
        -e CCL_TOPO_P2P_ACCESS=0 \
        -e MODELS_DIR=/models -e MODEL_FILE="$MODEL_FILE" -e MODEL_SHA256="$MODEL_SHA256" \
        -e SERVED="$SERVED" -e API_KEY_FILE=/run/secrets/dd_api_key \
        -e GPU_COUNT=2 -e CTX_SIZE_OVERRIDE="$CTX_SIZE" -e PARALLEL=1 \
        -e BATCH="$BATCH" -e UBATCH="$UBATCH" \
        -e ENABLE_MTP="$ENABLE_MTP" -e MTP_SPEC_TYPE="$MTP_SPEC_TYPE" \
        -e MTP_DRAFT_MAX_FLAG="$MTP_DRAFT_MAX_FLAG" \
        -e ENABLE_VISION=0 -e LAB_DOORS="$LAB_DOORS" \
        -e GGML_SYCL_QUANT_CENSUS="$GGML_SYCL_QUANT_CENSUS" \
        -e GGML_SYCL_QUANT_TIMING_SAMPLE="$GGML_SYCL_QUANT_TIMING_SAMPLE" \
        -e GGML_SYCL_QUANT_TIMING_SKIP="$GGML_SYCL_QUANT_TIMING_SKIP" \
        -e GGML_SYCL_QUANT_TIMING_MAX="$GGML_SYCL_QUANT_TIMING_MAX" \
        -e GGML_SYCL_FATTN_MMA=0 -e THREADS=8 \
        -p "$PORT:8010" \
        --entrypoint bash "$IMG" /entrypoint.sh >/dev/null || return 1
    wait_healthy || return 1
    coherence_gate || return 1
    auth_args
    curl -fsS --max-time 10 "${AUTH_H[@]}" "http://127.0.0.1:$PORT/v1/models" | jq .
    say "UP http://0.0.0.0:$PORT/v1 served=$SERVED"
}

stop() {
    docker stop --time 30 "$NAME" >/dev/null 2>&1 || true
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    say "stopped $NAME"
}

status() {
    docker ps --format '{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}' | grep -E "^${NAME}\\|" || true
    auth_args
    curl -fsS --max-time 10 "${AUTH_H[@]}" "http://127.0.0.1:$PORT/v1/models" | jq .
}

case "$ACTION" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    gate) coherence_gate ;;
    logs) docker logs --tail "${2:-200}" "$NAME" ;;
    *) echo "usage: $0 {start|stop|status|gate|logs [lines]}"; exit 2 ;;
esac
