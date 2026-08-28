#!/usr/bin/env bash
# Qualified single-card Qwen3.8 GPTQ INT4 control on the pinned vLLM 0.27.2
# XeCores image. BF16 KV is intentional and mandatory for this route.
# Run every start/stop sequence inside: bin/gpu-run --card "$DEVICE" ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
IMG="${IMG:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}"
HOST_CKPT="${HOST_CKPT:-$REPO/models/files/qwen3.8-27b/gptq-int4-mtp-bf16-9d189a60}"
PATCH_DIR="$REPO/vllm/patches/cookbook"
DEVICE="${DEVICE:-0}"
PORT="${PORT:-18080}"
MAXLEN="${MAXLEN:-4096}"
MAXSEQS="${MAXSEQS:-1}"
MAXBATCH="${MAXBATCH:-$MAXLEN}"
UTIL="${UTIL:-0.75}"
MTPTOK="${MTPTOK:-4}"
DRAFT_LMHEAD_INT4="${DRAFT_LMHEAD_INT4:-1}"
MIXED_SPLIT="${MIXED_SPLIT:-0}"
NAME="${NAME:-qwen38_gptq_int4_v0272_tp1}"
B70_LOGDIR="${B70_LOGDIR:-$ROOT}"

case "$DEVICE" in 0|1) ;; *) echo "DEVICE must be 0 or 1" >&2; exit 2 ;; esac
case "$MTPTOK" in ""|4) ;; *) echo "MTPTOK must be empty or 4" >&2; exit 2 ;; esac
case "$DRAFT_LMHEAD_INT4" in 0|1) ;; *) echo "DRAFT_LMHEAD_INT4 must be 0 or 1" >&2; exit 2 ;; esac
case "$MIXED_SPLIT" in 0|1) ;; *) echo "MIXED_SPLIT must be 0 or 1" >&2; exit 2 ;; esac
if [ -n "${KVDTYPE:-}" ] && [ "${KVDTYPE:-}" != auto ] && [ "${KVDTYPE:-}" != bf16 ]; then
  echo "This qualified route requires BF16 KV; KVDTYPE=${KVDTYPE}" >&2
  exit 2
fi
if [ "$DRAFT_LMHEAD_INT4" = 1 ] && [ -z "$MTPTOK" ]; then
  echo "DRAFT_LMHEAD_INT4=1 requires MTPTOK=4" >&2
  exit 2
fi
if [ "$MAXSEQS" -gt 1 ] && [ "$MIXED_SPLIT" != 1 ]; then
  echo "MAXSEQS>1 requires MIXED_SPLIT=1" >&2
  exit 2
fi

test -f "$HOST_CKPT/model.safetensors.index.json" || {
  echo "Missing checkpoint: $HOST_CKPT" >&2
  exit 1
}
for patch in patch_mtp_nightly.py patch_mtp_boundary.py \
  patch_draft_lmhead_int4.py patch_gdn_mixed_split_v5.py; do
  test -f "$PATCH_DIR/$patch" || { echo "Missing patch: $PATCH_DIR/$patch" >&2; exit 1; }
done

if [ -z "${SERVED:-}" ]; then
  if [ -z "$MTPTOK" ]; then
    SERVED="qwen3.8-27b-GPTQ-INT4-g128-target-bf16kv-vllm0272-tp1"
  else
    SERVED="qwen3.8-27b-GPTQ-INT4-g128-mtp4-bf16kv-vllm0272-tp1"
    if [ "$DRAFT_LMHEAD_INT4" = 1 ]; then
      SERVED="qwen3.8-27b-GPTQ-INT4-g128-mtp4-draft-lmhead-int4-bf16kv-vllm0272-tp1"
    fi
  fi
fi
case "$SERVED" in *bf16kv*vllm0272*tp1*) ;; *) echo "SERVED must encode bf16kv, vllm0272, and tp1" >&2; exit 2 ;; esac
if [ "$DRAFT_LMHEAD_INT4" = 1 ]; then
  case "$SERVED" in *draft-lmhead-int4*) ;; *) echo "SERVED must encode draft-lmhead-int4" >&2; exit 2 ;; esac
fi

stop_server() {
  local log="$B70_LOGDIR/b70_${NAME}.log"
  if docker inspect "$NAME" >/dev/null 2>&1; then
    docker logs "$NAME" >"$log" 2>&1 || true
    docker stop -t "${STOP_GRACE:-30}" "$NAME" >/dev/null 2>&1 || true
    docker rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  echo "stopped $NAME; log=$log"
}

wait_healthy() {
  local start now signature previous="" last_progress
  start="$(date +%s)"
  last_progress="$start"
  while :; do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      curl -s "http://localhost:$PORT/v1/models"
      return 0
    fi
    if docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited; then
      docker logs "$NAME" 2>&1 | tail -80
      return 1
    fi
    now="$(date +%s)"
    signature="$(docker logs "$NAME" 2>&1 | sha256sum | awk '{print $1}')"
    if [ "$signature" != "$previous" ]; then previous="$signature"; last_progress="$now"; fi
    if [ $((now - last_progress)) -ge "${HEALTH_STALL:-300}" ]; then
      echo "Server initialization stalled" >&2
      docker logs "$NAME" 2>&1 | tail -80
      return 1
    fi
    if [ $((now - start)) -ge "${HEALTH_TIMEOUT:-1200}" ]; then
      echo "Server initialization timed out" >&2
      docker logs "$NAME" 2>&1 | tail -80
      return 1
    fi
    sleep 5
  done
}

start_server() {
  local patch_cmd spec_text=""
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  patch_cmd='python /patches/patch_mtp_nightly.py; python /patches/patch_mtp_boundary.py;'
  if [ "$DRAFT_LMHEAD_INT4" = 1 ]; then
    patch_cmd+=' python /patches/patch_draft_lmhead_int4.py;'
  fi
  if [ "$MIXED_SPLIT" = 1 ]; then
    patch_cmd+=' python /patches/patch_gdn_mixed_split_v5.py;'
  fi
  if [ -n "$MTPTOK" ]; then
    spec_text=" --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$MTPTOK}'"
  fi
  docker run -d --name "$NAME" --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --shm-size 32g \
    --oom-score-adj 500 -p "$PORT:8000" \
    -v "$HOST_CKPT:/model:ro" -v "$PATCH_DIR:/patches:ro" \
    -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" \
    -v "$ROOT/tmp_ssd:/tmp_ssd" \
    -e ZE_AFFINITY_MASK="$DEVICE" -e VLLM_TARGET_DEVICE=xpu \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    -e B70_MTP_BF16_DRAFT=1 -e B70_DRAFT_LMHEAD_INT4="$DRAFT_LMHEAD_INT4" \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True -e HF_HOME=/hf_cache \
    -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
    -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd \
    -e VLLM_LOGGING_LEVEL=INFO --entrypoint bash "$IMG" -lc \
    "$patch_cmd exec vllm serve /model --host 0.0.0.0 --port 8000 --dtype float16 --max-model-len '$MAXLEN' --gpu-memory-utilization '$UTIL' --max-num-seqs '$MAXSEQS' --max-num-batched-tokens '$MAXBATCH' --served-model-name '$SERVED' --trust-remote-code --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --quantization gptq --language-model-only --generation-config vllm --reasoning-parser qwen3${spec_text}" >/dev/null
  wait_healthy
  echo "Serving $SERVED on port $PORT. Stop inside the same GPU lease."
}

case "${1:-start}" in
  start) start_server ;;
  stop) stop_server ;;
  logs) exec docker logs -f "$NAME" ;;
  *) echo "usage: $0 [start|stop|logs]" >&2; exit 2 ;;
esac
