#!/usr/bin/env bash
# Launch a single-card cookbook-style vLLM serve with MTP patches applied.
#
# Usage:
#   bash vllm/cookbook_campaign/launch.sh TRACK MODE CACHE [PORT] [CARD]
#
# TRACK: dense27-gptq | moe35-gptq | dense38-gptq | dense27-autoround | moe35-autoround
# MODE:  no-spec | mtp1 | mtp2 | mtp4
# CACHE: on | off
# PORT:  default 8000
# CARD:  0|1 default 0
#
# IMAGE: default = public cookbook digest for *-gptq tracks.
#        autoround tracks default to vllm-xpu-env:v0240 (our validated INT4 path).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRACK=${1:?usage: $0 TRACK MODE CACHE [PORT] [CARD]}
MODE=${2:?}
CACHE=${3:?}
PORT=${4:-8000}
CARD=${5:-0}
NAME="${NAME:-b70_cb_${TRACK}_${MODE}_${CACHE}}"
PATCH_DIR="$REPO/vllm/patches/cookbook"
MODELS_FILES="${MODELS_FILES:-$REPO/models/files}"
# 3.6 cookbook digest. Do NOT use this for 3.8 (f01e24f6) or Nemotron (1da0a954).
PUBLIC_IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
# 3.8 cookbook digest (vLLM 0.27.2rc1). Do NOT mix with 3.6 2c427ef.
PUBLIC_IMAGE_38='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MAXLEN="${MAXLEN:-131072}"
MAXBATCH="${MAXBATCH:-8192}"
# Dense MTP4 capture OOMs at max-seqs 64 on this box (graph buffers + 17 GiB weights).
# Default 8 unless caller overrides; no-spec can still raise MAXSEQS.
MAXSEQS="${MAXSEQS:-8}"
LANGUAGE_ONLY="${LANGUAGE_ONLY:-1}"

QUANT_FLAG=""
DTYPE="float16"
KVDTYPE="fp8"
IMAGE="${IMAGE:-}"
TOOL_PARSER="${TOOL_PARSER:-}"

case "$TRACK" in
  dense27-gptq)
    HOST_CKPT="$MODELS_FILES/community/qwen36-27b-gptq-mtp-preserved"
    SERVED="Qwen3.6-27B-MTP-Preserved-GPTQ-Int4"
    QUANT_FLAG="gptq"
    IMAGE="${IMAGE:-$PUBLIC_IMAGE}"
    TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
    ;;
  moe35-gptq)
    HOST_CKPT="$MODELS_FILES/community/qwen36-35b-gptq-mtp-preserved"
    SERVED="Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4"
    QUANT_FLAG="gptq"
    IMAGE="${IMAGE:-$PUBLIC_IMAGE}"
    TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
    ;;
  dense38-gptq)
    HOST_CKPT="$MODELS_FILES/community/qwen38-27b-gptq-mtp-preserved"
    SERVED="${SERVED:-qwen3.8-27b-GPTQ-Int4-mtp4}"
    QUANT_FLAG="gptq"
    IMAGE="${IMAGE:-$PUBLIC_IMAGE_38}"
    TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
    ;;
  dense27-autoround)
    HOST_CKPT="$MODELS_FILES/qwen3.6-27b/int4-autoround"
    SERVED="qwen36-27b-int4-autoround-mtp"
    # base :v0260 has broken XPU device discovery on this host; int8g bake is the working 0.26 image
    IMAGE="${IMAGE:-vllm-xpu-env:int8g-v0260}"
    QUANT_FLAG=""
    DTYPE="auto"
    KVDTYPE="fp8_e5m2"
    TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
    ;;
  moe35-autoround)
    HOST_CKPT="$MODELS_FILES/qwen3.6-35b-a3b/int4-autoround"
    SERVED="qwen36-35b-a3b-int4-autoround-mtp"
    IMAGE="${IMAGE:-vllm-xpu-env:int8g-v0260}"
    QUANT_FLAG=""
    DTYPE="auto"
    KVDTYPE="fp8_e5m2"
    TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
    ;;
  *)
    echo "TRACK must be dense27-gptq|moe35-gptq|dense38-gptq|dense27-autoround|moe35-autoround" >&2
    exit 2
    ;;
esac

HOST_CKPT="${CKPT_HOST:-$HOST_CKPT}"
if [ ! -d "$HOST_CKPT" ] || [ ! -f "$HOST_CKPT/config.json" ]; then
  echo "Model dir missing/incomplete: $HOST_CKPT" >&2
  exit 1
fi
sz=$(du -sm "$HOST_CKPT" | awk '{print $1}')
if [ "$sz" -lt 1000 ]; then
  echo "Model dir looks incomplete (${sz} MB): $HOST_CKPT" >&2
  exit 1
fi

case "$MODE" in
  no-spec)
    SPEC_JSON='{}'
    SPEC_USE=0
    GPU_UTIL="${UTIL:-0.90}"
    ;;
  mtp1|mtp2|mtp4)
    N=${MODE#mtp}
    SPEC_JSON="{\"method\":\"mtp\",\"num_speculative_tokens\":$N}"
    SPEC_USE=1
    if [ "$N" = "4" ] && [[ "$TRACK" == dense* ]]; then
      GPU_UTIL="${UTIL:-0.88}"
    else
      GPU_UTIL="${UTIL:-0.90}"
    fi
    if [[ "$TRACK" == moe* ]]; then GPU_UTIL="${UTIL:-0.85}"; fi
    ;;
  *)
    echo "MODE must be no-spec|mtp1|mtp2|mtp4" >&2
    exit 2
    ;;
esac

case "$CACHE" in
  on) CACHE_ARG="--enable-prefix-caching" ;;
  off) CACHE_ARG="--no-enable-prefix-caching" ;;
  *) echo "CACHE must be on|off" >&2; exit 2 ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true

RENDER_GID=$(stat -c '%g' /dev/dri/render* 2>/dev/null | sort -u | head -1 || true)
# Stable path: must outlive this script (container bind-mounts it).
SPEC_HOST="/tmp/b70_spec_${NAME}.json"
echo "$SPEC_JSON" > "$SPEC_HOST"

# Build the inner serve command as a single string.
# NOTE: vLLM 0.26 on XPU wants --speculative-config as a JSON *string*, not a
# path (path form is rejected by argparse). We cat /spec.json inside the container.
# PIECEWISE cudagraph avoids FULL-mode SYCL Graph + flash_attn scratch errors
# seen with default FULL_AND_PIECEWISE on this stack.
CGMODE="${CGMODE:-PIECEWISE}"
COMPILE_JSON="{\"cudagraph_mode\":\"${CGMODE}\"}"
SERVE_CMD="python /patches/apply_mtp_patches.py && "
SERVE_CMD+='SPEC_JSON=$(cat /spec.json) && '
SERVE_CMD+="vllm serve /model"
SERVE_CMD+=" --host 0.0.0.0 --port 8000"
SERVE_CMD+=" --dtype $DTYPE"
SERVE_CMD+=" --max-model-len $MAXLEN"
SERVE_CMD+=" --gpu-memory-utilization $GPU_UTIL"
SERVE_CMD+=" --max-num-seqs $MAXSEQS"
SERVE_CMD+=" --max-num-batched-tokens $MAXBATCH"
if [ -n "$TOOL_PARSER" ]; then
  SERVE_CMD+=" --enable-auto-tool-choice --tool-call-parser $TOOL_PARSER"
fi
SERVE_CMD+=" --served-model-name $SERVED"
SERVE_CMD+=" --trust-remote-code"
SERVE_CMD+=" --compilation-config '$COMPILE_JSON'"
SERVE_CMD+=" $CACHE_ARG"
if [ -n "$QUANT_FLAG" ]; then SERVE_CMD+=" --quantization $QUANT_FLAG"; fi
if [ -n "$KVDTYPE" ] && [ "$KVDTYPE" != "auto" ] && [ "$KVDTYPE" != "0" ]; then
  SERVE_CMD+=" --kv-cache-dtype $KVDTYPE"
fi
if [ "$LANGUAGE_ONLY" = 1 ]; then SERVE_CMD+=" --language-model-only"; fi
if [ "$SPEC_USE" = 1 ]; then
  SERVE_CMD+=' --speculative-config "$SPEC_JSON"'
fi
# Allow full eager fallback for bring-up
if [ "${EAGER:-0}" = 1 ]; then
  SERVE_CMD+=" --enforce-eager"
fi

echo "=== cookbook launch ==="
echo "  name=$NAME image=$IMAGE"
echo "  track=$TRACK mode=$MODE cache=$CACHE util=$GPU_UTIL maxlen=$MAXLEN"
echo "  ckpt=$HOST_CKPT (${sz} MB) card=$CARD port=$PORT"
echo "  cmd=$SERVE_CMD"

# Device/env match rdy_to_serve/_common/lib.sh (b70_serve). The cookbook's
# ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE + ONEAPI_DEVICE_SELECTOR combo can yield
# "XPU device count is zero" on this host -- pin with ZE_AFFINITY_MASK only.
SHM="${SHM:-32g}"
DOCKER_ARGS=(
  run -d --name "$NAME"
  --device /dev/dri
  -v /dev/dri/by-path:/dev/dri/by-path
  --ipc=host
  --shm-size "$SHM"
  -p "${PORT}:8000"
  -v "$HOST_CKPT:/model:ro"
  -v "$PATCH_DIR:/patches:ro"
  -v "$SPEC_HOST:/spec.json:ro"
  -e VLLM_TARGET_DEVICE=xpu
  -e B70_MTP_BF16_DRAFT=1
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1
  -e PYTORCH_ALLOC_CONF=expandable_segments:True
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e "ZE_AFFINITY_MASK=${CARD}"
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0
  -e VLLM_LOGGING_LEVEL=INFO
  --entrypoint bash
)
if [ -n "${RENDER_GID:-}" ]; then
  DOCKER_ARGS+=( --group-add "$RENDER_GID" )
fi

docker "${DOCKER_ARGS[@]}" "$IMAGE" -lc "set -e; $SERVE_CMD"

echo "Container: $NAME"
echo "Logs:      docker logs -f $NAME"
echo "Health:    curl -sf http://127.0.0.1:$PORT/health"
echo "Stop:      docker rm -f $NAME"
