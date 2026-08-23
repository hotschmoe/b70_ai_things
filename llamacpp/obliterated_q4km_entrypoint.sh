#!/usr/bin/env bash
# Qwen3.8-27B-OBLITERATED V3 Q4_K_M, one B70 replica.
# The outer DP=2 wrapper pins one process to each card and fronts them with nginx.
set -eo pipefail

# oneAPI's setvars references optional unset variables, so enable nounset after it.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

MODELS_DIR="${MODELS_DIR:-/models}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf}"
MODEL_SHA256="${MODEL_SHA256:-c5e4fe705883e244a468c9e445c8d6ba37fd310b0113e25d2b8a7f2d6f1243e8}"
SERVED="${SERVED:-hotschmoe-dd}"
PORT="${PORT:-8010}"
CTX_SIZE="${CTX_SIZE:-245760}"
PARALLEL="${PARALLEL:-1}"
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-256}"
KV_TYPE="${KV_TYPE:-q8_0}"
LAB_DOORS="${LAB_DOORS:-1}"
ENABLE_MTP="${ENABLE_MTP:-1}"
MTP_SIDECAR="${MTP_SIDECAR:-0}"
MTP_FILE="${MTP_FILE:-mtp-Qwen3.8-27B-Q4_0.gguf}"
MTP_SHA256="${MTP_SHA256:-051a1764cff8c4f3ee6ae8b00593a0364c7539c67fa50ffc58f3f96509fca38e}"
MTP_DRAFT_MAX="${MTP_DRAFT_MAX:-3}"
API_KEY_FILE="${API_KEY_FILE:-/run/secrets/dd_api_key}"

TARGET="$MODELS_DIR/$MODEL_FILE"
[ -s "$TARGET" ] || { echo "[entrypoint] missing $TARGET"; exit 2; }
echo "$MODEL_SHA256  $TARGET" | sha256sum -c -

KEY_ARGS=()
if [ -s "$API_KEY_FILE" ]; then
    KEY_ARGS=(--api-key-file "$API_KEY_FILE")
fi

DRAFT_ARGS=()
if [ "$ENABLE_MTP" = "1" ]; then
    DRAFT_ARGS=(
        --spec-type draft-mtp
        --spec-draft-n-max "$MTP_DRAFT_MAX"
        --cache-type-k-draft q8_0
        --cache-type-v-draft q8_0
    )
    if [ "$MTP_SIDECAR" = "1" ]; then
        DRAFT="$MODELS_DIR/$MTP_FILE"
        [ -s "$DRAFT" ] || { echo "[entrypoint] missing MTP sidecar $DRAFT"; exit 2; }
        echo "$MTP_SHA256  $DRAFT" | sha256sum -c -
        DRAFT_ARGS+=(--model-draft "$DRAFT" --gpu-layers-draft all --spec-draft-device SYCL0)
    else
        echo "[entrypoint] using the V3 GGUF embedded blk.64.nextn MTP head"
    fi
fi

# Baseline B70 SYCL doors from the measured 0xSero/lab Q4_K_M path.
export UR_L0_USE_IMMEDIATE_COMMANDLISTS=1
export UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1
export GGML_SYCL_COMM_SINGLE_KERNEL=1
export GGML_META_FUSE_ALLREDUCE_ADD=1
export GGML_META_FUSE_ALLREDUCE_ADD_RMS_MUL=1
export GGML_SYCL_COMM_FUSED_Q8=1
export GGML_SYCL_FUSED_SWIGLU_Q8=1
export GGML_SYCL_FUSED_ATTN_Q8=1
export GGML_SYCL_FUSED_GDN_Q8=1
export GGML_SYCL_FUSED_GDN_BETA_SIGMOID=1
export GGML_SYCL_FUSED_CONCAT_STATE=1
export GGML_SYCL_FUSED_GDN_STATE_IO=1
export GGML_SYCL_FUSED_CONV_STATE_IO=1
export GGML_SYCL_COMM_DIRECT_Q8=2
export GGML_SYCL_FUSED_ROPE_SET_ROWS=1
export GGML_SYCL_COMM_REDUCE_VEC4=1
export GGML_SYCL_FUSED_QK_NORM_ROPE=1
export GGML_SYCL_FUSED_CONV_SILU_L2=1
export GGML_SYCL_FUSE_EXT=31

if [ "$LAB_DOORS" = "1" ]; then
    echo "[entrypoint] LAB_DOORS=1 Q4K reorder and fused MMVQ"
    export GGML_SYCL_FATTN_MMA=0
    export GGML_SYCL_MMQ_Q4K_REORDER=1
    export GGML_SYCL_FUSED_MMVQ_SWIGLU_Q4K=1
    export GGML_SYCL_FUSED_MMVQ_PAIR=1
    export GGML_SYCL_FUSED_MMVQ_PAIR_GDN=1
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_ATTN=1
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_GDN=1
    export GGML_SYCL_FUSED_MMVQ_QUAD_GDN=1
else
    echo "[entrypoint] LAB_DOORS=0 published JIT quality guards"
    export GGML_SYCL_FATTN_MMA=0
    export GGML_SYCL_MMQ_Q4K_REORDER=0
    export GGML_SYCL_FUSED_MMVQ_SWIGLU_Q4K=0
    export GGML_SYCL_FUSED_MMVQ_PAIR=0
    export GGML_SYCL_FUSED_MMVQ_PAIR_GDN=0
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_ATTN=0
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_GDN=0
    export GGML_SYCL_FUSED_MMVQ_QUAD_GDN=0
fi

echo "[entrypoint] served=$SERVED ctx=$CTX_SIZE kv=$KV_TYPE batch=$BATCH/$UBATCH mtp=$ENABLE_MTP draft_max=$MTP_DRAFT_MAX"

exec /build/llama.cpp/build-sycl/bin/llama-server \
    --model "$TARGET" \
    "${DRAFT_ARGS[@]}" \
    --alias "$SERVED" \
    --host 0.0.0.0 --port "$PORT" \
    --device SYCL0 \
    --gpu-layers 99 \
    --flash-attn on \
    --batch-size "$BATCH" \
    --ubatch-size "$UBATCH" \
    --cache-type-k "$KV_TYPE" \
    --cache-type-v "$KV_TYPE" \
    --cache-ram 0 \
    --ctx-checkpoints 0 \
    --fit off \
    --parallel "$PARALLEL" \
    --cont-batching \
    --jinja \
    --reasoning off \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --temp 0 \
    --top-k 0 \
    --top-p 1.0 \
    --min-p 0.0 \
    --repeat-penalty 1.15 \
    --threads "${THREADS:-8}" \
    --poll 50 \
    --ctx-size "$CTX_SIZE" \
    --metrics \
    "${KEY_ARGS[@]}"
