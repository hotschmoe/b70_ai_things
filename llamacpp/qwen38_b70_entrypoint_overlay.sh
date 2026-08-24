#!/usr/bin/env bash
# Overlay for 0xSero qwen38-b70:latest. Same SHA check + llama-server, but
# runtime doors can match the lab Q4_K_M record (LAB_DOORS=1) instead of
# the published JIT quality-guard defaults (all Q4K fusions off).
#
# LAB_DOORS=1 is the 2026-08-15 AOT record config:
#   GGML_SYCL_MMQ_Q4K_REORDER=1 + GGML_SYCL_FUSED_MMVQ_SWIGLU_Q4K=1
# 0xSero says those corrupt stock ggml-org under JIT. Paris-gate first.
set -e
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1

MODELS_DIR="${MODELS_DIR:-/models}"
GPU_COUNT="${GPU_COUNT:-2}"
CTX_SIZE="${CTX_SIZE:-262144}"
PARALLEL="${PARALLEL:-1}"
BATCH="${BATCH:-8192}"
UBATCH="${UBATCH:-8192}"
PORT="${PORT:-8010}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_SPEC_TYPE="${MTP_SPEC_TYPE:-mtp}"
MTP_DRAFT_MAX_FLAG="${MTP_DRAFT_MAX_FLAG:---draft-max}"
ENABLE_VISION="${ENABLE_VISION:-0}"
LAB_DOORS="${LAB_DOORS:-0}"
SERVED="${SERVED:-}"
API_KEY_FILE="${API_KEY_FILE:-/run/secrets/dd_api_key}"

if [ "$GPU_COUNT" = "1" ]; then
    DEVICE_ARGS=(--device SYCL0)
    CTX_SIZE="${CTX_SIZE_OVERRIDE:-131072}"
else
    DEVICE_ARGS=(--device SYCL0,SYCL1 --split-mode tensor --tensor-split 1,1)
    CTX_SIZE="${CTX_SIZE_OVERRIDE:-$CTX_SIZE}"
fi

TARGET="$MODELS_DIR/$MODEL_FILE"
echo "$MODEL_SHA256  $TARGET" | sha256sum -c - || { echo "[entrypoint] SHA-256 mismatch!"; exit 1; }

DRAFT_ARGS=()
if [ "$ENABLE_MTP" = "1" ]; then
    DRAFT_ARGS=(
        --spec-type "$MTP_SPEC_TYPE"
        --model-draft "$MODELS_DIR/$MTP_FILE"
        "$MTP_DRAFT_MAX_FLAG" 8
    )
fi

ALIAS_ARGS=()
[ -n "$SERVED" ] && ALIAS_ARGS=(--alias "$SERVED")
KEY_ARGS=()
[ -s "$API_KEY_FILE" ] && KEY_ARGS=(--api-key-file "$API_KEY_FILE")

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
    echo "[entrypoint] LAB_DOORS=1 (Q4K reorder + SwiGLU + MMVQ pair/triple/quad ON)"
    export GGML_SYCL_FATTN_MMA="${GGML_SYCL_FATTN_MMA:-0}"
    export GGML_SYCL_MMQ_Q4K_REORDER=1
    export GGML_SYCL_FUSED_MMVQ_SWIGLU_Q4K=1
    export GGML_SYCL_FUSED_MMVQ_PAIR=1
    export GGML_SYCL_FUSED_MMVQ_PAIR_GDN=1
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_ATTN=1
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_GDN=1
    export GGML_SYCL_FUSED_MMVQ_QUAD_GDN=1
else
    echo "[entrypoint] LAB_DOORS=0 (0xSero JIT quality guards)"
    export GGML_SYCL_FATTN_MMA=0
    export GGML_SYCL_MMQ_Q4K_REORDER=0
    export GGML_SYCL_FUSED_MMVQ_SWIGLU_Q4K=0
    export GGML_SYCL_FUSED_MMVQ_PAIR=0
    export GGML_SYCL_FUSED_MMVQ_PAIR_GDN=0
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_ATTN=0
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_GDN=0
    export GGML_SYCL_FUSED_MMVQ_QUAD_GDN=0
fi

echo "[entrypoint] GPU_COUNT=$GPU_COUNT ctx=$CTX_SIZE batch=$BATCH ubatch=$UBATCH devices=${DEVICE_ARGS[*]}"

exec /build/llama.cpp/build-sycl/bin/llama-server \
    --model "$TARGET" \
    "${DRAFT_ARGS[@]}" \
    "${ALIAS_ARGS[@]}" \
    --host 0.0.0.0 --port "$PORT" \
    "${DEVICE_ARGS[@]}" \
    --gpu-layers 99 \
    --flash-attn on \
    --batch-size "$BATCH" \
    --ubatch-size "$UBATCH" \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --cache-ram 0 \
    --ctx-checkpoints 0 \
    --fit off \
    --reasoning off \
    --threads "${THREADS:-8}" \
    --poll 50 \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --metrics \
    "${KEY_ARGS[@]}"
