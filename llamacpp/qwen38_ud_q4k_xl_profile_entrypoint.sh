#!/usr/bin/env bash
# Profiling-only Qwen3.8 TP=2 entrypoint with optional embedded NEXTN MTP.
set -eo pipefail

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

MODELS_DIR="${MODELS_DIR:-/models}"
GPU_COUNT="${GPU_COUNT:-2}"
CTX_SIZE="${CTX_SIZE:-262144}"
PARALLEL="${PARALLEL:-1}"
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-256}"
PORT="${PORT:-8010}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_DRAFT_MAX="${MTP_DRAFT_MAX:-3}"
LAB_DOORS="${LAB_DOORS:-0}"
PROFILE_VERBOSE="${PROFILE_VERBOSE:-0}"
PROFILE_STATS="${PROFILE_STATS:-0}"
VTUNE_GPU_OFFLOAD="${VTUNE_GPU_OFFLOAD:-0}"
VTUNE_RESULT_DIR="${VTUNE_RESULT_DIR:-/profile/result}"
VTUNE_TARGET_GPU="${VTUNE_TARGET_GPU:-0:11:0.0,0:68:0.0}"
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
[ -s "$TARGET" ] || { echo "[profile-entrypoint] missing $TARGET"; exit 2; }
echo "$MODEL_SHA256  $TARGET" | sha256sum -c -

DRAFT_ARGS=()
if [ "$ENABLE_MTP" = "1" ]; then
    DRAFT_ARGS=(
        --spec-type draft-mtp
        --spec-draft-n-max "$MTP_DRAFT_MAX"
        --cache-type-k-draft f16
        --cache-type-v-draft f16
    )
    echo "[profile-entrypoint] embedded NEXTN MTP draft_max=$MTP_DRAFT_MAX"
fi

ALIAS_ARGS=()
[ -n "$SERVED" ] && ALIAS_ARGS=(--alias "$SERVED")
KEY_ARGS=()
[ -s "$API_KEY_FILE" ] && KEY_ARGS=(--api-key-file "$API_KEY_FILE")
LOG_ARGS=()
[ "$PROFILE_VERBOSE" = "1" ] && LOG_ARGS=(--verbose)

# LAB_DOORS=2 is a campaign-only evidence mode: production Q4K doors remain
# disabled while verbose and census logging are enabled. Keeping this separate
# prevents per-token logging from contaminating the timed arms.
if [ "$LAB_DOORS" = "2" ]; then
    PROFILE_VERBOSE=1
    PROFILE_STATS=1
    LOG_ARGS=(--verbose)
fi

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

if [ "$PROFILE_STATS" = "1" ]; then
    export GGML_META_ALLREDUCE_STATS=1
    export GGML_SYCL_FUSION_STATS=1
fi

if [ "$LAB_DOORS" = "1" ]; then
    echo "[profile-entrypoint] LAB_DOORS=1 Q4K-only experimental paths enabled"
    export GGML_SYCL_FATTN_MMA=0
    export GGML_SYCL_MMQ_Q4K_REORDER=1
    export GGML_SYCL_FUSED_MMVQ_SWIGLU_Q4K=1
    export GGML_SYCL_FUSED_MMVQ_PAIR=1
    export GGML_SYCL_FUSED_MMVQ_PAIR_GDN=1
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_ATTN=1
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_GDN=1
    export GGML_SYCL_FUSED_MMVQ_QUAD_GDN=1
else
    echo "[profile-entrypoint] LAB_DOORS=$LAB_DOORS production JIT quality guards"
    export GGML_SYCL_FATTN_MMA=0
    export GGML_SYCL_MMQ_Q4K_REORDER=0
    export GGML_SYCL_FUSED_MMVQ_SWIGLU_Q4K=0
    export GGML_SYCL_FUSED_MMVQ_PAIR=0
    export GGML_SYCL_FUSED_MMVQ_PAIR_GDN=0
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_ATTN=0
    export GGML_SYCL_FUSED_MMVQ_TRIPLE_GDN=0
    export GGML_SYCL_FUSED_MMVQ_QUAD_GDN=0
fi

echo "[profile-entrypoint] model=$MODEL_FILE gpu_count=$GPU_COUNT ctx=$CTX_SIZE batch=$BATCH/$UBATCH mtp=$ENABLE_MTP"

SERVER_CMD=(/build/llama.cpp/build-sycl/bin/llama-server \
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
    "${LOG_ARGS[@]}" \
    "${KEY_ARGS[@]}")

if [ "$VTUNE_GPU_OFFLOAD" = "1" ]; then
    [ "${GGML_SYCL_QUANT_TIMING_SAMPLE:-0}" = "0" ] || {
        echo "[profile-entrypoint] refuse VTune with SYCL queue profiling enabled"
        exit 2
    }
    [ "${GGML_SYCL_PROFILE:-0}" = "0" ] || {
        echo "[profile-entrypoint] refuse VTune with legacy SYCL profiling enabled"
        exit 2
    }
    VTUNE_BIN=/opt/intel/oneapi/vtune/2025.10/bin64/vtune
    [ -x "$VTUNE_BIN" ] || {
        echo "[profile-entrypoint] missing VTune CLI $VTUNE_BIN"
        exit 2
    }
    [ "$VTUNE_RESULT_DIR" = "/profile/result" ] || {
        echo "[profile-entrypoint] refuse unexpected VTune result path $VTUNE_RESULT_DIR"
        exit 2
    }
    [ -d /profile ] && [ -w /profile ] || {
        echo "[profile-entrypoint] /profile must be a writable bind mount"
        exit 2
    }
    [ ! -e "$VTUNE_RESULT_DIR" ] || {
        echo "[profile-entrypoint] refuse existing VTune result $VTUNE_RESULT_DIR"
        exit 2
    }
    "$VTUNE_BIN" -version > /profile/vtune_version.txt
    echo "[profile-entrypoint] VTune gpu-offload start-paused target=$VTUNE_TARGET_GPU"
    exec "$VTUNE_BIN" -collect gpu-offload \
        -knob collect-programming-api=true \
        -knob enable-tasks-stack-collection=false \
        -knob enable-stack-collection=false \
        -knob enable-characterization-insights=false \
        -knob dump-compute-task-binaries=false \
        -knob target-gpu="$VTUNE_TARGET_GPU" \
        -start-paused \
        -result-dir "$VTUNE_RESULT_DIR" -- "${SERVER_CMD[@]}"
fi

exec "${SERVER_CMD[@]}"
