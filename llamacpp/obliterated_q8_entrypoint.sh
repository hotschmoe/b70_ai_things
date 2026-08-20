#!/usr/bin/env bash
# Overlay for qwen38-b70:latest serving OBLITERATUS Qwen3.8-27B Q8_0.
# Q8 fused comm/swiglu/attn/gdn doors ON. Q4K reorder family OFF (wrong quant).
# MODEL_SHA256 optional: empty skips the pin (first fetch). FATTN_MMA stays 0.
set -e
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1

MODELS_DIR="${MODELS_DIR:-/models}"
GPU_COUNT="${GPU_COUNT:-2}"
CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-1}"
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-256}"
PORT="${PORT:-8010}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-OBLITERATED-Q8_0.gguf}"
Q8_DOORS="${Q8_DOORS:-1}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.15}"
TEMP="${TEMP:-0}"
# Pliny @elder_plinius 2026-08-20: temp 0, repetition_penalty 1.15,
# max_new_tokens >=2048, NO system prompt, thinking off.
TOP_K="${TOP_K:-0}"
TOP_P="${TOP_P:-1.0}"
MIN_P="${MIN_P:-0.0}"
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"

if [ "$GPU_COUNT" = "1" ]; then
    DEVICE_ARGS=(--device SYCL0)
    CTX_SIZE="${CTX_SIZE_OVERRIDE:-$CTX_SIZE}"
else
    DEVICE_ARGS=(--device SYCL0,SYCL1 --split-mode tensor --tensor-split 1,1)
    CTX_SIZE="${CTX_SIZE_OVERRIDE:-$CTX_SIZE}"
fi

TARGET="$MODELS_DIR/$MODEL_FILE"
[ -s "$TARGET" ] || { echo "[entrypoint] missing $TARGET"; ls -la "$MODELS_DIR"; exit 2; }
if [ -n "${MODEL_SHA256:-}" ]; then
    echo "$MODEL_SHA256  $TARGET" | sha256sum -c - || { echo "[entrypoint] SHA-256 mismatch"; exit 1; }
fi

# Lab Q8 fused doors (0xSero entrypoint). Decode M=1 GEMV + TP2 comm.
export UR_L0_USE_IMMEDIATE_COMMANDLISTS=1
export UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1
export GGML_SYCL_COMM_SINGLE_KERNEL=1
export GGML_META_FUSE_ALLREDUCE_ADD=1
export GGML_META_FUSE_ALLREDUCE_ADD_RMS_MUL=1
export GGML_SYCL_FUSED_GDN_BETA_SIGMOID=1
export GGML_SYCL_FUSED_CONCAT_STATE=1
export GGML_SYCL_FUSED_GDN_STATE_IO=1
export GGML_SYCL_FUSED_CONV_STATE_IO=1
export GGML_SYCL_FUSED_ROPE_SET_ROWS=1
export GGML_SYCL_COMM_REDUCE_VEC4=1
export GGML_SYCL_FUSED_QK_NORM_ROPE=1
export GGML_SYCL_FUSED_CONV_SILU_L2=1
export GGML_SYCL_FUSE_EXT=31
export GGML_SYCL_FATTN_MMA="${GGML_SYCL_FATTN_MMA:-0}"
# Q4K reorder/SwiGLU stay off (wrong quant). MMVQ pair/triple/quad are
# Q8_1-activation fusions that also help Q8_0 weight GEMV -- ON.
export GGML_SYCL_MMQ_Q4K_REORDER=0
export GGML_SYCL_FUSED_MMVQ_SWIGLU_Q4K=0
export GGML_SYCL_FUSED_MMVQ_PAIR=1
export GGML_SYCL_FUSED_MMVQ_PAIR_GDN=1
export GGML_SYCL_FUSED_MMVQ_TRIPLE_ATTN=1
export GGML_SYCL_FUSED_MMVQ_TRIPLE_GDN=1
export GGML_SYCL_FUSED_MMVQ_QUAD_GDN=1
export GGML_SYCL_QDEDUP_STATS="${GGML_SYCL_QDEDUP_STATS:-1}"
# Lab Q8 geometry. QUAD_SG24 is a no-op on 0xSero JIT (no symbol).
# SG32 exists in-image; default 0 (BMG uses SG16). Never FATTN_MMA=1 on JIT.
export GGML_SYCL_MMVQ_SG32="${GGML_SYCL_MMVQ_SG32:-0}"
export GGML_SYCL_MMVQ_Q8_QUAD_SG16="${GGML_SYCL_MMVQ_Q8_QUAD_SG16:-1}"
export GGML_SYCL_MMVQ_Q8_QUAD_SG24="${GGML_SYCL_MMVQ_Q8_QUAD_SG24:-1}"

if [ "$Q8_DOORS" = "1" ]; then
    echo "[entrypoint] Q8_DOORS=1 fused Q8 swiglu/attn/gdn/comm"
    export GGML_SYCL_COMM_FUSED_Q8=1
    export GGML_SYCL_FUSED_SWIGLU_Q8=1
    export GGML_SYCL_FUSED_ATTN_Q8=1
    export GGML_SYCL_FUSED_GDN_Q8=1
    export GGML_SYCL_COMM_DIRECT_Q8="${GGML_SYCL_COMM_DIRECT_Q8:-2}"
else
    echo "[entrypoint] Q8_DOORS=0 baseline"
    export GGML_SYCL_COMM_FUSED_Q8=0
    export GGML_SYCL_FUSED_SWIGLU_Q8=0
    export GGML_SYCL_FUSED_ATTN_Q8=0
    export GGML_SYCL_FUSED_GDN_Q8=0
    export GGML_SYCL_COMM_DIRECT_Q8=0
fi

# llama.cpp-only comm path. Do NOT set CCL_TOPO_P2P_ACCESS (vLLM TP>1 wedge).
if [ "${LLAMA_P2P:-0}" = "1" ]; then
    echo "[entrypoint] LLAMA_P2P=1 (llama.cpp SYCL only, not vLLM)"
    export GGML_SYCL_P2P=1
fi

echo "[entrypoint] GPU_COUNT=$GPU_COUNT ctx=$CTX_SIZE batch=$BATCH/$UBATCH file=$MODEL_FILE MMVQ_SG32=${GGML_SYCL_MMVQ_SG32:-0} QUAD_SG24=${GGML_SYCL_MMVQ_Q8_QUAD_SG24:-1}"

exec /build/llama.cpp/build-sycl/bin/llama-server \
    --model "$TARGET" \
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
    --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \
    --temp "$TEMP" \
    --top-k "$TOP_K" \
    --top-p "$TOP_P" \
    --min-p "$MIN_P" \
    --repeat-penalty "$REPEAT_PENALTY" \
    --threads "${THREADS:-8}" \
    --poll 50 \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --metrics
