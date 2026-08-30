#!/usr/bin/env bash
# Local safety port of the Neural.Download deterministic Qwen3.8-27B-FP8
# target-only launcher. Run only while bin/gpu-run holds both B70 leases.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-neural-download/vllm-openai-xpu:qwen38-fp8-collective-work-wait-r15}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:dce80db0a1ad861145e88b1c565f29172641912dc75a3b50d08e370f7d58e291}"
MODEL_DIR="${MODEL_DIR:-$REPO/models/files/qwen3.8-27b/fp8-official}"
CACHE_DIR="${CACHE_DIR:-}"
NAME="${NAME:-qwen38-fp8-neural-f02}"
SERVED="${SERVED:-qwen3.8-27b-FP8-official-W8A16-mtp0-p2p0-fp16kv-f02}"
PORT="${PORT:-18187}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MEMORY_GIB="${MEMORY_GIB:-32}"
ALLOW_EXISTING_CACHE="${ALLOW_EXISTING_CACHE:-0}"
COMMUNICATOR_SHA256="${COMMUNICATOR_SHA256:-5ab2ea5d9e049e6b53e2d56d1e3419ce01d1988e8be5295bab1f912a7fdbf74d}"
LAYERNORM_SHA256="${LAYERNORM_SHA256:-}"
SPECULATIVE_TOKENS="${SPECULATIVE_TOKENS:-0}"
SPECULATIVE_FORCE_REJECT="${SPECULATIVE_FORCE_REJECT:-0}"
RMS_PACKED_SERIAL_EXACT="${RMS_PACKED_SERIAL_EXACT:-0}"
GDN_PERSISTENT_SCRATCH="${GDN_PERSISTENT_SCRATCH:-0}"
INDUCTOR_COMBO_KERNELS="${INDUCTOR_COMBO_KERNELS:-1}"
INDUCTOR_BENCHMARK_COMBO_KERNEL="${INDUCTOR_BENCHMARK_COMBO_KERNEL:-1}"
INDUCTOR_MAX_AUTOTUNE="${INDUCTOR_MAX_AUTOTUNE:-1}"
INDUCTOR_COORDINATE_DESCENT_TUNING="${INDUCTOR_COORDINATE_DESCENT_TUNING:-1}"
INDUCTOR_AUTOTUNE_POINTWISE="${INDUCTOR_AUTOTUNE_POINTWISE:-1}"

EXPECTED_FILE_HASHES=(
  "f3273ccfb41be44c3c02080c26df10e8b200060366b900d940803f4221224c59  /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py"
  "$COMMUNICATOR_SHA256  /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py"
  "7c36e4a8dab4bfc06b1d5be2d8466e8cdc94099dd5409424fecc6dd8ffc2c208  /opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/scaled_mm/xpu.py"
  "7afb4de8b87d7f180d696f7cadad8b9d48d9ab7b706ae19616425c4f9456fb19  /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
)
RUNTIME_FILES=(
  /opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py
  /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py
  /opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/scaled_mm/xpu.py
  /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
)
if [ -n "$LAYERNORM_SHA256" ]; then
  EXPECTED_FILE_HASHES+=(
    "$LAYERNORM_SHA256  /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py"
    "$LAYERNORM_SHA256  /workspace/vllm/vllm/model_executor/layers/layernorm.py"
  )
  RUNTIME_FILES+=(
    /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py
    /workspace/vllm/vllm/model_executor/layers/layernorm.py
  )
fi

usage() {
  echo "usage: $0 --print-config | --verify-image | run"
}

positive_integer() {
  case "$2" in
    ''|*[!0-9]*|0) echo "$1 must be a positive integer" >&2; exit 2 ;;
  esac
}

verify_image() {
  local actual line observed
  actual="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  [ "$actual" = "$EXPECTED_IMAGE_ID" ] || {
    echo "image ID mismatch: actual=$actual expected=$EXPECTED_IMAGE_ID" >&2
    exit 1
  }
  observed="$(docker run --rm --entrypoint sha256sum "$IMAGE" "${RUNTIME_FILES[@]}")"
  for line in "${EXPECTED_FILE_HASHES[@]}"; do
    printf '%s\n' "$observed" | grep -Fxq "$line" || {
      echo "runtime file hash mismatch: expected=$line" >&2
      exit 1
    }
  done
  echo "image verification -> pass id=$actual files=${#EXPECTED_FILE_HASHES[@]}"
}

print_config() {
  echo "image=$IMAGE"
  echo "expected_image_id=$EXPECTED_IMAGE_ID"
  echo "communicator_sha256=$COMMUNICATOR_SHA256"
  echo "model_dir=$MODEL_DIR"
  echo "served_model=$SERVED"
  echo "container=$NAME"
  echo "port=$PORT"
  echo "tp=2"
  echo "p2p=0"
  echo "mtp=$SPECULATIVE_TOKENS"
  echo "speculative_force_reject=$SPECULATIVE_FORCE_REJECT"
  echo "rms_packed_serial_exact=$RMS_PACKED_SERIAL_EXACT"
  echo "gdn_persistent_scratch=$GDN_PERSISTENT_SCRATCH"
  echo "xpu_graph=0"
  echo "inductor=1"
  echo "inductor_combo_kernels=$INDUCTOR_COMBO_KERNELS"
  echo "inductor_benchmark_combo_kernel=$INDUCTOR_BENCHMARK_COMBO_KERNEL"
  echo "inductor_max_autotune=$INDUCTOR_MAX_AUTOTUNE"
  echo "inductor_coordinate_descent_tuning=$INDUCTOR_COORDINATE_DESCENT_TUNING"
  echo "inductor_autotune_pointwise=$INDUCTOR_AUTOTUNE_POINTWISE"
  echo "dtype=float16"
  echo "kv_cache_dtype=auto"
  echo "quantization=fp8"
  echo "gpu_memory_utilization=0.80"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "max_num_seqs=$MAX_NUM_SEQS"
  echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo "container_memory_gib=$MEMORY_GIB"
  echo "container_swap_extra_gib=0"
  echo "allow_existing_cache=$ALLOW_EXISTING_CACHE"
}

case "${1:-}" in
  --print-config) print_config; exit 0 ;;
  --verify-image) verify_image; exit 0 ;;
  run) ;;
  -h|--help|'') usage; exit 2 ;;
  *) usage >&2; exit 2 ;;
esac

for pair in \
  "PORT:$PORT" \
  "MAX_MODEL_LEN:$MAX_MODEL_LEN" \
  "MAX_NUM_SEQS:$MAX_NUM_SEQS" \
  "MAX_NUM_BATCHED_TOKENS:$MAX_NUM_BATCHED_TOKENS" \
  "MEMORY_GIB:$MEMORY_GIB"; do
  positive_integer "${pair%%:*}" "${pair#*:}"
done
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
case "$ALLOW_EXISTING_CACHE" in
  0|1) ;;
  *) echo "ALLOW_EXISTING_CACHE must be 0 or 1" >&2; exit 2 ;;
esac
case "$SPECULATIVE_TOKENS" in
  ''|*[!0-9]*) echo "SPECULATIVE_TOKENS must be a nonnegative integer" >&2; exit 2 ;;
esac
case "$SPECULATIVE_FORCE_REJECT" in
  0|1) ;;
  *) echo "SPECULATIVE_FORCE_REJECT must be 0 or 1" >&2; exit 2 ;;
esac
[ "$SPECULATIVE_FORCE_REJECT" -eq 0 ] || [ "$SPECULATIVE_TOKENS" -gt 0 ] || {
  echo "SPECULATIVE_FORCE_REJECT requires SPECULATIVE_TOKENS > 0" >&2
  exit 2
}
for pair in \
  "RMS_PACKED_SERIAL_EXACT:$RMS_PACKED_SERIAL_EXACT" \
  "GDN_PERSISTENT_SCRATCH:$GDN_PERSISTENT_SCRATCH" \
  "INDUCTOR_COMBO_KERNELS:$INDUCTOR_COMBO_KERNELS" \
  "INDUCTOR_BENCHMARK_COMBO_KERNEL:$INDUCTOR_BENCHMARK_COMBO_KERNEL" \
  "INDUCTOR_MAX_AUTOTUNE:$INDUCTOR_MAX_AUTOTUNE" \
  "INDUCTOR_COORDINATE_DESCENT_TUNING:$INDUCTOR_COORDINATE_DESCENT_TUNING" \
  "INDUCTOR_AUTOTUNE_POINTWISE:$INDUCTOR_AUTOTUNE_POINTWISE"; do
  case "${pair#*:}" in
    0|1) ;;
    *) echo "${pair%%:*} must be 0 or 1" >&2; exit 2 ;;
  esac
done
[ -d "$MODEL_DIR" ] || { echo "model directory is missing: $MODEL_DIR" >&2; exit 1; }
[ -n "$CACHE_DIR" ] || { echo "set CACHE_DIR to a new writable directory" >&2; exit 2; }
if [ "$ALLOW_EXISTING_CACHE" -eq 0 ]; then
  [ ! -e "$CACHE_DIR" ] || { echo "CACHE_DIR must be new: $CACHE_DIR" >&2; exit 1; }
else
  [ ! -e "$CACHE_DIR" ] || [ -d "$CACHE_DIR" ] || {
    echo "CACHE_DIR exists but is not a directory: $CACHE_DIR" >&2
    exit 1
  }
fi
docker inspect "$NAME" >/dev/null 2>&1 && {
  echo "container already exists: $NAME" >&2
  exit 1
}
verify_image
mkdir -p "$CACHE_DIR"

memory_bytes=$((MEMORY_GIB * 1024 * 1024 * 1024))
combo_kernels_json=false
benchmark_combo_kernel_json=false
autotune_pointwise_json=false
[ "$INDUCTOR_COMBO_KERNELS" -eq 0 ] || combo_kernels_json=true
[ "$INDUCTOR_BENCHMARK_COMBO_KERNEL" -eq 0 ] || benchmark_combo_kernel_json=true
[ "$INDUCTOR_AUTOTUNE_POINTWISE" -eq 0 ] || autotune_pointwise_json=true
compilation_config="{\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[1],\"max_cudagraph_capture_size\":1,\"inductor_compile_config\":{\"combo_kernels\":$combo_kernels_json,\"benchmark_combo_kernel\":$benchmark_combo_kernel_json,\"triton.autotune_pointwise\":$autotune_pointwise_json}}"
speculative_args=()
if [ "$SPECULATIVE_TOKENS" -gt 0 ]; then
  speculative_config="{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$SPECULATIVE_TOKENS}"
  if [ "$SPECULATIVE_FORCE_REJECT" -eq 1 ]; then
    speculative_config="{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$SPECULATIVE_TOKENS,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_rates\":[0.0]}"
  fi
  speculative_args=(
    --speculative-config
    "$speculative_config"
  )
fi
exec docker run --rm --name "$NAME" \
  --ulimit core=0 \
  --memory "$memory_bytes" --memory-swap "$memory_bytes" \
  --oom-score-adj 500 \
  --device /dev/dri:/dev/dri --group-add render \
  --cap-add SYS_PTRACE --security-opt label=disable \
  --ipc=host --shm-size=8g \
  --publish "127.0.0.1:${PORT}:8000" \
  --volume "$MODEL_DIR:/model:ro" \
  --volume "$CACHE_DIR:/root/.cache/vllm" \
  --env ZE_AFFINITY_MASK=0,1 \
  --env ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
  --env VLLM_TARGET_DEVICE=xpu \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  --env VLLM_XPU_GRAPH=0 \
  --env TORCHINDUCTOR_DETERMINISTIC=1 \
  --env VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE="$INDUCTOR_MAX_AUTOTUNE" \
  --env VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING="$INDUCTOR_COORDINATE_DESCENT_TUNING" \
  --env VLLM_XPU_FP8_BLOCK_W8A16=1 \
  --env VLLM_BATCH_INVARIANT=0 \
  --env VLLM_XPU_QWEN_GEMMA_RMSNORM_BATCH_INVARIANT=0 \
  --env VLLM_XPU_QWEN_GEMMA_RMSNORM_PACKED_SERIAL_EXACT="$RMS_PACKED_SERIAL_EXACT" \
  --env VLLM_XPU_GDN_NATIVE_SPEC_RECURRENT_SERIAL_EXACT=0 \
  --env VLLM_XPU_GDN_SPEC_PERSISTENT_SCRATCH="$GDN_PERSISTENT_SCRATCH" \
  --env VLLM_XPU_GDN_NATIVE_FALLBACK=1 \
  --env VLLM_XPU_MTP_SUPPRESS_BONUS_TOKEN=0 \
  --env VLLM_XPU_MTP_DRAFT_EAGER=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env CCL_ATL_TRANSPORT=ofi \
  --env FI_PROVIDER=tcp --env FI_TCP_IFACE=lo \
  --env CCL_ZE_IPC_EXCHANGE=pidfd \
  --env CCL_SEND=direct --env CCL_RECV=direct \
  --env CCL_TOPO_P2P_ACCESS=0 \
  --env CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
  --env CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
  --env CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
  --entrypoint vllm "$IMAGE" \
  serve /model \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --dtype float16 --quantization fp8 --kv-cache-dtype auto \
  --gpu-memory-utilization 0.80 \
  --max-model-len "$MAX_MODEL_LEN" --block-size 64 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --no-enable-prefix-caching --enable-prompt-tokens-details \
  --language-model-only \
  "${speculative_args[@]}" \
  --compilation-config "$compilation_config"
