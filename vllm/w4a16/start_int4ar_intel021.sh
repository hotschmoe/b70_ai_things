#!/usr/bin/env bash
# LOOP 37: S2b on intel/vllm:0.21.0-xpu (torch 2.11, SYCL-8, in-image 2021.17).
# Does NOT use lib.sh start: that --entrypoint vllm skips setvars, and
# xpu-health --entrypoint python3 on this image ImportErrors libccl
# (false WEDGED). Health this fire with int8g-v0260, then this script.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
MODELS_FILES="${MODELS_FILES:-$REPO/models/files}"
IMG="${IMG:-intel/vllm:0.21.0-xpu}"
NAME="${NAME:-qwen38_int4ar}"
PORT="${PORT:-18080}"
SERVED="${SERVED:-qwen3.8-27b-W4A16-autoround-mtp5}"
CKPT="${CKPT:-/models/qwen3.8-27b/int4-autoround}"
TP="${TP:-2}"
GRAPH="${GRAPH:-1}"
MTPTOK="${MTPTOK:-5}"
MAXLEN="${MAXLEN:-16384}"
UTIL="${UTIL:-0.88}"
MAXSEQS="${MAXSEQS:-8}"
DTYPE="${DTYPE:-float16}"
DEVICE="${DEVICE:-0}"
WRAP="$REPO/vllm/w4a16/intel021_vllm_entrypoint.sh"
chmod +x "$WRAP"

if [ "${1:-start}" = stop ]; then
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "stopped $NAME"
  exit 0
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$ROOT/vllm_cache/triton_intel021" "$ROOT/vllm_cache/intel021" "$ROOT/tmp_ssd" "$ROOT/hf_cache"

CC=()
EAGER=(--enforce-eager)
GDOCK=()
GENV=()
if [ "$GRAPH" = 1 ]; then
  EAGER=()
  GENV=(-e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e OMP_NUM_THREADS=8)
  GDOCK=(--pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556)
  # 0.21.1.dev18 CompilationConfig rejects the 0.26/0.27 pass_config keys
  # (fuse_rope_kvcache_cat_mla etc). Keep the 0.21 PIECEWISE shape only.
  CC=(--compilation-config '{"cudagraph_mode":"PIECEWISE","use_inductor_graph_partition":true,"cudagraph_capture_sizes":[1,2,4,5,6,8],"max_cudagraph_capture_size":8}')
fi

MGPU=()
SHM=16g
if [ "$TP" -gt 1 ]; then
  SHM=32g
  MGPU=(-e CCL_ENABLE_SYCL_KERNELS=1 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
        -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi
        -e VLLM_WORKER_MULTIPROC_METHOD=spawn
        -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd)
else
  MGPU=(-e ZE_AFFINITY_MASK="$DEVICE")
fi

ARGS=(serve "$CKPT" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT"
      --dtype "$DTYPE" --tensor-parallel-size "$TP" --max-model-len "$MAXLEN"
      --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL"
      --trust-remote-code --no-enable-prefix-caching
      --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTPTOK}}"
      --enable-auto-tool-choice --tool-call-parser qwen3_xml
      --reasoning-parser qwen3 --language-model-only
      "${EAGER[@]}" "${CC[@]}")
[ "$TP" -gt 1 ] && ARGS+=(--distributed-executor-backend mp)

echo "=== intel021 serve $SERVED IMG=$IMG TP=$TP GRAPH=$GRAPH ===" >&2
echo "vllm ${ARGS[*]}" >&2

docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size "$SHM" -p "${PORT}:${PORT}" "${GDOCK[@]}" \
  -v "$MODELS_FILES:/models:ro" \
  -v "$ROOT/hf_cache:/hf_cache" \
  -v "$ROOT/vllm_cache:/vllm_cache" \
  -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$WRAP:/opt/venv/bin/vllm:ro" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache/intel021 \
  -e XDG_CACHE_HOME=/vllm_cache/intel021 \
  -e TRITON_CACHE_DIR=/vllm_cache/triton_intel021 -e TMPDIR=/tmp_ssd \
  -e VLLM_LOGGING_LEVEL=INFO \
  "${MGPU[@]}" "${GENV[@]}" \
  --entrypoint /opt/venv/bin/vllm "$IMG" "${ARGS[@]}"
echo "started $NAME id=$(docker inspect -f '{{.Id}}' "$NAME" | cut -c1-12)"
