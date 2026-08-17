#!/usr/bin/env bash
# Serve official Qwen3.8-27B-FP8 + rmacy DSpark drafter via the published
# ghcr.io/rmacy/qwen38-fp8-dspark image (intel/llm-scaler#620).
#
# This is a RESEARCH serve, not a shelf entry. Recipe is isolated-C1:
#   TP=2, maxlen=8192, max-num-seqs=1, dflash k=4, XPU graphs OFF.
# Claimed: 72.2 tok/s median isolated C1 (temp=0, pp=0, n=16) with P2P=1.
#
# P2PACCESS defaults to 0. Our box has a documented oneCCL P2P-in-vLLM-TP
# wedge (P2P_GPU.md H.13). Their published recipe uses P2P=1; only flip
# that with I_KNOW_P2P_WEDGES=1 and be ready to reboot.
#
#   ./bin/gpu-run bash vllm/dflash/serve_qwen38_fp8_dspark.sh start
#   bash vllm/dflash/serve_qwen38_fp8_dspark.sh stop
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="${NAME:-qwen38_fp8_dspark}"
ACTION="${1:-start}"
PORT="${PORT:-8078}"
IMG="${IMG:-ghcr.io/rmacy/qwen38-fp8-dspark:v10-slim}"
TARGET="${TARGET:-$REPO/models/files/qwen3.8-27b/fp8}"
DRAFTER="${DRAFTER:-$REPO/models/files/qwen3.8-27b/dflash-drafter-fp8-b70}"
SERVED="${SERVED:-qwen3.8-27b-fp8-dspark}"
MAXLEN="${MAXLEN:-8192}"
UTIL="${UTIL:-0.90}"
MAXSEQS="${MAXSEQS:-1}"
MAXBATCH="${MAXBATCH:-4096}"
SPECTOK="${SPECTOK:-4}"
P2PACCESS="${P2PACCESS:-0}"
# Their published serve.sh uses drmfd. On this box drmfd fails
# mem_to_ipc_handle (device_fd invalid) at the xpu_worker warmup all_reduce.
# pidfd is the known-good exchange (lib.sh IPCX default).
IPCX="${IPCX:-pidfd}"
ATL="${ATL:-ofi}"
# Image ships oneCCL 2021.15.9 which dies at TP=2 warmup all_reduce
# (ze_handle_manager mem_to_ipc_handle: device_fd is invalid) for
# pidfd/sockets/drmfd alike. Same bug we patched in int8g-v0251.
# Bind-mount our 2021.17.2 tree over the 2021.15 path so venv
# libccl.so.1 -> /opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1.0
# resolves to 2021.17.
CCL217="${CCL217:-/mnt/vm_8tb/b70/ccl_2021.17/2021.17}"
LOG="${LOG:-/mnt/vm_8tb/b70/dd-logs/${NAME}.log}"

if [ "$ACTION" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
  exit 0
fi

if [ ! -f "$TARGET/config.json" ]; then
  echo "missing target $TARGET (hf: Qwen/Qwen3.8-27B-FP8)"; exit 1
fi
if [ ! -f "$DRAFTER/config.json" ]; then
  echo "missing drafter $DRAFTER (hf: rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70)"; exit 1
fi
if [ ! -e "$CCL217/lib/libccl.so.1.0" ]; then
  echo "missing oneCCL 2021.17 at $CCL217 (needed to replace image 2021.15 TP=2 bug)"; exit 1
fi

if [ "$P2PACCESS" = 1 ] && [ "${I_KNOW_P2P_WEDGES:-0}" != 1 ]; then
  echo "[GUARD] P2PACCESS=1 refused (P2P_GPU.md H.13). Set I_KNOW_P2P_WEDGES=1 to override."
  exit 2
fi

mkdir -p "$(dirname "$LOG")"
docker rm -f "$NAME" >/dev/null 2>&1 || true

# Device passthrough matches lib.sh (headless box: card0+card1). Their
# published serve.sh pins card1/card2 because that host has an iGPU as card0.
echo "=== $NAME  IMG=$IMG  port=$PORT  P2P=$P2PACCESS  IPCX=$IPCX  k=$SPECTOK  maxlen=$MAXLEN ==="
docker run -d --name "$NAME" \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --network host --shm-size 32g --ipc=host \
  -e ZE_AFFINITY_MASK=0,1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_USE_AOT_COMPILE=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e CCL_ROOT=/opt/intel/oneapi/ccl/2021.15 \
  -e CCL_CONFIGURATION=cpu_gpu_dpcpp \
  -e CCL_ATL_TRANSPORT="$ATL" \
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e CCL_TOPO_P2P_ACCESS="$P2PACCESS" \
  -e CCL_ZE_IPC_EXCHANGE="$IPCX" \
  -e FI_PROVIDER_PATH=/opt/venv/lib:/usr/lib/x86_64-linux-gnu/libfabric \
  -e LD_LIBRARY_PATH=/opt/intel/oneapi/ccl/2021.15/lib:/opt/venv/lib \
  -v "$CCL217":/opt/intel/oneapi/ccl/2021.15:ro \
  -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
  -e CCL_SYCL_ALLREDUCE_TMP_BUF=0 \
  -e CCL_ENABLE_SYCL_KERNELS=1 \
  -e CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 \
  -e CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$TARGET":/models/target:ro \
  -v "$DRAFTER":/models/drafter:ro \
  --entrypoint /opt/venv/bin/vllm \
  "$IMG" \
  serve --host 127.0.0.1 --port "$PORT" \
    --model /models/target \
    --served-model-name "$SERVED" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAXLEN" \
    --max-num-batched-tokens "$MAXBATCH" \
    --max-num-seqs "$MAXSEQS" \
    --block-size 64 \
    --dtype bfloat16 \
    --mamba-ssm-cache-dtype float16 \
    --async-scheduling \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/drafter\",\"num_speculative_tokens\":${SPECTOK}}"

cid=$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)
if [ "$cid" != true ]; then
  echo "docker run failed to start $NAME"
  docker ps -a --filter name="$NAME" --format '{{.Status}}'
  exit 1
fi
echo "started $NAME  logs: docker logs -f $NAME"
echo "endpoint: http://127.0.0.1:${PORT}/v1  id=$SERVED"
