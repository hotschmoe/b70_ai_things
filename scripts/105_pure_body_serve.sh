#!/usr/bin/env bash
# Minimal pure-body serve of the 27B W8A8 graft dir: TP=2, EAGER, NO speculative-config, NO mtp shim.
# Isolates the W8A8 body load/apply from the MTP graft+shim. Used to verify the ignore-list fix.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
CKPT="${CKPT:-/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft}"
NAME="${NAME:-vllm_w8a8_purebody}"
PORT="${PORT:-8000}"
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
GDN_SO=$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so
GDN_LIB=$ROOT/vllm-xpu-kernels/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g -p "${PORT}:${PORT}" \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$GDN_SO:$PKGD/_xpu_C.abi3.so:ro" -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache -e TMPDIR=/tmp_ssd \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e CCL_ENABLE_SYCL_KERNELS=0 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e CCL_ATL_TRANSPORT=ofi -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE=pidfd \
  --entrypoint vllm vllm-xpu-env:int8g \
  serve "$CKPT" --served-model-name w8a8pure --host 0.0.0.0 --port "$PORT" \
  --dtype auto --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching --trust-remote-code --distributed-executor-backend mp \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enforce-eager >/dev/null
echo "launched $NAME"
