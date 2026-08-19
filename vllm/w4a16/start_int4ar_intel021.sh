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
# D13 overlay: Steve GDN spec fallback for the 8df6feb7d image only.
# 44fc8fde0 has native GDN spec -- do not clobber it with the old files.
VLLM_SRC="${VLLM_SRC:-}"
GDNFB="${GDNFB:-/mnt/vm_8tb/b70/qwen38-w8a8-dspark/intel021_gdnfb}"
CACHE_NAME="${CACHE_NAME:-intel021}"
EXTRA_MOUNTS=()
if [ -n "$VLLM_SRC" ] && [ -d "$VLLM_SRC/vllm" ]; then
  EXTRA_MOUNTS+=( -v "$VLLM_SRC:/opt/vllm:ro" )
  echo "=== overlay vLLM source <- $VLLM_SRC ===" >&2
fi
PKGD=/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels
if [ -n "${XPU_C_SO:-}" ] && [ -f "$XPU_C_SO" ]; then
  EXTRA_MOUNTS+=( -v "$XPU_C_SO:$PKGD/_xpu_C.abi3.so:ro" )
  echo "=== overlay _xpu_C <- $XPU_C_SO ===" >&2
fi
if [ -n "${GDN_LIB:-}" ] && [ -f "$GDN_LIB" ]; then
  EXTRA_MOUNTS+=( -v "$GDN_LIB:$PKGD/libgdn_attn_kernels_xe_2.so:ro" )
  echo "=== overlay gdn lib <- $GDN_LIB ===" >&2
fi
# D14: Steve 4ceafd1 oneCCL (SYCL-8). Bind over 2021.15/2021.17 so torch RPATH cannot win.
CCL4CE="${CCL4CE:-}"
if [ -n "$CCL4CE" ] && [ -f "$CCL4CE/lib/libccl.so.1.0" ]; then
  EXTRA_MOUNTS+=( -v "$CCL4CE:/opt/ccl4ce:ro" )
  EXTRA_MOUNTS+=( -v "$CCL4CE/lib/libccl.so.1.0:/opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1.0:ro" )
  EXTRA_MOUNTS+=( -v "$CCL4CE/lib/libccl.so.1.0:/opt/intel/oneapi/ccl/2021.15/lib/libccl.so.1:ro" )
  EXTRA_MOUNTS+=( -v "$CCL4CE/lib/libccl.so.1.0:/opt/intel/oneapi/ccl/2021.17/lib/libccl.so.1.0:ro" )
  EXTRA_MOUNTS+=( -v "$CCL4CE/lib/libccl.so.1.0:/opt/intel/oneapi/ccl/2021.17/lib/libccl.so.1:ro" )
  echo "=== overlay oneCCL 4ceafd1 <- $CCL4CE ===" >&2
fi
if [ -z "${VLLM_SRC:-}" ] && [ -f "$GDNFB/_xpu_ops.py" ]; then
  EXTRA_MOUNTS+=( -v "$GDNFB/_xpu_ops.py:/opt/vllm/vllm/_xpu_ops.py:ro" )
  echo "=== overlay _xpu_ops.py <- $GDNFB (GDN spec fallback) ===" >&2
  if [ -f "$GDNFB/gdn_linear_attn.py" ]; then
    EXTRA_MOUNTS+=( -v "$GDNFB/gdn_linear_attn.py:/opt/vllm/vllm/model_executor/layers/mamba/gdn_linear_attn.py:ro" )
    echo "=== overlay gdn_linear_attn.py <- $GDNFB ===" >&2
  fi
fi

if [ "${1:-start}" = stop ]; then
  docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "stopped $NAME"
  exit 0
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$ROOT/vllm_cache/triton_intel021" "$ROOT/vllm_cache/${CACHE_NAME}" "$ROOT/tmp_ssd" "$ROOT/hf_cache"

CC=()
EAGER=(--enforce-eager)
GDOCK=()
GENV=()
if [ "$GRAPH" = 1 ]; then
  EAGER=()
  GENV=(-e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e OMP_NUM_THREADS=8)
  # 44fc8fde0: allow PIECEWISE capture on TP>1 (Steve 101.922 path).
  if [ "$TP" -gt 1 ]; then
    GENV+=(-e VLLM_XPU_FORCE_GRAPH_WITH_COMM=1)
  fi
  GDOCK=(--pids-limit=-1 --ulimit nofile=1048576:1048576 --ulimit nproc=63556:63556)
  # 44fc8fde0 defaults fuse_rope_kvcache_cat_mla True but does not import
  # MLARoPEKVCacheCatFusionPass on XPU (NameError). Steve's recipe sets false.
  # Older 8df6feb7d rejects this key -- only send it when overlaying 44fc.
  if [ -n "${VLLM_SRC:-}" ]; then
    CC=(--compilation-config '{"cudagraph_mode":"PIECEWISE","use_inductor_graph_partition":true,"cudagraph_capture_sizes":[1,2,4,5,6,8],"max_cudagraph_capture_size":8,"pass_config":{"fuse_rope_kvcache_cat_mla":false}}')
  else
    CC=(--compilation-config '{"cudagraph_mode":"PIECEWISE","use_inductor_graph_partition":true,"cudagraph_capture_sizes":[1,2,4,5,6,8],"max_cudagraph_capture_size":8}')
  fi
fi

MGPU=()
SHM=16g
if [ "$TP" -gt 1 ]; then
  SHM=32g
  MGPU=(-e CCL_ENABLE_SYCL_KERNELS=1 -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
        -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi
        -e VLLM_WORKER_MULTIPROC_METHOD=spawn
        -e CCL_TOPO_P2P_ACCESS=0 -e CCL_ZE_IPC_EXCHANGE="${IPCX:-pidfd}")
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
  "${EXTRA_MOUNTS[@]}" \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache/${CACHE_NAME} \
  -e XDG_CACHE_HOME=/vllm_cache/${CACHE_NAME} \
  -e TRITON_CACHE_DIR=/vllm_cache/triton_intel021 -e TMPDIR=/tmp_ssd \
  -e VLLM_LOGGING_LEVEL=INFO \
  "${MGPU[@]}" "${GENV[@]}" \
  --entrypoint /opt/venv/bin/vllm "$IMG" "${ARGS[@]}"
echo "started $NAME id=$(docker inspect -f '{{.Id}}' "$NAME" | cut -c1-12)"
