#!/usr/bin/env bash
# Productionized: serve a DENSE compressed-tensors W8A8 model with TRUE int8 (DPAS via triton_scaled_mm) on
# vllm-xpu-env:v0230 -- no :int8 image needed. Mounts the sitecustomize hook (registers the task-c
# XPUInt8TritonScaledMMLinearKernel for the CompressedTensorsW8A8Int8 path) + the task-c quark.py (kernel source).
# Env: MODEL, SERVED, PORT(18081), DEVICE(1=GPU1), GRAPH(1=PIECEWISE capture / 0=eager), NAME, UTIL, MAXLEN, MAXSEQS.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:v0230}"
MODEL="${MODEL:-/models/Qwen3-14B-W8A8-autoround}"
SERVED="${SERVED:-qwen3-14b-w8a8-v0230int8}"
PORT="${PORT:-18081}"; DEVICE="${DEVICE:-1}"; GRAPH="${GRAPH:-1}"
NAME="${NAME:-vllm_dense_w8a8_v0230}"; UTIL="${UTIL:-0.60}"; MAXLEN="${MAXLEN:-4096}"; MAXSEQS="${MAXSEQS:-8}"
CAPS="${CAPSIZES:-1,2,4,8,16,32,64}"
Q="$ROOT/rdy_to_serve/qwen36-35b-a3b-quark-w8a8-int8/patches/quark.py"   # source of _b70_register_xpu_int8_kernel
SITE="$ROOT/b70site"                                                     # sitecustomize that wraps init_int8_linear_kernel
docker rm -f "$NAME" 2>/dev/null || true
GENV=(); CC=(--enforce-eager)
if [ "$GRAPH" = 1 ]; then
  GENV=(-e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e OMP_NUM_THREADS="${OMP:-8}")
  PASS='"pass_config":{"fuse_rope_kvcache_cat_mla":false,"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"fuse_rope_kvcache":false,"enable_qk_norm_rope_fusion":false}'
  CC=(--compilation-config "{\"cudagraph_mode\":\"PIECEWISE\",\"use_inductor_graph_partition\":true,\"compile_sizes\":[1],\"cudagraph_capture_sizes\":[$CAPS],$PASS}")
fi
docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --shm-size 16g \
  --pids-limit=-1 --ulimit nofile=1048576:1048576 -p ${PORT}:${PORT} \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/vllm_cache:/vllm_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" \
  -v "$Q":/workspace/vllm/vllm/model_executor/layers/quantization/quark/quark.py:ro \
  -v "$Q":/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/quark/quark.py:ro \
  -v "$SITE":/b70site:ro -e PYTHONPATH=/b70site -e B70_INT8_LINEAR=triton \
  -e HF_HOME=/hf_cache -e VLLM_CACHE_ROOT=/vllm_cache -e XDG_CACHE_HOME=/vllm_cache \
  -e TRITON_CACHE_DIR=/vllm_cache/triton -e TMPDIR=/tmp_ssd -e ZE_AFFINITY_MASK="$DEVICE" "${GENV[@]}" \
  --entrypoint vllm "$IMG" serve "$MODEL" --served-model-name "$SERVED" --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 1 --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" --gpu-memory-utilization "$UTIL" \
  --no-enable-prefix-caching --trust-remote-code "${CC[@]}"
echo "=== $NAME on GPU$DEVICE:$PORT GRAPH=$GRAPH; waiting readiness (~14min, first compile slow) ==="
for i in $(seq 1 168); do curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { echo "HEALTHY $SERVED"; break; }; docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q exited && { echo "EXITED EARLY"; docker logs "$NAME" 2>&1 | tail -25; break; }; sleep 5; done
docker logs "$NAME" 2>&1 | grep -iE "Selected.*Int8|registered XPU int8|Capturing|Model loading took|error|traceback" | tail -8
