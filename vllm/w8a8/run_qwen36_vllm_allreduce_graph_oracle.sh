#!/usr/bin/env bash
# One guarded TP2 transaction for the vLLM custom-op/graph integration layer.
# Usage: ./bin/gpu-run env I_KNOW_P2P_WEDGES=1 bash \
#   vllm/w8a8/run_qwen36_vllm_allreduce_graph_oracle.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMG="${IMG:-intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/oneccl_oracle}"
RESULT_FILE="${RESULT_FILE:-$RESULT_DIR/qwen36_tp2_vllm_allreduce_graph_${STAMP}.json}"
LOG_FILE="${LOG_FILE:-$RESULT_DIR/qwen36_tp2_vllm_allreduce_graph_${STAMP}.log}"
CACHE_DIR="${CACHE_DIR:-/mnt/vm_8tb/b70/vllm_oracle_cache}"
NAME="${NAME:-qwen36_vllm_allreduce_graph_oracle}"
EXPECTED_CCL_SHA256="${EXPECTED_CCL_SHA256:-542142aca8f3d318616eae0f300aaa47dc62b217831599cb1461212f8aa4dc76}"
EXPECTED_XPU_C_SHA256="${EXPECTED_XPU_C_SHA256:-ae330affe0315a5be4ac50478cc15c7874ae6e8fa9fa71cf64d5e5dff158968b}"
EXPECTED_CCL_KERNELS_SHA256="${EXPECTED_CCL_KERNELS_SHA256:-0d549c35a558f1b216cb7d1efeaa9f86d7596ffc47b383644e075290d314f0c9}"

if [ "${I_KNOW_P2P_WEDGES:-0}" != 1 ]; then
  echo "Refusing direct P2P oracle without I_KNOW_P2P_WEDGES=1" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR" "$CACHE_DIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "config -> image=$IMG shapes=[1,2048],[4,2048],[8192,2048] custom_op=s2b_all_reduce_clone graph=XPUGraph collectives=81 ipc=default"
docker run --rm --name "$NAME" --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path --ipc=host \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$SCRIPT_DIR/qwen36_s2b_sitecustomize.py:/opt/b70_qwen36_site/sitecustomize.py:ro" \
  -v "$SCRIPT_DIR/qwen36_vllm_allreduce_graph_oracle.py:/opt/qwen36_vllm_allreduce_graph_oracle.py:ro" \
  -v "$RESULT_DIR:/results" -v "$CACHE_DIR:/vllm_oracle_cache" \
  -e B70_ORACLE_IMAGE="$IMG" \
  -e PYTHONPATH=/opt/b70_qwen36_site \
  -e TORCHINDUCTOR_CACHE_DIR=/vllm_oracle_cache/torchinductor \
  -e TRITON_CACHE_DIR=/vllm_oracle_cache/triton \
  -e CCL_ATL_TRANSPORT=ofi \
  -e CCL_TOPO_P2P_ACCESS=1 \
  -e CCL_LOG_LEVEL=info \
  -e CCL_KERNEL_PATH=/opt/ccl4ce/lib/ccl/kernels \
  -e EXPECTED_CCL_SHA256="$EXPECTED_CCL_SHA256" \
  -e EXPECTED_XPU_C_SHA256="$EXPECTED_XPU_C_SHA256" \
  -e EXPECTED_CCL_KERNELS_SHA256="$EXPECTED_CCL_KERNELS_SHA256" \
  -e FI_TCP_IFACE=eth0 \
  -e CCL_KVS_IFACE=eth0 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
  -e ZE_AFFINITY_MASK=0,1 \
  -e VLLM_USE_V1=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e XPU_GRAPH=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_XPU_FORCE_GRAPH_WITH_COMM=1 \
  -e VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1 \
  -e VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES=1 \
  -e VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1 \
  -e VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT=1 \
  -e VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1 \
  -e LD_PRELOAD=/opt/ccl4ce/lib/libccl.so.1.0 \
  -e LD_LIBRARY_PATH=/opt/ccl4ce/lib:/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  --entrypoint /opt/venv/bin/torchrun "$IMG" \
  --standalone --nproc-per-node=2 /opt/qwen36_vllm_allreduce_graph_oracle.py \
  --output "/results/$(basename "$RESULT_FILE")" 2>&1 | tee "$LOG_FILE"
python3 -c 'import json, sys; assert json.load(open(sys.argv[1]))["passed"]' "$RESULT_FILE"
if rg -qi 'output tensor.*alias|aliasing an input tensor' "$LOG_FILE"; then
  echo "FAIL: custom-op alias warning found in $LOG_FILE" >&2
  exit 1
fi
echo "result -> $RESULT_FILE log=$LOG_FILE"
