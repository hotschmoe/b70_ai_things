#!/usr/bin/env bash
# One guarded TP=2 transaction: June profile MoE, then compiled oneCCL.
# Run through: ./bin/gpu-run env I_KNOW_P2P_WEDGES=1 bash <this-script>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMG="intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94"
JUNE_RUNTIME="/mnt/vm_8tb/b70/steve-repro/june-xpuc-bmg-g21-a0-20260825/runtime-candidate"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/oneccl_oracle}"
RESULT_FILE="$RESULT_DIR/qwen36_tp2_moe_collective_boundary_${STAMP}.json"
LOG_FILE="$RESULT_DIR/qwen36_tp2_moe_collective_boundary_${STAMP}.log"
CACHE_DIR="/mnt/vm_8tb/b70/vllm_oracle_cache/qwen36_tp2_moe_collective_boundary_${STAMP}"
EXPECTED_XPU_C_SHA256="2d931484ee0aadd4c9fb6abf494e147a5275210a216426a1eb56add0158bef0d"

if [ "${I_KNOW_P2P_WEDGES:-0}" != 1 ]; then
  echo "Refusing direct P2P boundary probe without I_KNOW_P2P_WEDGES=1" >&2
  exit 2
fi
mkdir -p "$RESULT_DIR" "$CACHE_DIR"

post_health() {
  local run_rc=$? health_rc
  trap - EXIT
  set +e
  "$REPO_ROOT/bin/xpu-health" 2>&1 | tee "$RESULT_DIR/qwen36_tp2_moe_collective_boundary_${STAMP}.health.log"
  health_rc="${PIPESTATUS[0]}"
  set -e
  if [ "$run_rc" = 0 ] && [ "$health_rc" != 0 ]; then
    run_rc="$health_rc"
  fi
  exit "$run_rc"
}
trap post_health EXIT
"$REPO_ROOT/bin/xpu-health"

echo "config -> tp=2 rows=8192 routed_rows=65536 native_moe=june-base next_op=compiled-s2b-allreduce p2p=1"
docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$JUNE_RUNTIME:/opt/runtime:ro" \
  -v "$SCRIPT_DIR/qwen36_s2b_sitecustomize.py:/opt/b70_qwen36_site/sitecustomize.py:ro" \
  -v "$SCRIPT_DIR/qwen36_june_fused_moe_single.py:/opt/b70_qwen36_site/qwen36_june_fused_moe_single.py:ro" \
  -v "$SCRIPT_DIR/qwen36_tp2_moe_collective_boundary.py:/opt/boundary.py:ro" \
  -v "$RESULT_DIR:/results" -v "$CACHE_DIR:/cache" \
  -e PYTHONPATH=/opt/runtime:/opt/b70_qwen36_site \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e CCL_ATL_TRANSPORT=ofi -e CCL_TOPO_P2P_ACCESS=1 \
  -e CCL_LOG_LEVEL=info -e CCL_KERNEL_PATH=/opt/ccl4ce/lib/ccl/kernels \
  -e FI_TCP_IFACE=eth0 -e CCL_KVS_IFACE=eth0 \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 -e ZE_AFFINITY_MASK=0,1 \
  -e VLLM_USE_V1=1 -e VLLM_TARGET_DEVICE=xpu \
  -e VLLM_XPU_INT8_MOE_MIXED_WORKSPACE=0 \
  -e VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES=1 \
  -e VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1 \
  -e VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT=1 \
  -e VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1 \
  -e EXPECTED_XPU_C_SHA256="$EXPECTED_XPU_C_SHA256" \
  -e LD_PRELOAD=/opt/ccl4ce/lib/libccl.so.1.0 \
  -e LD_LIBRARY_PATH=/opt/ccl4ce/lib:/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  --entrypoint /opt/venv/bin/torchrun "$IMG" \
  --standalone --nproc-per-node=2 /opt/boundary.py \
  --output "/results/$(basename "$RESULT_FILE")" 2>&1 | tee "$LOG_FILE"
python3 -c 'import json, sys; assert json.load(open(sys.argv[1]))["passed"]' "$RESULT_FILE"
echo "verdict -> boundary probe passed: $RESULT_FILE"
