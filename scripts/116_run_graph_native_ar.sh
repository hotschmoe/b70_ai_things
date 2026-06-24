#!/usr/bin/env bash
# Build + run 116_graph_native_ar.cpp: per-rank SYCL command_graph push all-reduce with the cross-device
# sync injected as L0 event signal/wait/reset via ext_codeplay_enqueue_native_command (the exact mechanism
# torch-xpu XPUGraph capture will record). docs/P2P_GPU.md K.4. timeout-guarded vs a replay deadlock.
# Usage: ./bin/gpu-run bash scripts/116_run_graph_native_ar.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/116_graph_native_ar.cpp"
echo "=== graph_native_ar :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f gnar 2>/dev/null || true
docker run --rm --name gnar --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    icpx -fsycl -O2 '$SRC' -o /tmp/graph_native_ar -lze_loader && echo 'BUILD OK'
    timeout 180 /tmp/graph_native_ar || echo 'RUN TIMEOUT/FAIL (replay deadlock / native-cmd-in-graph unsupported?)'
  " 2>&1
echo "=== graph_native_ar exit $? ==="
