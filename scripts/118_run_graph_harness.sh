#!/usr/bin/env bash
# Build libxpu_push_ar_graph.so + run the torch XPUGraph harness (capturable push all-reduce). docs/P2P_GPU K.6.
# Decisive pre-serve de-risk: does native-cmd event sync record into torch.xpu.graph + replay correctly, 2 procs.
# Usage: ./bin/gpu-run bash scripts/118_run_graph_harness.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:int8g}"
SRC="$REPO/scripts/118_xpu_push_ar_graph.cpp"
echo "=== graph_harness :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f graphhar 2>/dev/null || true
docker run --rm --name graphhar --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    icpx -fsycl -O2 -fPIC -shared '$SRC' -o /tmp/libxpu_push_ar_graph.so -lze_loader -lrt && echo 'BUILD .so OK'
    SO=/tmp/libxpu_push_ar_graph.so timeout 180 python3 '$REPO/scripts/118_graph_harness.py' \
      || echo 'RUN TIMEOUT/FAIL'
  " 2>&1
echo "=== graph_harness exit $? ==="
