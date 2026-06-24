#!/usr/bin/env bash
# Build + run 114_graph_allreduce.cpp: is the push all-reduce SYCL-graph-capturable+replayable on B70?
# (the decode-capture question, docs/P2P_GPU.md J.9 / handoff_decode_push_ar.md). `timeout` guards a
# mis-recorded cross-device graph edge against a replay hang.
# Usage: ./bin/gpu-run bash scripts/114_run_graph_allreduce.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/114_graph_allreduce.cpp"
echo "=== graph_allreduce :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f graphar 2>/dev/null || true
docker run --rm --name graphar --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    icpx -fsycl -O2 '$SRC' -o /tmp/graph_allreduce -lze_loader && echo 'BUILD OK'
    timeout 180 /tmp/graph_allreduce || echo 'RUN TIMEOUT/FAIL (graph replay hang or cross-device edge unsupported?)'
  " 2>&1
echo "=== graph_allreduce exit $? ==="
