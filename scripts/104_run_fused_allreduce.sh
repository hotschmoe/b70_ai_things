#!/usr/bin/env bash
# Build + run 104_fused_allreduce.cpp: A baseline (J.7) vs B cross-queue-events vs C fused device-flag.
# Measures decode-sized all-reduce LATENCY. `timeout` guards mode C's device spin-wait against a hang.
# Usage: ./bin/gpu-run bash scripts/104_run_fused_allreduce.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/104_fused_allreduce.cpp"
echo "=== fused_allreduce :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f fusedar 2>/dev/null || true
docker run --rm --name fusedar --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    icpx -fsycl -O2 '$SRC' -o /tmp/fused_allreduce && echo 'BUILD OK'
    timeout 120 /tmp/fused_allreduce || echo 'RUN TIMEOUT/FAIL (mode C spin-wait hang?)'
  " 2>&1
echo "=== fused_allreduce exit $? ==="
