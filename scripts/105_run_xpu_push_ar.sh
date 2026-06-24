#!/usr/bin/env bash
# Build libxpu_push_ar.so and run the 2-process torch.distributed harness (105_ar_harness.py).
# Proves the deployable vLLM custom-op core: independent procs, named-socket IPC fd-pass, full all-reduce.
# Usage: ./bin/gpu-run bash scripts/105_run_xpu_push_ar.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/105_xpu_push_ar.cpp"
PY="$REPO/scripts/105_ar_harness.py"
echo "=== xpu_push_ar :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f xpupushar 2>/dev/null || true
docker run --rm --name xpupushar --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    icpx -fsycl -O2 -fPIC -shared '$SRC' -o /tmp/libxpu_push_ar.so -lze_loader -lrt && echo 'BUILD OK'
    rm -f /tmp/ar_ipc.sock
    timeout 180 python3 '$PY' || echo 'HARNESS TIMEOUT/FAIL'
  " 2>&1
echo "=== xpu_push_ar exit $? ==="
