#!/usr/bin/env bash
# Build libxpu_push_ar_torch.so and run the real-torch-tensor 2-process harness (106_ar_torch_harness.py).
# Proves the custom all-reduce runs in TORCH's L0 context on a torch tensor's data_ptr -- the live-serve bind.
# Usage: ./bin/gpu-run bash scripts/106_run_ar_torch.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/106_xpu_push_ar_torch.cpp"; PY="$REPO/scripts/106_ar_torch_harness.py"
echo "=== ar_torch :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f artorch 2>/dev/null || true
docker run --rm --name artorch --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    icpx -fsycl -O2 -fPIC -shared '$SRC' -o /tmp/libxpu_push_ar_torch.so -lze_loader -lrt && echo 'BUILD OK'
    rm -f /tmp/ar_torch.sock
    timeout 180 python3 '$PY' || echo 'HARNESS TIMEOUT/FAIL'
  " 2>&1
echo "=== ar_torch exit $? ==="
