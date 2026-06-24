#!/usr/bin/env bash
# Rebuild libxpu_push_ar_torch.so (now with bf16 reduce) and run the vLLM-call-pattern test (107).
# Proves the op is a correct+fast drop-in for XpuCommunicator.all_reduce on bf16 prefill/decode shapes.
# Usage: ./bin/gpu-run bash scripts/107_run_ar_vllm_pattern.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/106_xpu_push_ar_torch.cpp"; PY="$REPO/scripts/107_ar_vllm_pattern.py"
echo "=== ar_vllm_pattern :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f arvllm 2>/dev/null || true
docker run --rm --name arvllm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    icpx -fsycl -O2 -fPIC -shared '$SRC' -o /tmp/libxpu_push_ar_torch.so -lze_loader -lrt && echo 'BUILD OK'
    rm -f /tmp/ar_vllm.sock
    timeout 180 python3 '$PY' || echo 'HARNESS TIMEOUT/FAIL'
  " 2>&1
echo "=== ar_vllm_pattern exit $? ==="
