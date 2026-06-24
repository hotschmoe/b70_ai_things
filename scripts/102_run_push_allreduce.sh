#!/usr/bin/env bash
# Build + run hand-rolled 2-rank push all-reduce (102_push_allreduce.cpp). Compare algbw vs oneCCL H.12.
# Usage: ./bin/gpu-run bash scripts/102_run_push_allreduce.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/102_push_allreduce.cpp"
echo "=== push_allreduce :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f pushar 2>/dev/null || true
docker run --rm --name pushar --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    icpx -fsycl -O2 '$SRC' -o /tmp/push_allreduce && echo 'BUILD OK'
    /tmp/push_allreduce
  " 2>&1
echo "=== push_allreduce exit $? ==="
