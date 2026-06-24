#!/usr/bin/env bash
# Build + run 115_ze_event_sync.c: KEYSTONE -- is a cross-device command-streamer L0-event wait correct +
# replayable on B70? (the graph-capturable decode sync question, docs/P2P_GPU.md K.3). timeout-guarded
# (a broken cross-device wait could deadlock the two closed lists).
# Usage: ./bin/gpu-run bash scripts/115_run_event_sync.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/115_ze_event_sync.c"
echo "=== ze_event_sync :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f zeevsync 2>/dev/null || true
docker run --rm --name zeevsync --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e
    gcc '$SRC' -o /tmp/ze_event_sync -lze_loader && echo 'BUILD OK'
    timeout 120 /tmp/ze_event_sync || echo 'RUN TIMEOUT/FAIL (cross-device wait deadlock?)'
  " 2>&1
echo "=== ze_event_sync exit $? ==="
