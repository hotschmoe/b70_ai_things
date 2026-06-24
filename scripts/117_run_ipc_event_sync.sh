#!/usr/bin/env bash
# Build + run 117_ipc_event_sync.c: cross-PROCESS IPC event pool + cross-device command-streamer wait,
# replayed (the 2-TP-worker topology). docs/P2P_GPU.md K.5. timeout-guarded vs a cross-process deadlock.
# Usage: ./bin/gpu-run bash scripts/117_run_ipc_event_sync.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/117_ipc_event_sync.c"
echo "=== ipc_event_sync :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f ipcevsync 2>/dev/null || true
docker run --rm --name ipcevsync --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e
    gcc '$SRC' -o /tmp/ipc_event_sync -lze_loader && echo 'BUILD OK'
    timeout 120 /tmp/ipc_event_sync || echo 'RUN TIMEOUT/FAIL (IPC event open fail or cross-process deadlock?)'
  " 2>&1
echo "=== ipc_event_sync exit $? ==="
