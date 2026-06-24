#!/usr/bin/env bash
# Build + run the 2-process IPC push exchange (103_ipc_push_allreduce.c). Proves the cross-process
# Level-Zero IPC peer-write transport that a real vLLM TP worker pair needs. Compare push GB/s vs the
# single-context J.7 (102: ~10.6 GB/s) and oneCCL (H.12: 9.7 GB/s).
# Usage: ./bin/gpu-run bash scripts/103_run_ipc_push_allreduce.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/103_ipc_push_allreduce.c"
echo "=== ipc_push_allreduce :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f ipcpushar 2>/dev/null || true
docker run --rm --name ipcpushar --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g -e ZE_AFFINITY_MASK=0,1 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    gcc -O2 '$SRC' -o /tmp/ipc_push_allreduce -lze_loader && echo 'BUILD OK'
    /tmp/ipc_push_allreduce
  " 2>&1
echo "=== ipc_push_allreduce exit $? ==="
