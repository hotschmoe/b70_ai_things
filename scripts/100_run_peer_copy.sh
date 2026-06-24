#!/usr/bin/env bash
# Build + run the raw Level-Zero DIRECT peer-copy benchmark (100_ze_peer_copy.c) on both B70s.
# Reprofile 2026-06-24 (new kernel 7.0 + new BIOS, IOMMU off). Route via gpu-run (locks both cards).
# Usage:  cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run bash scripts/100_run_peer_copy.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
P2PACCESS="${P2PACCESS:-1}"   # CCL not used here, but keep for parity/logging
SRC="$REPO/scripts/100_ze_peer_copy.c"

echo "=== ze_peer_copy :: IMG=$IMG P2PACCESS=$P2PACCESS kernel=$(uname -r) ==="
docker rm -f zepeer 2>/dev/null || true
docker run --rm --name zepeer \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g \
  -e ZE_AFFINITY_MASK=0,1 \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys="${NEOReadDebugKeys:-0}" \
  ${ENABLE_P2P:+-e EnableP2P=1} \
  ${ENABLE_XDEV:+-e EnableCrossDeviceAccess=1} \
  -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e
    echo '--- enum sanity (ze_api.h) ---'
    grep -hE 'ZE_STRUCTURE_TYPE_(COMMAND_QUEUE_GROUP_PROPERTIES|COMMAND_QUEUE_DESC|COMMAND_LIST_DESC|CONTEXT_DESC|DEVICE_MEM_ALLOC_DESC|HOST_MEM_ALLOC_DESC) ' /usr/include/level_zero/ze_api.h | head
    echo '--- build ---'
    gcc '$SRC' -o /tmp/ze_peer_copy -lze_loader -O2 && echo 'BUILD OK'
    echo '--- run ---'
    /tmp/ze_peer_copy
  " 2>&1
echo "=== ze_peer_copy exit $? ==="
