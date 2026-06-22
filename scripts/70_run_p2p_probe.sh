#!/usr/bin/env bash
# Run the B70<->B70 P2P probe in the :int8 image with BOTH cards. Route via gpu-run (flock).
# Usage (host): cd /mnt/vm_8tb/b70 && ./gpu-run bash 70_run_p2p_probe.sh
# Env A/B knobs (pass through): P2PACCESS (CCL_TOPO_P2P_ACCESS), IPCX (CCL_ZE_IPC_EXCHANGE: pidfd|sockets|drmfd)
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8}"
P2PACCESS="${P2PACCESS:-1}"
IPCX="${IPCX:-drmfd}"
SCRIPT="$ROOT/70_xpu_p2p_probe.py"
echo "=== P2P probe :: IMG=$IMG P2PACCESS=$P2PACCESS IPCX=$IPCX ==="
docker rm -f p2pprobe 2>/dev/null || true
docker run --rm --name p2pprobe \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g \
  -e ZE_AFFINITY_MASK=0,1 \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e CCL_TOPO_P2P_ACCESS="$P2PACCESS" \
  -e CCL_ZE_IPC_EXCHANGE="$IPCX" \
  -v "$ROOT:$ROOT" -v "$SCRIPT:$SCRIPT:ro" \
  --entrypoint bash "$IMG" -lc "
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    echo '--- ze_peer present? ---'
    which ze_peer 2>/dev/null || find / -name 'ze_peer*' -type f 2>/dev/null | head -3 || echo 'no ze_peer'
    echo '--- sysfs p2p / iommu hints ---'
    ls -d /sys/kernel/iommu_groups/*/devices/*0a:00* 2>/dev/null | head -2 || true
    echo '--- torch p2p probe ---'
    python3 '$SCRIPT'
  " 2>&1
echo "=== p2p probe exit $? ==="
