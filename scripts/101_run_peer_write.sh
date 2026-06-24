#!/usr/bin/env bash
# Build + run the SYCL EU peer-write microkernel (101_peer_write_kernel.cpp) and DUMP its Xe ISA.
# Usage: cd /mnt/vm_8tb/github/b70_ai_things && ./bin/gpu-run bash scripts/101_run_peer_write.sh
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
IMG="${IMG:-vllm-xpu-env:v0230}"
SRC="$REPO/scripts/101_peer_write_kernel.cpp"

echo "=== peer_write microkernel :: IMG=$IMG kernel=$(uname -r) ==="
docker rm -f peerwrite 2>/dev/null || true
docker run --rm --name peerwrite \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 8g \
  -e ZE_AFFINITY_MASK=0,1 \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e NEOReadDebugKeys=1 -e EnableCrossDeviceAccess=1 \
  -v "$REPO:$REPO" \
  --entrypoint bash "$IMG" -lc "
    set -e
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    echo '--- build ---'
    icpx -fsycl -O2 '$SRC' -o /tmp/peer_write && echo 'BUILD OK'
    echo '--- run ---'
    /tmp/peer_write
    echo '--- dump Xe ISA (IGC) ---'
    rm -rf /tmp/igc && mkdir -p /tmp/igc
    IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/tmp/igc /tmp/peer_write >/dev/null 2>&1 || true
    echo 'asm files:'; ls /tmp/igc/*.asm 2>/dev/null | head
    f=\$(ls /tmp/igc/*.asm 2>/dev/null | head -1)
    if [ -n \"\$f\" ]; then echo \"=== first 60 lines of \$f ===\"; head -60 \"\$f\";
      echo '=== store/send (peer-write) instrs ==='; grep -nE 'send|store|st_' \"\$f\" | head -20; fi
  " 2>&1
echo "=== peer_write exit $? ==="
