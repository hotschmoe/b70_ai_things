#!/usr/bin/env bash
# Run the raw Level-Zero P2P ctypes probe (71_ze_p2p_ctypes.py) across a DOZEN env variations in ONE
# container -- find which (if any) settings flip B70<->B70 peer access on. Route via gpu-run (flock).
# Usage (host): cd /mnt/vm_8tb/b70 && ./gpu-run bash 71_run_ze_matrix.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8}"
SCRIPT="$ROOT/71_ze_p2p_ctypes.py"

# the dozen variations: label :: env assignments (newline-separated; ZE_AFFINITY_MASK=0,1 unless overridden)
read -r -d '' VARIANTS <<'EOF'
01-baseline :: ZE_AFFINITY_MASK=0,1
02-debugkeys-crossdev :: ZE_AFFINITY_MASK=0,1 NEOReadDebugKeys=1 EnableCrossDeviceAccess=1
03-debugkeys-enablep2p :: ZE_AFFINITY_MASK=0,1 NEOReadDebugKeys=1 EnableP2P=1
04-hier-FLAT :: ZE_AFFINITY_MASK=0,1 ZE_FLAT_DEVICE_HIERARCHY=FLAT
05-hier-COMPOSITE :: ZE_AFFINITY_MASK=0,1 ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
06-hier-COMBINED :: ZE_AFFINITY_MASK=0,1 ZE_FLAT_DEVICE_HIERARCHY=COMBINED
07-ccl-p2p-on :: ZE_AFFINITY_MASK=0,1 CCL_TOPO_P2P_ACCESS=1
08-l0-v2 :: ZE_AFFINITY_MASK=0,1 SYCL_UR_USE_LEVEL_ZERO_V2=1
09-crossdev-FLAT :: ZE_AFFINITY_MASK=0,1 NEOReadDebugKeys=1 EnableCrossDeviceAccess=1 ZE_FLAT_DEVICE_HIERARCHY=FLAT
10-crossdev-p2p-FLAT :: ZE_AFFINITY_MASK=0,1 NEOReadDebugKeys=1 EnableCrossDeviceAccess=1 EnableP2P=1 ZE_FLAT_DEVICE_HIERARCHY=FLAT
11-no-affinity :: NEOReadDebugKeys=1 EnableCrossDeviceAccess=1
12-noimplicitscaling :: ZE_AFFINITY_MASK=0,1 NEOReadDebugKeys=1 EnableImplicitScaling=0 EnableCrossDeviceAccess=1
EOF

echo "=== ZE P2P matrix :: IMG=$IMG ==="
docker rm -f zematrix 2>/dev/null || true
docker run --rm --name zematrix \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --shm-size 8g \
  -v "$ROOT:$ROOT" -v "$SCRIPT:$SCRIPT:ro" \
  --entrypoint bash "$IMG" -lc "
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    echo '--- libze present ---'; ldconfig -p | grep -i ze_loader || ls -l /usr/lib/x86_64-linux-gnu/libze_loader* 2>/dev/null || true
    while IFS= read -r line; do
      [ -z \"\$line\" ] && continue
      lbl=\${line%% ::*}; envs=\${line#*:: }
      echo; echo '##################################################################'
      echo \"### VARIANT \$lbl  ::  \$envs\"
      echo '##################################################################'
      env -i PATH=\$PATH LD_LIBRARY_PATH=\$LD_LIBRARY_PATH \$envs python3 '$SCRIPT' 2>&1 | sed 's/^/   /'
    done <<'VARS'
$VARIANTS
VARS
  " 2>&1
echo "=== ze matrix exit $? ==="
