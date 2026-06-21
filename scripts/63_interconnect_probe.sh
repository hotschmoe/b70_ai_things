#!/usr/bin/env bash
# Interconnect TRUTH probe for dual B70 -- settles the "Gen1 x1" question with a
# positive H2D number and checks GPU P2P readiness/reality on THIS rig.
# GPU job -> wrap in gpu-run (wait forever for the lease, we are queueing behind a campaign):
#   B70_GPU_LOCK_TIMEOUT=0 ./gpu-run bash 63_interconnect_probe.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
IMG="${IMG:-vllm-xpu-env:v0230}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/results/interconnect_probe_${TS}.txt"
mkdir -p "$ROOT/results"

# ---- host-side context (read-only; the REAL link + P2P readiness) ----
{
echo "######## INTERCONNECT PROBE $TS ########"
echo "== kernel (xe GPU<->GPU P2P needs Linux 7.0+ multi-device SVM) =="
uname -r
echo
echo "== REAL PCIe link = on-card switch UPSTREAM bridge (Intel KB 000094587; endpoint lies) =="
for b in 08:00.0 42:00.0; do
  printf "  %s  " "$b"; lspci -vvv -s "$b" 2>/dev/null | grep -E "LnkSta:" | sed 's/^[[:space:]]*//'
done
echo "== (for contrast) the ARTIFACT nodes that falsely read Gen1 x1 =="
for b in 0a:00.0 44:00.0; do
  printf "  %s  " "$b"; lspci -vvv -s "$b" 2>/dev/null | grep -E "LnkSta:" | sed 's/^[[:space:]]*//'
done
echo
echo "== IOMMU state (cross-die P2P needs IOMMU out of path: amd_iommu=off or iommu=pt) =="
echo "  cmdline: $(cat /proc/cmdline)"
echo "  iommu groups: $(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l) ; AMD-Vi: $(dmesg 2>/dev/null | grep -ci 'AMD-Vi')"
echo
echo "== Resizable BAR / full-VRAM aperture (P2P DMAs into BAR2; want ~32G, not 256M) =="
for d in 0000:0a:00.0 0000:44:00.0; do
  echo "  $d:"; lspci -vv -s "${d#0000:}" 2>/dev/null | grep -E "Region (0|2):|Memory at .* size=" | sed 's/^/    /'
done
echo
echo "######## torch-xpu bandwidth + P2P probe (in $IMG) ########"
} | tee "$OUT"

# ---- the GPU touch: H2D/D2H/D2D bandwidth + can_device_access_peer ----
docker rm -f icprobe 2>/dev/null || true
docker run --rm --name icprobe --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g \
  -v "$ROOT/bw_p2p_probe.py:/bw_p2p_probe.py:ro" \
  --entrypoint bash "$IMG" -lc 'python /bw_p2p_probe.py' 2>&1 | tee -a "$OUT"

echo "######## DONE -> $OUT ########" | tee -a "$OUT"