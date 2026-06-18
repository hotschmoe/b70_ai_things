#!/usr/bin/env bash
# Box reconnaissance for B70 optimization project. Run via: ssh b70 bash -s < recon.sh
set -uo pipefail

sec() { printf '\n===== %s =====\n' "$1"; }

sec "IDENTITY"
whoami; hostname
cat /etc/unraid-version 2>/dev/null || echo "no unraid-version file"

sec "CPU / MEM"
nproc
free -h | head -2

sec "GPU (lspci)"
lspci -nn | grep -iE 'vga|display|3d controller' || echo "no display devices via lspci"

sec "INTEL / ARC specifically"
lspci -nn | grep -iE 'intel|arc|battlemage|bmg' || echo "no intel match"

sec "/dev/dri"
ls -la /dev/dri 2>/dev/null || echo "no /dev/dri"
ls -la /dev/dri/by-path 2>/dev/null || true

sec "i915 / xe DRIVER MODULES"
lsmod | grep -iE 'i915|^xe|xe ' || echo "no i915/xe module loaded"
dmesg 2>/dev/null | grep -iE 'i915|\bxe\b|drm' | tail -20 || echo "(dmesg not readable)"

sec "DOCKER"
docker version --format '{{.Server.Version}}' 2>/dev/null || docker --version || echo "no docker"
docker info 2>/dev/null | grep -iE 'storage driver|docker root dir|runtimes' || true

sec "STORAGE (df)"
df -h | grep -vE 'tmpfs|loop|overlay' | sort -k6

sec "UNRAID MOUNTS (/mnt)"
ls -la /mnt 2>/dev/null
echo "--- disk sizes under /mnt ---"
for d in /mnt/*/; do
  [ -d "$d" ] || continue
  sz=$(df -h "$d" 2>/dev/null | awk 'NR==2{print $2" used:"$3" free:"$4" fs:"$1}')
  echo "$d -> $sz"
done

sec "BLOCK DEVICES (rotational? SSD vs HDD)"
lsblk -o NAME,SIZE,TYPE,ROTA,MOUNTPOINT,MODEL 2>/dev/null || lsblk

sec "DOCKER IMAGES PRESENT"
docker images 2>/dev/null | head -20 || true

sec "INTEL GPU TOOLS"
command -v intel_gpu_top && echo "have intel_gpu_top" || echo "no intel_gpu_top"
command -v clinfo && clinfo 2>/dev/null | grep -iE 'platform name|device name' | head -10 || echo "no clinfo"
command -v xpu-smi && echo "have xpu-smi" || echo "no xpu-smi"

sec "DONE"
