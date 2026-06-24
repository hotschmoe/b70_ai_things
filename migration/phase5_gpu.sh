#!/usr/bin/env bash
# Phase 5 -- Intel B70 GPU userspace + group access.
# Run with: ! sudo bash /home/hotschmoe/phase5_gpu.sh
# xe kernel driver is already up on both B70s (0b:00.0 and 44:00.0). This adds
# the userspace runtimes (OpenCL + Level Zero) and grants the login user device access.
set -euo pipefail

USER_NAME="${SUDO_USER:-hotschmoe}"

echo "==> apt update"
apt-get update -y

echo "==> Installing Intel GPU userspace (26.04 archive packages)"
# intel-opencl-icd  : OpenCL compute runtime (NEO)
# libze1            : oneAPI Level Zero loader (was 'level-zero')
# libze-intel-gpu1  : Intel L0 GPU driver (was 'intel-level-zero-gpu')
# clinfo            : OpenCL device lister (verification)
# intel-gpu-tools   : intel_gpu_top etc.
apt-get install -y \
  intel-opencl-icd \
  libze1 \
  libze-intel-gpu1 \
  clinfo \
  intel-gpu-tools

echo "==> Adding $USER_NAME to render + video groups"
usermod -aG render,video "$USER_NAME"
echo "  groups for $USER_NAME now: $(id -nG "$USER_NAME")"
echo "  NOTE: group change takes effect on next login (or 'newgrp render' in a shell)."

echo; echo "==== clinfo: number of OpenCL platforms/devices (run as root, sees devices now) ===="
clinfo -l || true

echo; echo "==== Level Zero driver files installed ===="
ls -1 /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so* 2>/dev/null || echo "  (libze_intel_gpu not where expected -- check dpkg -L libze-intel-gpu1)"

echo; echo "==== /dev/dri ===="
ls -la /dev/dri/

echo; echo "Phase 5 GPU userspace done."
echo "Expect TWO Intel devices in 'clinfo -l' (Battlemage card0 + card1)."
echo "After you re-login, 'clinfo -l' as $USER_NAME (no sudo) should list both."
