#!/usr/bin/env bash
# Create the project working tree on the 8TB VM SSD. All heavy data lives here
# (bind-mounted into containers), never inside the 50GB docker.img.
set -euo pipefail
ROOT=/mnt/vm_8tb/b70

mkdir -p "$ROOT"/{models,hf_cache,results,logs,scripts,docker}
# models/: GGUF + safetensors weights. hf_cache/: HF_HOME for downloads.
# results/: benchmark json/csv. logs/: container logs. docker/: compose files on-box.

echo "===== tree under $ROOT ====="
ls -la "$ROOT"
echo "===== free space ====="
df -h "$ROOT" | tail -1
echo "===== write+read speed sanity (1GB, on SSD) ====="
dd if=/dev/zero of="$ROOT/.ddtest" bs=1M count=1024 oflag=direct 2>&1 | tail -1 || \
  dd if=/dev/zero of="$ROOT/.ddtest" bs=1M count=1024 conv=fdatasync 2>&1 | tail -1
sync
rm -f "$ROOT/.ddtest"
echo "===== DONE ====="
