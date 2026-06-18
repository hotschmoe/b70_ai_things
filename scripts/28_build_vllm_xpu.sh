#!/usr/bin/env bash
# Build upstream vLLM-XPU from source per Intel's "Run Gemma 4 on Arc" guide.
# Image: vllm-xpu-env (commit 3ca6ca2). Clone on SSD; build cache in docker.img (131G free).
set -uo pipefail
BUILD=/mnt/vm_8tb/b70/build
mkdir -p "$BUILD"
cd "$BUILD"

if [ ! -d vllm/.git ]; then
  echo "=== cloning vllm ==="
  git clone https://github.com/vllm-project/vllm.git
fi
cd vllm
echo "=== checkout c51df4300 (user-proven version: vllm 0.20.2rc1.dev2, NATIVE gemma4) ==="
git fetch --all -q || true
git checkout c51df4300 2>&1 | tail -3
git log -1 --oneline

echo "=== docker build -f docker/Dockerfile.xpu -t vllm-xpu-env (LONG: 30-60min) ==="
ls docker/Dockerfile.xpu || { echo "Dockerfile.xpu not found at this commit"; ls docker/ | head; exit 1; }
time docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g . 2>&1 | tail -60

echo "=== result ==="
docker images vllm-xpu-env --format '{{.Repository}}:{{.Tag}} {{.Size}}'
df -h /var/lib/docker | tail -1
echo "=== DONE ==="
