#!/usr/bin/env bash
# Reclaim docker.img space (SAFE items only - do NOT touch user's running-container
# images: nextcloud/mariadb/syncthing/clamav). Then inspect the llm-scaler vLLM image.
set -uo pipefail
IMG="intel/llm-scaler-vllm:0.14.0-b8.3"

echo "=== BEFORE ==="; df -h /var/lib/docker | tail -1
echo "=== prune build cache (safe) ==="
docker builder prune -f 2>&1 | tail -2
echo "=== remove python:3.10 (unused; 3.11 kept for downloads) ==="
docker rmi python:3.10 2>&1 | tail -2 || echo "(python:3.10 not removable/in use)"
echo "=== AFTER ==="; df -h /var/lib/docker | tail -1

echo; echo "=== llm-scaler image interface ==="
docker run --rm --entrypoint bash "$IMG" -c '
  echo "--- vllm version ---"; python -c "import vllm; print(vllm.__version__)" 2>&1 | tail -1
  echo "--- vllm CLI ---"; which vllm; vllm --help 2>&1 | head -3
  echo "--- entrypoint default ---"; echo "$0"
  echo "--- intel int4 kernel present? ---"; find / -name "*int4*multi_arc*" 2>/dev/null | head -3
  echo "--- xpu-smi present? ---"; which xpu-smi 2>/dev/null || echo "no xpu-smi in image"
  echo "--- key envs/tools ---"; python -c "import torch; print(\"torch\", torch.__version__)" 2>&1 | tail -1
' 2>&1 | head -40
echo "=== DONE ==="
