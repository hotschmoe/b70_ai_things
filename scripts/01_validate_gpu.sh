#!/usr/bin/env bash
# Gate test: does the Arc Pro B70 pass into a container and get seen by the
# Intel compute runtime (Level-Zero / SYCL)? The llama.cpp:full-intel image uses
# a wrapper entrypoint, so override it with --entrypoint to run real tools.
set -uo pipefail
IMG="ghcr.io/ggml-org/llama.cpp:full-intel"

echo "===== sycl-ls (SYCL/Level-Zero device enumeration) ====="
docker run --rm --device /dev/dri --entrypoint sycl-ls "$IMG" 2>&1 || echo "[sycl-ls failed]"

echo "===== ONEAPI_DEVICE_SELECTOR=level_zero:* sycl-ls ====="
docker run --rm --device /dev/dri -e ONEAPI_DEVICE_SELECTOR='level_zero:*' --entrypoint sycl-ls "$IMG" 2>&1 || echo "[failed]"

echo "===== clinfo (OpenCL backend) ====="
docker run --rm --device /dev/dri --entrypoint bash "$IMG" -c 'clinfo 2>/dev/null | grep -iE "platform name|device name|board name|global memory size|driver version" | head -30 || echo "clinfo not present"'

echo "===== llama-bench --version + listed devices ====="
docker run --rm --device /dev/dri --entrypoint llama-bench "$IMG" --version 2>&1 | head -10 || echo "[no --version]"

echo "===== DONE ====="
