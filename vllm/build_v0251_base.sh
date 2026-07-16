#!/usr/bin/env bash
# Build the vLLM v0.25.1 (torch 2.12) XPU base image from the upstream tag, canonical
# multi-stage docker/Dockerfile.xpu (rust-build + vllm-base + ucx-nixl + vllm-openai).
# Compile-only, NO GPU lease needed -- runs alongside the daily driver.
#
#   Stage 1 of the vLLM v0.24.0->v0.25.1 upgrade (2026-07-16, user request "newest vllm").
#   KEY: v0.25.1 still pins torch==2.12.0 (requirements/xpu.txt) == SAME as v0.24.0, so
#   NO torch ABI bump -- the custom .so (int8 gemm, nvfp4 gemm, GDN) should load without a
#   rebuild; the only drift risk is vLLM-internal API (0.24->0.25) affecting mount patches.
#
# Result tag: vllm-xpu-env:v0251
set -uo pipefail
SRC=/mnt/vm_8tb/b70/build24/vllm
TAG=vllm-xpu-env:v0251
LOG=/mnt/vm_8tb/b70/build24/build_v0251.log
cd "$SRC" || { echo "no src $SRC"; exit 1; }
echo "=== vLLM $(git describe --tags 2>/dev/null) -> $TAG ==="
echo "=== docker build -f docker/Dockerfile.xpu (LONG: base pull + rust + ucx/nixl + vllm compile) ==="
date
time docker build -f docker/Dockerfile.xpu -t "$TAG" --shm-size=8g . 2>&1 | tee "$LOG" | tail -1
rc=${PIPESTATUS[0]}
echo "=== build rc=$rc ==="
docker images "$TAG" --format '{{.Repository}}:{{.Tag}} {{.Size}} {{.CreatedAt}}'
echo "=== tail of log ==="
tail -25 "$LOG"
exit $rc
