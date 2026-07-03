#!/usr/bin/env bash
# Build the vLLM v0.24.0 (torch 2.12) XPU base image from the upstream tag, canonical
# multi-stage docker/Dockerfile.xpu (rust-build + vllm-base + ucx-nixl + vllm-openai).
# Compile-only, NO GPU lease needed -- runs alongside the daily driver.
#
#   Stage 1 of the vLLM v0.23.0->v0.24.0 rebase (docs/20260702_vllm_v0230_rebase_plan.md,
#   re-targeted to v0.24.0 per user 2026-07-03). torch 2.11->2.12 ABI bump: every custom
#   .so (int8 gemm, GDN) must rebuild against this base (Stage 2, separate).
#
#   Provenance (Stage 0, DONE 2026-07-03): current :v0230/:int8g ARE genuinely vLLM
#   0.23.0 / torch 2.11.0+xpu; xpu-kernels clone already at v0.1.10 tag.
#
# Result tag: vllm-xpu-env:v0240   (immutable dated alias recorded in JOURNAL)
set -uo pipefail
SRC=/mnt/vm_8tb/b70/build24/vllm
TAG=vllm-xpu-env:v0240
LOG=/mnt/vm_8tb/b70/build24/build_v0240.log
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
