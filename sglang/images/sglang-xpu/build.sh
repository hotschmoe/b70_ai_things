#!/usr/bin/env bash
# Build the official SGLang Intel XPU image for the dual Arc B70 (Battlemage/Xe2) box.
#
# Source: github.com/sgl-project/sglang docker/xpu.Dockerfile (saved verbatim alongside).
# Base:   intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04 (oneAPI 2025.3.2).
# Driver: ppa:kobuk-team/intel-graphics (consumer Arc/Battlemage UMD).
# torch:  2.12.0+xpu ; sgl-kernel built from sgl-kernel-xpu (DPCPP_SYCL_TARGET=bmg => Battlemage AOT).
# Models: registry carries qwen3_5.py / qwen3_5_mtp.py / qwen3_next.py (our GDN class).
#
# Captured upstream SHAs at first build (2026-06-27):
#   sglang main         = 09ca4fc96b3c
#   sgl-kernel-xpu main = 6cd2a07bef5b
#
# GPU-FREE build (driver+torch+kernel compile only; no device needed) -> no gpu-run lease required.
#   bash build.sh            # build -> tag sglang-xpu:bmg
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${TAG:-sglang-xpu:bmg}"
LOG="${LOG:-$SCRIPT_DIR/build.log}"

echo "=== docker build $TAG (ctx=$SCRIPT_DIR, log=$LOG) ===" | tee "$LOG"
docker build \
  -t "$TAG" \
  -f "$SCRIPT_DIR/xpu.Dockerfile" \
  --build-arg http_proxy="${http_proxy:-}" \
  --build-arg https_proxy="${https_proxy:-}" \
  --build-arg no_proxy="${no_proxy:-}" \
  "$SCRIPT_DIR" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "=== build exit=$rc ===" | tee -a "$LOG"
exit "$rc"
