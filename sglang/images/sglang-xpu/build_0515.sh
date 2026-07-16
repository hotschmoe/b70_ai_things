#!/usr/bin/env bash
# Build the SGLang Intel XPU base image pinned to the v0.5.15.post1 release (2026-07-14).
#
# Same recipe as build.sh but with SG_LANG_BRANCH pinned to the tag instead of moving 'main'.
# Delta vs the historical 0.5.6 base (SHA 09ca4fc):
#   - sglang cloned at tag v0.5.15.post1 (was floating main).
#   - pyproject_xpu.toml pins transformers 5.8.1 -> 5.12.1 and mistral_common 1.11.0 -> 1.11.5
#     (resolved automatically by the in-tree pyproject; torch/torchao/torchvision/torchaudio +xpu
#      and xgrammar 0.1.33 pins are UNCHANGED).
# GPU-FREE build (driver+torch+kernel compile only; no device needed) -> no gpu-run lease required.
#   bash build_0515.sh            # build -> tag sglang-xpu:bmg-0515
# See ../../SGLANG_0515_UPGRADE.md.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TAG="${TAG:-sglang-xpu:bmg-0515}"
export SG_LANG_BRANCH="${SG_LANG_BRANCH:-v0.5.15.post1}"
export LOG="${LOG:-$SCRIPT_DIR/build_0515.log}"
# SAFETY (2026-07-16): bound the sgl-kernel SYCL AOT compile (it thrashed the box for 8h at default
# parallelism). MAX_JOBS=4 caps peak RAM; BUILD_MEM + classic builder (DOCKER_BUILDKIT=0) hard-cap
# build RAM so an overrun OOM-kills the step. RUN WITH THE DAILY DRIVER DOWN (frees ~69 GB host RAM).
export MAX_JOBS="${MAX_JOBS:-4}"
export BUILD_MEM="${BUILD_MEM:-80g}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}"
exec bash "$SCRIPT_DIR/build.sh"
