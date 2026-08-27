#!/usr/bin/env bash
# Prepare and build the Qwen3.8 NVFP4 operator against the vLLM 0.28 torch ABI.
# Run this through bin/gpu-run because SYCL AOT compilation is a leased XPU
# build workload even though the container does not open a device.
set -euo pipefail

ROOT="${ROOT:-/mnt/vm_8tb/b70}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SOURCE_REPO="${SOURCE_REPO:-$ROOT/steve-s2b/vllm-xpu-kernels}"
SOURCE_COMMIT="${SOURCE_COMMIT:-a397c58eb7781e6fe0d6b3fb7c25d21b5f658784}"
SRC="${SRC:-$ROOT/vllm-xpu-kernels-nvfp4-v028-build}"
OUT="${OUT:-$ROOT/nvfp4_kernel_v028}"
BUILD_IMAGE="${BUILD_IMAGE:-b70-sglang-xpu:20260826-bede6bc-2d10888-torch213-umd2622}"

case "$SRC" in
  "$ROOT"/vllm-xpu-kernels-nvfp4-*) ;;
  *) echo "FAIL: SRC must be a dedicated NVFP4 tree under $ROOT" >&2; exit 1 ;;
esac
case "$OUT" in
  "$ROOT"/nvfp4_*) ;;
  *) echo "FAIL: OUT must be a dedicated NVFP4 directory under $ROOT" >&2; exit 1 ;;
esac

test -d "$SOURCE_REPO/.git" || {
  echo "FAIL: missing retained source repository: $SOURCE_REPO" >&2
  exit 1
}
docker image inspect "$BUILD_IMAGE" >/dev/null 2>&1 || {
  echo "FAIL: missing build image: $BUILD_IMAGE" >&2
  exit 1
}
test ! -e "$SRC" || {
  echo "FAIL: dedicated source path already exists: $SRC" >&2
  exit 1
}
test ! -e "$OUT" || {
  echo "FAIL: output path already exists: $OUT" >&2
  exit 1
}

git clone --no-checkout "$SOURCE_REPO" "$SRC"
git -C "$SRC" fetch origin refs/remotes/upstream/main
git -C "$SRC" checkout --detach "$SOURCE_COMMIT"
cp "$REPO/kernels/nvfp4_gemm_w4a16.h" \
  "$SRC/csrc/xpu/onednn/nvfp4_gemm_w4a16.h"
git -C "$SRC" apply "$REPO/kernels/nvfp4_v028_integration.patch"
git -C "$SRC" diff --check

docker run --rm \
  -v "$SRC:/build/vllm-xpu-kernels" \
  -v "$REPO:/repo:ro" \
  --entrypoint bash "$BUILD_IMAGE" \
  /repo/vllm/nvfp4/build_nvfp4_v028_inside.sh

mkdir "$OUT"
cp "$SRC/vllm_xpu_kernels/_xpu_C.abi3.so" "$OUT/"
cp "$SRC/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so" "$OUT/"
sha256sum "$OUT/_xpu_C.abi3.so" "$OUT/libgdn_attn_kernels_xe_2.so"
