#!/usr/bin/env bash
# Build a dedicated Qwen3.6 NVFP4 serve kernel with native E4M3 block scales.
# The production nvfp4_fused_kernel_gdn directory is never modified.
set -euo pipefail

ROOT="${ROOT:-/mnt/vm_8tb/b70}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BASE="${BASE:-vllm-xpu-env:int8g-v0251}"
SRC_BASE="${SRC_BASE:-$ROOT/vllm-xpu-kernels-v0240}"
SRC="$ROOT/vllm-xpu-kernels-nvfp4-f8scale"
OUT="$ROOT/nvfp4_f8scale_kernel_gdn"

docker image inspect "$BASE" >/dev/null 2>&1 || {
  echo "FAIL: base image is missing: $BASE"
  exit 1
}
test -f "$SRC_BASE/csrc/xpu/onednn/nvfp4_gemm_w4a16.h" || {
  echo "FAIL: base NVFP4 source tree is missing: $SRC_BASE"
  exit 1
}

rm -rf "$SRC"
rsync -a --exclude=build --exclude=.deps --exclude='*.so' --exclude=.git \
  "$SRC_BASE/" "$SRC/"
cp "$REPO/kernels/nvfp4_gemm_w4a16.h" \
  "$SRC/csrc/xpu/onednn/nvfp4_gemm_w4a16.h"
patch -p1 -d "$SRC" < "$REPO/kernels/nvfp4_f8scale_integration.patch"

docker rm -f nvfp4_f8scale_gdn_build >/dev/null 2>&1 || true
docker run --name nvfp4_f8scale_gdn_build \
  -v "$SRC:/build/vllm-xpu-kernels" \
  -v "$REPO:/repo:ro" \
  --entrypoint bash "$BASE" \
  -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; bash /repo/vllm/nvfp4/build_nvfp4_f8scale_inside.sh' \
  2>&1 | tee "$ROOT/nvfp4_f8scale_build_gdn.log"
rc=${PIPESTATUS[0]}
docker rm -f nvfp4_f8scale_gdn_build >/dev/null 2>&1 || true
test "$rc" -eq 0 || exit "$rc"

mkdir -p "$OUT"
cp "$SRC/vllm_xpu_kernels/_xpu_C.abi3.so" "$OUT/"
cp "$SRC/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so" "$OUT/"
sha256sum "$OUT/_xpu_C.abi3.so" "$OUT/libgdn_attn_kernels_xe_2.so"
echo "OK: $OUT"
