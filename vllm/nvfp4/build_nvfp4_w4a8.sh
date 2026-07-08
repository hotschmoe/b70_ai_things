#!/usr/bin/env bash
# Build a dedicated _xpu_C.abi3.so carrying the NEW nvfp4_gemm_w4a8 op (block-scaled
# INT8: s8 src per-token x s8 weight per-16-K-group scale -> INT8 XMX). Isolated source
# tree + output so the DD's nvfp4_fused_kernel_gdn .so is UNTOUCHED.
# GDN OFF (microbench only, ~6-8 min). XPU_SPECIFIC ops only.
# Output: /mnt/vm_8tb/b70/nvfp4pref_kernel/_xpu_C.abi3.so
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
BASE="${BASE:-vllm-xpu-env:v0240}"
SRC="$ROOT/vllm-xpu-kernels-nvfp4pref"
OUT="$ROOT/nvfp4pref_kernel"
docker image inspect "$BASE" >/dev/null 2>&1 || { echo "FAIL: base $BASE missing"; exit 1; }
grep -q 'dnnl_matmul_nvfp4_w4a8' "$SRC/csrc/xpu/onednn/nvfp4_gemm_w4a8.h" || { echo "FAIL: new kernel not in $SRC"; exit 1; }
grep -q 'nvfp4_gemm_w4a8' "$SRC/csrc/xpu/torch_bindings.cpp" || { echo "FAIL: binding not wired in $SRC"; exit 1; }

cat > "$SRC/_build_w4a8.sh" <<'BUILD'
set -uo pipefail
cd /build/vllm-xpu-kernels
pip install --no-build-isolation -q setuptools_scm 2>/dev/null || true
export VLLM_VERSION_OVERRIDE=0.1.10
rm -rf build .deps vllm_xpu_kernels/_xpu_C*.so
export BASIC_KERNELS_ENABLED=OFF FA2_KERNELS_ENABLED=OFF MOE_KERNELS_ENABLED=OFF
export MQA_LOGITS_KERNELS_ENABLED=OFF XPUMEM_ALLOCATOR_ENABLED=OFF
export XPU_SPECIFIC_KERNELS_ENABLED=ON
export GDN_KERNELS_ENABLED=OFF BUILD_SYCL_TLA_KERNELS=OFF
export VLLM_XPU_AOT_DEVICES=bmg VLLM_XPU_XE2_AOT_DEVICES=bmg
export MAX_JOBS=16 PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/xpu
echo "==== BUILD START $(date -u +%H:%M:%S) ===="
python setup.py build_ext --inplace; RC=$?
echo "==== BUILD END $(date -u +%H:%M:%S) RC=$RC ===="
find . -name '_xpu_C*.so' | xargs -r ls -la
exit $RC
BUILD

docker rm -f nvfp4pref_build 2>/dev/null || true
mkdir -p "$ROOT/build24"
docker run --name nvfp4pref_build -v "$SRC:/build/vllm-xpu-kernels" --entrypoint bash "$BASE" -c \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; bash /build/vllm-xpu-kernels/_build_w4a8.sh' 2>&1 \
  | tee "$ROOT/build24/build_nvfp4_w4a8.log" | tail -25
RC=${PIPESTATUS[0]}
docker rm -f nvfp4pref_build >/dev/null 2>&1 || true
mkdir -p "$OUT"
cp -f "$SRC/vllm_xpu_kernels/_xpu_C.abi3.so" "$OUT/" 2>/dev/null && echo "OK -> $OUT/_xpu_C.abi3.so" || { echo "MISSING _xpu_C.abi3.so"; exit 1; }
echo "=== build RC=$RC ==="
