#!/usr/bin/env bash
# Build a dedicated _xpu_C.abi3.so carrying the NVFP4 K-group fix to int8_gemm_w8a16
# (is_block_quant now uses {grp_k, grp_n} inferred from the scale shape, so a
# [K/16, N] NVFP4 weight scale groups {16,1} = K-grouped, per-N-channel).
# int8 ops ONLY (XPU_SPECIFIC) -- NO GDN/TLA -> ~5-8 min compile, not 25.
# Output: /mnt/vm_8tb/b70/nvfp4_kernel/_xpu_C.abi3.so  (SEPARATE from w8a8_kernel_v0240,
# so the daily driver's kernel is untouched.)
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
BASE="${BASE:-vllm-xpu-env:v0240}"
SRC="$ROOT/vllm-xpu-kernels-nvfp4"     # patched tree (K-group fix already applied)
OUT="$ROOT/nvfp4_kernel"
docker image inspect "$BASE" >/dev/null 2>&1 || { echo "FAIL: base $BASE missing"; exit 1; }
grep -q 'grp_k, grp_n' "$SRC/csrc/xpu/onednn/int8_gemm_w8a16.h" || { echo "FAIL: K-group fix not in $SRC"; exit 1; }

cat > "$SRC/_build_nvfp4.sh" <<'BUILD'
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

docker rm -f nvfp4_build 2>/dev/null || true
docker run --name nvfp4_build -v "$SRC:/build/vllm-xpu-kernels" --entrypoint bash "$BASE" -c \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; bash /build/vllm-xpu-kernels/_build_nvfp4.sh' 2>&1 \
  | tee "$ROOT/build24/build_nvfp4_kernel.log" | tail -20
RC=${PIPESTATUS[0]}
docker rm -f nvfp4_build >/dev/null 2>&1 || true
mkdir -p "$OUT"
cp -f "$SRC/vllm_xpu_kernels/_xpu_C.abi3.so" "$OUT/" 2>/dev/null && echo "OK -> $OUT/_xpu_C.abi3.so" || { echo "MISSING _xpu_C.abi3.so"; exit 1; }
echo "=== build RC=$RC ==="
