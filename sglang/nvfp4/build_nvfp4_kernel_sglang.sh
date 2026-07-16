#!/usr/bin/env bash
# Build the nvfp4_gemm_w4a16 oneDNN op (+ the other XPU_SPECIFIC int8/int4 ops) against
# the SGLANG image's torch 2.12.0+xpu ABI, mirroring sglang/W4A8_BUILD.md but for the NVFP4
# op. Produces a runtime _xpu_C.abi3.so that LOADS into sglang-xpu:{woq,mtp} and registers
# torch.ops._xpu_C.nvfp4_gemm_w4a16 (the bit-exact NVFP4 f4_e2m1 weight-decompression matmul).
#
# NO GPU needed (compile-only). Output (git-ignored runtime artifact, NOT repo content):
#   /mnt/vm_8tb/b70/nvfp4_kernel_sglang/_xpu_C.abi3.so
#
# Source: an ISOLATED copy of vllm-xpu-kernels-v0240 (the tree that already carries the
# nvfp4_gemm_w4a16 op + binding, see vllm/nvfp4/NVFP4_KERNEL_BUILD.md). GDN OFF: sglang
# supplies its own triton GDN/linear-attn backend (unlike vLLM, which needs the vendored
# gdn_attention_core .so), so the sglang serve does NOT mount a GDN sidecar. XPU_SPECIFIC
# ON = the oneDNN int8/int4/nvfp4 gemms. Same minimal-scope env as w4a8_kernel/build_xpu_c.sh.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
BASE="${BASE:-sglang-xpu:woq}"          # torch 2.12.0+xpu, Intel DPC++ 2025.3 toolchain, no GPU
SRC="${SRC:-$ROOT/vllm-xpu-kernels-nvfp4-sglang}"   # isolated copy (see the rsync in NVFP4_PORT.md)
OUT="${OUT:-$ROOT/nvfp4_kernel_sglang}"

docker image inspect "$BASE" >/dev/null 2>&1 || { echo "FAIL: base $BASE missing"; exit 1; }
[ -f "$SRC/csrc/xpu/onednn/nvfp4_gemm_w4a16.h" ] || { echo "FAIL: nvfp4 op header not in $SRC"; exit 1; }
grep -q 'nvfp4_gemm_w4a16' "$SRC/csrc/xpu/torch_bindings.cpp" || { echo "FAIL: nvfp4 binding not wired in $SRC"; exit 1; }

cat > "$SRC/_build_nvfp4_sglang.sh" <<'BUILD'
set -uo pipefail
cd /build/vllm-xpu-kernels
echo "==== ENV PROBE ===="
icpx --version | head -1
python3 -c "import torch;print('torch',torch.__version__)"
pip install --no-build-isolation -q setuptools_scm 2>/dev/null || pip install -q setuptools_scm 2>/dev/null || true
export VLLM_VERSION_OVERRIDE=0.1.10
rm -rf build .deps vllm_xpu_kernels/_xpu_C*.so
# MINIMAL SCOPE: only the XPU_SPECIFIC oneDNN gemms (incl nvfp4_gemm_w4a16). Everything else OFF.
export BUILD_SYCL_TLA_KERNELS=OFF BASIC_KERNELS_ENABLED=OFF FA2_KERNELS_ENABLED=OFF
export MOE_KERNELS_ENABLED=OFF GDN_KERNELS_ENABLED=OFF MQA_LOGITS_KERNELS_ENABLED=OFF
export XPUMEM_ALLOCATOR_ENABLED=OFF
export XPU_SPECIFIC_KERNELS_ENABLED=ON
export VLLM_XPU_AOT_DEVICES=bmg VLLM_XPU_XE2_AOT_DEVICES=bmg
export MAX_JOBS=16 PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/xpu
echo "==== BUILD START $(date -u +%H:%M:%S) ===="
python setup.py build_ext --inplace; RC=$?
echo "==== BUILD END $(date -u +%H:%M:%S) RC=$RC ===="
find . -name '_xpu_C*.so' | xargs -r ls -la
exit $RC
BUILD

mkdir -p "$ROOT/build24"
docker rm -f nvfp4_sglang_build 2>/dev/null || true
docker run --name nvfp4_sglang_build -v "$SRC:/build/vllm-xpu-kernels" --entrypoint bash "$BASE" -c \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; bash /build/vllm-xpu-kernels/_build_nvfp4_sglang.sh' 2>&1 \
  | tee "$ROOT/build24/build_nvfp4_kernel_sglang.log" | tail -25
RC=${PIPESTATUS[0]}
docker rm -f nvfp4_sglang_build >/dev/null 2>&1 || true
mkdir -p "$OUT"
cp -f "$SRC/vllm_xpu_kernels/_xpu_C.abi3.so" "$OUT/" 2>/dev/null \
  && echo "OK -> $OUT/_xpu_C.abi3.so" || { echo "MISSING _xpu_C.abi3.so"; exit 1; }
echo "=== build RC=$RC ==="
