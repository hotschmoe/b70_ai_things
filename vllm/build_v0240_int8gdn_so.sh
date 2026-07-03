#!/usr/bin/env bash
# Stage 2 of the vLLM v0.24.0 rebase: build ONE combined _xpu_C.abi3.so (+ libgdn_attn_kernels_xe_2.so)
# against the v0240 base image's torch 2.12 ABI, from the patched vllm-xpu-kernels source tree that
# carries our int8 W8A8 ops. Both kernel groups the 27B W8A8 serve path mounts:
#   - XPU_SPECIFIC_KERNELS_ENABLED=ON  -> int8_gemm_w8a8 / int8_gemm_w8a16 / dynamic_per_token_int8_quant (oneDNN)
#   - GDN_KERNELS_ENABLED=ON           -> qwen_gdn_attention_core (+ libgdn_attn_kernels_xe_2.so sidecar, TLA)
# Compile-only, NO GPU. ~20-25 min (GDN TLA compile). Output: /mnt/vm_8tb/b70/w8a8_kernel_v0240/
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
BASE="${BASE:-vllm-xpu-env:v0240}"
SRC_MASTER="$ROOT/vllm-xpu-kernels-w8a8"          # the patched tree (int8 ops applied)
SRC="$ROOT/vllm-xpu-kernels-v0240"                # fresh writable copy (clean cmake cache)
OUT="$ROOT/w8a8_kernel_v0240"
docker image inspect "$BASE" >/dev/null 2>&1 || { echo "FAIL: base $BASE not present (run Stage 1 first)"; exit 1; }

echo "=== fresh copy of patched tree -> $SRC (clean cmake cache) ==="
rm -rf "$SRC"; rsync -a --exclude='build' --exclude='.deps' --exclude='*.so' --exclude='.git' "$SRC_MASTER/" "$SRC/"
# sanity: int8 ops must be in the copy
grep -q 'int8_gemm_w8a16' "$SRC/csrc/xpu/onednn/onednn_matmul.cpp" || { echo "FAIL: int8 patch missing in $SRC"; exit 1; }

cat > "$SRC/_build_int8gdn.sh" <<'BUILD'
set -uo pipefail
cd /build/vllm-xpu-kernels
which icpx cmake ninja >/dev/null || true
python3 -c "import torch;print('torch',torch.__version__)"
pip install --no-build-isolation -q setuptools_scm 2>/dev/null || pip install -q setuptools_scm 2>/dev/null || true
export VLLM_VERSION_OVERRIDE=0.1.10
rm -rf build .deps vllm_xpu_kernels/_xpu_C*.so vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so
# combined scope: int8 (XPU_SPECIFIC) + GDN (needs TLA). Everything heavy else OFF.
export BASIC_KERNELS_ENABLED=OFF FA2_KERNELS_ENABLED=OFF MOE_KERNELS_ENABLED=OFF
export MQA_LOGITS_KERNELS_ENABLED=OFF XPUMEM_ALLOCATOR_ENABLED=OFF
export XPU_SPECIFIC_KERNELS_ENABLED=ON
export GDN_KERNELS_ENABLED=ON BUILD_SYCL_TLA_KERNELS=ON
export VLLM_XPU_AOT_DEVICES=bmg VLLM_XPU_XE2_AOT_DEVICES=bmg
export MAX_JOBS=16 VERBOSE=1 PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/xpu
echo "==== BUILD START $(date -u +%H:%M:%S) ===="
python setup.py build_ext --inplace; RC=$?
echo "==== BUILD END $(date -u +%H:%M:%S) RC=$RC ===="
find . -name '_xpu_C*.so' -o -name 'libgdn_attn_kernels_xe_2.so' | xargs -r ls -la
exit $RC
BUILD

echo "=== build inside $BASE (torch 2.12), tree -> /build/vllm-xpu-kernels ==="
docker rm -f w8a8gdn_v0240 2>/dev/null || true
docker run --name w8a8gdn_v0240 -v "$SRC:/build/vllm-xpu-kernels" --entrypoint bash "$BASE" -c \
  'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; bash /build/vllm-xpu-kernels/_build_int8gdn.sh' 2>&1 | tee "$ROOT/build24/build_int8gdn.log" | tail -30
RC=${PIPESTATUS[0]}
echo "=== docker RC=$RC ==="
mkdir -p "$OUT"
cp -f "$SRC/vllm_xpu_kernels/_xpu_C.abi3.so" "$OUT/" 2>/dev/null && echo "copied _xpu_C.abi3.so" || echo "MISSING _xpu_C.abi3.so"
cp -f "$SRC/vllm_xpu_kernels/libgdn_attn_kernels_xe_2.so" "$OUT/" 2>/dev/null && echo "copied libgdn" || echo "MISSING libgdn"
ls -la "$OUT/"
exit $RC
