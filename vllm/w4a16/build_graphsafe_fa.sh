#!/usr/bin/env bash
# LOOP 47: Steve graph-safe FA vs 2dd55f38 (Qwen3.8 head256 focused).
# CPU only. Skip completion-barrier (already in 2dd55f38). --full so.
set -euo pipefail
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
SRC="${SRC:-$ROOT/steve-s2b/vllm-xpu-kernels}"
FA="${FA:-$ROOT/b70-optimization-lab-main/experiments/qwen27_graphsafe_flash_attention}"
OUT="${OUT:-$ROOT/steve-s2b/fa-graphsafe}"
IMG="${IMG:-intel/vllm:0.21.0-xpu}"
NAME="${NAME:-loop47_fabuild}"
JOBS="${JOBS:-2}"
LOG="${LOG:-$ROOT/qwen38-w8a8-dspark/loop47_fabuild.log}"
mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "=== LOOP 47 FA build $(date -u +%Y-%m-%dT%H%M%SZ) ==="
test -f "$SRC/csrc/xpu/attn/xe_2/chunk_prefill.hpp"
test -f "$FA/qwen27-chunk-prefill-local-accessor.patch"
echo "kernels=$(git -C "$SRC" rev-parse HEAD)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
# No GPU. AGASYNC stays up.
docker run -d --name "$NAME" --entrypoint bash \
  -v "$SRC:/src:ro" \
  -v "$FA:/fa:ro" \
  -v "$OUT:/out" \
  -e JOBS="$JOBS" \
  "$IMG" -lc 'sleep 7200'
docker exec "$NAME" bash -lc '
set -euo pipefail
JOBS="${JOBS:-2}"
stage=/tmp/fa-src
build=/tmp/fa-build
rm -rf "$stage" "$build"
mkdir -p "$stage" "$build"
tar -C /src --exclude .git --exclude build --exclude ".deps" -cf - . | tar -C "$stage" -xf -
patch -d "$stage" -p1 < /fa/qwen27-chunk-prefill-local-accessor.patch
patch -d "$stage" -p1 < /fa/qwen27-force-chunk-decode.patch
patch -d "$stage" -p1 < /fa/qwen27-force-chunk-decode-python.patch
set +u
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh >/dev/null
set -u
cmake -S "$stage" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$stage/cmake/toolchain.cmake" \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2025.3/bin/icx \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2025.3/bin/icpx \
  -DVLLM_PYTHON_EXECUTABLE=/opt/venv/bin/python3 \
  -DVLLM_TARGET_DEVICE=xpu \
  -DBUILD_SYCL_TLA_KERNELS=ON \
  -DVLLM_XPU_ENABLE_XE2=ON \
  -DVLLM_XPU_ENABLE_XE_DEFAULT=OFF \
  -DVLLM_CHUNK_PREFILL_CONFIG=/fa/qwen38-head256-chunk.conf \
  -DVLLM_PAGED_DECODE_CONFIG=/fa/qwen38-head256-paged.conf \
  -DBASIC_KERNELS_ENABLED=OFF \
  -DFA2_KERNELS_ENABLED=ON \
  -DMOE_KERNELS_ENABLED=OFF \
  -DGDN_KERNELS_ENABLED=OFF \
  -DMQA_LOGITS_KERNELS_ENABLED=OFF \
  -DXPU_SPECIFIC_KERNELS_ENABLED=OFF \
  -DXPUMEM_ALLOCATOR_ENABLED=OFF
cmake --build "$build" --target attn_kernels_xe_2 _vllm_fa2_C --parallel "$JOBS"
mkdir -p /out/vllm_xpu_kernels
shopt -s nullglob
ext=("$build"/_vllm_fa2_C*.so)
test "${#ext[@]}" -eq 1
install -m 0755 "${ext[0]}" /out/vllm_xpu_kernels/
install -m 0755 "$build/libattn_kernels_xe_2.so" /out/vllm_xpu_kernels/
install -m 0644 "$stage/vllm_xpu_kernels/flash_attn_interface.py" /out/vllm_xpu_kernels/
sha256sum /out/vllm_xpu_kernels/*
ls -la /out/vllm_xpu_kernels/
echo FA_BUILD_OK
'
echo "=== FA build done $(date -u +%Y-%m-%dT%H%M%SZ) ==="
docker rm -f "$NAME" >/dev/null 2>&1 || true
