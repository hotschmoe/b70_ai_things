#!/usr/bin/env bash
# Build ONLY libtorch_xpu.so from the patched pytorch source (submit_without_event fix),
# ABI-matched to the prebuilt torch 2.12.0+xpu. Runs INSIDE a container off
# vllm-xpu-env:int8g-v0240. NOTE: setvars.sh 'return's at -c top level (acts like exit),
# so we capture its env from a subshell instead of sourcing inline.
set -o pipefail
SRC=/work/pytorch
BUILD=/work/build-torch-xpu
PY=/opt/venv/bin/python
echo "===== BUILD START $(date -u +%FT%TZ) ====="

cd "$SRC" || { echo "no source"; exit 2; }
grep -q 'submit_without_event' aten/src/ATen/xpu/XPUGraph.cpp && echo "PATCH PRESENT" || { echo "PATCH MISSING"; exit 2; }

echo "--- capturing oneAPI env (subshell; setvars returns at top level) ---"
while IFS='=' read -r k v; do
  case "$k" in
    PATH|LD_LIBRARY_PATH|CPATH|LIBRARY_PATH|CMAKE_PREFIX_PATH|PKG_CONFIG_PATH|CMPLR_ROOT|\
MKLROOT|MKL_ROOT|TBBROOT|CCL_ROOT|CCL_CONFIGURATION|FI_PROVIDER_PATH|OCL_ICD_FILENAMES|\
ONEAPI_ROOT|DPL_ROOT|NLSPATH|DIAGUTIL_PATH|INTELFPGAOCLSDKROOT|SETVARS_COMPLETED|\
GDB_INFO|CMAKE_PREFIX_PATH|SYCL_*|UR_*|ZE_*|INTEL_*)
      export "$k=$v" ;;
  esac
done < <(bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; env')

export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
export CMPLR_ROOT=/opt/intel/oneapi/compiler/2025.3
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-28}"

# git safe.directory (root container, hotschmoe-owned mount)
git config --global --add safe.directory '*' 2>/dev/null || true

# The image ships cmake 4.3.4 which REMOVED compat with cmake_minimum_required<3.5 (breaks
# pytorch 2.12's old FP16/NNPACK/XNNPACK/qnnpack subdirs) and the single-arg FetchContent_Populate
# used by torch-xpu-ops. Downgrade to cmake 3.31 (still supports the old projects).
echo "--- installing cmake 3.31 (image has $(cmake --version|head -1)) ---"
$PY -m pip install --quiet "cmake==3.31.*" 2>&1 | tail -2 || true
hash -r
echo "icpx: $(command -v icpx)  g++-13: $(command -v g++-13)  cmake: $(command -v cmake) $(cmake --version|head -1)"
icpx --version | head -1
echo "CMPLR_ROOT=$CMPLR_ROOT ; MKLROOT=${MKLROOT:-unset}"
rm -rf "$BUILD"   # fresh configure

echo "--- configure ($(date -u +%FT%TZ)) ---"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DPython_EXECUTABLE="$PY" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
  -DGLIBCXX_USE_CXX11_ABI=1 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF -DBUILD_BINARY=OFF -DINSTALL_TEST=OFF \
  -DUSE_XPU=ON -DUSE_CUDA=OFF -DUSE_ROCM=OFF \
  -DUSE_FLASH_ATTENTION=OFF -DUSE_MEM_EFF_ATTENTION=OFF \
  -DUSE_XNNPACK=OFF -DUSE_QNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DUSE_NNPACK=OFF -DUSE_FBGEMM=OFF \
  -DUSE_MKL=ON -DBLAS=MKL -DUSE_MKLDNN=ON \
  -DUSE_KINETO=ON -DUSE_DISTRIBUTED=ON -DUSE_XCCL=ON
rc=$?; echo "configure rc=$rc ($(date -u +%FT%TZ))"; [ $rc -ne 0 ] && { echo "CONFIGURE FAILED"; exit 3; }

echo "--- build torch_xpu (MAX_JOBS=$MAX_JOBS) ($(date -u +%FT%TZ)) ---"
cmake --build "$BUILD" --target torch_xpu --parallel "$MAX_JOBS"
rc=$?; echo "build rc=$rc ($(date -u +%FT%TZ))"
if [ $rc -ne 0 ]; then echo "PARALLEL BUILD FAILED; serial retry for clean error"; cmake --build "$BUILD" --target torch_xpu --parallel 1; exit 4; fi

ls -la "$BUILD/lib/libtorch_xpu.so" || { echo "NO OUTPUT .so"; exit 5; }
readelf -d "$BUILD/lib/libtorch_xpu.so" | egrep 'SONAME|NEEDED|RPATH|RUNPATH' | sort
echo "===== BUILD DONE $(date -u +%FT%TZ) ====="
