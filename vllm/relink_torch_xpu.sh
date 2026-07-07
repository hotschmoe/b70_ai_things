#!/usr/bin/env bash
# Incremental: recompile the one changed file (XPUGraph.cpp) + relink libtorch_xpu.so, reusing the
# existing build tree (do NOT rm -rf / reconfigure). Runs inside a container off the image.
set -o pipefail
BUILD=/work/build-torch-xpu
[ -d "$BUILD" ] || { echo "no build dir -- run the full build first"; exit 1; }
echo "===== RELINK START $(date -u +%FT%TZ) ====="

# capture oneAPI env (setvars returns at top level)
while IFS='=' read -r k v; do
  case "$k" in
    PATH|LD_LIBRARY_PATH|CPATH|LIBRARY_PATH|CMAKE_PREFIX_PATH|PKG_CONFIG_PATH|CMPLR_ROOT|MKLROOT|MKL_ROOT|\
TBBROOT|CCL_ROOT|OCL_ICD_FILENAMES|ONEAPI_ROOT|SYCL_*|UR_*|ZE_*|INTEL_*) export "$k=$v" ;;
  esac
done < <(bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; env')
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"
if [ -f /opt/intel/oneapi/ccl/2021.17/lib/libccl.so.2.0 ]; then export CCL_ROOT=/opt/intel/oneapi/ccl/2021.17; fi
export LD_LIBRARY_PATH="$CCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
git config --global --add safe.directory '*' 2>/dev/null || true
# ensure cmake 3.31 (build tree was configured with it)
/opt/venv/bin/python -m pip install --quiet "cmake==3.31.*" 2>&1 | tail -1 || true
hash -r

echo "--- confirm patch (execute_graph) ---"
grep -n 'execute_graph(queue' /work/pytorch/aten/src/ATen/xpu/XPUGraph.cpp || { echo "patch missing"; exit 2; }

echo "--- ninja incremental torch_xpu ($(date -u +%FT%TZ)) ---"
cmake --build "$BUILD" --target torch_xpu --parallel "${MAX_JOBS:-28}"
rc=$?; echo "relink rc=$rc ($(date -u +%FT%TZ))"
[ $rc -ne 0 ] && { echo "RELINK FAILED"; exit 4; }
ls -la "$BUILD/lib/libtorch_xpu.so" || { echo "NO .so"; exit 5; }
echo "===== RELINK DONE $(date -u +%FT%TZ) ====="
