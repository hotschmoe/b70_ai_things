#!/usr/bin/env bash
# Compile libb70_lowbit_xpu.so with oneAPI 2026.1 against torch 2.14+xpu.
# No GPU. The .so stays outside the git tree.
set -euo pipefail

ROOT="${ROOT:-/mnt/vm_8tb/github/b70_ai_things}"
SRC="$ROOT/vllm/flashnext/plugins/b70_lowbit_xpu/csrc"
OUT="${OUT:-/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu}"
VENV="${VENV:-/mnt/vm_8tb/b70/build/vllm-flashnext/venv}"
TORCH="$VENV/lib/python3.12/site-packages/torch"
PYINC="${PYINC:-/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12}"
IMG="${IMG:-b70/flashnext-llamacpp-build:oneapi2026.1-de7fa0a}"

mkdir -p "$OUT"
test -f "$SRC/lowbit_kernels.cpp"
test -f "$TORCH/include/c10/xpu/XPUStream.h"
test -f "$PYINC/Python.h"
echo "build.sh start $(date '+%F %T') src=$SRC"

# AOT for BMG needs llvm-foreach from the compiler tree and the host ocloc
# (26.22). The oneAPI image does not ship a matching ocloc.
OCLIB=/usr/lib/x86_64-linux-gnu
MOUNT_OCL=()
if [ -e /usr/bin/ocloc ]; then
  MOUNT_OCL+=(-v /usr/bin/ocloc:/usr/local/bin/ocloc:ro)
fi
if [ -e "$OCLIB/libocloc.so" ]; then
  MOUNT_OCL+=(-v "$OCLIB/libocloc.so:$OCLIB/libocloc.so:ro")
fi
if [ -e "$OCLIB/libocloc.so.2" ]; then
  MOUNT_OCL+=(-v "$OCLIB/libocloc.so.2:$OCLIB/libocloc.so.2:ro")
fi

docker run --rm -i --network none \
  --user "$(id -u):$(id -g)" \
  --memory 48g --cpus 4 \
  -e RPATH_VENV="$VENV/lib" \
  -e RPATH_TORCH="$TORCH/lib" \
  -v "$SRC":/src:ro \
  -v "$OUT":/work \
  -v "$TORCH":/torch:ro \
  -v "$VENV/lib":/venvlib:ro \
  -v "$PYINC":/pyinc:ro \
  "${MOUNT_OCL[@]}" \
  --entrypoint bash "$IMG" -s <<'EOS'
# setvars exits the shell under `set -u` / `set -e`. Relax those, then restore.
set +e
set +u
export HOME=/work
mkdir -p /work/tmp
export TMPDIR=/work/tmp
source /opt/intel/oneapi/setvars.sh --force > /work/setvars.log 2>&1
echo "setvars_rc=$?"
set -e
set -u
export PATH="/opt/intel/oneapi/compiler/2026.1/bin/compiler:${PATH}"
command -v llvm-foreach
command -v ocloc
ocloc --version | head -1
if ! command -v icpx >/dev/null 2>&1; then
  if [ -x /opt/intel/oneapi/compiler/2026.1/bin/icpx ]; then
    export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$PATH"
  fi
fi
command -v icpx >/dev/null || { echo NO_icpx; exit 2; }
icpx --version | head -1
echo '=================== BUILD libb70_lowbit_xpu.so ==================='
set +e
# per_kernel split: this TU has plain SYCL kernels and ESIMD kernels.
icpx -fsycl -fPIC -shared -std=c++20 -O2 \
  -fsycl-device-code-split=per_kernel \
  -fsycl-targets=intel_gpu_bmg_g31 \
  -I/torch/include \
  -I/torch/include/torch/csrc/api/include \
  -I/pyinc \
  /src/lowbit_kernels.cpp \
  -L/torch/lib -L/venvlib \
  -Wl,-rpath,"$RPATH_VENV" \
  -Wl,-rpath,"$RPATH_TORCH" \
  -Wl,--no-as-needed \
  -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu \
  -o /work/libb70_lowbit_xpu.so \
  2> /work/build_lowbit.err
rc=$?
set -e
echo "compile rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo '--- errors (tail) ---'
  tail -160 /work/build_lowbit.err
  exit "$rc"
fi
ls -l /work/libb70_lowbit_xpu.so
echo '=== BUILD OK ==='
EOS
