#!/usr/bin/env bash
# Compile libb70_kvpark.so with oneAPI 2026.1 against torch 2.14+xpu.
# No GPU. The .so stays outside the git tree.
set -euo pipefail

ROOT="${ROOT:-/mnt/vm_8tb/github/b70_ai_things}"
SRC="$ROOT/vllm/flashnext/plugins/b70_kvpark/csrc"
OUT="${OUT:-/mnt/vm_8tb/b70/build/vllm-flashnext/b70_kvpark}"
VENV="${VENV:-/mnt/vm_8tb/b70/build/vllm-flashnext/venv}"
TORCH="$VENV/lib/python3.12/site-packages/torch"
PYINC="${PYINC:-/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12}"
IMG="${IMG:-b70/flashnext-llamacpp-build:oneapi2026.1-de7fa0a}"

mkdir -p "$OUT"
test -f "$SRC/copy_batch.cpp"
test -f "$TORCH/include/c10/xpu/XPUStream.h"
test -f "$PYINC/Python.h"
SO_NAME="${SO_NAME:-libb70_kvpark.so}"
echo "build.sh start $(date '+%F %T') src=$SRC so=$SO_NAME"

docker run --rm -i --network none \
  --user "$(id -u):$(id -g)" \
  --memory 48g --cpus 4 \
  -e RPATH_VENV="$VENV/lib" \
  -e RPATH_TORCH="$TORCH/lib" \
  -e SO_NAME="$SO_NAME" \
  -v "$SRC":/src:ro \
  -v "$OUT":/work \
  -v "$TORCH":/torch:ro \
  -v "$VENV/lib":/venvlib:ro \
  -v "$PYINC":/pyinc:ro \
  --entrypoint bash "$IMG" -s <<'EOS'
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
if ! command -v icpx >/dev/null 2>&1; then
  if [ -x /opt/intel/oneapi/compiler/2026.1/bin/icpx ]; then
    export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$PATH"
  fi
fi
command -v icpx >/dev/null || { echo NO_icpx; exit 2; }
icpx --version | head -1
echo "=================== BUILD ${SO_NAME} ==================="
set +e
icpx -fsycl -fPIC -shared -std=c++20 -O2 \
  -I/torch/include \
  -I/torch/include/torch/csrc/api/include \
  -I/pyinc \
  /src/copy_batch.cpp \
  -L/torch/lib -L/venvlib \
  -Wl,-rpath,"$RPATH_VENV" \
  -Wl,-rpath,"$RPATH_TORCH" \
  -Wl,--no-as-needed \
  -ltorch -ltorch_cpu -ltorch_xpu -lc10 -lc10_xpu \
  -o "/work/${SO_NAME}" \
  2> /work/build_kvpark.err
rc=$?
set -e
echo "compile rc=$rc"
if [ "$rc" -ne 0 ]; then
  echo '--- errors (tail) ---'
  tail -160 /work/build_kvpark.err
  exit "$rc"
fi
ls -l "/work/${SO_NAME}"
echo '=== BUILD OK ==='
EOS
