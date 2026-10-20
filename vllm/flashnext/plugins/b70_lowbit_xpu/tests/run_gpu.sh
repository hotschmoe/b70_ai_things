#!/usr/bin/env bash
# Card-0 bench for b70_lowbit. Caller holds the gpu-run --card 0 lease.
# ZE_AFFINITY_MASK=0. LD_LIBRARY_PATH has torch and a libsycl-only dir,
# not the whole venv lib (that directory's OpenCL ICD overrides host UMD).
set -uo pipefail

ROOT="${ROOT:-/mnt/vm_8tb/github/b70_ai_things}"
PLUGIN="$ROOT/vllm/flashnext/plugins/b70_lowbit_xpu"
VENV="${VENV:-/mnt/vm_8tb/b70/build/vllm-flashnext/venv}"
PY="${PY:-/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3}"
OUT="${OUT:-/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu}"
HOSTLIBS="$OUT/hostlibs"
TORCHLIB="$VENV/lib/python3.12/site-packages/torch/lib"
SO="$OUT/libb70_lowbit_xpu.so"

test -x "$PY"
test -f "$SO"
test -f "$TORCHLIB/libtorch_xpu.so"

mkdir -p "$HOSTLIBS"
ln -sfn "$VENV/lib/libsycl.so" "$HOSTLIBS/libsycl.so"
ln -sfn "$VENV/lib/libsycl.so.9" "$HOSTLIBS/libsycl.so.9"
ln -sfn "$VENV/lib/libsycl.so.9.0.0" "$HOSTLIBS/libsycl.so.9.0.0"
# libsycl needs the unified runtime loader. Do not add all of venv/lib:
# that directory's OpenCL ICD overrides the host UMD.
ln -sfn "$VENV/lib/libur_loader.so" "$HOSTLIBS/libur_loader.so"
ln -sfn "$VENV/lib/libur_loader.so.0" "$HOSTLIBS/libur_loader.so.0"
ln -sfn "$VENV/lib/libur_loader.so.0.12.0" "$HOSTLIBS/libur_loader.so.0.12.0"
# Intel compiler runtime that torch._C needs. Not libOpenCL.
for lib in libimf.so libsvml.so libirng.so libintlc.so libintlc.so.5; do
  ln -sfn "$VENV/lib/$lib" "$HOSTLIBS/$lib"
done

SITE="$VENV/lib/python3.12/site-packages"
# Plugin source first so it shadows the non-editable site-packages copy.
export PYTHONPATH="$PLUGIN:$SITE${PYTHONPATH:+:$PYTHONPATH}"
export ZE_AFFINITY_MASK=0
export B70_LOWBIT_SO="$SO"
export B70_LOWBIT_PORT="${B70_LOWBIT_PORT:-29671}"
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR="${SYCL_CACHE_DIR:-/mnt/vm_8tb/b70/build/vllm-flashnext/cache/sycl}"
export TMPDIR="${TMPDIR:-/mnt/vm_8tb/b70/build/vllm-flashnext/tmp}"
# libsycl-only dir first, then torch. Do not prepend venv/lib.
export LD_LIBRARY_PATH="$HOSTLIBS:$TORCHLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$SYCL_CACHE_DIR" "$TMPDIR"
cd "$PLUGIN"

echo "=== xpu-health before ==="
"$ROOT/bin/xpu-health" --card 0
health_before=$?
echo "xpu-health before rc=$health_before"
if [ "$health_before" -eq 1 ]; then
  echo "card 0 wedged before the bench; not launching kernels"
  exit 1
fi

echo "=== bench ==="
"$PY" -u tests/bench_lowbit.py
bench_rc=$?
echo "bench rc=$bench_rc"

echo "=== xpu-health after ==="
"$ROOT/bin/xpu-health" --card 0
health_after=$?
echo "xpu-health after rc=$health_after"

if [ "$health_after" -eq 1 ]; then
  echo "card 0 wedged after the bench"
  exit 1
fi
exit "$bench_rc"
