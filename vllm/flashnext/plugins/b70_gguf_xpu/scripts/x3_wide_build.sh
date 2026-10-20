#!/usr/bin/env bash
# Additive C13 wide dequant library. Does not replace the serving binary.
# Invoke under bin/cpu-run; do not nest another cpu-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONEAPI="${ONEAPI:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1}"
ICPX="${ICPX:-$ONEAPI/bin/icpx}"
VENV="${VENV:-/mnt/vm_8tb/b70/build/vllm-flashnext/venv}"
TORCH="$VENV/lib/python3.12/site-packages/torch"
PY="${PY:-/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu}"
TC="${TC:-/mnt/vm_8tb/b70/build/gguf-xpu-toolchain}"
OUT="${B70_PREFILL_OUT:-$ROOT/b70_gguf_xpu/_C.x3wide.so}"
CXXFLAGS_STD=(
    --gcc-toolchain="$TC"
    -isystem "$TC/include/c++/15"
    -isystem "$TC/include/x86_64-linux-gnu/c++/15"
)
echo "x3 wide build: $OUT"
"$ICPX" -fsycl -std=c++20 -O3 -fPIC -fexceptions -shared \
    -fsycl-device-code-split=per_kernel \
    -fsycl-max-parallel-link-jobs=4 \
    "${CXXFLAGS_STD[@]}" \
    -DB70_PREFILL_SMOKE=1 \
    -I"$ROOT/csrc" \
    -I"$TORCH/include" \
    -I"$TORCH/include/torch/csrc/api/include" \
    -I"$PY/include/python3.12" \
    -DTORCH_EXTENSION_NAME=_C \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    "$ROOT/csrc/dequant_wide.cpp" "$ROOT/csrc/x3_wide_bind.cpp" \
    -L"$TORCH/lib" -L"$VENV/lib" \
    -ltorch_xpu -ltorch_cpu -ltorch -lc10 -lc10_xpu -lsycl \
    -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$VENV/lib" -Wl,-rpath,"$ONEAPI/lib" \
    -o "$OUT"
ls -l "$OUT"
echo X3_WIDE_BUILD_OK
