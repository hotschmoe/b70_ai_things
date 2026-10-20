#!/usr/bin/env bash
# Build libdequant_cpu.so (no GPU) and the SYCL torch extension _C.so.
# The SYCL link must run under bin/cpu-run. This script itself is the command
# cpu-run invokes; do not nest another cpu-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONEAPI="${ONEAPI:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1}"
ICPX="${ICPX:-$ONEAPI/bin/icpx}"
VENV="${VENV:-/mnt/vm_8tb/b70/build/vllm-flashnext/venv}"
TORCH="$VENV/lib/python3.12/site-packages/torch"
PY="${PY:-/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu}"
# Host has libstdc++.so.6 (GCC 15) but no g++ headers and no sudo. Headers are
# copied from strata-sycl-dev:020 (also GCC 15). See reports/1A2_gguf.md.
# Host has libstdc++.so.6 (GCC 15) but no g++ and no sudo. Headers, crt, and
# the libstdc++ linker input are copied from strata-sycl-dev:020 under $TC.
TC="${TC:-/mnt/vm_8tb/b70/build/gguf-xpu-toolchain}"
CXXFLAGS_STD=(
    --gcc-toolchain="$TC"
    -isystem "$TC/include/c++/15"
    -isystem "$TC/include/x86_64-linux-gnu/c++/15"
)
OUT="$ROOT/b70_gguf_xpu"
mkdir -p "$OUT"

echo "build: host dequant"
"$ICPX" -std=c++20 -O2 -fPIC -shared "${CXXFLAGS_STD[@]}" \
    "$ROOT/csrc/host_dequant.cpp" \
    -Wl,-rpath,"$ONEAPI/lib" \
    -o "$OUT/libdequant_cpu.so"

if [[ "${B70_GGUF_HOST_ONLY:-0}" == 1 ]]; then
    echo "build: host only, skipping SYCL"
    exit 0
fi

# C2: grouped_prefill.cpp. B70_PREFILL_SMOKE=1 builds IQ3_XXS+Q2_0 fp16 only
# and writes _C.smoke.so so a failed smoke link does not replace _C.so.
PREFILL_DEFS=()
OUT_SO="$OUT/_C.so"
if [[ "${B70_PREFILL_SMOKE:-0}" == "1" ]]; then
    PREFILL_DEFS+=(-DB70_PREFILL_SMOKE=1)
    OUT_SO="${B70_PREFILL_SMOKE_OUT:-$OUT/_C.smoke.so}"
    echo "build: prefill smoke instantiations -> $OUT_SO"
elif [[ -n "${B70_PREFILL_OUT:-}" ]]; then
    OUT_SO="$B70_PREFILL_OUT"
fi
echo "build: SYCL extension (MAX_JOBS=${MAX_JOBS:-1})"
"$ICPX" -fsycl -std=c++20 -O3 -fPIC -fexceptions -shared \
    -fsycl-device-code-split=per_kernel \
    "${CXXFLAGS_STD[@]}" \
    "${PREFILL_DEFS[@]}" \
    -I"$ROOT/csrc" \
    -I"$TORCH/include" \
    -I"$TORCH/include/torch/csrc/api/include" \
    -I"$PY/include/python3.12" \
    -DTORCH_EXTENSION_NAME=_C \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    "$ROOT/csrc/kernels.cpp" "$ROOT/csrc/grouped_prefill.cpp" "$ROOT/csrc/bind.cpp" \
    -L"$TORCH/lib" -L"$VENV/lib" \
    -ltorch_xpu -ltorch_cpu -ltorch -lc10 -lc10_xpu -lsycl \
    -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$VENV/lib" -Wl,-rpath,"$ONEAPI/lib" \
    -o "$OUT_SO"
echo "build: wrote $OUT/libdequant_cpu.so and $OUT_SO"
