#!/usr/bin/env bash
# Build b70_ep_xpu/_C.so and, unless B70_EP_SKIP_GGUF=1, the slotted GGUF
# library _C.ep.so. Does not replace _C.so or _C.graph.so.
# Run under bin/cpu-run. Do not nest another cpu-run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONEAPI="${ONEAPI:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1}"
ICPX="${ICPX:-$ONEAPI/bin/icpx}"
VENV="${VENV:-/mnt/vm_8tb/b70/build/vllm-flashnext/venv}"
TORCH="$VENV/lib/python3.12/site-packages/torch"
PY="${PY:-/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu}"
TC="${TC:-/mnt/vm_8tb/b70/build/gguf-xpu-toolchain}"
ZE_INC="${ZE_INC:-/mnt/vm_8tb/b70/build/vllm-flashnext/level-zero-sdk/include}"
CXXFLAGS_STD=(
    --gcc-toolchain="$TC"
    -isystem "$TC/include/c++/15"
    -isystem "$TC/include/x86_64-linux-gnu/c++/15"
)
OUT="$ROOT/b70_ep_xpu"
mkdir -p "$OUT"
GGUF_ROOT="$(cd "$ROOT/../b70_gguf_xpu" && pwd)"
EP_SO="${B70_EP_GGUF_SO:-$GGUF_ROOT/b70_gguf_xpu/_C.ep.so}"

echo "build: EP sidecar first (MAX_JOBS=${MAX_JOBS:-1})"
"$ICPX" -fsycl -std=c++20 -O3 -fPIC -fexceptions -shared \
    -fsycl-device-code-split=per_kernel \
    "${CXXFLAGS_STD[@]}" \
    -I"$ROOT/csrc" \
    -I"$ZE_INC" \
    -I"$TORCH/include" \
    -I"$TORCH/include/torch/csrc/api/include" \
    -I"$PY/include/python3.12" \
    -DTORCH_EXTENSION_NAME=_C \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    "$ROOT/csrc/ep_kernels.cpp" "$ROOT/csrc/probe.cpp" "$ROOT/csrc/moe_ep.cpp" "$ROOT/csrc/bind.cpp" \
    -L"$TORCH/lib" -L"$VENV/lib" -L/usr/lib/x86_64-linux-gnu \
    -ltorch_xpu -ltorch_cpu -ltorch -lc10 -lc10_xpu -lsycl -l:libze_loader.so.1 -ldl \
    -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$VENV/lib" -Wl,-rpath,"$ONEAPI/lib" \
    -o "$OUT/_C.so"
echo "build: wrote $OUT/_C.so"

if [[ "${B70_EP_SKIP_GGUF:-0}" != "1" ]]; then
    echo "build: slotted GGUF -> $EP_SO"
    B70_PREFILL_OUT="$EP_SO" "$GGUF_ROOT/scripts/build.sh"
fi
ls -l "$OUT/_C.so" "$EP_SO"
