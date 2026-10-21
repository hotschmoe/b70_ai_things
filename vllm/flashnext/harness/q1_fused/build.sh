#!/usr/bin/env bash
# Build isolated b70_q1_packcompact_r1 ops (no GPU access). Run under bin/cpu-run.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
ROOT=${Q1_FUSED_ROOT:-/mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r1}
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1
VENV=/mnt/vm_8tb/b70/build/vllm-flashnext/venv
TORCH="$VENV/lib/python3.12/site-packages/torch"
PY=/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu
TC=/mnt/vm_8tb/b70/build/gguf-xpu-toolchain
OUT=${1:-/mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r1/libq1_fused.so}
export MAX_JOBS=4
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
"$ONEAPI/bin/icpx" -fsycl -std=c++20 -O3 -fPIC -fexceptions -shared -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen -Xs "-device bmg" \
    --gcc-toolchain="$TC" -isystem "$TC/include/c++/15" -isystem "$TC/include/x86_64-linux-gnu/c++/15" \
    -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PY/include/python3.12" \
    -DTORCH_EXTENSION_NAME=_C -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    "$SRC/fused.cpp" \
    -L"$TORCH/lib" -L"$VENV/lib" -ltorch_xpu -ltorch_cpu -ltorch -lc10 -lc10_xpu -lsycl \
    -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$VENV/lib" -Wl,-rpath,"$ONEAPI/lib" -o "$OUT"
ls -l "$OUT"; sha256sum "$OUT"

"$ONEAPI/bin/icpx" -std=c++20 -O2 --gcc-toolchain="$TC" -isystem "$TC/include/c++/15" -isystem "$TC/include/x86_64-linux-gnu/c++/15" "$SRC/cpu_keys.cpp" -o "$ROOT/cpu_keys"
