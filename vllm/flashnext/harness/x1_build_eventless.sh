#!/usr/bin/env bash
# CPU-only build; invoke under bin/cpu-run MAX_JOBS<=4.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
BUILD=/mnt/vm_8tb/b70/build/vllm-flashnext/x1/eventless
VENV=/mnt/vm_8tb/b70/build/vllm-flashnext/venv
TORCH=$VENV/lib/python3.12/site-packages/torch
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1
TC=/mnt/vm_8tb/b70/build/gguf-xpu-toolchain
mkdir -p "$BUILD/tmp"
export TMPDIR="$BUILD/tmp"
"$ONEAPI/bin/icpx" -fsycl -std=c++20 -O2 -fPIC -shared \
  --gcc-toolchain="$TC" -isystem "$TC/include/c++/15" \
  -isystem "$TC/include/x86_64-linux-gnu/c++/15" \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  "$REPO/vllm/flashnext/harness/patches/x1_eventless.cpp" \
  -L"$TORCH/lib" -L"$VENV/lib" \
  -ltorch_xpu -ltorch_cpu -lc10_xpu -lc10 -lsycl \
  -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$VENV/lib" -Wl,-rpath,"$ONEAPI/lib" \
  -o "$BUILD/libx1_eventless.so"
echo X1_EVENTLESS_BUILD_OK
