#!/usr/bin/env bash
# Offline BMG AOT build; call under bin/cpu-run with MAX_JOBS<=4.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
OUT=/mnt/vm_8tb/b70/build/vllm-flashnext/x5/q8-fast
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1
VENV=/mnt/vm_8tb/b70/build/vllm-flashnext/venv
TORCH=$VENV/lib/python3.12/site-packages/torch
TC=/mnt/vm_8tb/b70/build/gguf-xpu-toolchain
mkdir -p "$OUT/tmp"
export MAX_JOBS=4 TMPDIR="$OUT/tmp"
"$ONEAPI/bin/icpx" -fsycl -std=c++20 -O3 -fPIC -fexceptions -shared \
  -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen -Xs '-device bmg' \
  --gcc-toolchain="$TC" -isystem "$TC/include/c++/15" -isystem "$TC/include/x86_64-linux-gnu/c++/15" \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -D_GLIBCXX_USE_CXX11_ABI=1 \
  "$REPO/vllm/flashnext/plugins/b70_x5_xpu/csrc/q8_linear.cpp" \
  -L"$TORCH/lib" -L"$VENV/lib" -ltorch_xpu -ltorch_cpu -lc10_xpu -lc10 -lsycl \
  -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$VENV/lib" -Wl,-rpath,"$ONEAPI/lib" \
  -o "$OUT/libb70_x5_xpu.so"
sha256sum "$OUT/libb70_x5_xpu.so"
