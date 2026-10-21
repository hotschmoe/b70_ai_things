#!/usr/bin/env bash
# P1: unique native bucket overlay generated from a frozen control source.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ROOT="$REPO/vllm/flashnext/plugins/b70_gguf_xpu"
P1_SOURCE="$REPO/vllm/flashnext/plugins/b70_p1_xpu/csrc"
export TMPDIR="${P1_TMPDIR:-/mnt/vm_8tb/b70/build/vllm-flashnext/p1/tmp}"
export MAX_JOBS=4
mkdir -p "$TMPDIR"
ONEAPI=/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi/compiler/2026.1
ICPX=$ONEAPI/bin/icpx
VENV=/mnt/vm_8tb/b70/build/vllm-flashnext/venv
TORCH="$VENV/lib/python3.12/site-packages/torch"
PY=/home/hotschmoe/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu
TC=/mnt/vm_8tb/b70/build/gguf-xpu-toolchain
OUT=${1:-/mnt/vm_8tb/b70/build/vllm-flashnext/p1/_C.p1bucket.so}
GENERATED=/mnt/vm_8tb/b70/build/vllm-flashnext/p1/bucket_overlay.cpp
python3 "$REPO/vllm/flashnext/plugins/b70_p1_xpu/generate_bucket_overlay.py" --out "$GENERATED"
"$ICPX" -fsycl -std=c++20 -O3 -fPIC -fexceptions -shared -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=4 \
    --gcc-toolchain="$TC" -isystem "$TC/include/c++/15" -isystem "$TC/include/x86_64-linux-gnu/c++/15" \
    -DB70_PREFILL_SMOKE=1 -I"$ROOT/csrc" -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PY/include/python3.12" \
    -DTORCH_EXTENSION_NAME=_C -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    "$GENERATED" \
    -L"$TORCH/lib" -L"$VENV/lib" -ltorch_xpu -ltorch_cpu -ltorch -lc10 -lc10_xpu -lsycl \
    -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$VENV/lib" -Wl,-rpath,"$ONEAPI/lib" -o "$OUT"
ls -l "$OUT"
sha256sum "$OUT"
echo P1_BUCKET_OVERLAY_BUILD_OK
