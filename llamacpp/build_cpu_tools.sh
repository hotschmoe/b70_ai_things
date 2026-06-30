#!/usr/bin/env bash
# llamacpp/build_cpu_tools.sh -- build a CPU-ONLY llama-quantize (GGML_SYCL=OFF).
#
# WHY: the SYCL build links the SYCL backend, and llama-quantize calls llama_backend_init() at startup,
# which initializes ggml-sycl -> dpct::dev_mgr, which ABORTS ("can not find preferred GPU platform") unless
# a GPU is visible. Quantization itself is a CPU operation, so we keep a separate GGML_SYCL=OFF binary that
# needs no GPU at all -- this keeps the convert/quantize pipeline fully GPU-free (no lease, no enumeration of
# the cards while the daily driver holds them). Serving still uses the SYCL build (build_sycl.sh).
#
# Output: $SRC/build-cpu/bin/llama-quantize  (git-ignored runtime artifact).
set -euo pipefail
SRC="${SRC:-/mnt/vm_8tb/b70/llama.cpp}"
IMG="${IMG:-sglang-xpu:mtp}"
JOBS="${JOBS:-$(nproc)}"
echo "=== llama.cpp CPU-only tools build (GGML_SYCL=OFF)  $(date) ==="
docker run --rm --entrypoint bash -v "$SRC:/llama" "$IMG" -lc "
  set -e
  cd /llama
  cmake -B build-cpu -G Ninja -DGGML_SYCL=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=OFF
  cmake --build build-cpu --target llama-quantize -j$JOBS
  ls -la build-cpu/bin/llama-quantize
"
echo "=== done $(date) -> $SRC/build-cpu/bin/llama-quantize ==="
