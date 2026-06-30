#!/usr/bin/env bash
# llamacpp/build_sycl.sh -- build llama.cpp with the SYCL backend for Intel Arc Pro B70 (Battlemage/Xe2).
#
# We do NOT install a native oneAPI toolchain on the host; we build INSIDE the existing oneAPI image
# (sglang-xpu:mtp) which already ships oneAPI 2025.3 + icx/icpx + Level-Zero + oneMKL + oneDNN + cmake +
# ninja (verified 2026-06-30). This mirrors how vLLM/sglang kernels are built per-backend (kernels/README.md).
#
# Pure COMPILATION -- no GPU compute, no /dev/dri. Safe to run any time (does not need the GPU lease).
#
# Upstream source (git-ignored runtime clone, NOT repo content): /mnt/vm_8tb/b70/llama.cpp  (HEAD 86b94708).
# Built binaries land at /mnt/vm_8tb/b70/llama.cpp/build/bin/ (git-ignored runtime artifacts).
#
# Build flags rationale (see REVIEW_intel_arch.md sec 1, docs/backend/SYCL.md:313-336):
#   GGML_SYCL=ON GGML_SYCL_TARGET=INTEL  -- the Intel SYCL path.
#   CMAKE_{C,CXX}_COMPILER=icx/icpx       -- oneAPI DPC++ (required for SYCL).
#   LLAMA_CURL=OFF                        -- weights are local; image has no libcurl-dev.
#   (NO GGML_SYCL_DEVICE_ARCH)            -- default JIT/SPIR-V: portable, runs on any Intel GPU incl. B70.
#                                            AOT (bmg_g21) is a perf knob but couples with corruption bug
#                                            #21893 (F16+AOT garbage unless GGML_SYCL_DISABLE_OPT=1). Stay JIT.
#   GGML_SYCL_F16 (env, default OFF)      -- the validated baseline artifact is F16=OFF. F16=ON is the Intel
#                                            reference recipe and a perf candidate, but must be coherence-
#                                            sweep-gated on B70 first (see #21893). Set F16=1 to opt in.
set -euo pipefail
SRC="${SRC:-/mnt/vm_8tb/b70/llama.cpp}"
IMG="${IMG:-sglang-xpu:mtp}"
F16="${F16:-0}"          # 1 -> -DGGML_SYCL_F16=ON (sweep-gate on B70 before trusting; #21893)
JOBS="${JOBS:-$(nproc)}"
F16_FLAG="OFF"; [ "$F16" = 1 ] && F16_FLAG="ON"

echo "=== llama.cpp SYCL build  src=$SRC  img=$IMG  F16=$F16_FLAG  jobs=$JOBS  $(date) ==="
docker run --rm --entrypoint bash -v "$SRC:/llama" "$IMG" -lc "
  set -e
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  cd /llama
  echo \"icx: \$(icx --version | head -1)\"
  rm -rf build
  cmake -B build -G Ninja \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_TARGET=INTEL \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL_F16=$F16_FLAG \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_SERVER=ON
  cmake --build build --config Release -j$JOBS
  echo '=== built binaries ==='
  ls build/bin/ | grep -E '^llama-(cli|server|quantize|bench|mtmd)' || ls build/bin/ | head
"
echo "=== done $(date) -> $SRC/build/bin/ ==="
