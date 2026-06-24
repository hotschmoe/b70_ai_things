#!/usr/bin/env bash
# Build ONLY the _xpu_C extension of vllm-xpu-kernels (skips flash-attn/MoE/GDN/basic/cutlass-TLA
# via the *_KERNELS_ENABLED=OFF toggles -> minutes instead of 1-2h) to get our int8_gemm_w8a8 op
# compiled + registered. Repo mounted at /src to match the .deps cmake caches. Full log on SSD;
# prints a filtered summary + the hasattr registration check. Synchronous (run via runremote in bg).
set -uo pipefail
R=/mnt/vm_8tb/b70/vllm-xpu-kernels
STAMP="$(date +%Y%m%d_%H%M%S)"; LOG=/mnt/vm_8tb/b70/results/int8_build_${STAMP}.log
docker rm -f int8_build 2>/dev/null || true
echo "=== minimal _xpu_C build; full log -> $LOG ==="

docker run --rm --name int8_build \
  -v "$R":/src -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70 \
  --entrypoint bash vllm-xpu-env:v0230 -c '
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    export CCACHE_DIR=/mnt/vm_8tb/b70/.ccache TMPDIR=/mnt/vm_8tb/b70/.tmp PIP_CACHE_DIR=/mnt/vm_8tb/b70/.pipcache
    export FA2_KERNELS_ENABLED=OFF MOE_KERNELS_ENABLED=OFF GDN_KERNELS_ENABLED=ON \
           MQA_LOGITS_KERNELS_ENABLED=OFF BASIC_KERNELS_ENABLED=OFF XPUMEM_ALLOCATOR_ENABLED=OFF \
           XPU_SPECIFIC_KERNELS_ENABLED=ON
    cd /src
    echo "MINIMAL_BUILD_START $(date)"
    pip install --no-build-isolation -e . > "'"$LOG"'" 2>&1
    RC=$?
    echo "MINIMAL_BUILD_RC=$RC"
    if [ "$RC" = 0 ]; then
      echo "=== HASATTR CHECK ==="
      python -c "import torch, vllm._xpu_ops as x; print(\"int8_gemm_w8a8 registered:\", hasattr(torch.ops._xpu_C, \"int8_gemm_w8a8\"))" 2>&1
    fi
  '
RC=$?
echo "=== build wrapper exit: $RC ==="
echo "=== which extensions got configured (should be ONLY _xpu_C) ==="
grep -iE "KERNELS_ENABLED|Building extension|--target|Configuring done|Generating done" "$LOG" 2>/dev/null | head -15
echo "=== ERRORS (if any) ==="
grep -iE "CMake Error|error:|FAILED|undefined reference|fatal error|cannot|No such file" "$LOG" 2>/dev/null | grep -viE "Wno-|-Werror|no-error" | head -25
echo "=== last 25 log lines ==="
tail -25 "$LOG" 2>/dev/null
echo "=== built .so ==="
find "$R" -name "*_xpu_C*.so" -newermt "$STAMP" 2>/dev/null | head; find "$R" -name "_xpu_C*.so" 2>/dev/null | head
