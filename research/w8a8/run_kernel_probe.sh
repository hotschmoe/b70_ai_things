#!/usr/bin/env bash
# run_kernel_probe.sh -- run w8a8_kernel_probe.py on card 0 in the sglang torch-2.12 image.
# Card-0 microbench only (no serve, no TP=2) -> no wedge risk; takes the single-card lease.
set -euo pipefail
ROOT="${ROOT:-/mnt/vm_8tb/b70}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMG="${IMG:-sglang-xpu:woq}"
KERNEL_DIR="${KERNEL_DIR:-$ROOT/w4a8_kernel}"
LOG="${LOG:-$REPO/w8a8/kernel_probe.log}"

exec ./bin/gpu-run --card 0 docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g -e ZE_AFFINITY_MASK=0 \
  -v "$ROOT/models:/models:ro" \
  -v "$KERNEL_DIR:/work/w4a8_kernel:ro" \
  -v "$REPO/w8a8:/work" \
  -e B70_XPU_C_SO=/work/w4a8_kernel/_xpu_C.abi3.so \
  "$IMG" bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; \
    python3 /work/w8a8_kernel_probe.py" 2>&1 | tee "$LOG"
