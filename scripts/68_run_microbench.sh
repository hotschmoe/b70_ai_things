#!/usr/bin/env bash
# INT8 vs BF16 GEMM/GEMV microbench for B70 (Xe2/Battlemage).
# Runs 68_int8_gemm_gemv_microbench.py inside the int8 image via gpu-run.
#
# Usage (on Unraid host):
#   cd /mnt/vm_8tb/b70 && ./gpu-run bash 68_run_microbench.sh
#
# Optional env:
#   IMG    docker image (default vllm-xpu-env:int8)
#   STAMP  output CSV timestamp suffix (default: date +%Y%m%d_%H%M%S)

set -uo pipefail

ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:int8}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SCRIPT="$ROOT/scripts/68_int8_gemm_gemv_microbench.py"
LOG="$ROOT/results/microbench_gemm_gemv_${STAMP}.log"

mkdir -p "$ROOT/results"

[ -f "$SCRIPT" ] || { echo "MISSING $SCRIPT"; exit 1; }

echo "=== INT8 vs BF16 GEMM/GEMV microbench :: IMG=$IMG STAMP=$STAMP ==="
echo "=== log=$LOG ==="

docker rm -f mb68 2>/dev/null || true
docker run --rm --name mb68 \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g \
  -e ZE_AFFINITY_MASK=0 \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e OMP_NUM_THREADS=32 \
  -v "$ROOT:$ROOT" \
  -e HF_HOME="$ROOT/hf_cache" \
  -e TMPDIR="$ROOT/tmp_ssd" \
  -v "$SCRIPT:$SCRIPT:ro" \
  --entrypoint bash "$IMG" -lc "
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    echo \"torch: \$(python -c 'import torch;print(torch.__version__)')\"
    python -c 'import torch; assert torch.xpu.is_available(); print(\"xpu ok, device:\", torch.xpu.get_device_name(0))'
    python '$SCRIPT' '$STAMP'
  " 2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
echo "=== exit $rc ==="
echo "=== CSV: $ROOT/results/microbench_gemm_gemv_${STAMP}.csv ==="
exit $rc
