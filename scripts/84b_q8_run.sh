#!/usr/bin/env bash
# QUANTS Q8 runner. SMOKE=1 (default) -> 10-min toolchain validate (iters=2); SMOKE=0 -> full 4-8h run.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
SMOKE="${SMOKE:-1}"
if [ "$SMOKE" = 1 ]; then OUT=/models/Qwable-5-27B-Coder-int4-AutoRound-smoke; ITERS=2; NSAMPLES=8; TAG=smoke
else OUT=/models/Qwable-5-27B-Coder-int4-AutoRound; ITERS=200; NSAMPLES=128; TAG=full; fi
LOGF="$ROOT/results/q8_${TAG}.log"; mkdir -p "$ROOT/results" "$ROOT/pip_cache"
docker rm -f q8_qwable_int4 2>/dev/null || true
echo "=== Q8 $TAG: SRC=Qwable OUT=$OUT iters=$ITERS nsamples=$NSAMPLES ==="
docker run --rm --name q8_qwable_int4 --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 32g \
  -v "$ROOT/models:/models" -v "$ROOT/hf_cache:/hf_cache" -e HF_HOME=/hf_cache \
  -v "$ROOT/pip_cache:/pipcache" -e PIP_CACHE_DIR=/pipcache \
  -v "$ROOT/q8_qwable_int4.py:/work/q8_qwable_int4.py:ro" \
  -e CCL_ENABLE_SYCL_KERNELS=0 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi \
  -e SRC=/models/DJLougen_Qwable-5-27B-Coder -e OUT="$OUT" -e DEVMAP=0,1 \
  -e ITERS="$ITERS" -e NSAMPLES="$NSAMPLES" -e SEQLEN=2048 \
  --entrypoint bash vllm-xpu-env:v0230 -lc '
    set -e
    python -c "import auto_round" 2>/dev/null || pip install -q --no-deps auto-round
    python -c "import accelerate, datasets, py_cpuinfo" 2>/dev/null || pip install -q accelerate datasets py-cpuinfo
    python -c "import torch;assert torch.xpu.is_available(),\"no xpu\""
    python /work/q8_qwable_int4.py
  ' > "$LOGF" 2>&1 || echo "(docker returned nonzero)"
echo "=== q8 $TAG done; log $LOGF ==="
grep -iE "RESULT_Q8|SAVED|FATAL|error|traceback|MLLM|blocked|construct|auto_round |layer_config|quantize" "$LOGF" | tail -25
