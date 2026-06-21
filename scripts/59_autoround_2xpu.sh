#!/usr/bin/env bash
# Experimental: try to load a model across BOTH B70s for quantization (no public XPU precedent).
# Runs aq2x.py in the :v0230 image with both /dev/dri cards exposed. pip-installs auto-round at runtime
# (no image bakes it). GPU job -> wrap in gpu-run:  ./gpu-run bash 59_autoround_2xpu.sh
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
IMG="${IMG:-vllm-xpu-env:v0230}"
MODELP="${MODELP:-/models/Qwen_Qwen3-0.6B}"
docker rm -f aq2x 2>/dev/null || true
# Both cards visible (no ZE_AFFINITY pin); multi-GPU CCL stability env in case auto_round/accelerate uses xccl.
docker run --rm --name aq2x --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --shm-size 16g \
  -v "$ROOT/models:/models:ro" -v "$ROOT/hf_cache:/hf_cache" -v "$ROOT/tmp_ssd:/tmp_ssd" -v "$ROOT/aq2x.py:/aq2x.py:ro" \
  -e HF_HOME=/hf_cache -e TMPDIR=/tmp_ssd -e MODELP="$MODELP" \
  -e CCL_ENABLE_SYCL_KERNELS=0 -e SYCL_UR_USE_LEVEL_ZERO_V2=0 -e CCL_ATL_TRANSPORT=ofi \
  --entrypoint bash "$IMG" -lc '
    echo "=== torch already present: $(python -c "import torch;print(torch.__version__)") ==="
    echo "=== pip install auto-round (no torch clobber) ==="
    pip install -q --no-deps auto-round 2>&1 | tail -3 || echo "(no-deps install failed)"
    # auto-round runtime deps that may be missing (small, pure-python); ignore if already present
    pip install -q "py-cpuinfo" "accelerate>=0.30" 2>&1 | tail -2 || true
    python -c "import torch;assert torch.xpu.is_available(),\"XPU LOST after pip\";print(\"xpu OK after pip, count\",torch.xpu.device_count())"
    python /aq2x.py
  '
