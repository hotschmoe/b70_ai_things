#!/usr/bin/env bash
# Post-build sanity for sglang-xpu:bmg. Two stages:
#  1) CPU-only: torch is the +xpu build, triton-xpu present, sglang imports, qwen3_5 registered.
#     (No GPU needed; run anytime after build.)
#  2) GPU: torch.xpu sees both B70s. MUST hold the lease: ../bin/gpu-run bash sglang/verify_image.sh gpu
set -uo pipefail
IMG="${IMG:-sglang-xpu:bmg}"
stage="${1:-cpu}"

if [ "$stage" = cpu ]; then
  docker run --rm "$IMG" bash -lc '
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    python - <<PY
import torch, importlib.util
print("torch:", torch.__version__)
assert "xpu" in torch.__version__, "TORCH IS NOT +xpu BUILD -> CUDA contamination!"
print("has torch.xpu:", hasattr(torch, "xpu"))
print("triton-xpu:", importlib.util.find_spec("triton") is not None)
import sglang; print("sglang:", sglang.__version__)
from sglang.srt.models import qwen3_5  # our GDN class
print("qwen3_5 model module: OK")
try:
    from sglang.srt.models import qwen3_next; print("qwen3_next module: OK")
except Exception as e:
    print("qwen3_next import:", e)
PY'
elif [ "$stage" = gpu ]; then
  docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path "$IMG" bash -lc '
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    python - <<PY
import torch
print("torch:", torch.__version__)
print("xpu.is_available:", torch.xpu.is_available())
print("xpu.device_count:", torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    print(" ", i, torch.xpu.get_device_name(i))
x=torch.randn(1024,1024,device="xpu"); y=(x@x).sum().item()
print("matmul on xpu OK, sum finite:", y==y)
PY'
else echo "usage: $0 {cpu|gpu}"; exit 2; fi
