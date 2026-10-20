#!/usr/bin/env bash
# Inner command for harness/xpu-run.sh. Times 4-submit vs 2-submit moe_gemv.
# The 1-submit kernel is a separate process so a hang can be killed.
set -uo pipefail

PLUGIN=/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_lowbit_xpu
export PYTHONPATH="${PLUGIN}${PYTHONPATH:+:${PYTHONPATH}}"
export B70_LOWBIT_SO="${B70_LOWBIT_SO:-/mnt/vm_8tb/b70/build/vllm-flashnext/b70_lowbit_xpu/libb70_lowbit_xpu.so}"
export B70_LOWBIT_PORT="${B70_LOWBIT_PORT:-29671}"
cd "$PLUGIN"
# First-launch AOT load is seconds. A hung 2-submit kernel must not keep the card.
timeout 300 python3 -u tests/moe_launch_bench.py
rc=$?
echo "LAUNCH2_RC:$rc"
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi
# 1-submit: one work-group per token. No grid barrier. Still isolated.
timeout 60 python3 - <<'PY'
import os, sys
sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_lowbit_xpu")
os.environ["B70_LOWBIT_MOE_LAUNCHES"] = "1"
import torch
from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import plane2_stride, plane_hi_stride

def planes(e, n, k, bits, device):
    s2 = plane2_stride(k)
    shi = plane_hi_stride(k, bits)
    p2 = torch.randint(0, 256, (e, n, s2), dtype=torch.uint8, device=device)
    hi = torch.randint(0, 256, (e, n, shi), dtype=torch.uint8, device=device)
    scale = torch.randn(e, n, k // 128, dtype=torch.bfloat16, device=device)
    return p2, hi, scale

device = torch.device("xpu")
e, h, i, t, topk, bits = 4, 256, 128, 1, 4, 4
x = torch.randn(t, h, dtype=torch.bfloat16, device=device)
ids = torch.randint(0, e, (t, topk), dtype=torch.int32, device=device)
router = torch.rand(t, topk, device=device)
w13 = planes(e, i * 2, h, bits, device)
w2 = planes(e, h, i, bits, device)
os.environ["B70_LOWBIT_MOE_LAUNCHES"] = "4"
y4 = ops.moe_gemv(x, ids, router, w13[0], w13[1], w13[2], w2[0], w2[1], w2[2], bits)
os.environ["B70_LOWBIT_MOE_LAUNCHES"] = "1"
y1 = ops.moe_gemv(x, ids, router, w13[0], w13[1], w13[2], w2[0], w2[1], w2[2], bits)
torch.xpu.synchronize()
d = (y4.float() - y1.float()).abs()
print(f"correctness 1-submit max_abs={float(d.max()):.6f}", flush=True)
if float(d.max()) > 0.05:
    raise SystemExit(4)
print("LAUNCH1_OK", flush=True)
PY
echo "LAUNCH1_RC:$?"
exit $?
