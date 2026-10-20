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
# 1-submit: one work-group per token. No grid barrier. AOT can take a minute.
# 180s covers compile plus a short Ornith-shaped timing. A hang exits 124.
timeout 180 python3 - <<'PY'
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

def once(x, ids, router, w13, w2, bits, launches):
    os.environ["B70_LOWBIT_MOE_LAUNCHES"] = launches
    return ops.moe_gemv(
        x, ids, router, w13[0], w13[1], w13[2], w2[0], w2[1], w2[2], bits
    )

def time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iters

device = torch.device("xpu")
bits = 4
e, h, i, t, topk = 4, 256, 128, 1, 4
x = torch.randn(t, h, dtype=torch.bfloat16, device=device)
ids = torch.randint(0, e, (t, topk), dtype=torch.int32, device=device)
router = torch.rand(t, topk, device=device)
w13 = planes(e, i * 2, h, bits, device)
w2 = planes(e, h, i, bits, device)
print("launch1: small correctness", flush=True)
y4 = once(x, ids, router, w13, w2, bits, "4")
y1 = once(x, ids, router, w13, w2, bits, "1")
torch.xpu.synchronize()
d = (y4.float() - y1.float()).abs()
print(f"correctness 1-submit small max_abs={float(d.max()):.6f}", flush=True)
if float(d.max()) > 0.05:
    raise SystemExit(4)
e, h, i, t, topk = 256, 2048, 512, 1, 8
x = torch.randn(t, h, dtype=torch.bfloat16, device=device)
ids = torch.randint(0, e, (t, topk), dtype=torch.int32, device=device)
router = torch.softmax(torch.randn(t, topk, device=device), dim=-1)
w13 = planes(e, i * 2, h, bits, device)
w2 = planes(e, h, i, bits, device)
print("launch1: ornith correctness", flush=True)
y4 = once(x, ids, router, w13, w2, bits, "4")
y1 = once(x, ids, router, w13, w2, bits, "1")
torch.xpu.synchronize()
d = (y4.float() - y1.float()).abs()
print(f"correctness 1-submit ornith max_abs={float(d.max()):.6f}", flush=True)
if float(d.max()) > 0.05:
    raise SystemExit(5)
ms4 = time_ms(lambda: once(x, ids, router, w13, w2, bits, "4"), 2, 10)
ms1 = time_ms(lambda: once(x, ids, router, w13, w2, bits, "1"), 2, 10)
print(f"TIME launches=1 ms={ms1:.4f} us={ms1 * 1000:.1f}", flush=True)
print(f"TIME launches=4-repeat ms={ms4:.4f} us={ms4 * 1000:.1f}", flush=True)
winner = "1" if ms1 < ms4 else "4"
print(f"WINNER_1VS4={winner}", flush=True)
print("LAUNCH1_OK", flush=True)
PY
echo "LAUNCH1_RC:$?"
exit $?
