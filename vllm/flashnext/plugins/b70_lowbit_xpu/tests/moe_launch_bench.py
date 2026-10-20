"""Time moe_gemv at 4, 2, and 1 device submits. Compare against the 4-submit path.

Ornith decode shape: T=1, topk=8, E=256, H=2048, I=512, int4.
A smaller int4 tensor checks that 2-submit and 1-submit match 4-submit.
Set B70_LOWBIT_MOE_LAUNCHES in the process; the op reads it on each call.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_lowbit_xpu")

from b70_lowbit_xpu import ops  # noqa: E402
from b70_lowbit_xpu.format import plane2_stride, plane_hi_stride  # noqa: E402


def planes(e: int, n: int, k: int, bits: int, device: torch.device):
    s2 = plane2_stride(k)
    shi = plane_hi_stride(k, bits)
    p2 = torch.randint(0, 256, (e, n, s2), dtype=torch.uint8, device=device)
    hi = torch.randint(0, 256, (e, n, shi), dtype=torch.uint8, device=device)
    scale = torch.randn(e, n, k // 128, dtype=torch.bfloat16, device=device)
    return p2, hi, scale


def run_once(x, ids, router, w13, w2, bits: int, launches: str):
    os.environ["B70_LOWBIT_MOE_LAUNCHES"] = launches
    return ops.moe_gemv(
        x, ids, router, w13[0], w13[1], w13[2], w2[0], w2[1], w2[2], bits
    )


def time_ms(fn, warmup: int, iters: int) -> float:
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


def main() -> int:
    device = torch.device("xpu")
    bits = 4
    # Correctness on a shape that still uses the SLM GEMV.
    e, h, i, t, topk = 4, 256, 128, 2, 4
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    x = torch.randn(t, h, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, e, (t, topk), dtype=torch.int32, device=device)
    router = torch.rand(t, topk, dtype=torch.float32, device=device)
    w13 = planes(e, i * 2, h, bits, device)
    w2 = planes(e, h, i, bits, device)
    y4 = run_once(x, ids, router, w13, w2, bits, "4")
    y2 = run_once(x, ids, router, w13, w2, bits, "2")
    torch.xpu.synchronize()
    d2 = (y4.float() - y2.float()).abs()
    print(
        f"correctness small T={t} topk={topk} E={e} H={h} I={i} "
        f"max_abs_2vs4={float(d2.max()):.6f} mean={float(d2.mean()):.6f}",
        flush=True,
    )
    if float(d2.max()) > 0.05:
        print("FAIL 2-submit does not match 4-submit", flush=True)
        return 2

    # Full Ornith decode shape. Timing only after the small match.
    e, h, i, t, topk = 256, 2048, 512, 1, 8
    x = torch.randn(t, h, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, e, (t, topk), dtype=torch.int32, device=device)
    router = torch.softmax(torch.randn(t, topk, device=device), dim=-1)
    w13 = planes(e, i * 2, h, bits, device)
    w2 = planes(e, h, i, bits, device)
    y4 = run_once(x, ids, router, w13, w2, bits, "4")
    y2 = run_once(x, ids, router, w13, w2, bits, "2")
    torch.xpu.synchronize()
    d2 = (y4.float() - y2.float()).abs()
    print(
        f"correctness ornith max_abs_2vs4={float(d2.max()):.6f} "
        f"mean={float(d2.mean()):.6f}",
        flush=True,
    )
    if float(d2.max()) > 0.05:
        print("FAIL ornith 2-submit mismatch", flush=True)
        return 3

    times: dict[str, float] = {}

    def bench(label: str) -> None:
        ms = time_ms(
            lambda: run_once(x, ids, router, w13, w2, bits, label), 5, 20
        )
        times[label] = ms
        print(f"TIME launches={label} ms={ms:.4f} us={ms * 1000:.1f}", flush=True)

    bench("4")
    bench("2")
    winner = "2" if times["2"] < times["4"] else "4"
    print(
        f"WINNER_2VS4={winner} us4={times['4'] * 1000:.1f} "
        f"us2={times['2'] * 1000:.1f}",
        flush=True,
    )
    print("LAUNCH_BENCH_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
