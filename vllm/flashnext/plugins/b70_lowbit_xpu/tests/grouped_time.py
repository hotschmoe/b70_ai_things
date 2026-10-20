"""One-layer device times. Not a serve tok/s.

Shapes are the 35B expert: H=2048, I=512, E=256, top-8, bits=3.
Planes are random bytes. The question is milliseconds, not numerics.
"""

from __future__ import annotations

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import plane2_stride, plane_hi_stride


def ms(fn, warm=1, iters=3) -> float:
    for _ in range(warm):
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


def planes(e, n, k, bits, device):
    p2 = torch.randint(0, 256, (e, n, plane2_stride(k)), dtype=torch.uint8, device=device)
    hi = torch.randint(
        0, 256, (e, n, plane_hi_stride(k, bits)), dtype=torch.uint8, device=device
    )
    sc = torch.randn(e, n, k // 128, dtype=torch.bfloat16, device=device)
    return p2, hi, sc


def main() -> None:
    device = torch.device("xpu")
    bits = 3
    e, h, i, topk = 256, 2048, 512, 8
    p13, hi13, s13 = planes(e, 2 * i, h, bits, device)
    p2, hi2, s2 = planes(e, h, i, bits, device)
    print("planes_ready", flush=True)
    for t_tokens in (128, 1024, 8192):
        g = torch.Generator(device="cpu").manual_seed(t_tokens)
        x = torch.randn(t_tokens, h, dtype=torch.bfloat16, device=device)
        ids = torch.randint(0, e, (t_tokens, topk), generator=g, device="cpu").to(device)
        router = torch.rand(t_tokens, topk, device=device)

        def run(x=x, ids=ids, router=router):
            return ops.moe_prefill(
                x, ids, router, p13, hi13, s13, p2, hi2, s2, bits
            )

        layer_ms = ms(run)
        print(
            f"moe_prefill T={t_tokens} ms={layer_ms:.2f} "
            f"tok/s_if_40layers={t_tokens / (40 * layer_ms / 1e3):.1f}",
            flush=True,
        )

    m = 256
    x = torch.randn(m, h, dtype=torch.bfloat16, device=device)
    one_ms = ms(lambda: ops.prefill_wna16(x, p13[0], hi13[0], s13[0], bits))
    i8_ms = ms(lambda: ops.prefill_wxa8(x, p13[0], hi13[0], s13[0], bits))
    b = torch.randn(h, 2 * i, dtype=torch.bfloat16, device=device)
    mm_ms = ms(lambda: torch.matmul(x, b))
    print(
        f"one_expert M=256 K=2048 N=1024 wna16={one_ms:.3f} "
        f"wxa8={i8_ms:.3f} matmul={mm_ms:.3f}",
        flush=True,
    )
    # 256 independent matmuls, the expert count of one projection.
    xs = torch.randn(e, m, h, dtype=torch.bfloat16, device=device)
    bs = torch.randn(e, h, 2 * i, dtype=torch.bfloat16, device=device)

    def loop_mm():
        for expert in range(e):
            torch.matmul(xs[expert], bs[expert])

    loop_ms = ms(loop_mm, warm=1, iters=1)
    print(f"matmul_loop E=256 M=256 ms={loop_ms:.2f}", flush=True)
    print("GROUPED_TIME_OK", flush=True)


if __name__ == "__main__":
    main()
