"""Unpack-to-bf16 plus vendor matmul. Not a serve tok/s.

Numeric gate is rel_l2 < 0.02 against prefill_wna16. Timing uses the
35B shapes: one q-proj (M=8192, K=2048, N=2048) and one expert w13.
"""

from __future__ import annotations

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import dequant, quantize_and_pack

PASS_REL = 0.02
FAILS: list[str] = []


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


def check(name, got, ref) -> None:
    torch.xpu.synchronize()
    g = got.float().reshape(-1)
    r = ref.float().reshape(-1)
    denom = torch.linalg.vector_norm(r).clamp_min(1e-8)
    rel = float(torch.linalg.vector_norm(g - r) / denom)
    ok = rel < PASS_REL
    print(f"  {'PASS' if ok else 'FAIL'} {name} rel_l2={rel:.4e}", flush=True)
    if not ok:
        FAILS.append(name)


def one(device, bits, m, n, k, seed) -> None:
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(n, k, generator=gen)
    p2, hi, sc = quantize_and_pack(weight, bits)
    p2, sc = p2.to(device), sc.to(device)
    hi = None if hi is None else hi.to(device)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    unpacked = ops.unpack_bf16(p2, hi, sc, bits)
    ref_w = dequant(p2.cpu(), None if hi is None else hi.cpu(), sc.cpu(), bits, k)
    check(
        f"b{bits} unpack N={n} K={k}",
        unpacked,
        ref_w.to(torch.bfloat16).to(device),
    )
    got = torch.nn.functional.linear(x, unpacked)
    ref = ops.prefill_wna16(x, p2, hi, sc, bits)
    check(f"b{bits} linear M={m} N={n} K={k}", got, ref)
    wna = ms(lambda: ops.prefill_wna16(x, p2, hi, sc, bits))
    vendor = ms(lambda: torch.nn.functional.linear(x, ops.unpack_bf16(p2, hi, sc, bits)))
    print(
        f"  time b{bits} M={m} N={n} K={k} wna16={wna:.3f} ms "
        f"unpack+matmul={vendor:.3f} ms",
        flush=True,
    )


def main() -> None:
    device = torch.device("xpu")
    print("so", __import__("os").environ.get("B70_LOWBIT_SO"), flush=True)
    one(device, 3, 32, 128, 256, 1)
    one(device, 2, 32, 128, 256, 2)
    one(device, 4, 32, 128, 256, 3)
    one(device, 3, 8192, 2048, 2048, 4)
    if FAILS:
        raise SystemExit("MATMUL_PROBE_FAIL " + ",".join(FAILS))
    print("MATMUL_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
