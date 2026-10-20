"""Grouped unpack + one vendor bmm versus the per-expert loop.

Not a serve tok/s. Random int3 planes, 35B expert shape. Also times
moe_gemv at bits 2/3/4 and reports whether torch.bmm / grouped_mm run.
"""

from __future__ import annotations

import os

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import plane2_stride, plane_hi_stride
from b70_lowbit_xpu.moe import drain_timer_ms, expert_prefill_bmm


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
    hi = None
    if bits != 2:
        hi = torch.randint(
            0, 256, (e, n, plane_hi_stride(k, bits)), dtype=torch.uint8, device=device
        )
    sc = torch.randn(e, n, k // 128, dtype=torch.bfloat16, device=device) * 0.01
    return p2, hi, sc


def expert_loop(x, ids, router, w13, w2, intermediate):
    """The served matmul loop, without a second unpack."""
    num_experts = w13.shape[0]
    topk = ids.shape[1]
    hidden = x.shape[1]
    flat = ids.reshape(-1)
    key = torch.where(flat >= 0, flat, torch.full_like(flat, num_experts))
    counts = torch.bincount(key, minlength=num_experts + 1)
    order = torch.argsort(key, stable=True)
    counts_cpu = counts.tolist()
    y = torch.zeros(x.shape[0], hidden, dtype=torch.float32, device=x.device)
    cursor = 0
    for expert, n_tok in enumerate(counts_cpu[:num_experts]):
        n_hit = int(n_tok)
        if n_hit == 0:
            continue
        sel = order[cursor : cursor + n_hit]
        cursor += n_hit
        tok = torch.div(sel, topk, rounding_mode="floor")
        xe = x.index_select(0, tok)
        gu = torch.nn.functional.linear(xe, w13[expert])
        gate = gu[:, :intermediate]
        up = gu[:, intermediate:]
        hid = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
        down = torch.nn.functional.linear(hid, w2[expert])
        weight = router.reshape(-1)[sel].unsqueeze(-1)
        y.index_add_(0, tok, down.float() * weight)
    return y.to(torch.bfloat16)


def rel(a, b) -> float:
    d = (a.float() - b.float()).norm()
    den = b.float().norm().clamp_min(1e-8)
    return float((d / den).item())


def check_small(device) -> None:
    bits = 3
    e, h, i, topk, t = 4, 256, 128, 4, 32
    p13, hi13, s13 = planes(e, 2 * i, h, bits, device)
    p2, hi2, s2 = planes(e, h, i, bits, device)
    g = torch.Generator(device="cpu")
    g.manual_seed(2)
    x = torch.randn(t, h, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, e, (t, topk), generator=g).to(device=device, dtype=torch.int32)
    # Force one expert empty and one token onto expert 0 twice.
    ids[0, 0] = 0
    ids[0, 1] = 0
    ids[:, 2] = 1
    router = torch.rand(t, topk, device=device)
    os.environ["B70_LOWBIT_FWD_SPLIT"] = "0"
    w13 = ops.unpack_bf16(
        p13.reshape(e * (2 * i), -1), hi13.reshape(e * (2 * i), -1), s13.reshape(e * (2 * i), -1), bits
    ).view(e, 2 * i, h)
    w2 = ops.unpack_bf16(
        p2.reshape(e * h, -1), hi2.reshape(e * h, -1), s2.reshape(e * h, -1), bits
    ).view(e, h, i)
    ref = expert_loop(x, ids, router, w13, w2, i)
    got = expert_prefill_bmm(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
    err = rel(got, ref)
    print(f"small_rel {err:.6e}", flush=True)
    if err > 2e-2:
        raise SystemExit(f"BMM_REL {err}")
    print("SMALL_BMM_OK", flush=True)


def time_shape(device, tokens: int) -> None:
    bits = 3
    e, h, i, topk = 256, 2048, 512, 8
    print(f"alloc T={tokens}", flush=True)
    p13, hi13, s13 = planes(e, 2 * i, h, bits, device)
    p2, hi2, s2 = planes(e, h, i, bits, device)
    g = torch.Generator(device="cpu")
    g.manual_seed(1)
    x = torch.randn(tokens, h, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, e, (tokens, topk), generator=g).to(device=device, dtype=torch.int32)
    router = torch.rand(tokens, topk, device=device)
    w13 = ops.unpack_bf16(
        p13.reshape(e * (2 * i), -1),
        hi13.reshape(e * (2 * i), -1),
        s13.reshape(e * (2 * i), -1),
        bits,
    ).view(e, 2 * i, h)
    w2 = ops.unpack_bf16(
        p2.reshape(e * h, -1), hi2.reshape(e * h, -1), s2.reshape(e * h, -1), bits
    ).view(e, h, i)
    loop_ms = ms(lambda: expert_loop(x, ids, router, w13, w2, i), warm=1, iters=1)
    os.environ["B70_LOWBIT_FWD_SPLIT"] = "1"
    from b70_lowbit_xpu import moe as moe_mod

    moe_mod._TIMER_SPANS.clear()

    def once():
        return expert_prefill_bmm(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)

    bmm_ms = ms(once, warm=1, iters=1)
    spans = drain_timer_ms()
    got = once()
    torch.xpu.synchronize()
    ref = expert_loop(x, ids, router, w13, w2, i)
    err = rel(got, ref)
    print(
        f"T={tokens} loop_ms {loop_ms:.1f} bmm_ms {bmm_ms:.1f} rel {err:.6e}",
        flush=True,
    )
    for name in sorted(spans):
        print(f"T={tokens} span {name} {spans[name]:.2f}", flush=True)
    if err > 2e-2:
        raise SystemExit(f"BMM_REL_T{tokens} {err}")


def time_gemv(device) -> None:
    e, h, i, topk = 256, 2048, 512, 8
    x = torch.randn(1, h, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, e, (1, topk), device=device, dtype=torch.int32)
    router = torch.rand(1, topk, device=device)
    for bits in (4, 3, 2):
        p13, hi13, s13 = planes(e, 2 * i, h, bits, device)
        p2, hi2, s2 = planes(e, h, i, bits, device)
        if hi13 is None:
            hi13 = torch.empty(0, dtype=torch.uint8, device=device)
            hi2 = torch.empty(0, dtype=torch.uint8, device=device)
        def run(p13=p13, hi13=hi13, s13=s13, p2=p2, hi2=hi2, s2=s2, bits=bits):
            return ops.moe_gemv(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
        took = ms(run, warm=2, iters=5)
        print(f"moe_gemv bits {bits} ms {took:.4f} rows {os.environ.get('B70_LOWBIT_GEMV_ROWS', '0')}", flush=True)


def grouped_mm_probe(device) -> None:
    name = "none"
    if hasattr(torch.nn.functional, "grouped_mm"):
        name = "grouped_mm"
    print(f"grouped_api {name}", flush=True)
    try:
        a = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
        b = torch.randn(2, 64, 32, dtype=torch.bfloat16, device=device)
        offs = torch.tensor([16, 32], dtype=torch.int32, device=device)
        if hasattr(torch.nn.functional, "grouped_mm"):
            y = torch.nn.functional.grouped_mm(a, b, offs=offs)
            torch.xpu.synchronize()
            print(f"grouped_mm shape {tuple(y.shape)}", flush=True)
        else:
            print("grouped_mm absent", flush=True)
    except Exception as exc:
        print(f"grouped_mm fail {type(exc).__name__} {exc}", flush=True)


def main() -> None:
    device = torch.device("xpu")
    print("probe", flush=True)
    grouped_mm_probe(device)
    check_small(device)
    time_shape(device, 2048)
    time_shape(device, 8192)
    time_gemv(device)
    print("E2_BMM_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
