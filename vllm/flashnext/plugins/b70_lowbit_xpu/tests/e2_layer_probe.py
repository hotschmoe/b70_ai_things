"""New bmm layout plus ESIMD unpack, one expert layer. Not a serve tok/s."""

from __future__ import annotations

import os

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import plane2_stride, plane_hi_stride
from b70_lowbit_xpu.moe import drain_timer_ms, expert_prefill_bmm


def sync():
    torch.xpu.synchronize()


def planes(e, n, k, bits, device):
    p2 = torch.randint(0, 256, (e, n, plane2_stride(k)), dtype=torch.uint8, device=device)
    hi = torch.randint(
        0, 256, (e, n, plane_hi_stride(k, bits)), dtype=torch.uint8, device=device
    )
    sc = torch.randn(e, n, k // 128, dtype=torch.bfloat16, device=device) * 0.02
    return p2, hi, sc


def expert_loop(x, ids, router, w13, w2, intermediate):
    num_experts = w13.shape[0]
    topk = ids.shape[1]
    hidden = x.shape[1]
    flat = ids.reshape(-1)
    key = torch.where(flat >= 0, flat, torch.full_like(flat, num_experts))
    order = torch.argsort(key, stable=True)
    counts = torch.bincount(key.to(torch.int64), minlength=num_experts + 1).tolist()
    y = torch.zeros(x.shape[0], hidden, dtype=torch.float32, device=x.device)
    cursor = 0
    for expert, n_tok in enumerate(counts[:num_experts]):
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


def main() -> None:
    device = torch.device("xpu")
    bits = 3
    os.environ["B70_LOWBIT_FWD_SPLIT"] = "0"
    # Small, including an empty expert and a repeated id.
    e, h, i, topk, t = 4, 256, 128, 4, 32
    p13, hi13, s13 = planes(e, 2 * i, h, bits, device)
    p2, hi2, s2 = planes(e, h, i, bits, device)
    g = torch.Generator(device="cpu")
    g.manual_seed(2)
    x = torch.randn(t, h, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, e, (t, topk), generator=g).to(device, dtype=torch.int32)
    ids[0, 0] = 0
    ids[0, 1] = 0
    ids[:, 2] = 1
    router = torch.rand(t, topk, device=device)
    w13 = ops.unpack_bf16(
        p13.reshape(e * (2 * i), -1), hi13.reshape(e * (2 * i), -1), s13.reshape(e * (2 * i), -1), bits
    ).view(e, 2 * i, h)
    w2 = ops.unpack_bf16(
        p2.reshape(e * h, -1), hi2.reshape(e * h, -1), s2.reshape(e * h, -1), bits
    ).view(e, h, i)
    ref = expert_loop(x, ids, router, w13, w2, i)
    got = expert_prefill_bmm(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
    sync()
    err = rel(got, ref)
    print(f"small_rel {err:.6e}", flush=True)
    if err > 2e-2:
        raise SystemExit(f"LAYER_REL {err}")

    e, h, i, topk, tokens = 256, 2048, 512, 8, 8192
    print("alloc 8192", flush=True)
    p13, hi13, s13 = planes(e, 2 * i, h, bits, device)
    p2, hi2, s2 = planes(e, h, i, bits, device)
    g.manual_seed(1)
    x = torch.randn(tokens, h, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, e, (tokens, topk), generator=g).to(device, dtype=torch.int32)
    router = torch.rand(tokens, topk, device=device)
    os.environ["B70_LOWBIT_FWD_SPLIT"] = "1"
    from b70_lowbit_xpu import moe as moe_mod

    expert_prefill_bmm(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
    sync()
    moe_mod._TIMER_SPANS.clear()
    a = torch.xpu.Event(enable_timing=True)
    b = torch.xpu.Event(enable_timing=True)
    a.record()
    got = expert_prefill_bmm(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)
    b.record()
    sync()
    spans = drain_timer_ms()
    print(f"layer_ms {a.elapsed_time(b):.2f}", flush=True)
    for name in sorted(spans):
        print(f"span {name} {spans[name]:.2f}", flush=True)
    w13 = ops.unpack_bf16(
        p13.reshape(e * (2 * i), -1), hi13.reshape(e * (2 * i), -1), s13.reshape(e * (2 * i), -1), bits
    ).view(e, 2 * i, h)
    w2 = ops.unpack_bf16(
        p2.reshape(e * h, -1), hi2.reshape(e * h, -1), s2.reshape(e * h, -1), bits
    ).view(e, h, i)
    ref = expert_loop(x, ids, router, w13, w2, i)
    sync()
    err = rel(got, ref)
    print(f"T8192_rel {err:.6e}", flush=True)
    if err > 2e-2:
        raise SystemExit(f"LAYER_REL_8K {err}")
    print("E2_LAYER_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
