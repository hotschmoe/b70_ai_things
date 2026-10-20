"""Where an 8k prefill layer goes. Not a serve tok/s.

Random int3 planes, 35B shapes. Times one expert-loop layer, the
unpack alone, four dense projections, and one decode GEMV layer.
"""

from __future__ import annotations

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import plane2_stride, plane_hi_stride


def ms(fn, warm=1, iters=2) -> float:
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


def expert_layer(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits):
    t_tokens, hidden = x.shape
    n13 = p13.shape[1]
    intermediate = n13 // 2
    num_experts = p13.shape[0]
    topk = ids.shape[1]
    w13 = ops.unpack_bf16(
        p13.reshape(num_experts * n13, -1),
        hi13.reshape(num_experts * n13, -1),
        s13.reshape(num_experts * n13, -1),
        bits,
    ).view(num_experts, n13, hidden)
    w2 = ops.unpack_bf16(
        p2.reshape(num_experts * hidden, -1),
        hi2.reshape(num_experts * hidden, -1),
        s2.reshape(num_experts * hidden, -1),
        bits,
    ).view(num_experts, hidden, intermediate)
    flat = ids.reshape(-1)
    key = torch.where(flat >= 0, flat, torch.full_like(flat, num_experts))
    counts = torch.bincount(key, minlength=num_experts + 1)
    order = torch.argsort(key, stable=True)
    counts_cpu = counts.tolist()
    y = torch.zeros(t_tokens, hidden, dtype=torch.float32, device=x.device)
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
        y.index_add_(0, tok, down.float() * weight.to(down.dtype))
    return y


def main() -> None:
    device = torch.device("xpu")
    bits = 3
    e, h, i, topk = 256, 2048, 512, 8
    p13, hi13, s13 = planes(e, 2 * i, h, bits, device)
    p2, hi2, s2 = planes(e, h, i, bits, device)
    t_tokens = 8192
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(t_tokens, h, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, e, (t_tokens, topk), generator=g).to(device)
    router = torch.rand(t_tokens, topk, device=device)
    print("ready", flush=True)
    unpack_ms = ms(
        lambda: ops.unpack_bf16(
            p13.reshape(e * (2 * i), -1),
            hi13.reshape(e * (2 * i), -1),
            s13.reshape(e * (2 * i), -1),
            bits,
        )
    )
    layer_ms = ms(
        lambda: expert_layer(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits),
        warm=1,
        iters=1,
    )
    print(f"expert_unpack_w13 ms={unpack_ms:.2f}", flush=True)
    print(
        f"expert_layer T=8192 ms={layer_ms:.2f} "
        f"x40={40 * layer_ms / 1000:.2f}s",
        flush=True,
    )
    # Four dense projections, q-proj shape.
    n = 2048
    pd, hid, sd = planes(1, n, h, bits, device)
    xd = x
    dense_ms = ms(
        lambda: torch.nn.functional.linear(
            xd, ops.unpack_bf16(pd[0], hid[0], sd[0], bits)
        )
    )
    print(
        f"dense_one M=8192 N=2048 ms={dense_ms:.3f} x160={160 * dense_ms / 1000:.2f}s",
        flush=True,
    )
    # Decode: one token, one moe_gemv, one gemv.
    x1 = x[:1]
    ids1 = ids[:1]
    r1 = router[:1]
    dec_moe = ms(
        lambda: ops.moe_gemv(x1, ids1, r1, p13, hi13, s13, p2, hi2, s2, bits)
    )
    dec_lin = ms(lambda: ops.gemv(x1, pd[0], hid[0], sd[0], bits))
    print(f"decode_moe_gemv ms={dec_moe:.3f} x40={40 * dec_moe:.2f}", flush=True)
    print(f"decode_gemv ms={dec_lin:.3f} x160={160 * dec_lin:.2f}", flush=True)
    print("PREFILL_SPLIT_OK", flush=True)


if __name__ == "__main__":
    main()
