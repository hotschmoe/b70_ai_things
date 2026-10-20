"""One-call breakdown of bmm vs grouped_mm at the 35B expert shape.

Not a serve tok/s. Weights are random int3 planes unpacked once.
Spans are drained after warmup so they match the timed call.
"""

from __future__ import annotations

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import plane2_stride, plane_hi_stride
from b70_lowbit_xpu.moe import drain_timer_ms, expert_prefill_bmm


def sync():
    torch.xpu.synchronize()


def ms(fn, warm=2, iters=3) -> float:
    for _ in range(warm):
        fn()
    sync()
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    sync()
    return start.elapsed_time(end) / iters


def planes(e, n, k, bits, device):
    p2 = torch.randint(0, 256, (e, n, plane2_stride(k)), dtype=torch.uint8, device=device)
    hi = torch.randint(
        0, 256, (e, n, plane_hi_stride(k, bits)), dtype=torch.uint8, device=device
    )
    sc = torch.randn(e, n, k // 128, dtype=torch.bfloat16, device=device) * 0.01
    return p2, hi, sc


def unpack(p2, hi, sc, bits, rows, k):
    e = p2.shape[0]
    return ops.unpack_bf16(
        p2.reshape(e * rows, -1),
        hi.reshape(e * rows, -1),
        sc.reshape(e * rows, -1),
        bits,
    ).view(e, rows, k)


def group_offs(ids, num_experts):
    flat = ids.reshape(-1).to(torch.int32)
    key = torch.where(flat >= 0, flat, torch.full_like(flat, num_experts))
    order = torch.argsort(key, stable=True)
    counts = torch.bincount(key, minlength=num_experts + 1)
    active = counts[:num_experts]
    ends = torch.cumsum(active, dim=0).to(torch.int32)
    return order, ends, int(flat.shape[0])


def main() -> None:
    device = torch.device("xpu")
    bits = 3
    e, h, i, topk = 256, 2048, 512, 8
    for tokens in (2048, 8192):
        print(f"alloc T={tokens}", flush=True)
        p13, hi13, s13 = planes(e, 2 * i, h, bits, device)
        p2, hi2, s2 = planes(e, h, i, bits, device)
        g = torch.Generator(device="cpu")
        g.manual_seed(1)
        x = torch.randn(tokens, h, dtype=torch.bfloat16, device=device)
        ids = torch.randint(0, e, (tokens, topk), generator=g).to(device, dtype=torch.int32)
        router = torch.rand(tokens, topk, device=device)
        w13 = unpack(p13, hi13, s13, bits, 2 * i, h)
        w2 = unpack(p2, hi2, s2, bits, h, i)
        order, ends, slots = group_offs(ids, e)
        tok = torch.div(order, topk, rounding_mode="floor")
        xs = x.index_select(0, tok)
        # grouped_mm wants [E, K, N]. Unpack writes [E, N, K].
        w13_t = w13.transpose(-2, -1).contiguous()
        w2_t = w2.transpose(-2, -1).contiguous()
        sync()

        def bmm_only():
            # Match expert_prefill_bmm's GEMM, weights already unpacked.
            mmax = int(ends.diff().max().item()) if tokens else int(ends[0].item())
            # Use the full function so the span list is the serve path.
            return expert_prefill_bmm(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)

        def gemm_grouped():
            return torch.nn.functional.grouped_mm(xs, w13_t, offs=ends)

        def gemm_bmm_nt():
            # One bmm, no pad: not valid for ragged M. Reference cost of a
            # dense batched GEMM at the mean row count is measured separately.
            return torch.bmm(w13, xs[: e * (slots // e)].view(e, slots // e, h).transpose(1, 2))

        print(f"T={tokens} slots {slots} mmax_prep", flush=True)
        # Drop the first grouped_mm compile from the timed window.
        y = gemm_grouped()
        sync()
        print(f"T={tokens} grouped_out {tuple(y.shape)}", flush=True)
        gms = ms(gemm_grouped, warm=1, iters=3)
        # Transpose cost, weights already unpacked.
        tms = ms(lambda: w13.transpose(-2, -1).contiguous(), warm=1, iters=3)
        bms = ms(gemm_bmm_nt, warm=1, iters=3)
        print(
            f"T={tokens} grouped_w13_ms {gms:.2f} transpose_w13_ms {tms:.2f} "
            f"bmm_even_w13_ms {bms:.2f}",
            flush=True,
        )

        from b70_lowbit_xpu import moe as moe_mod

        moe_mod._TIMER_SPANS.clear()
        for _ in range(1):
            bmm_only()
        sync()
        moe_mod._TIMER_SPANS.clear()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        got = bmm_only()
        end.record()
        sync()
        spans = drain_timer_ms()
        print(
            f"T={tokens} full_bmm_ms {start.elapsed_time(end):.2f} out {tuple(got.shape)}",
            flush=True,
        )
        for name in sorted(spans):
            print(f"T={tokens} span {name} {spans[name]:.2f}", flush=True)
        # Empty-expert offs: duplicate an offset.
        if tokens == 2048:
            ends2 = ends.clone()
            ends2[3] = ends2[2]
            try:
                z = torch.nn.functional.grouped_mm(xs, w13_t, offs=ends2)
                sync()
                print(f"empty_expert_grouped {tuple(z.shape)}", flush=True)
            except Exception as exc:
                print(f"empty_expert_fail {type(exc).__name__} {exc}", flush=True)
    print("E2_GROUPED_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
