"""ESIMD unpack vs format.dequant, and multi-row moe_gemv vs 1-row.

Not a serve tok/s. B70_LOWBIT_SO should be libb70_lowbit_xpu_e2u.so.
B70_LOWBIT_UNPACK and B70_LOWBIT_GEMV_ROWS are read on each launch.
"""

from __future__ import annotations

import os

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import dequant, plane2_stride, plane_hi_stride


def sync():
    torch.xpu.synchronize()


def ms(fn, warm=2, iters=5) -> float:
    for _ in range(warm):
        fn()
    sync()
    a = torch.xpu.Event(enable_timing=True)
    b = torch.xpu.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    sync()
    return a.elapsed_time(b) / iters


def planes(n, k, bits, device):
    p2 = torch.randint(0, 256, (n, plane2_stride(k)), dtype=torch.uint8, device=device)
    hi = None
    if bits != 2:
        hi = torch.randint(
            0, 256, (n, plane_hi_stride(k, bits)), dtype=torch.uint8, device=device
        )
    sc = torch.randn(n, k // 128, dtype=torch.bfloat16, device=device) * 0.02
    return p2, hi, sc


def check_unpack(device) -> None:
    for bits in (2, 3, 4):
        n, k = 32, 512
        p2, hi, sc = planes(n, k, bits, device)
        ref = dequant(
            p2.cpu(), None if hi is None else hi.cpu(), sc.cpu(), bits, k
        ).to(torch.bfloat16)
        for mode in ("esimd", "scalar"):
            os.environ["B70_LOWBIT_UNPACK"] = "" if mode == "esimd" else "scalar"
            got = ops.unpack_bf16(
                p2,
                hi if hi is not None else torch.empty(0, dtype=torch.uint8, device=device),
                sc,
                bits,
            )
            sync()
            err = (got.cpu().float() - ref.float()).abs()
            print(
                f"unpack bits {bits} {mode} max_abs {float(err.max()):.6e} "
                f"mean {float(err.mean()):.6e}",
                flush=True,
            )
            if float(err.max()) > 0:
                # bf16 ulp at these magnitudes is tiny; more than 1 ulp fails.
                raise SystemExit(f"UNPACK_MISMATCH bits {bits} {mode}")


def time_unpack(device) -> None:
    bits = 3
    e, n, k = 256, 1024, 2048
    p2, hi, sc = planes(e * n, k, bits, device)
    for mode in ("esimd", "scalar"):
        os.environ["B70_LOWBIT_UNPACK"] = "" if mode == "esimd" else "scalar"

        def run(p2=p2, hi=hi, sc=sc):
            return ops.unpack_bf16(p2, hi, sc, bits)

        took = ms(run, warm=1, iters=3)
        print(f"unpack_w13 {mode} ms {took:.2f}", flush=True)
    # down projection shape: E*H rows, K=I
    p2, hi, sc = planes(e * 2048, 512, bits, device)
    for mode in ("esimd", "scalar"):
        os.environ["B70_LOWBIT_UNPACK"] = "" if mode == "esimd" else "scalar"

        def run(p2=p2, hi=hi, sc=sc):
            return ops.unpack_bf16(p2, hi, sc, bits)

        took = ms(run, warm=1, iters=3)
        print(f"unpack_w2 {mode} ms {took:.2f}", flush=True)


def time_gemv(device) -> None:
    os.environ["B70_LOWBIT_MOE_LAUNCHES"] = "2"
    e, h, i, topk = 256, 2048, 512, 8
    x = torch.randn(1, h, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, e, (1, topk), device=device, dtype=torch.int32)
    router = torch.rand(1, topk, device=device)
    for bits in (4, 3, 2):
        p13, hi13, s13 = planes(e * (2 * i), h, bits, "cpu")
        p2, hi2, s2 = planes(e * h, i, bits, "cpu")
        # Expert-major views: [E, N, stride]
        p13 = p13.view(e, 2 * i, -1).to(device)
        p2 = p2.view(e, h, -1).to(device)
        s13 = s13.view(e, 2 * i, -1).to(device)
        s2 = s2.view(e, h, -1).to(device)
        if hi13 is None:
            hi13 = torch.empty(0, dtype=torch.uint8, device=device)
            hi2 = torch.empty(0, dtype=torch.uint8, device=device)
        else:
            hi13 = hi13.view(e, 2 * i, -1).to(device)
            hi2 = hi2.view(e, h, -1).to(device)
        def run():
            return ops.moe_gemv(x, ids, router, p13, hi13, s13, p2, hi2, s2, bits)

        os.environ["B70_LOWBIT_GEMV_ROWS"] = "0"
        y0 = run()
        sync()
        took0 = ms(run, warm=2, iters=5)
        for rows in ("2", "4"):
            os.environ["B70_LOWBIT_GEMV_ROWS"] = rows
            y = run()
            sync()
            err = (y.float() - y0.float()).abs()
            took = ms(run, warm=2, iters=5)
            print(
                f"gemv bits {bits} rows {rows} max_abs {float(err.max()):.6e} "
                f"ms {took:.4f} rows0_ms {took0:.4f}",
                flush=True,
            )
            if float(err.max()) > 1e-3:
                raise SystemExit(f"GEMV_MISMATCH bits {bits} rows {rows}")
        del p13, hi13, s13, p2, hi2, s2, y0
        sync()


def layout_bmm(device) -> None:
    """Split the w13 bmm span. Also one expert_prefill_bmm call at T=8192."""
    from b70_lowbit_xpu.moe import drain_timer_ms, expert_prefill_bmm

    bits = 3
    e, h, i, topk, tokens = 256, 2048, 512, 8, 8192
    n13 = 2 * i
    p2, hi, sc = planes(e * n13, h, bits, device)
    w = ops.unpack_bf16(p2, hi, sc, bits).view(e, n13, h)
    g = torch.Generator(device="cpu")
    g.manual_seed(1)
    ids = torch.randint(0, e, (tokens, topk), generator=g).to(device, dtype=torch.int32)
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat.to(torch.int64), minlength=e)
    mmax = int(counts.max().item())
    mmax_pad = (mmax + 7) & ~7
    slots = tokens * topk
    pos = torch.arange(slots, device=device, dtype=torch.int32)
    starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.cumsum(counts[:-1], 0),
        )
    )
    local = (pos.to(torch.int64) - starts[flat.to(torch.int64)[order]]).to(torch.int32)
    expert = flat[order]
    tok = torch.div(order, topk, rounding_mode="floor")
    x = torch.randn(tokens, h, dtype=torch.bfloat16, device=device)
    xpad = torch.empty(e, mmax_pad, h, dtype=torch.bfloat16, device=device)
    flat_i = expert.to(torch.int64) * mmax_pad + local.to(torch.int64)
    xpad.view(e * mmax_pad, h).index_copy_(0, flat_i, x.index_select(0, tok))
    sync()
    print(f"layout mmax {mmax}", flush=True)

    def bmm_contig():
        return torch.bmm(w, xpad.transpose(1, 2).contiguous())

    def bmm_view():
        return torch.bmm(w, xpad.transpose(1, 2))

    def bmm_wt():
        return torch.bmm(xpad, w.transpose(1, 2))

    y_nm = bmm_view()
    sync()
    print(f"bmm_xt_contig_ms {ms(bmm_contig, warm=1, iters=3):.2f}", flush=True)
    print(f"bmm_xt_view_ms {ms(bmm_view, warm=1, iters=3):.2f}", flush=True)
    try:
        y_mn = bmm_wt()
        sync()
        print(
            f"bmm_x_wt_view_ms {ms(bmm_wt, warm=1, iters=3):.2f} shape {tuple(y_mn.shape)}",
            flush=True,
        )
    except Exception as exc:
        y_mn = None
        print(f"bmm_x_wt_view_fail {type(exc).__name__} {exc}", flush=True)

    def transpose_out():
        return y_nm.transpose(1, 2).contiguous()

    print(f"transpose_out_ms {ms(transpose_out, warm=1, iters=3):.2f}", flush=True)
    try:
        def gather_nm():
            return y_nm[expert.to(torch.int64), :, local.to(torch.int64)]

        gth = gather_nm()
        sync()
        print(f"gather_nm_ms {ms(gather_nm, warm=1, iters=3):.2f} shape {tuple(gth.shape)}", flush=True)
    except Exception as exc:
        print(f"gather_nm_fail {type(exc).__name__} {exc}", flush=True)
    if y_mn is not None:
        def gather_mn():
            return y_mn.reshape(e * mmax_pad, n13).index_select(0, flat_i)

        print(f"gather_mn_ms {ms(gather_mn, warm=1, iters=3):.2f}", flush=True)

    # Full path the serve will call, one layer, spans after warmup.
    p2b, hi2, s2 = planes(e * h, i, bits, device)
    router = torch.rand(tokens, topk, device=device)
    os.environ["B70_LOWBIT_FWD_SPLIT"] = "1"
    from b70_lowbit_xpu import moe as moe_mod

    for _ in range(1):
        expert_prefill_bmm(x, ids, router, p2.view(e, n13, -1), hi.view(e, n13, -1), sc.view(e, n13, -1), p2b.view(e, h, -1), hi2.view(e, h, -1), s2.view(e, h, -1), bits)
    sync()
    moe_mod._TIMER_SPANS.clear()
    a = torch.xpu.Event(enable_timing=True)
    b = torch.xpu.Event(enable_timing=True)
    a.record()
    expert_prefill_bmm(
        x, ids, router,
        p2.view(e, n13, -1), hi.view(e, n13, -1), sc.view(e, n13, -1),
        p2b.view(e, h, -1), hi2.view(e, h, -1), s2.view(e, h, -1),
        bits,
    )
    b.record()
    sync()
    print(f"serve_bmm_layer_ms {a.elapsed_time(b):.2f}", flush=True)
    spans = drain_timer_ms()
    for name in sorted(spans):
        print(f"serve_span {name} {spans[name]:.2f}", flush=True)


def main() -> None:
    device = torch.device("xpu")
    print("fast_probe", flush=True)
    check_unpack(device)
    time_unpack(device)
    time_gemv(device)
    layout_bmm(device)
    print("E2_FAST_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
