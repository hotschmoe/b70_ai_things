"""Where the 8.8 ms bmm_w13 span goes, and what index_add costs.

Not a serve tok/s. Random int3, T=8192, 35B expert shape.
Uses whatever unpack the loaded .so provides.
"""

from __future__ import annotations

import torch

from b70_lowbit_xpu import ops
from b70_lowbit_xpu.format import plane2_stride, plane_hi_stride


def sync():
    torch.xpu.synchronize()


def ms(fn, warm=2, iters=3) -> float:
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


def main() -> None:
    device = torch.device("xpu")
    bits = 3
    e, h, i, topk, tokens = 256, 2048, 512, 8, 8192
    n13 = 2 * i
    p2 = torch.randint(0, 256, (e, n13, plane2_stride(h)), dtype=torch.uint8, device=device)
    hi = torch.randint(
        0, 256, (e, n13, plane_hi_stride(h, bits)), dtype=torch.uint8, device=device
    )
    sc = torch.randn(e, n13, h // 128, dtype=torch.bfloat16, device=device) * 0.02
    print("unpack", flush=True)

    def do_unpack():
        return ops.unpack_bf16(
            p2.reshape(e * n13, -1),
            hi.reshape(e * n13, -1),
            sc.reshape(e * n13, -1),
            bits,
        )

    w = do_unpack().view(e, n13, h)
    sync()
    print(f"unpack_w13_ms {ms(do_unpack):.2f}", flush=True)
    g = torch.Generator(device="cpu")
    g.manual_seed(1)
    ids = torch.randint(0, e, (tokens, topk), generator=g).to(device, dtype=torch.int32)
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=e)
    mmax = int(counts.max().item())
    mmax_pad = (mmax + 7) & ~7
    slots = tokens * topk
    pos = torch.arange(slots, device=device, dtype=torch.int32)
    if e > 1:
        starts = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(counts[:-1], 0).to(torch.int32),
            )
        )
    else:
        starts = torch.zeros(1, dtype=torch.int32, device=device)
    local = pos - starts[flat[order]]
    tok = torch.div(order, topk, rounding_mode="floor")
    x = torch.randn(tokens, h, dtype=torch.bfloat16, device=device)
    gathered = x.index_select(0, tok)
    xpad = torch.empty(e, mmax_pad, h, dtype=torch.bfloat16, device=device)
    flat_i = flat[order].to(torch.int64) * mmax_pad + local.to(torch.int64)
    xpad.view(e * mmax_pad, h).index_copy_(0, flat_i, gathered)
    sync()
    print(f"mmax {mmax} mmax_pad {mmax_pad}", flush=True)

    def bmm_xt():
        return torch.bmm(w, xpad.transpose(1, 2).contiguous())

    def bmm_xt_view():
        return torch.bmm(w, xpad.transpose(1, 2))

    def bmm_wt():
        return torch.bmm(xpad, w.transpose(1, 2).contiguous())

    def bmm_wt_view():
        return torch.bmm(xpad, w.transpose(1, 2))

    y_nm = bmm_xt()
    sync()
    print(f"bmm_w_xt_contig_ms {ms(bmm_xt):.2f} shape {tuple(y_nm.shape)}", flush=True)
    y_view = bmm_xt_view()
    sync()
    print(f"bmm_w_xt_view_ms {ms(bmm_xt_view):.2f}", flush=True)
    y_mn = bmm_wt()
    sync()
    print(f"bmm_x_wt_contig_ms {ms(bmm_wt):.2f} shape {tuple(y_mn.shape)}", flush=True)
    try:
        y_wv = bmm_wt_view()
        sync()
        print(f"bmm_x_wt_view_ms {ms(bmm_wt_view):.2f} shape {tuple(y_wv.shape)}", flush=True)
    except Exception as exc:
        print(f"bmm_x_wt_view_fail {type(exc).__name__} {exc}", flush=True)

    def transpose_out():
        return y_nm.transpose(1, 2).contiguous()

    def gather_nm():
        return y_nm[flat[order], :, local]

    def gather_mn(buf=y_mn):
        return buf.reshape(e * mmax_pad, n13).index_select(0, flat_i)

    print(f"transpose_out_ms {ms(transpose_out):.2f}", flush=True)
    try:
        gth = gather_nm()
        sync()
        print(f"gather_nm_ms {ms(gather_nm):.2f} shape {tuple(gth.shape)}", flush=True)
    except Exception as exc:
        print(f"gather_nm_fail {type(exc).__name__} {exc}", flush=True)
    g2 = gather_mn()
    sync()
    print(f"gather_mn_ms {ms(gather_mn):.2f} shape {tuple(g2.shape)}", flush=True)

    router = torch.rand(tokens, topk, device=device)
    weight = router.reshape(-1)[order].unsqueeze(-1)
    down = torch.randn(slots, h, dtype=torch.bfloat16, device=device)
    y = torch.zeros(tokens, h, dtype=torch.float32, device=device)

    def reduce_fp32():
        y.zero_()
        y.index_add_(0, tok, (down.float() * weight))

    yb = torch.zeros(tokens, h, dtype=torch.bfloat16, device=device)

    def reduce_bf16():
        yb.zero_()
        yb.index_add_(0, tok, down * weight.to(torch.bfloat16))

    print(f"reduce_fp32_ms {ms(reduce_fp32, warm=1, iters=3):.2f}", flush=True)
    try:
        print(f"reduce_bf16_ms {ms(reduce_bf16, warm=1, iters=3):.2f}", flush=True)
    except Exception as exc:
        print(f"reduce_bf16_fail {type(exc).__name__} {exc}", flush=True)
    print("E2_LAYOUT_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
