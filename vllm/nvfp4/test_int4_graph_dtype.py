#!/usr/bin/env python3
"""O2: int4_gemm_w4a16 is fp16-out; GRAPH dummy_run wants layer dtype.

Opaque b70::int4_gemm_w4a16_cast must capture on bf16 x without Half!=BF16.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/opt/nvfp4_shim")
import vllm_xpu_kernels._xpu_C  # noqa: F401, E402
import draft_mtp_int4 as d  # noqa: E402


def main():
    assert hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"), "no int4_gemm_w4a16"
    dev = "xpu:0"
    torch.xpu.set_device(0)
    n_out, k_in = 64, 128
    w = torch.randn(n_out, k_in, dtype=torch.bfloat16, device=dev)
    qweight, scales, qzeros, gs = d.quantize_to_int4(w)
    x = torch.randn(2, k_in, dtype=torch.bfloat16, device=dev)

    flat = x.reshape(-1, k_in).to(torch.float16)
    raw = torch.ops._xpu_C.int4_gemm_w4a16(
        flat, qweight, None, scales, qzeros, gs, None
    )
    print(f"raw_out_dtype={raw.dtype} expect=torch.float16")
    if raw.dtype != torch.float16:
        print("FAIL raw kernel dtype")
        sys.exit(1)

    d._ensure_cast_op()
    assert hasattr(torch.ops.b70, "int4_gemm_w4a16_cast"), "no cast op"
    cast = torch.ops.b70.int4_gemm_w4a16_cast(x, qweight, scales, qzeros, gs)
    print(f"cast_out_dtype={cast.dtype} expect={x.dtype}")
    if cast.dtype != x.dtype:
        print("FAIL cast dtype")
        sys.exit(1)

    meth = d._Int4LinearMethod(qweight, scales, qzeros, gs)

    class _L:
        pass

    y = meth.apply(_L(), x, None)
    print(f"apply_out_dtype={y.dtype} shape={tuple(y.shape)}")
    if y.dtype != x.dtype or y.shape[-1] != n_out:
        print("FAIL apply")
        sys.exit(1)

    out_g = torch.empty_like(y)
    g = torch.xpu.XPUGraph()
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            z = meth.apply(_L(), x, None)
            out_g.copy_(z)
        torch.xpu.current_stream().synchronize()
        with torch.xpu.graph(g):
            z = meth.apply(_L(), x, None)
            out_g.copy_(z)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()
    print(f"graph_out_dtype={out_g.dtype}")
    g.replay()
    torch.xpu.synchronize()
    print("capture OK")
    print("PASS")


if __name__ == "__main__":
    main()
