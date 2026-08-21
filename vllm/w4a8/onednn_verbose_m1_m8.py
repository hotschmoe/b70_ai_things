#!/usr/bin/env python3
# K4: ONEDNN_VERBOSE dump of int4_gemm_w4a8 at M=1 vs M=8 on down_proj.
# Campaign: is DPAS even on at the DSpark verify tile? Not a serve.
# ASCII only. Set ONEDNN_VERBOSE=1 in the environment before launch.
import os
import sys
import ctypes
import torch

DEV = "xpu"
GROUP = 128
K, N = 17408, 5120
SO = os.environ.get("B70_XPU_C_SO", "")


def pack_int4(w_i8):
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=w_i8.device)
    u = (w_i8.to(torch.int32) + 8).reshape(w_i8.shape[0], w_i8.shape[1] // 8, 8)
    return ((u & 0xF) << shifts[None, None, :]).sum(dim=2).to(torch.int32)


def main():
    print("torch", torch.__version__, "xpu", torch.xpu.is_available(), flush=True)
    if SO:
        ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)
        print("CDLL OK", SO, flush=True)
    ops = torch.ops._xpu_C
    if not hasattr(ops, "int4_gemm_w4a8"):
        print("FAIL: int4_gemm_w4a8 missing", flush=True)
        sys.exit(3)
    w_i8 = torch.randint(-8, 8, (N, K), device=DEV, dtype=torch.int8)
    wq = pack_int4(w_i8)
    ws = (torch.rand(N, K // GROUP, device=DEV, dtype=torch.float16) * 0.05 + 0.001)
    qweight = wq.t()
    assert qweight.stride()[0] == 1, qweight.stride()
    wscale = ws.t().contiguous()
    wzp = torch.tensor([8], dtype=torch.int8, device=DEV)
    ms = os.environ.get("ONLY_MS", "1,8")
    MS = [int(x) for x in ms.split(",") if x.strip()]
    print(f"down_proj K={K} N={N} packed {tuple(wq.shape)} NT {tuple(qweight.shape)} "
          f"stride {qweight.stride()} ONEDNN_VERBOSE={os.environ.get('ONEDNN_VERBOSE')} MS={MS}",
          flush=True)
    for M in MS:
        x = torch.randn(M, K, device=DEV, dtype=torch.float16) * 0.05
        amax = x.abs().amax(-1, keepdim=True).clamp_(min=1e-5)
        xs = (amax / 127.0).to(torch.float16).contiguous()
        xq = (x / xs).round().clamp_(-127, 127).to(torch.int8).contiguous()
        xz = torch.zeros(M, 1, dtype=torch.int32, device=DEV).contiguous()
        print(f"=== CALL M={M} int4_gemm_w4a8 ===", flush=True)
        for i in range(3):
            y = ops.int4_gemm_w4a8(xq, xs, xz, qweight, wscale, wzp, GROUP, None, None)
            torch.xpu.synchronize()
            if i == 0:
                print(f"  y {tuple(y.shape)} {y.dtype}", flush=True)
        print(f"=== CALL M={M} int4_gemm_w4a16 ===", flush=True)
        y2 = ops.int4_gemm_w4a16(x, qweight, None, wscale, wzp, GROUP, None)
        torch.xpu.synchronize()
        print(f"  y {tuple(y2.shape)} {y2.dtype}", flush=True)
    print("DONE_K4_ONEDNN", flush=True)


if __name__ == "__main__":
    main()
