#!/usr/bin/env python3
# int4_gemm_w4a8_probe.py -- microbench vLLM's torch.ops._xpu_C.int4_gemm_w4a8
# (oneDNN/SYCL fused int4-weight x int8-activation GEMM) on a single Arc B70.
#
# Goal: decide if this op is a viable fast W4A8 prefill kernel to port into sglang.
# Compares against bf16 matmul at the same shape, and reports correctness relerr.
#
# Shape = real checkpoint layer down_proj (group-128):
#   in (K) = 17408, out (N) = 5120
#   weight (packed)  I32 [OUT, IN/8] = [5120, 2176]
#   weight_scale     BF16 [OUT, IN/group] = [5120, 136]
#
# Run inside vllm-xpu-env:v0230 with /models mounted, card 0 pinned.
# ASCII only.

import time
import torch

DEV = "xpu"
IN = 17408
OUT = 5120
GROUP = 128
CKPT = "/models/Qwen3.6-27B-W4A8-sqgptq-prepacked/model.safetensors"
KEY_W = "model.language_model.layers.20.mlp.down_proj.weight"
KEY_S = "model.language_model.layers.20.mlp.down_proj.weight_scale"

WARMUP = 20
ITERS = 60


def sync():
    torch.xpu.synchronize()


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e3  # ms/call


def unpack_weight_to_fp32(wq_i32, ws_bf16):
    # wq_i32: [OUT, IN/8] int32 packed (8 nibbles per int32, nibble = value+8)
    # ws_bf16: [OUT, IN/group] bf16
    OUT_, K8 = wq_i32.shape
    shifts = torch.arange(0, 32, 4, device=wq_i32.device, dtype=torch.int32)
    nib = (wq_i32.unsqueeze(-1) >> shifts) & 0xF        # [OUT, K8, 8] in 0..15
    vals = (nib.to(torch.int32) - 8).reshape(OUT_, K8 * 8)  # [-8,7], [OUT, IN]
    ws_full = ws_bf16.to(torch.float32).repeat_interleave(GROUP, dim=1)  # [OUT, IN]
    return vals.to(torch.float32) * ws_full             # [OUT, IN] fp32


def main():
    print("=== int4_gemm_w4a8 probe on B70 ===", flush=True)
    print("torch", torch.__version__, "xpu_avail", torch.xpu.is_available(), flush=True)
    print("device:", torch.xpu.get_device_name(0), flush=True)

    import vllm_xpu_kernels._xpu_C  # noqa: registers torch.ops._xpu_C
    from vllm._xpu_ops import xpu_ops as ops

    has = hasattr(torch.ops._xpu_C, "int4_gemm_w4a8")
    print("HAS_OP int4_gemm_w4a8:", has, flush=True)
    assert has

    from safetensors import safe_open
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        wq = f.get_tensor(KEY_W)   # [5120, 2176] int32
        ws = f.get_tensor(KEY_S)   # [5120, 136]  bf16
    print("ckpt weight", tuple(wq.shape), wq.dtype, "scale", tuple(ws.shape), ws.dtype, flush=True)

    wq = wq.to(DEV)
    ws = ws.to(DEV)

    # Layout exactly as vllm XPUW4A8IntLinearKernel.apply_weights feeds the op:
    qweight = wq.t()                    # [IN/8, OUT] = [2176, 5120] int32 (non-contig view)
    weight_scale = ws.t().contiguous()  # [IN/group, OUT] = [136, 5120] bf16
    weight_zp = torch.tensor([8], dtype=torch.int8, device=DEV)

    # bf16 reference weight (dequantized) for baseline matmul + correctness
    w_fp32 = unpack_weight_to_fp32(wq, ws)   # [OUT, IN] fp32
    w_bf16 = w_fp32.to(torch.bfloat16)       # [OUT, IN] for bf16 baseline
    w_fp16 = w_fp32.to(torch.float16)        # [OUT, IN] for fp16 baseline

    results = {}
    for M in (1, 2048):
        print(f"\n--- M={M} ---", flush=True)
        # fp16 activation (op native: produces fp16, recommends fp16 dtype)
        x16 = torch.randn(M, IN, device=DEV, dtype=torch.float16)
        xbf = x16.to(torch.bfloat16)

        # pre-quantize activation once (for op-only timing)
        qx, xs, xz = ops.dynamic_per_token_int8_quant_ref(x16, True, 8)
        qx = qx.contiguous()
        print("  quant_x", tuple(qx.shape), qx.dtype, "x_scale", tuple(xs.shape), xs.dtype,
              "x_zero", tuple(xz.shape), xz.dtype, flush=True)

        def op_only():
            return torch.ops._xpu_C.int4_gemm_w4a8(
                qx, xs, xz, qweight, weight_scale, weight_zp, GROUP, None, None)

        def full_path():
            q, s, z = ops.dynamic_per_token_int8_quant_ref(x16, True, 8)
            return torch.ops._xpu_C.int4_gemm_w4a8(
                q, s, z, qweight, weight_scale, weight_zp, GROUP, None, None)

        def bf16_mm():
            return torch.nn.functional.linear(xbf, w_bf16)

        def fp16_mm():
            return torch.nn.functional.linear(x16, w_fp16)

        # correctness: kernel out vs fp32(qx*xs) @ w_fp32.T
        out = op_only()
        sync()
        print("  op out", tuple(out.shape), out.dtype, flush=True)
        ref = (qx.to(torch.float32) * xs.to(torch.float32)) @ w_fp32.t()  # [M, OUT]
        relerr = (out.to(torch.float32) - ref).norm() / ref.norm()
        # also vs a pure bf16 matmul (includes activation-quant error -- looser)
        ref_bf = (xbf.to(torch.float32)) @ w_fp32.t()
        relerr_vs_bf = (out.to(torch.float32) - ref_bf).norm() / ref_bf.norm()
        print(f"  relerr(op vs dequant-consistent) = {relerr.item():.4e}", flush=True)
        print(f"  relerr(op vs full-bf16)          = {relerr_vs_bf.item():.4e}", flush=True)

        t_op = bench(op_only)
        t_full = bench(full_path)
        t_bf = bench(bf16_mm)
        t_fp = bench(fp16_mm)
        print(f"  int4_gemm_w4a8 (op only)   : {t_op:.4f} ms", flush=True)
        print(f"  int4_gemm_w4a8 (op+quant)  : {t_full:.4f} ms", flush=True)
        print(f"  bf16 linear  baseline      : {t_bf:.4f} ms", flush=True)
        print(f"  fp16 linear  baseline      : {t_fp:.4f} ms", flush=True)
        print(f"  speedup op-only vs bf16    : {t_bf / t_op:.3f}x", flush=True)
        print(f"  speedup op+quant vs bf16   : {t_bf / t_full:.3f}x", flush=True)
        results[M] = (t_op, t_full, t_bf, t_fp, relerr.item())

    print("\n=== SUMMARY (down_proj K=17408 N=5120 g128) ===", flush=True)
    for M, (t_op, t_full, t_bf, t_fp, re) in results.items():
        print(f"M={M:5d}: w4a8_op={t_op:.4f}ms  w4a8_full={t_full:.4f}ms  "
              f"bf16={t_bf:.4f}ms  op_vs_bf16={t_bf/t_op:.3f}x  relerr={re:.2e}", flush=True)


if __name__ == "__main__":
    main()
