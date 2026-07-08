# INT8-XMX prefill CEILING probe for the NVFP4 27B MLP (Track: reclaim 2x prefill).
#
# Question: today the fused serve runs nvfp4_gemm_w4a16 (f4_e2m1 weight-decompress,
# bf16 COMPUTE ~183 TFLOPS) for BOTH decode AND prefill. Prefill is compute-bound;
# B70 INT8 XMX is ~367 TOPS = 2x. Is there a real INT8-XMX prefill win to reclaim?
#
# This measures the pure SPEED ceiling on the REAL MLP shapes at prefill M, IGNORING
# numerics for (b)/(d) (random data -> GEMM time is data-independent for these ops):
#   (a) nvfp4_gemm_w4a16  -- CURRENT path (bf16 compute, 4-bit resident weight)
#   (b) int8_gemm_w8a8    -- s8 x s8, resident s8 weight + per-channel scale (INT8 XMX ceiling)
#   (c) bf16 F.linear     -- the naive bf16 baseline
#   (d) int4_gemm_w4a8    -- s8 x int4 weight + per-16-K-group scale (block-scaled INT8 probe;
#                            keeps weight 4-bit resident; speed signal only, NVFP4 can't use int4 numerics)
#
# Run inside vllm-xpu-env:int8g-v0240 with nvfp4_fused_kernel_gdn/_xpu_C.abi3.so mounted
# (it registers ALL of nvfp4_gemm_w4a16 / int8_gemm_w8a8 / int8_gemm_w8a16 / int4_gemm_w4a8).
import os
import time
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401  (registers torch.ops._xpu_C.*)

DEV = "xpu"
torch.manual_seed(0)

# Real 27B MLP shapes (per config: hidden 5120, intermediate 17408).
SHAPES = {
    "gate/up (N=17408 K=5120)": (17408, 5120),
    "down    (N=5120 K=17408)": (5120, 17408),
}
# Lean default M set (override with MS="256,512,..."); DO_I4=1 adds the int4_w4a8 probe.
MS = [int(x) for x in (os.environ.get("MS") or "512,2048,8192").split(",")]
DO_I4 = os.environ.get("DO_I4", "0") == "1"
ITERS = int(os.environ.get("ITERS", "20"))


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.time() - t) / iters * 1e3  # ms


def make_weights(N, K):
    W = {}
    # (a) nvfp4: packed f4_e2m1 [N, K/2] uint8 + [K/16, N] bf16 scale
    W["packed"] = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=DEV)
    W["nv_scale"] = (torch.rand(K // 16, N, device=DEV, dtype=torch.bfloat16) * 0.1 + 0.01)
    # (b) int8 w8a8: resident s8 weight [N,K] + per-channel [N] f32 scale
    W["s8"] = torch.randint(-12, 13, (N, K), dtype=torch.int8, device=DEV)
    W["pc_scale"] = (torch.rand(N, device=DEV, dtype=torch.float32) * 0.01 + 0.001)
    # (c) bf16 weight [N,K]
    W["bf16"] = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02
    # (d) int4 w4a8: weight int4 packed [N, K/8] int32 + [K/16,N] scale + [K/16, N/8] zp
    W["i4"] = torch.randint(-(2**31), 2**31, (N, K // 8), dtype=torch.int32, device=DEV)
    W["i4_scale"] = (torch.rand(K // 16, N, device=DEV, dtype=torch.bfloat16) * 0.1 + 0.01)
    W["i4_zp"] = torch.zeros(K // 16, N // 8, dtype=torch.int32, device=DEV)
    return W


print("device:", torch.xpu.get_device_name(0) if hasattr(torch.xpu, "get_device_name") else "xpu")
for label, (N, K) in SHAPES.items():
    print(f"\n==================== {label} ====================")
    W = make_weights(N, K)
    print(f"{'M':>6} | {'bf16 F.linear':>14} | {'nvfp4_w4a16(a)':>15} | {'int8_w8a8(b)':>13} | "
          f"{'int4_w4a8(d)':>13} | {'b/a':>5} {'a/bf16':>7} {'b/bf16':>7}")
    for M in MS:
        xb = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1
        xs8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=DEV)
        asc = (torch.rand(M, 1, device=DEV, dtype=torch.float32) * 0.01 + 0.001)
        azp = torch.zeros(M, 1, dtype=torch.int8, device=DEV)

        # (c) bf16
        t_bf16 = bench(lambda: torch.nn.functional.linear(xb, W["bf16"]))

        # (a) nvfp4_gemm_w4a16 (current path)
        Bp = W["packed"].t()  # [K/2, N] NT view
        try:
            _ = torch.ops._xpu_C.nvfp4_gemm_w4a16(xb, Bp, None, W["nv_scale"], 16)
            t_a = bench(lambda: torch.ops._xpu_C.nvfp4_gemm_w4a16(xb, Bp, None, W["nv_scale"], 16))
        except Exception as e:
            t_a = float("nan"); print("  a FAIL:", type(e).__name__, str(e)[:120])

        # (b) int8_gemm_w8a8 (resident s8 + per-channel; INT8 XMX ceiling)
        Bs8 = W["s8"].t()  # [K,N] NT view
        try:
            _ = torch.ops._xpu_C.int8_gemm_w8a8(xs8, asc, None, Bs8, W["pc_scale"], None, None,
                                                torch.bfloat16)
            t_b = bench(lambda: torch.ops._xpu_C.int8_gemm_w8a8(xs8, asc, None, Bs8, W["pc_scale"],
                                                                None, None, torch.bfloat16))
        except Exception as e:
            t_b = float("nan"); print("  b FAIL:", type(e).__name__, str(e)[:140])

        # (d) int4_gemm_w4a8 (s8 x int4, per-16-K-group scale = block-scaled INT8 probe)
        t_d = float("nan")
        if DO_I4:
            Bi4 = W["i4"].t()  # [K/8, N] NT view
            try:
                _ = torch.ops._xpu_C.int4_gemm_w4a8(xs8, asc, azp, Bi4, W["i4_scale"], W["i4_zp"],
                                                    16, None, None)
                t_d = bench(lambda: torch.ops._xpu_C.int4_gemm_w4a8(xs8, asc, azp, Bi4, W["i4_scale"],
                                                                    W["i4_zp"], 16, None, None))
            except Exception as e:
                t_d = float("nan"); print("  d FAIL:", type(e).__name__, str(e)[:140])

        int8_vs_nvfp4 = (t_a / t_b) if (t_a == t_a and t_b == t_b and t_b > 0) else float("nan")
        spa = (t_bf16 / t_a) if t_a == t_a else float("nan")
        spb = (t_bf16 / t_b) if t_b == t_b else float("nan")
        print(f"{M:>6} | {t_bf16:>14.3f} | {t_a:>15.3f} | {t_b:>13.3f} | {t_d:>13.3f} | "
              f"{int8_vs_nvfp4:>5.2f} {spa:>7.2f} {spb:>7.2f}")

print("\nlegend: times in ms (lower=faster). 'b/a' col = nvfp4_w4a16_ms / int8_w8a8_ms")
print("        = INT8-XMX speedup over the current path (>1.3 = a real prefill win to reclaim).")
print("        a/bf16, b/bf16 = each path's speedup over naive bf16 F.linear.")
