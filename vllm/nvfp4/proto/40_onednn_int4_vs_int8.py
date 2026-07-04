# 40_onednn_int4_vs_int8.py -- EXPERIMENT A.1 + C (decisive): does the SHIPPED oneDNN
# 4-bit weight-decompression matmul (int4_gemm_w4a16) actually read weights at 4-bit
# footprint and ~halve decode latency vs the int8 path (int8_gemm_w8a16)?
#
# My naive Triton fused-4bit kernel was 50x off oneDNN int8 (run1). The FAIR test of the
# "4-bit-in-VRAM wins decode" thesis is the OPTIMIZED oneDNN int4 op vs the OPTIMIZED
# oneDNN int8 op on identical real MLP shapes. Run with ONEDNN_VERBOSE=dispatch to see
# which impl each dispatches to (jit:gemm:xe DPAS vs reference).
#
# NOTE on NVFP4: int4_gemm_w4a16 decodes 4-bit as (u4 - zero_point) LINEAR ints -> it
# CANNOT represent NVFP4's E2M1 float LUT ({0,.5,1,1.5,2,3,4,6}; *2 needs +/-12 > s4).
# So this measures the CEILING a well-optimized 4-bit path reaches on B70; using it for
# NVFP4 needs EITHER a requant to linear-int4 (lossy) OR a custom E2M1-LUT oneDNN op.
import time, os
import torch
import vllm_xpu_kernels._xpu_C  # noqa
def line(*a): print(*a, flush=True)
DEV = "xpu"

def bench(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.xpu.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.xpu.synchronize()
    return (time.time() - t0) / iters * 1e3

def rand_int4_packed(k, n):
    # [k//8, n] int32, each int32 holds 8 int4 (per the upstream test)
    rand = torch.randint(-128, 128, [(k * n) // 2], device=DEV).to(torch.int8)
    return rand.view(dtype=torch.int32).reshape(k // 8, n)

SHAPES = [
    ("27B gate/up K5120  N17408", 5120, 17408),
    ("27B down    K17408 N5120 ", 17408, 5120),
    ("8B  gate/up K4096  N14336", 4096, 14336),
    ("8B  down    K14336 N4096 ", 14336, 4096),
]

def run():
    line("=== oneDNN int4_gemm_w4a16 vs int8_gemm_w8a16 (decode M=1,8) ===")
    line("weight bytes: int4 = N*K/2, int8 = N*K. Lower ms = faster decode.")
    for M in (1, 8):
        line(f"\n########## M={M} ##########")
        for tag, K, N in SHAPES:
            dt = torch.float16
            x = torch.rand(M, K, device=DEV, dtype=dt)
            gs = min(128, K); gnum = K // gs

            # ---- int4 path ----
            try:
                wq = rand_int4_packed(K, N)                          # [K/8, N] i32
                weight_ba = wq.transpose(0, 1).contiguous().transpose(0, 1)
                scale4 = torch.rand(gnum, N, device=DEV, dtype=dt)
                zp = torch.tensor([8], dtype=torch.int8, device=DEV)  # symmetric
                y4 = torch.ops._xpu_C.int4_gemm_w4a16(x, weight_ba, torch.Tensor().to(DEV),
                                                      scale4, zp, gs, None)
                t4 = bench(lambda: torch.ops._xpu_C.int4_gemm_w4a16(
                    x, weight_ba, torch.Tensor().to(DEV), scale4, zp, gs, None))
                gb4 = (N * K / 2) / (t4 / 1e3) / 1e9
                s4 = f"int4 {t4:8.4f}ms  {gb4:6.1f}GB/s  shape_ok"
            except Exception as e:
                s4 = f"int4 FAIL {type(e).__name__}: {str(e)[:110]}"

            # ---- int8 path ----
            try:
                w8 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=DEV)
                wt = w8.t().contiguous()                             # [K,N]
                sc8 = torch.ones(N, device=DEV, dtype=dt)
                t8 = bench(lambda: torch.ops._xpu_C.int8_gemm_w8a16(x, wt, sc8, None))
                gb8 = (N * K) / (t8 / 1e3) / 1e9
                s8 = f"int8 {t8:8.4f}ms  {gb8:6.1f}GB/s"
            except Exception as e:
                s8 = f"int8 FAIL {type(e).__name__}: {str(e)[:110]}"

            line(f"{tag}")
            line(f"    {s8}")
            line(f"    {s4}")
            # speed ratio
            try:
                line(f"    -> int4/int8 latency ratio {t4/t8:.2f}x  (want <1.0 = int4 faster; ~0.5 = ideal BW win)")
            except Exception:
                pass
    line("DONE")

if __name__ == "__main__":
    run()
