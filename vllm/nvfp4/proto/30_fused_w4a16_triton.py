# 30_fused_w4a16_triton.py -- EXPERIMENT C (HEADLINE): fused 4-bit-in-VRAM dequant GEMM.
#
# Keep NVFP4 weights 4-bit-PACKED in VRAM (HALF the int8 footprint: 27B = 21.9GB vs
# 31.1GB int8 -- int8-repack does NOT fit one B70 card, 4-bit does). Unpack the E2M1
# nibbles via a LUT + apply the per-16-K-group scale in REGISTERS, then GEMV.
#
# E2M1 is a FLOAT LUT {0,.5,1,1.5,2,3,4,6} (x sign), NOT twos-complement int4 -- so we
# decode nibble -> sign * magtable, where magtable*2 = {0,1,2,3,4,6,8,12}. We fold the
# /2 into the scale (as the int8-repack path does), so the LUT emits {0,1,2,3,4,6,8,12}.
#
# PACKING (half-split, so all activation loads are contiguous -- the real ckpt is
# interleaved k=2j/2j+1; a load-time repack to this layout is a one-time cost, noted):
#   byte[n, j] = nibble(k=j)  |  nibble(k=j + K/2) << 4     for j in [0, K/2)
#   sum over k of x[k]*w[k] is order-independent, so this matches the natural ref.
#
# Three ways benched on real Qwen MLP shapes:
#   (1) triton W4 : packed uint8 [N, K/2]  -- 4-bit in VRAM (HALF bytes)   <-- the play
#   (2) triton W8 : int8        [N, K]     -- same kernel, full bytes (isolates the byte-count lever)
#   (3) oneDNN int8_gemm_w8a16             -- the existing production int8-XMX path
# Report wall-clock ms + EFFECTIVE weight-read GB/s vs 608 GB/s HBM roofline.
import time
import torch
import triton
import triton.language as tl

DEV = "xpu"
def line(*a): print(*a, flush=True)

MAG = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12], dtype=torch.float32)  # E2M1 mag * 2


@triton.jit
def _dec(nib):
    # nib: tensor of 4-bit codes. return signed E2M1*2 magnitude as float.
    mag_idx = nib & 0x7
    sign = tl.where((nib & 0x8) != 0, -1.0, 1.0)
    m = tl.where(mag_idx <= 4, mag_idx.to(tl.float32),
        tl.where(mag_idx == 5, 6.0, tl.where(mag_idx == 6, 8.0, 12.0)))
    return sign * m


@triton.jit
def gemm_w4(x_ptr, wq_ptr, sc_ptr, y_ptr, M, N, K,
            stride_xm, stride_xk, stride_wn, stride_wj, stride_sn, stride_sg,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_KH: tl.constexpr,
            GROUP: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    KH = K // 2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for j0 in range(0, KH, BLOCK_KH):
        offs_j = j0 + tl.arange(0, BLOCK_KH)
        # packed weight bytes [BLOCK_N, BLOCK_KH]
        wb = tl.load(wq_ptr + offs_n[:, None] * stride_wn + offs_j[None, :] * stride_wj)
        w_lo = _dec(wb & 0xF)          # weights for k = j          [BN, BKH]
        w_hi = _dec((wb >> 4) & 0xF)   # weights for k = j + KH
        # scales
        g_lo = offs_j // GROUP
        g_hi = (KH + offs_j) // GROUP
        s_lo = tl.load(sc_ptr + offs_n[:, None] * stride_sn + g_lo[None, :] * stride_sg)
        s_hi = tl.load(sc_ptr + offs_n[:, None] * stride_sn + g_hi[None, :] * stride_sg)
        w_lo = w_lo * s_lo
        w_hi = w_hi * s_hi
        # activations: contiguous halves
        x_lo = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_j[None, :] * stride_xk,
                       mask=offs_m[:, None] < M, other=0.0).to(tl.float32)
        x_hi = tl.load(x_ptr + offs_m[:, None] * stride_xm + (KH + offs_j)[None, :] * stride_xk,
                       mask=offs_m[:, None] < M, other=0.0).to(tl.float32)
        acc += tl.dot(x_lo, tl.trans(w_lo), out_dtype=tl.float32)
        acc += tl.dot(x_hi, tl.trans(w_hi), out_dtype=tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=offs_m[:, None] < M)


@triton.jit
def gemm_w8(x_ptr, w_ptr, sc_ptr, y_ptr, M, N, K,
            stride_xm, stride_xk, stride_wn, stride_wk, stride_sn, stride_sg,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk).to(tl.float32)
        g = offs_k // GROUP
        sc = tl.load(sc_ptr + offs_n[:, None] * stride_sn + g[None, :] * stride_sg)
        w = w * sc
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=offs_m[:, None] < M, other=0.0).to(tl.float32)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=offs_m[:, None] < M)


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.xpu.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.xpu.synchronize()
    return (time.time() - t0) / iters * 1e3


def make_weights(N, K, GROUP=16):
    codes = torch.randint(0, 16, (N, K), dtype=torch.uint8)          # nibble codes 0..15
    mag_idx = (codes & 0x7).long()
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    w_int8_f = sign * MAG[mag_idx]                                   # [N,K] int8 value as float
    w_int8 = w_int8_f.to(torch.int8)
    KH = K // 2
    lo = codes[:, :KH]                                               # k in [0,KH)
    hi = codes[:, KH:]                                              # k in [KH,K)
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()          # [N, K/2]
    G = K // GROUP
    scale = (torch.rand(N, G) * 0.02 + 0.001).to(torch.float32)
    scale_exp = scale.repeat_interleave(GROUP, dim=1)               # [N,K]
    w_deq = (w_int8_f * scale_exp)                                  # [N,K]
    return packed, w_int8, scale, w_deq


SHAPES = [
    ("27B gate/up K5120  N17408", 5120, 17408),
    ("27B down    K17408 N5120 ", 17408, 5120),
    ("8B  gate/up K4096  N14336", 4096, 14336),
    ("8B  down    K14336 N4096 ", 14336, 4096),
]
BLOCK_M = 16
BLOCK_N = 64
BLOCK_KH = 128     # half-K block for W4; W8 uses 2x = 256


def main():
    import vllm_xpu_kernels._xpu_C  # noqa: registers torch.ops._xpu_C.int8_gemm_w8a16
    GROUP = 16
    line("=== EXPERIMENT C: 4-bit-in-VRAM dequant GEMM vs int8 (decode M=1,8) ===")
    line("roofline 608 GB/s HBM. weight bytes W4=N*K/2  W8=N*K")
    for M in (1, 8):
        line(f"\n########## M={M} ##########")
        for tag, K, N in SHAPES:
            packed, w_int8, scale, w_deq = make_weights(N, K, GROUP)
            packed = packed.to(DEV); w_int8_d = w_int8.to(DEV); scale_d = scale.to(DEV)
            w_deq_d = w_deq.to(DEV)
            x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.bfloat16)
            ref = (x.float() @ w_deq_d.float().t())
            grid = (triton.cdiv(N, BLOCK_N),)
            refmax = ref.abs().max().item() + 1e-9
            wb4 = N * K / 2; wb8 = N * K

            try:
                y4 = torch.zeros(BLOCK_M, N, device=DEV, dtype=torch.float32)
                def run4():
                    gemm_w4[grid](x, packed, scale_d, y4, M, N, K,
                                  x.stride(0), x.stride(1), packed.stride(0), packed.stride(1),
                                  scale_d.stride(0), scale_d.stride(1),
                                  BLOCK_M, BLOCK_N, BLOCK_KH, GROUP)
                run4(); torch.xpu.synchronize()
                err4 = (y4[:M].float() - ref).abs().max().item() / refmax
                t4 = bench(run4); gb4 = wb4/(t4/1e3)/1e9
                s4 = f"W4pack {t4:8.4f}ms {gb4:6.1f}GB/s err {err4:.4f}"
            except Exception as e:
                s4 = f"W4pack FAIL {type(e).__name__}: {str(e)[:80]}"

            try:
                y8 = torch.zeros(BLOCK_M, N, device=DEV, dtype=torch.float32)
                def run8():
                    gemm_w8[grid](x, w_int8_d, scale_d, y8, M, N, K,
                                  x.stride(0), x.stride(1), w_int8_d.stride(0), w_int8_d.stride(1),
                                  scale_d.stride(0), scale_d.stride(1),
                                  BLOCK_M, BLOCK_N, 2*BLOCK_KH, GROUP)
                run8(); torch.xpu.synchronize()
                err8 = (y8[:M].float() - ref).abs().max().item() / refmax
                t8 = bench(run8); gb8 = wb8/(t8/1e3)/1e9
                s8 = f"W8trit {t8:8.4f}ms {gb8:6.1f}GB/s err {err8:.4f}"
            except Exception as e:
                s8 = f"W8trit FAIL {type(e).__name__}: {str(e)[:80]}"

            try:
                wt = w_int8_d.t().contiguous()
                sc_kg = scale_d.t().contiguous().to(torch.bfloat16)
                to = bench(lambda: torch.ops._xpu_C.int8_gemm_w8a16(x, wt, sc_kg, None))
                yo = torch.ops._xpu_C.int8_gemm_w8a16(x, wt, sc_kg, None)
                erro = (yo.float() - ref).abs().max().item() / refmax
                gbo = wb8/(to/1e3)/1e9
                so = f"oneDNN {to:8.4f}ms {gbo:6.1f}GB/s err {erro:.4f}"
            except Exception as e:
                so = f"oneDNN FAIL {type(e).__name__}: {str(e)[:80]}"

            line(f"{tag}")
            line(f"    {s4}")
            line(f"    {s8}")
            line(f"    {so}")
    line("DONE")


if __name__ == "__main__":
    main()
