# 20_spoof_raw_byte.py -- EXPERIMENT B.1 + B.2: the "spoof" ideas.
#
# B.1 RAW-BYTE SPOOF: feed the packed NVFP4 uint8 [N, K/2] straight into the int8-XMX
#     GEMM as if each byte were ONE int8 weight (no unpack). The GEMM then computes
#     sum_j x_j * (lo_j + 16*hi_j) -- two different 4-bit weights ENTANGLED into one
#     lane with the wrong (x_j vs x_{j+*}) pairing. No scale can disentangle it, so it
#     is numerically wrong; the POINT is to measure the SPEED/bandwidth of reading the
#     weight at its 4-bit (half-int8) footprint through the existing fast path, and to
#     confirm the byte-count -- not the math -- is what the spoof buys. Data point for
#     "why you must unpack in-kernel (Exp C)".
#
# B.2 DOUBLE-PUMP note+test: can int8 DPAS compute two 4-bit MACs per lane? We test
#     whether feeding a byte packing two nibbles and a matching activation recovers
#     both products. It cannot on Xe2 (no 2xint4-in-int8 dot; the cross term 16*lo*x
#     pollutes). We show the numeric pollution to log the deadend.
import time
import torch
import numpy as np
import vllm_xpu_kernels._xpu_C  # noqa
def line(*a): print(*a, flush=True)

DEV = "xpu"
MAG = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int8)

def bench(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.xpu.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.xpu.synchronize()
    return (time.time() - t0) / iters * 1e3

SHAPES = [("27B gate/up K5120 N17408", 5120, 17408),
          ("27B down    K17408 N5120", 17408, 5120)]

line("=== EXPERIMENT B.1: raw-byte spoof (packed uint8 -> int8 GEMM, no unpack) ===")
for M in (1, 8):
    line(f"\n### M={M} ###")
    for tag, K, N in SHAPES:
        # build packed weight [N, K/2] and true int8+scale for the honest reference
        codes = torch.randint(0, 16, (N, K), dtype=torch.uint8)
        mi = (codes & 0x7).long(); sign = torch.where((codes & 0x8) != 0, -1, 1)
        w_int8_f = (sign * torch.from_numpy(MAG)[mi]).float()
        w_int8 = w_int8_f.to(torch.int8).to(DEV)
        KH = K // 2
        packed = (codes[:, :KH] | (codes[:, KH:] << 4)).to(torch.uint8).contiguous()
        # reinterpret packed bytes as SIGNED int8 (values 0..255 -> -128..127)
        packed_i8 = packed.view(torch.int8).to(DEV)           # [N, K/2] s8 (garbage weight)
        x = (torch.randn(M, K, device=DEV) * 0.1).to(torch.bfloat16)
        xh = x[:, :KH].contiguous()                            # activation of length K/2

        # honest int8 full-K reference speed (per-channel scale=1 for pure speed signal)
        sc_full = torch.ones(N, device=DEV, dtype=torch.bfloat16)
        wt_full = w_int8.t().contiguous()
        t_full = bench(lambda: torch.ops._xpu_C.int8_gemm_w8a16(x, wt_full, sc_full, None))

        # spoof: half-length int8 GEMM over the packed bytes
        sc_half = torch.ones(N, device=DEV, dtype=torch.bfloat16)
        wt_half = packed_i8.t().contiguous()                  # [K/2, N]
        try:
            y = torch.ops._xpu_C.int8_gemm_w8a16(xh, wt_half, sc_half, None)
            t_half = bench(lambda: torch.ops._xpu_C.int8_gemm_w8a16(xh, wt_half, sc_half, None))
            # honest reference of what a real 4-bit GEMV would produce (for err)
            ref = (x.float() @ w_int8.float().t())
            err = (y.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            gb_full = (N*K)/(t_full/1e3)/1e9
            gb_half = (N*KH)/(t_half/1e3)/1e9
            line(f"{tag}")
            line(f"    int8 full-K : {t_full:8.4f}ms  {gb_full:6.1f}GB/s")
            line(f"    spoof half-K: {t_half:8.4f}ms  {gb_half:6.1f}GB/s  speedup {t_full/t_half:.2f}x  rel-err {err:.3f} (GARBAGE math, speed-only)")
        except Exception as e:
            line(f"{tag}: spoof FAIL {type(e).__name__}: {str(e)[:100]}")

line("\n=== EXPERIMENT B.2: double-pump pollution demo (why int8 lane != 2x int4 MAC) ===")
# one lane byte = lo | hi<<4. int8 dot with activation a gives a*(lo + 16*hi).
# We want a1*lo + a2*hi. Setting a=a1 recovers a1*lo but adds 16*a1*hi cross term.
lo = torch.tensor([3.0]); hi = torch.tensor([5.0]); a1 = torch.tensor([2.0]); a2 = torch.tensor([7.0])
byte_val = lo + 16*hi
want = a1*lo + a2*hi
got_single_lane = a1 * byte_val            # what one int8 MAC yields with act=a1
line(f"  want a1*lo+a2*hi = {want.item():.1f}; one-lane int8 (act=a1) = {got_single_lane.item():.1f} -> cross-term pollution {abs(got_single_lane.item()-want.item()):.1f}")
line("  => no scalar fixup recovers both; Xe2 has no 2xint4-in-int8 dot. DEADEND (documented).")
line("DONE")
