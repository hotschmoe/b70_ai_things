#!/usr/bin/env python3
"""O4: HC GEMV shapes, oneDNN bf16 vs Triton bf16 split-K, XPUGraph timing.

down: x[M, 10240] @ W[324, 10240]^T   up: a[M, 320] @ W[10240, 320]^T
Random data (Xe2 compresses constant buffers). 96 distinct weights per
shape so the stack does not sit in L2.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_splitk(X, W, P, M, N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr):
    # P[s, m, n] = sum over k in split s of X[m, k] * W[n, k] (fp32).
    pn = tl.program_id(0)
    s = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    nm = n < N
    m = tl.arange(0, MP)
    mm = m < M
    acc = tl.zeros((MP, BN), tl.float32)
    k0 = s * KS
    for kk in range(0, KS, BK):
        k = k0 + kk + tl.arange(0, BK)
        km = k < K
        w = tl.load(W + n[:, None] * K + k[None, :], nm[:, None] & km[None, :], 0.0).to(tl.float32)
        x = tl.load(X + m[:, None] * K + k[None, :], mm[:, None] & km[None, :], 0.0).to(tl.float32)
        acc += tl.sum(x[:, None, :] * w[None, :, :], 2)
    tl.store(P + (s * M + m[:, None]) * N + n[None, :], acc, mm[:, None] & nm[None, :])


@triton.jit
def _gemv_rows(X, W, Y, M, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr):
    pn = tl.program_id(0)
    n = pn * BN + tl.arange(0, BN)
    nm = n < N
    m = tl.arange(0, MP)
    mm = m < M
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        km = k < K
        w = tl.load(W + n[:, None] * K + k[None, :], nm[:, None] & km[None, :], 0.0).to(tl.float32)
        x = tl.load(X + m[:, None] * K + k[None, :], mm[:, None] & km[None, :], 0.0).to(tl.float32)
        acc += tl.sum(x[:, None, :] * w[None, :, :], 2)
    tl.store(Y + m[:, None] * N + n[None, :], acc.to(tl.bfloat16), mm[:, None] & nm[None, :])


def gtime(fn, n=20):
    fn()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    torch.xpu.synchronize()
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / n


def main():
    L = 96
    for M in (1, 2, 4):
        for name, N, K in (("down", 324, 10240), ("up", 10240, 320)):
            Ws = [torch.randn(N, K, device="xpu", dtype=torch.bfloat16) * 0.02 for _ in range(L)]
            x = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
            byt = N * K * 2 * L
            ref = torch.nn.functional.linear(x, Ws[0])

            def onednn():
                for W in Ws:
                    torch.nn.functional.linear(x, W)

            t = gtime(onednn)
            print(f"M={M} {name} onednn {t/L*1e3:.2f} us/call {byt/(t*1e-3)/1e9:.0f} GB/s", flush=True)
            MP = max(2, triton.next_power_of_2(M))
            if name == "down":
                for BN, BK, SPL in ((4, 512, 20), (8, 512, 10), (4, 256, 40), (8, 256, 20), (16, 256, 10)):
                    KS = K // SPL
                    P = torch.empty(SPL, M, N, device="xpu", dtype=torch.float32)

                    def tri():
                        for W in Ws:
                            _gemv_splitk[(triton.cdiv(N, BN), SPL)](x, W, P, M, N, K, KS, BN, BK, MP)

                    try:
                        t = gtime(tri)
                        _gemv_splitk[(triton.cdiv(N, BN), SPL)](x, Ws[0], P, M, N, K, KS, BN, BK, MP)
                        err = (P.sum(0).to(torch.bfloat16).float() - ref.float()).abs().max().item()
                        print(f"   triton splitk BN={BN} BK={BK} S={SPL} {t/L*1e3:.2f} us {byt/(t*1e-3)/1e9:.0f} GB/s err={err:.2e}", flush=True)
                    except Exception as ex:  # noqa: BLE001
                        print(f"   triton BN={BN} BK={BK} S={SPL} fail {ex!r}"[:300], flush=True)
            else:
                Y = torch.empty(M, N, device="xpu", dtype=torch.bfloat16)
                for BN, BK in ((16, 64), (32, 64), (16, 128), (32, 128), (64, 64), (8, 512)):
                    def tri():
                        for W in Ws:
                            _gemv_rows[(triton.cdiv(N, BN),)](x, W, Y, M, N, K, BN, BK, MP)

                    try:
                        t = gtime(tri)
                        _gemv_rows[(triton.cdiv(N, BN),)](x, Ws[0], Y, M, N, K, BN, BK, MP)
                        err = (Y.float() - ref.float()).abs().max().item()
                        print(f"   triton rows BN={BN} BK={BK} {t/L*1e3:.2f} us {byt/(t*1e-3)/1e9:.0f} GB/s err={err:.2e}", flush=True)
                    except Exception as ex:  # noqa: BLE001
                        print(f"   triton BN={BN} BK={BK} fail {ex!r}"[:300], flush=True)
            del Ws
            torch.xpu.empty_cache()


if __name__ == "__main__":
    main()
