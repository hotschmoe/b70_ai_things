#!/usr/bin/env python3
# Interconnect truth probe for dual B70: measure REAL PCIe bandwidth + test GPU P2P.
# Purpose: settle the "Gen1 x1" question with a positive number, and check whether
# xpu0<->xpu1 copies go direct (P2P) or bounce through host (host-staged).
#
# A real Gen3 x16 link ~= 12 GB/s H2D. A genuine Gen1 x1 link would be ~0.20-0.25 GB/s.
# If D2D (xpu0->xpu1) ~= H2D/2 it is host-staged (D->H->D, no P2P). If D2D >> that,
# something is doing direct peer DMA.
#
# Runs single-process inside vllm-xpu-env (torch 2.11+xpu). No CCL / collectives here.
import time
import torch

def bw(fn, nbytes, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    dt = time.perf_counter() - t0
    return nbytes * iters / dt / 1e9  # GB/s

def main():
    print("torch:", torch.__version__)
    if not torch.xpu.is_available():
        print("ERROR: torch.xpu not available"); return
    n = torch.xpu.device_count()
    print("xpu device_count:", n)
    for i in range(n):
        print(f"  xpu:{i} = {torch.xpu.get_device_name(i)}")

    # --- P2P capability query (API may not exist on this build) ---
    print("\n=== P2P capability ===")
    if n >= 2:
        try:
            ok = torch.xpu.can_device_access_peer(0, 1)
            print(f"torch.xpu.can_device_access_peer(0,1) = {ok}")
        except Exception as e:
            print(f"torch.xpu.can_device_access_peer: NOT available ({type(e).__name__}: {e})")
    else:
        print("only 1 xpu visible -- skipping P2P")

    # --- buffers ---
    MB = 1024 * 1024
    sz = 256 * MB              # 256 MiB payload
    elems = sz // 2           # fp16
    cpu = torch.empty(elems, dtype=torch.float16).pin_memory() if False else torch.empty(elems, dtype=torch.float16)
    g0 = torch.empty(elems, dtype=torch.float16, device="xpu:0")

    # --- H2D / D2H (the headline: disproves Gen1 x1) ---
    print("\n=== H2D / D2H bandwidth (xpu:0) ===")
    h2d = bw(lambda: g0.copy_(cpu), sz)
    d2h = bw(lambda: cpu.copy_(g0), sz)
    print(f"H2D: {h2d:6.2f} GB/s     D2H: {d2h:6.2f} GB/s     (payload {sz//MB} MiB)")
    print(f"  -> Gen1 x1 would be ~0.20-0.25 GB/s; Gen3 x16 ~10-12 GB/s; Gen5 x16 ~25-50 GB/s")

    # --- D2D (xpu:0 -> xpu:1): host-staged or P2P? ---
    if n >= 2:
        print("\n=== D2D bandwidth (xpu:0 -> xpu:1) ===")
        g1 = torch.empty(elems, dtype=torch.float16, device="xpu:1")
        try:
            d2d = bw(lambda: g1.copy_(g0), sz)
            print(f"D2D: {d2d:6.2f} GB/s")
            print(f"  interpretation: ~H2D/2 => host-staged (no P2P); ~H2D or faster => direct peer DMA")
            print(f"  H2D/2 reference = {h2d/2:6.2f} GB/s")
        except Exception as e:
            print(f"D2D copy FAILED ({type(e).__name__}: {e})")

    # --- a tiny correctness check (peer copy integrity) ---
    if n >= 2:
        a = torch.arange(8, dtype=torch.float16, device="xpu:0") + 1
        b = a.to("xpu:1")
        print("\npeer copy sanity:", b.cpu().tolist(), "(expect [1..8])")

if __name__ == "__main__":
    main()
