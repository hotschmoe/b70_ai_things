# B70 GEMV/GEMM roofline + int4/int8 kernel characterization (2026-06-23)

Decode-GEMV / prefill-GEMM optimization study on ONE Arc Pro B70 (`gpu-run --card 0`, vLLM images
`:v0230` for int4, `:int8g` for int8). Goal: find where the real B70 kernel headroom is before tuning.
Method: isolated microbench (warm loop, multi-trial median), 27B MLP shapes (hidden 5120, intermediate 17408).

## The B70 roofline (measured)
- **Read BW ceiling ~581 GB/s** (bf16 `.sum()` over 512 MB); copy (r+w) 510 GB/s.
- A dense **bf16 GEMV hits the roofline at M=1** (583 GB/s, both shapes) -> the roofline IS reachable at M=1;
  any decode GEMV below it is a kernel-efficiency gap, not a hardware limit.
- int8 **GEMM prefill (M=2048) ~250-256 int8 TOPS** (gate_up 250, down 256) -- near the XMX int8 peak.

## THE KERNELS THEMSELVES ARE NEAR-OPTIMAL (do NOT tune these)
M=1 decode, % of the 581 GB/s roofline (on each format's own weight bytes):
```
  kernel                     gate_up(K5120,N34816)   down(K17408,N5120)
  int4_gemm_w4a16 (W4A16)    544 GB/s  94%           543 GB/s  93%
  int8_gemm_w8a8  (gemm-only)510 GB/s  88%           578 GB/s 100%
```
The int4 W4A16 GEMV and the int8 W8A8 GEMM are both memory-bandwidth-bound and within 6-12% of peak at M=1.
The earlier "gate_up M=1 penalty" (208us) was MEASUREMENT NOISE -- careful medians show 163us / 94% (no
penalty, no M=1->M=2 pad win). => the int4/int8 GEMM kernels are NOT a meaningful optimization target.

## THE REAL int8-DECODE BOTTLENECK: `dynamic_per_token_int8_quant` (the activation quant)
Decompose the int8 W8A8 M=1 decode (quant + gemm):
```
  shape       quant-only   gemm-only(=%roof)   full     quant share
  gate_up     37.7us       349.5us (88%)       390.3us  10%
  down        101.0us      154.1us (100%)      286.9us  35%
```
The GEMM is roofline-perfect; the **per-token activation quant is the killer** -- 101us to quantize a tiny
`[1, 17408]` activation (memory-bound ideal <1us). It does NOT parallelize the per-token max|x| reduction:
**M=1 and M=64 take the SAME time** (101 vs 104us). Linear fit (K=5120->40us, K=17408->101us): ~15us launch
floor + ~5 ns/element of SERIAL reduction work (1 work-item reducing K elements per row). That reduction work
is ~1000x off memory-bound and **persists under graph capture** (it is compute, not launch).
- Plain-torch quant (abs/amax/div/round/cast) is WORSE: 0.17-0.50x (it's ~5 kernel launches; ~210us fixed).
  -> not a drop-in win; the fix is a kernel.

## Findings / direction
1. **int4 decode wins decisively** over int8 at M=1 (half the weight bytes, both near roofline): int4 GEMV
   ~82-164us vs int8 full-path ~287-390us. For DECODE, prefer int4 weights; int8 is for prefill GEMM / quality.
2. **The GEMM/GEMV kernels are near-optimal -- don't tune them.** The decode lever is FEWER LAUNCHES + the
   quant: (a) graph capture removes per-op launch overhead (~15-40us/op on XPU -> why capture is the dominant
   decode lever, cf. MTP_TODO/localmaxxing 9.5x); (b) a **split-K / multi-work-item `dynamic_per_token_int8_quant`**
   (parallelize each row's K-reduction) OR fuse the quant into the int8 GEMM prologue -- this is the one kernel
   with real, capture-persistent headroom (down_proj: fixing 101->~5us takes the layer 287->~160us = 1.8x int8 decode).
3. int8 GEMM prefill ~250 TOPS is near XMX peak -> little prefill headroom there.

NEXT TEST (proposed): a Triton split-K per-token int8 quant vs `dynamic_per_token_int8_quant` at M=1, large K.
All work card-0 only (another agent on card 1; never the default `gpu-run` which locks both).
