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

## SPLIT-K QUANT TEST (2026-06-23, card 0, Triton on `:int8g`) -- algorithm WORKS, Triton dispatch caps it
Wrote a Triton per-token int8 quant, grid `(M,)` with `num_warps` so each row's K-reduction parallelizes
across the work-group (vs the stock op's serial 1-work-item reduction). CORRECT (max |xq diff| = 1 rounding,
scale rel-err 0). Result (M=1):
```
  K       stock     triton(best)   speedup
  5120    39.7us    60.4us         0.66x  (LOSES -- Triton floor > stock)
  17408   101.1us   60.4us         1.68x  (WINS)
```
- **The serial-reduction problem IS fixed**: Triton time is FLAT ~60us regardless of K (5120 & 17408) and M
  (1, 5, 64) -- the parallel reduction works. Stock scales 40->101us with K (serial). So at the large-K
  down_proj, Triton is **1.68x** even in eager.
- BUT there is a **~60us fixed Triton-XPU dispatch floor** (NOT tunable: swept num_warps 1..32 x BLOCK_K
  1024/4096, all 60-67us). The stock NATIVE op's launch is lower (~15-40us), so Triton loses at small K and
  its big-K win is floor-capped.
CONCLUSION: the split-K *algorithm* is the right fix, but Triton is the wrong vehicle on XPU (dispatch floor).
Production path, in order of cleanliness:
  1. **FUSE the per-token quant into the int8 GEMM prologue** -- the fused kernel reads bf16 `x` once,
     computes the per-token scale + quantizes inline, then does the s8s8 GEMM. ZERO extra dispatch, and the
     reduction is parallel within the GEMM's tiling. This is the real win (removes the whole quant launch).
  2. The same split-K reduction in the NATIVE contrib int8 kernel (`contrib/vllm_int8_xpu`) -- lower dispatch
     than Triton, beats stock at all K.
  3. Graph capture removes the dispatch for BOTH the stock and a parallel-reduction quant -> then the parallel
     reduction wins; but fusion (1) is better because it also kills the GEMM's separate launch.
Net for B70 int8 decode: don't tune the GEMMs (near-roofline); FUSE quant+GEMM (one launch, parallel reduce).

## PER-SCHEME IMPACT of the activation-quant fusion + a 14B W8A8 sim (2026-06-23)
Which quants the fused activation-quant helps (verified by reading each kernel's `apply_weights`):
- **W8A8** (`XPUInt8ScaledMMLinearKernel`): int8 per-token act-quant (`dynamic_per_token_int8_quant`) -> BENEFITS.
- **W4A8** (`XPUW4A8IntLinearKernel`): int8 per-token act-quant too -- AND it uses the SLOW
  `dynamic_per_token_int8_quant_ref` (pure-torch ~210us, ~5 launches) -> benefits EVEN MORE (or just switch
  it to the kernel quant for a free win). Weight GEMM differs (`int4_gemm_w4a8`) but the act-quant is shared.
- **W4A16** (`XPUwNa16LinearKernel`): `int4_gemm_w4a16(x,...)` on BF16 x, NO activation quant -> ZERO benefit.

14B W8A8 linear-path per-token sim (real ops/dims H=5120 I=17408 qkvN=7168 x40 layers, EAGER M=1):
```
  per-layer int8 GEMMs   610.8us  (constant)
  quant-path  unfused    286.8us  ->  fused 174.2us  (-39%; rms-fusion 2x + silu-fusion 1.5x)
  per token   unfused    35.90ms (27.9 t/s) -> fused 31.40ms (31.8 t/s) = 1.14x
```
The quant path is 32% of the linear path; fusing it cuts ~12% -> ~1.14x EAGER on the linear path.
CAVEATS: (1) the serve runs GRAPH=1 PIECEWISE -> capture already removes the LAUNCH part of the win, so the
captured end-to-end gain is SMALLER than 1.14x; (2) attention/GDN add constant time -> further dilution.
=> modest win for W8A8; the live captured number needs the wiring (rms_norm_dynamic_per_token_quant into the
int8 linear's RMSNorm-fed inputs). Bigger decode levers remain MTP + FULL capture. W4A8's slow-ref quant is
the cheapest standalone win here.

All work card-0 only (another agent on card 1; never the default `gpu-run` which locks both).
