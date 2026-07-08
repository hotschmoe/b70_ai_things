# NVFP4 prefill on INT8 XMX -- block-scaled int8 DPAS prototype + verdict (2026-07-08)

QUALIFIED DEAD-END. The goal was to reroute NVFP4 27B MLP PREFILL from the current bf16-compute
path (torch.ops._xpu_C.nvfp4_gemm_w4a16, oneDNN f4->bf16 weight-decompression) onto INT8 XMX for
the ~2x that W8A8 gets. E2M1x2 is exact s8 ({0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}), so the repack is
lossless. The 2x does NOT materialize for NVFP4 on Battlemage/Xe2, for a fundamental reason. This
dir preserves the prototypes and the evidence.

## The fundamental wall: NVFP4 group_size=16 < int8 DPAS K-depth=32

Xe2 int8 DPAS (SystolicDepth=8, OpsPerChannel(s8)=4) reduces K=32 per instruction; that K=32 depth
IS where int8's 2x-over-bf16 comes from (bf16 DPAS reduces only K=16). NVFP4's block scale changes
every 16 K-elements, and a per-16 scale cannot be applied to an s32 accumulator that has already
summed 32 elements across two different-scaled blocks. So a correct NVFP4 block-scaled int8 GEMM is
forced to a K=16 DPAS (zero-pad the upper 16, rescale per group) -- which does the SAME useful MACs
per instruction as bf16. The int8 2x is spent exactly on the sub-K-depth blocking.

(MXFP8 group=32 would NOT hit this -- block==DPAS-K. It is specific to NVFP4's group=16.)

## Evidence (all on B70 card 0, int8g-v0240)

### 1. Pure INT8-XMX ceiling is real, but per-channel (numerically WRONG for NVFP4)
vllm/nvfp4/int8_prefill_ceiling.py -- real oneDNN ops on real gate/up (N=17408 K=5120) shapes:
  M=512/2048/8192  int8_w8a8 vs bf16 F.linear = 1.93x / 2.23x / 2.02x ; vs nvfp4_w4a16 = 2.24x-2.61x.
So plain per-channel int8 = ~2x bf16. But per-channel collapses NVFP4's per-16-K block scale ->
wrong numerics. (Also notable: the CURRENT nvfp4_w4a16 path is 0.83-0.86x bf16 on gate/up, i.e.
~14% SLOWER than bf16 at prefill -- the price of 4-bit residency.)

### 2. Block-scale compute penalty (register-resident, decode amortized) -- bench_penalty.cpp
Isolates the DPAS penalty of group=16 vs per-channel (same useful MACs, no memory in hot loop):
  PCI per-channel full-K32 (ILP)     = 239.3 useful TOPS   (the int8 ceiling)
  PC  per-channel full-K32 (chained) = 136.6
  BS  block-scaled group=16          = 156.9 useful TOPS   <- correct NVFP4 numerics, compute-only
  BSD block-scaled + E2M1 decode     =  12.7  (M=8; decode not amortized -- artifact of tiny M)
=> correct block-scaled int8 tops out ~157 useful TOPS ~= bf16's 183 peak. The 2x is gone. Best case
   ~1.3x bf16 in the compute-bound limit (BS 157 vs bf16-achieved ~120-137).

### 3. oneDNN cannot do fast block-scaled int8 -- validate_w4a8.py + kernels/nvfp4_gemm_w4a8.h
A oneDNN W4A8 op (s8 weight-decompress + s8 act + per-16-K group scale) was built and is
NUMERICALLY CORRECT (relerr 3.2-3.8e-3 vs the exact ref) but runs at ~459ms/M=512 (~570x SLOWER
than bf16): oneDNN falls to a reference kernel for grouped-scale int8. Confirms the research finding
that oneDNN has no native block-scaled s8xs8->s32 GEMM.

### 4. Hand-written tiled ESIMD GEMM -- CORRECT but unoptimized -- bs_dpas_m3.cpp
Full tiled [TM=8 x TN=64] block-scaled int8 GEMM, 4-bit-resident E2M1 decode in-register, real
gate/up shape, 4-way ILP. Correct (relerr 2.7e-3 vs fp64). Speed M=512: BS 9.56ms (9.6 TOPS) /
per-channel-ceiling-of-same-tiling PC 8.70ms (10.5 TOPS) -- i.e. ~13x SLOWER than bf16. A naive
first-cut GEMM (no 2D block loads / prefetch / SLM staging / register blocking) sits at ~10 TOPS,
far from the 157-TOPS register-resident ceiling. Reaching even the ~1.3x-bf16 ceiling needs a full
XeTLA/CUTLASS-quality mainloop -- a large kernel-engineering effort.

## Verdict

NVFP4 int8-XMX prefill on B70 is NOT worth pursuing to production: the theoretical ceiling is only
~1.3x bf16 (not 2x -- group16<K32 eats half), oneDNN refuses (reference-slow), and realizing the
~1.3x needs a from-scratch optimized ESIMD/XeTLA GEMM. The current bf16-compute nvfp4_gemm_w4a16
path stays as the NVFP4 prefill path. The user's "E2M1x2 -> s8 -> int8_gemm_w8a8" idea was sound
(the repack is exact) but is defeated by the DPAS granularity mismatch, not by an implementation bug.

## Future lever (documented, unbuilt): the dot32 + correction trick
For a pair of adjacent groups g0,g1: exact = bg0*dot16_g0 + bg1*dot16_g1
  = bg1*dot32 + (bg0-bg1)*dot16_g0, where dot32 is one FULL-efficiency K=32 DPAS over both groups.
So 1 full K=32 DPAS + 1 half K=16 DPAS = 1.5 DPAS-equiv per 2 groups (vs 2.0 for naive BS) -> could
lift correct block-scaled int8 from ~1.3x toward ~1.5-1.7x bf16. Still requires the optimized GEMM
foundation (which #4 shows is the hard part). Only worth it if NVFP4 prefill ever becomes the
bottleneck AND someone builds the XeTLA-quality mainloop first.

## Files
- bench_penalty.cpp   -- the compute-bound block-scale penalty microbench (PC/BS/BSD/PCI modes)
- bs_dpas_m1/m2/m3.cpp -- ESIMD block-scaled int8 DPAS: m1 single-tile correctness -> m3 tiled GEMM
- build.sh / run.sh   -- AOT build (intel_gpu_bmg_g31) + card-0 run wrappers
- ../int8_prefill_ceiling.py -- real oneDNN op ceiling (int8 vs bf16 vs w4a16 on real shapes)
- ../validate_w4a8.py, ../build_nvfp4_w4a8.sh, ../onednn_dispatch_probe.py -- the oneDNN W4A8 route
- ../../kernels/nvfp4_gemm_w4a8.h -- the (correct-but-reference-slow) oneDNN W4A8 header
