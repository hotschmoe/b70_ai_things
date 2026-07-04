# Sub-int8 / NVFP4-decode kernel experiments on B70 (2026-07-04)

Empirical probe of whether Intel Arc B70 (Xe2) has ANY sub-int8 fast path, whether a
4-bit "spoof" can trick a fast path into eating 4-bit data, and whether keeping NVFP4
weights 4-bit-resident in VRAM (unpack+dequant in-kernel) actually raises DECODE
throughput vs the int8 path. All runs: card 1 only, TP=1, image `vllm-xpu-env:int8g-v0240`,
oneDNN ops from `nvfp4_kernel/_xpu_C.abi3.so`. Roofline: 608 GB/s HBM, ~367 INT8 TOPS.

Proto scripts: `vllm/nvfp4/proto/` (00 discover, 05 ops, 10 triton-dot, 20 spoof,
30 fused-triton, 40 int4-vs-int8, 50 verbose-dispatch).

Motivating sizing (27B, from the real ckpt): 4-bit resident = 21.9 GB (FITS one ~30 GB
card with KV headroom); int8-repack = 31.1 GB (does NOT fit); bf16 dequant = 56.7 GB.
So for the 27B the ONLY viable single-card fast serve is a 4-bit-in-VRAM path.

## EXPERIMENT A -- empirical sub-int8 capability

### A.1 oneDNN int4 / u4 weight-decompression matmul -- EXISTS and is FAST
The `XPU_SPECIFIC_KERNELS_ENABLED=ON` build already ships:
`int4_gemm_w4a16(A, B, bias, B_scale, B_zp, group_size, g_idx)` and `int4_gemm_w4a8`.
They resolve and run. Exact oneDNN dispatch (ONEDNN_VERBOSE=dispatch,profile_exec,
K5120 N17408 M1):

    int8_gemm_w8a16: exec,gpu,matmul,jit:gemm:any,  src:f16 wei:s8::blocked:ab
                     attr-scales:wei:2:f16                              t=0.719 (single-shot)
    int4_gemm_w4a16: exec,gpu,matmul,jit:gemm:any,  src:f16 wei:u4::blocked:ba
                     attr-scales:wei:3:f16:128x1 attr-zero-points:wei:0:s8  t=0.474

Impl string for BOTH = `jit:gemm:any` (oneDNN GemmStone JIT GEMM). The systolic path
`jit:xe_hp:gemm` is "skipping or dispatching to another implementation" at M=1 -- i.e.
DPAS/XMX systolic is NOT used at decode; the generic jit gemm is, and it is
bandwidth-bound. VERDICT: there IS a working sub-int8 GEMM on B70 (u4 weight
decompression), and at decode it is bandwidth-bound (as theory predicts), not systolic.

### A.2 torch / IPEX int4 ops
`torch.ops.aten._weight_int4pack_mm` and `_weight_int8pack_mm` exist but are the
tinygemm/CPU-oriented packs (int4pack has a `_for_cpu` sibling); not the XPU fast path.
IPEX is NOT installed in this image. The real XPU int4 path is the `_xpu_C` oneDNN op above.

### A.3 Triton tl.dot operand width (triton 3.7.1, Intel XPUDriver)
    i8 x i8  -> i32 : OK    rel-err 0.0000   0.047 ms   (DPAS-eligible)
    f16 x f16-> f32 : OK    rel-err 0.0000   0.045 ms
    i16 x i16-> i32 : FAIL  CompilationError (tl.dot refuses int16)
int8 is the NARROWEST tl.dot Triton-XPU accepts. There is NO int4 tl.dot; 4-bit must be
loaded as packed bytes and unpacked to >=int8/float in registers before tl.dot.

## EXPERIMENT B -- the spoofs

### B.1 raw-byte spoof (feed packed uint8 straight into int8 GEMM, no unpack)
Interpret packed NVFP4 bytes [N, K/2] as int8 weight and run int8_gemm_w8a16 over K/2:

    27B gate/up M1: int8 full-K 0.147ms 604 GB/s | spoof half-K 0.087ms 511 GB/s  1.69x  rel-err 9.2
    27B down    M1: int8 full-K 0.137ms 648 GB/s | spoof half-K 0.087ms 513 GB/s  1.58x  rel-err 9.0
    (M8 similar: 1.57-1.75x, rel-err ~8-9)

The spoof IS ~1.6-1.7x faster (it reads half the bytes) but the math is GARBAGE
(rel-err ~9) -- one lane holds sum x_j*(lo_j + 16*hi_j), two different weights entangled
with the wrong activation pairing; no scale disentangles it. VERDICT: the spoof proves
byte-count is the decode lever, but you MUST unpack correctly; the fast path cannot be
tricked into doing the E2M1 decode for free.

### B.2 double-pump (two 4-bit MACs per int8 lane)
Toy: want a1*lo + a2*hi = 41; one int8 lane (act=a1) yields a1*(lo+16*hi) = 166 ->
cross-term pollution 125. Xe2 has no 2x-int4-in-int8 dot. VERDICT: deadend.

## EXPERIMENT C -- the real decode lever: 4-bit-in-VRAM fused dequant-GEMM

### C.1 naive Triton fused kernel (register E2M1-LUT unpack + tl.dot) -- CORRECT but SLOW
`proto/30_fused_w4a16_triton.py`. Reads packed uint8 [N,K/2], LUT-decodes E2M1 nibbles
(sign * {0,1,2,3,4,6,8,12}), applies [K/16,N] group scale, tl.dot. M=1 results:

    27B gate/up: W4pack 4.20ms  10.6 GB/s  err 2e-4  | W8trit 2.45ms 36.4 GB/s | oneDNN-int8 0.16ms 550 GB/s
    27B down   : W4pack 4.84ms   9.2 GB/s  err 2e-4  | W8trit 2.79ms 32.0 GB/s | oneDNN-int8 0.16ms 573 GB/s

Numerically exact (err 2e-4) but ~50x slower than oneDNN int8: the select-chain LUT
decode + fp32 tl.dot + un-tuned loads are terrible codegen. At M=8 BOTH triton kernels
CRASH the Intel-Triton compiler: `TritonIntelRemoveMasks` pass -> `llvm::cast<CmpIOp>`
assertion (masking codegen bug, triton 3.7.1). VERDICT: naive Triton-XPU is a deadend
for this kernel -- both too slow and compiler-broken at M>1.

### C.2 the FAIR 4-bit test -- optimized oneDNN int4 vs int8 (identical shapes)
Since the naive Triton kernel is not representative, the honest test of "does 4-bit
halve decode latency" is the OPTIMIZED oneDNN int4 op vs the OPTIMIZED oneDNN int8 op.
`proto/40_onednn_int4_vs_int8.py`:

    shape                 M   int8 ms / GB/s    int4 ms / GB/s    int4/int8 latency
    27B gate/up K5120     1   0.131 / 679       0.081 / 548       0.62x
    27B down    K17408    1   0.131 / 682       0.079 / 565       0.60x
    8B  gate/up K4096     1   0.091 / 645       0.052 / 569       0.57x
    8B  down    K14336    1   0.092 / 635       0.050 / 583       0.55x
    27B gate/up           8   0.139 / 642       0.081 / 548       0.58x
    27B down              8   0.134 / 667       0.080 / 557       0.60x
    8B  gate/up           8   0.095 / 617       0.052 / 563       0.55x
    8B  down              8   0.095 / 617       0.050 / 585       0.53x

int4 is 0.53-0.62x the int8 latency = **1.6-1.9x faster decode** across all real MLP
shapes at M=1 and M=8. int4 sustains 548-585 GB/s (near HBM roofline) while moving HALF
the weight bytes. VERDICT: 4-bit-in-VRAM DOES roughly double the decode ceiling on B70,
via the optimized oneDNN jit gemm -- the byte-count is real and the hardware delivers it.

## Bottom-line verdicts

(i)  Sub-int8 fast path on B70? YES. oneDNN's u4 weight-decompression matmul
     (`int4_gemm_w4a16`, jit:gemm:any) runs coherently and is 1.6-1.9x faster than int8
     at decode. It is bandwidth-bound (not systolic DPAS) at M=1..8, which is exactly
     right since decode is BW-bound. Triton-XPU has no sub-int8 tl.dot (int8 is the floor).

(ii) Did any spoof work? NO (for correctness). The raw-byte spoof buys the 1.6x speed but
     rel-err ~9 garbage; the byte-count is the lever, the E2M1 decode cannot be skipped.
     Double-pump is a deadend (no 2x-int4-in-int8 dot on Xe2).

(iii) Does fused 4-bit-in-VRAM beat int8, and by how much? YES via optimized oneDNN:
     ~1.8x (0.55x latency) at decode, and it FITS the 27B on one card (21.9 GB vs int8's
     31.1 GB). NO via naive Triton (50x slower + compiler crash at M>1).

## Strongest recommendation -- the next kernel to build

Build `nvfp4_gemm_w4a16`: an oneDNN weight-decompression matmul that mirrors
`int4_gemm_w4a16.h` but sets the WEIGHT data type to `dnnl::memory::data_type::f4_e2m1`
(the bundled oneDNN already exposes `f4_e2m1` and `e8m0`), with:
  - weight  : NVFP4 packed 4-bit [K/2, N] bytes, viewed as f4_e2m1 (NT layout like int4)
  - scale   : [K/16, N] bf16 (or e4m3), oneDNN set_scales(WEIGHTS, mask (1<<0)+(1<<1),
              {group_size=16, 1})  -- this is the K-group-16 grouping we already fixed for
              the int8 path
  - zero_pt : NONE (E2M1 is symmetric -- drop the s8/u4 zp that int4_gemm_w4a16 carries)
  - activation: f16/bf16 (W4A16); joint dtype e.g. bf16_f4e2m1 added to onednn_ext.h
This is BIT-EXACT NVFP4 (no requant), needs the 27B's weights only at 4-bit (fits one
card), and should land at the ~0.55x-int8 latency the shipped u4 op already demonstrates
(1.8x decode). It is a small port: copy int4_gemm_w4a16.h -> swap the weight dtype to
f4_e2m1, set group_size 16, remove the zero-point args, add the joint_dtypes_t enum +
primitive-cache wiring, rebuild the `_xpu_C.abi3.so` (~8 min, int8-only build flags).

Forward call (per linear, one op, mirrors int4_gemm_w4a16):
    y = torch.ops._xpu_C.nvfp4_gemm_w4a16(x_bf16,           # [M, K]
                                          w_f4e2m1_packed,   # [K/2, N] uint8 (2 nibbles)
                                          bias_or_none,
                                          w_scale_bf16,      # [K/16, N]
                                          group_size=16)

Fallback if the jit gemm turns out not to support f4_e2m1 weights on XPU (the fp4_gemm
op today only wires mxfp4 = e8m0 scale @ group 32, W4A4): requantize NVFP4 -> LINEAR int4
and reuse the shipped `int4_gemm_w4a16`. This is lossy -- NVFP4's non-uniform E2M1 spacing
({..,1,1.5,2,3,4,6}) cannot be captured exactly by uniform int4 buckets -- so prefer the
f4_e2m1 op. Do NOT reuse int4_gemm for NVFP4 directly: its linear (u4 - zp) decode with
zp=8 covers -8..7, but NVFP4-as-int8 needs +/-12; and even folded, the decode is linear,
not the E2M1 LUT.

## Deadends logged
- Naive Triton-XPU fused 4-bit kernel: 50x slower than oneDNN int8; Intel-Triton
  `TritonIntelRemoveMasks` compiler crash at M=8. Not the route.
- Raw-byte spoof and int8 double-pump: fast but numerically wrong; no free E2M1 decode.
- Native int4 DPAS systolic at M=1: xe_hp systolic SKIPS; decode uses jit:gemm:any
  (bandwidth-bound) -- which is fine, but "int4 DPAS at decode" is a non-goal.
- int16 tl.dot: refused by triton. int8 is the Triton dot floor.
