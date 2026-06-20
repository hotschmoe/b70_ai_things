# 10 - INT8 W8A8 GEMM hand-tuning plan: make it the best in the world on the B70

Deep, ranked, executable plan to push our int8 W8A8 GEMM to the roofline on the Intel Arc Pro B70
(Xe2/Battlemage, 32 GB, **608 GB/s**, **367 INT8 TOPS** via XMX/DPAS, sub-group **16**, no native FP8).
Two regimes, separate levers:
- **Prefill (pp)**: m>=512, COMPUTE-bound -> target the 367 INT8 TOPS XMX peak.
- **Decode (tg)**: m=1, BW-bound -> target 608 GB/s (15.3 GiB int8 weights -> ~40 t/s ceiling).

Builds on `04_decode_optimization.md` (levers A/B/C + verified code-pointer table), `05` (oneDNN/IPEX
deltas), `06` (the int4-GEMV design), `01/02` (the W8A8 kernel we shipped). **Does not repeat them.**
This doc adds: (a) current-kernel gap analysis + a microbench DESIGN, (b) ranked HAND-TUNING levers for
pp and tg with the principle behind each, (c) two codex-drafted hand-written SYCL kernel skeletons
(prefill DPAS GEMM + decode dp4a GEMV), (d) a verified source-pointer table, (e) the first experiment.

Conventions: **VERIFIED** = read directly from our source on the box, or fetched from upstream source
(URL given). **PROPOSED** = our design choice / hypothesis, not yet measured. ASCII only, `->` not arrows.

GPU rule: this doc is RESEARCH + DRAFTS. Every GPU run is the lead's job, serialized through
`scripts/gpu-run` (CLAUDE.md). None of the kernels here are compiled or benchmarked.

================================================================================================
## 0. TL;DR -- the gaps and the top levers
================================================================================================

**Suspected prefill gap (pp).** Our W8A8 prefill is already strong (6353 t/s at 4096, 1.6x FP8 --
`docs/kernel/02`), BUT the per-GEMM XMX utilization is UNMEASURED and there is one concrete, VERIFIED
inefficiency: our oneDNN wrapper feeds the **weight with explicit strides**, never `format_tag::any`.
That **pins oneDNN to the user's plain s8 weight and forbids it from choosing its internal
blocked/VNNI-crosspacked weight layout** -- the exact layout DPAS wants. oneDNN's own inference guide
and IPEX `QMatmul.h` both do `format_tag::any` + a one-time cached `weights_desc()` reorder; we don't.
Likely 1.1x-1.5x prefill on the big GEMMs, near-zero risk, ~1 day. **This is the single highest-EV change.**

**Suspected decode gap (tg).** W8A8 decode is ~26.7 t/s with PIECEWISE graph capture = ~67% of the
~40 t/s int8 ceiling (15.3 GiB @ 608 GB/s). oneDNN at m=1 runs the **general JIT GEMM doing GEMV**
(`jit:gemm:any`, confirmed for the int4 sibling in `04`); the systolic array is ~1/16 utilized and the
plain weight layout is read **uncoalesced** (row-major `[k*N+n]` strides by N across the lanes that
should share a cache line). The residual ~33% to the ceiling is (a) the layout/coalescing tax and (b)
the act-quant + dispatch tax that graph capture already mostly fixed. A purpose-built **dp4a GEMV with a
reordered (column-contiguous) weight** can target >=90%.

**Top-3 pp levers:** (1) `format_tag::any` weights + cached reorder [VERIFIED-as-IPEX-recipe];
(2) measure XMX util via `ONEDNN_VERBOSE=2` + roofline, then a hand DPAS GEMM (atom
`XE_8x16x32_S32S8S8S32_TT`) with SLM double-buffer + offline VNNI-packed B if oneDNN underperforms;
(3) fold `A_scale*B_scale` at quant time + a fused dequant epilogue (keep the K-loop pure int32).

**Top-3 tg levers:** (1) offline **column-contiguous weight reorder** for coalesced GEMV loads
[the int4 doc-06 (d) idea, ported to int8]; (2) a hand **dp4a GEMV** (one sub-group / output column,
SLM-staged activation reuse, sub-group reduce) replacing the oneDNN GEMV-emulation; (3) push toward
**FULL graph capture** (`04` lever A2) so the last dispatch/attention tax disappears.

**Single highest-EV first experiment:** the `format_tag::any` weight-reorder prototype on the W8A8
prefill GEMM, gated behind an `ONEDNN_VERBOSE=2` microbench that first MEASURES the current XMX util and
which impl fires. One day, library does the systolic work, and it tells us whether we even need a hand
kernel for prefill. See section (e).

================================================================================================
## (a) Current-kernel analysis + the suspected gap + a microbench DESIGN
================================================================================================

### a.1 What our kernel actually does today [VERIFIED from source on the box]

Path: vLLM `XPUInt8ScaledMMLinearKernel.apply_weights` (`contrib/vllm_int8_xpu/xpu_int8.py`)
-> fused `dynamic_per_token_int8_quant` SYCL op -> `torch.ops._xpu_C.int8_gemm_w8a8`
-> `onednn_matmul.cpp::int8_gemm_w8a8` -> `int8_gemm_w8a8.h::dnnl_matmul_w8a8_int8`
-> `matmul_primitive_create_and_cache` (`onednn_ext.h`) -> oneDNN `dnnl::matmul`.

Key facts, all read from the live files:
- oneDNN version bundled = **v3.9.1** (`/opt/intel/oneapi/2025.3`, `dnnl_version.h`). Newer than docs
  assumed -- v3.8's "improved int8 matmul perf with src+weight zero-points" is already in. Lever B5
  (bump oneDNN) is MOOT; we are ahead.
- Joint dtype `s8_s8_f16` / `s8_s8_bf16` are wired (`onednn_ext.h`). s8s8s32 is native on Battlemage XMX.
- Per-token src scale (mask `(1<<0)+(1<<1)`, `{1,k}`) + per-channel weight scale (mask `1<<1`).
  **Symmetric -> no src/weight zero-points** (clean; the int4 sibling's wasteful src-zp is NOT present
  here -- good).
- **THE GAP (VERIFIED):** the weight memory descriptor is built with **explicit strides**, not
  `format_tag::any`:
  ```
  onednn_ext.h:795   auto wei_md = memory::desc({k, n}, wei_dt, wei_strides);   // wei_strides = {1, ldb} for NT
  ```
  There is no `format_tag::any` anywhere in the int8 path. oneDNN therefore cannot return a packed
  `weights_desc()` for us to reorder once; it must run a plain-layout kernel or repack internally.
- The fused act-quant kernel (`dynamic_per_token_int8_quant.cpp`) uses **sub_group_size=32**, one
  work-group (one sub-group) per row, two passes (absmax reduce, then quantize). Correct, but on Xe2 the
  native sub-group is **16** -- a SG=32 kernel runs as 2 native SIMD16 sub-groups; minor, note for later.

### a.2 Why explicit strides cost prefill throughput [VERIFIED upstream]

- oneDNN inference guide: pass `format_tag::any` for src/weights/dst, query the primitive for the
  recommended format, **reorder weights once** and reuse the reordered form.
  -> https://github.com/uxlfoundation/oneDNN/blob/main/doc/usage_models/inference.md
- oneDNN matmul dev guide: "the less information about shapes or format is available at creation, the
  less performant the execution." Explicit strides = MORE pinned info = LESS freedom = slower path.
  -> https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html
- oneDNN intel JIT GEMM generator: B is consumed by DPAS in a *packed* (crosspacked) systolic layout;
  the generator "always use 1D addressing for packed inputs" and gates 2D-block prefetch on
  `!isPacked(problem.B.layout)`. A plain user B takes a worse access path or an internal repack each call.
  -> https://github.com/uxlfoundation/oneDNN/blob/main/src/gpu/intel/gemm/jit/generator/strategy.cpp
- **IPEX does exactly the fix:** `QMatmul.h` (int8 path) sets `m2_md = memory::desc(..., format_tag::any)`,
  queries `matmul_pd.weights_desc()`, reorders the weight only if the user layout differs, and caches it.
  -> https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/csrc/gpu/oneDNN/QMatmul.h  (L222-335)

So our W8A8 forfeits the blocked/VNNI s8 weight layout that DPAS wants. Confidence: VERIFIED that we use
explicit strides and that IPEX/oneDNN recommend `format_tag::any`; the *size* of the win is PROPOSED
(measure it -- see the microbench) but the direction is certain.

### a.3 Why decode (m=1) sits at ~67% [VERIFIED reasoning, value PROPOSED]

- At m=1 the GEMM is pure weight bandwidth. oneDNN's int8 GEMV lands on the general JIT GEMM
  (`jit:gemm:any` -- confirmed for the int4 sibling in `04`, same code path) which split-K-emulates a
  GEMV across the systolic array -> ~1/16 lanes busy, plus scheduling overhead.
- Worse, with the plain row-major weight `[k*N+n]`, the 16 lanes that cooperate on one output column read
  addresses **N elements apart** -> uncoalesced, many cache lines per step. (This is the exact mechanism
  `06_sycl_int4_gemv_design.md (d)` identified for int4; it applies identically to int8.)
- Graph capture (`04` A1) already removed the dispatch/act-quant tax (the +13% to 26.7 t/s). The residual
  is the GEMM's own layout/coalescing inefficiency -> a hand GEMV with reordered weights is the fix.

### a.4 Microbench DESIGN for the lead (run via `scripts/gpu-run`)

GOAL: (1) MEASURE current XMX util at prefill and BW% at decode, (2) confirm which oneDNN impl fires and
whether the weight is being internally reordered, (3) give a baseline the hand kernels must beat.

Shapes (the real Qwen3-14B W8A8 GEMMs; K x N, NT weight):
```
  qkv/o_proj   5120  x 5120        (square, attention)
  gate/up      5120  x 17408       (MLP up, wide N)
  down_proj    17408 x 5120        (MLP down, wide K)
```
Drive m across both regimes: m in {1, 8, 64, 512, 2048, 4096} (1/8 = decode, 512+ = prefill).

Run 1 -- impl + reorder visibility (free, do FIRST):
```
  ONEDNN_VERBOSE=2  (or =all)  on a tiny script that calls torch.ops._xpu_C.int8_gemm_w8a8 directly
  for each shape, m in {1, 512, 4096}.
  LOOK FOR:
    - the impl string: jit:gemm:* (good) vs ref:* (disaster) vs gemm:any.
    - a "reorder" line on the WEIGHTS md appearing INSIDE the timed loop (= per-call repack tax) vs
      only once at warmup. If it repeats per call -> the explicit-strided md is forcing repacks.
    - the chosen weight format tag in the matmul primitive (is it a blocked/ab... tag or plain?).
    - confirm v3.9.1 and that no src/weight zero-point correction term is present (symmetric path).
```
Run 2 -- prefill XMX util roofline:
```
  Time int8_gemm_w8a8 alone (CUDA-graph or 100-iter median, warmup discarded) at m in {512,2048,4096}.
  util% = (2 * m * K * N flops) / time / 367e12.
  A clean DPAS GEMM at these shapes should hit 60-85% of 367 TOPS. If we are <40%, the layout/util gap
  is large and a hand DPAS GEMM (or format_tag::any) has big headroom. Record per-shape.
```
Run 3 -- decode BW roofline:
```
  Time at m=1 (median of many). BW_eff = (K*N bytes weight + small) / time. % = BW_eff / 608e9.
  Current full-model decode ~67% of the 40 t/s ceiling; this isolates the per-GEMM number.
  Compare to the ggml q8_0 GEMV BW ceiling (run llama.cpp test-backend-ops perf -o MUL_MAT at m=1,
  q8_0 model -- the recipe in 06 (e), swap q4_0 -> q8_0). That is the "what's actually on the table" number.
```
DELIVERABLE: a JOURNAL.md table -- per shape, per m: impl string, reorder-per-call yes/no, prefill XMX%,
decode BW%. This decides whether pp needs `format_tag::any` only, or a full hand DPAS GEMM.

================================================================================================
## (b) Ranked hand-tuning levers -- PREFILL (compute-bound, target 367 TOPS)
================================================================================================

Each lever: expected gain, effort, risk, and THE PRINCIPLE (the thing to learn). Levers are independent
unless noted. Sequence by EV (cheap measurement -> library win -> hand kernel).

**PP-0. Measure first (the microbench above).** [free] PRINCIPLE: never hand-write a kernel to beat an
unmeasured baseline. oneDNN on Xe2 may already be at 70%+ for these shapes -- if so, the EV of a hand GEMM
collapses and PP-1 (a one-day library tweak) is the whole win. Gates every lever below.

**PP-1. `format_tag::any` weights + one-time cached reorder.** [~1 day, LOW risk, HIGH EV]
Change `int8_gemm_w8a8.h` / `onednn_ext.h` so the WEIGHT md is `format_tag::any`; query
`matmul_pd.weights_desc()`; reorder the s8 weight once in `process_weights_after_loading` into the blocked
format; cache and reuse. Mirror IPEX `QMatmul.h` exactly (it does this for int8 already).
EXPECTED: 1.1x-1.5x on the big GEMMs (oneDNN picks its VNNI-crosspacked DPAS layout instead of a plain
kernel/per-call repack). PRINCIPLE: **let the library pick the operand layout the systolic array wants;
hand it `any`, not a fixed stride.** The weight is read every token -> a one-time repack amortizes to zero.
RISK: confirm the reordered md is stable across our shapes and that bf16/f16 dst still selects an optimized
impl (Run-1 verbose confirms). This is the lowest-risk, highest-EV pp change.

**PP-2. Hand DPAS GEMM (joint_matrix) with SLM double-buffer + offline VNNI-packed B.** [1-2 wk, MED risk]
ONLY if PP-0 shows oneDNN <~50-60% XMX util after PP-1. Skeleton in section (c.1). Atom (VERIFIED, two
primary sources): **`XE_8x16x32_S32S8S8S32_TT`** = s8[M<=8,K=32] x s8[K=32,N=16] -> s32[M=8,N=16], B
**VNNI-packed** (`layout::ext_intel_packed`). Tile hierarchy ported from CUTLASS s8 (CTA 128x128xKtile ->
sub-group tile -> DPAS atom) + Marlin/Machete's offline-prepack idea.
EXPECTED: match or beat oneDNN; the upside is full control of the pipeline (more stages, fused epilogue,
no per-call repack). PRINCIPLE: **a compute-bound GEMM is a three-level tile (WG/SLM -> sub-group -> DPAS
atom) plus a software pipeline that hides the global load behind the systolic math.** CUTLASS settles at
3 smem stages, Marlin at 4; pick by SLM budget.
RISK: joint_matrix correctness (see c.1 CAVEATS). MED -- joint_matrix is the supported portable surface
for int8 on BMG (the one case where llama.cpp's "no joint_matrix" stance does not apply, since we are
compute-bound here, not BW-bound).

**PP-3. Fold A_scale*B_scale at quant time + fused dequant epilogue.** [hours, LOW risk]
Keep the entire K-reduction in **int32**; apply `acc_s32 * A_scale[row] * B_scale[col] (+bias)` ONCE in
the epilogue. Optionally pre-multiply per-channel B_scale into a single fused scale where possible.
oneDNN already does dst scaling; the lever is making sure a hand kernel (PP-2) never touches float in the
K-loop. PRINCIPLE: **scaling in int32-land is free; do it once at the end, never per-MAC.** (CUTLASS/Marlin
both do exactly this -- the int32 accumulator is the canonical place to defer dequant.)

**PP-4. Tile-shape dispatch by M (IPEX hgemm_policy pattern).** [days, on top of PP-2]
Small-M prefill (m~128) wants a different tile than m>=2048. IPEX dispatches a `(wg_m x wg_n, sg, slm_ks)`
policy per (M,N,K) via a hash table: thin wg_m + K-slicing for small M, 256x256 wg for large M.
PRINCIPLE: **one tile shape is not optimal across the M range; dispatch on shape.** Low EV until PP-2
exists; logged for completeness.
-> https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/csrc/gpu/aten/operators/xetla/kernels/GEMM/hgemm_policy_xehpc.cpp

================================================================================================
## (b') Ranked hand-tuning levers -- DECODE (BW-bound, target 608 GB/s)
================================================================================================

**TG-0. Confirm the ceiling (ggml q8_0 GEMV bench).** [free, lead] Run llama.cpp `test-backend-ops perf
-o MUL_MAT` at m=1 with a q8_0 model (recipe `06 (e)`, q4_0 -> q8_0). Gives the real int8 GEMV BW% on the
B70. If ggml hits ~90%+ and we are at ~67%, the coalescing/layout tax is the gap -> TG-1/TG-2 justified.
PRINCIPLE: **measure the achievable BW on this exact card before writing a GEMV.**

**TG-1. Offline column-contiguous weight reorder for coalesced GEMV loads.** [~2 days, LOW risk, HIGH EV]
The single highest-leverage decode item. Reorder the s8 weight offline (once, at load) from row-major
`W[k*N+n]` to column-major-over-K `W_reorder[n*K+k]`, so the 16 lanes of the sub-group computing column n
read **16 contiguous int8 bytes** of that column = one cache line, fully coalesced. (Direct int8 analog of
`06 (d)` for int4; ggml's reorder path PR #12035 is the precedent.)
EXPECTED: this is what moves 67% -> 90% BW. PRINCIPLE: **at m=1 you are read-port-bound; the ONLY thing
that matters is that consecutive lanes touch consecutive bytes (coalescing).** A reorder is a pure index
permutation, zero per-decode cost.
NOTE: TG-1 only pays off with a kernel that reads it that way -> bundle with TG-2 (oneDNN won't consume an
arbitrary reordered layout). Standalone, TG-1 is the data prep for TG-2.

**TG-2. Hand dp4a GEMV (one sub-group / output column, SLM activation reuse, sub-group reduce).**
[1-2 wk, MED risk] Skeleton in section (c.2). NOT DPAS -- at m=1 the systolic array is ~1/16 utilized so
a vectorized 4-wide int8 FMA (dp4a-style) saturates the read port with less overhead. This is exactly
ggml-sycl's `mul_mat_vec_q` structure (VERIFIED): one sub-group per row/column, dp4a partials, butterfly
sub-group reduce (`for mask=8,4,2,1: tmp += permute_sub_group_by_xor`), lane-0 write, `reqd_sub_group_size(16)`.
Our additions over ggml: stage the per-token activation into SLM once per work-group (all sub-groups in
the WG reuse the same x[K]) and consume the TG-1 reordered weight.
EXPECTED: >=90% of 608 GB/s = the residual 1.3x-1.4x decode win on top of graph capture.
PRINCIPLE: **a BW-bound GEMV does not need matrix cores; it needs coalesced loads + a cheap reduction +
deferred dequant (one float multiply at the end, not per-element).**
RISK: MED. ggml-sycl proves the idiom works at SG=16 on Intel; our risk is the reorder ABI + plumbing it
as a custom op alongside `int8_gemm_w8a8` (an m==1 fast-path branch in the wrapper).

**TG-3. FULL graph capture (TRITON_ATTN) -- finish the dispatch story.** [hours-days, lead]
`04` lever A2: PIECEWISE leaves attention eager; FULL (via `--attention-backend TRITON_ATTN`, supported
per PR #34482) captures attention too. Independent of the GEMM but lifts decode further and flips
spec-decode positive. PRINCIPLE: **at m=1, non-GEMM dispatch is half the time; capture all of it.** Already
the standing top untested experiment in `04`; restated here because it co-optimizes tg with TG-1/TG-2.

**TG-4. m==1 fast-path branch in the wrapper.** [hours, glue] `int8_gemm_w8a8` should detect m==1 and
dispatch the hand GEMV (TG-2) instead of oneDNN; m>1 stays on oneDNN (PP-1) / the hand GEMM (PP-2).
PRINCIPLE: **decode and prefill are different kernels; branch on M at the call site** (Marlin's
`m_block_size_8` decode flag is the same idea).

================================================================================================
## (c) Codex-drafted hand-written SYCL kernel skeletons
================================================================================================

Both DRAFTED via `codex exec` (gpt-5.5), then reviewed against the verified atom/reduce facts. **NEITHER
IS COMPILED.** They are design review artifacts: land correctness first (the CAVEATS), then the perf TODOs.
The atom shapes, the VNNI-packed-B rule, the joint_matrix_mad template order, and the sub-group reduce
idiom were all cross-checked against primary sources (sycl-tla + the joint_matrix spec + ggml-sycl).

------------------------------------------------------------------------------------------------
### (c.1) DRAFT prefill int8 GEMM -- joint_matrix/DPAS, SLM double-buffer (m>=512)
------------------------------------------------------------------------------------------------

VERIFIED atom: `XE_8x16x32_S32S8S8S32_TT` == joint_matrix s8 [M<=8,K=32] x s8 [K=32,N=16] -> s32 [8,16],
B = `layout::ext_intel_packed` (VNNI: 4 int8 packed along K per 32-bit channel). joint_matrix_mad template
order is (M,K,N). sycl-tla's BMG TileShape is `Shape<_256,_256,_32>` (K-tile 32 == the DPAS K).

```cpp
#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace draft_int8_prefill {

namespace jm = sycl::ext::oneapi::experimental::matrix;

using s8  = std::int8_t;
using s32 = std::int32_t;
using f16 = sycl::half;

template <int WG_M = 128, int WG_N = 128, int K_STEP = 32, int SG_TILE_M = 16, int SG_TILE_N = 32>
class draft_int8_gemm_prefill_kernel;

template <int WG_M = 128, int WG_N = 128, int K_STEP = 32, int SG_TILE_M = 16, int SG_TILE_N = 32>
void draft_int8_gemm_prefill(
    sycl::queue& q,
    const s8* A,             // [M, K], row-major, per-token quantized
    const float* A_scale,    // [M]
    const s8* B_vnni,        // weight, OFFLINE VNNI-packed for use::b (see CAVEAT 2)
    const float* B_scale,    // [N]
    const f16* bias,         // optional [N], may be nullptr
    f16* D,                  // [M, N], row-major
    int M, int N, int K) {
  static_assert(WG_M % 8 == 0,  "WG_M multiple of DPAS atom M=8");
  static_assert(WG_N % 16 == 0, "WG_N multiple of DPAS atom N=16");
  static_assert(K_STEP == 32,   "Battlemage int8 DPAS K atom = 32");
  static_assert(SG_TILE_M % 8 == 0  && SG_TILE_N % 16 == 0, "sub-tile multiples of atom");

  // CTA tile 128x128: enough reuse for compute-bound prefill, less reg/SLM pressure than 256x128.
  constexpr int SG_ROWS = SG_TILE_M / 8;
  constexpr int SG_COLS = SG_TILE_N / 16;
  constexpr int SGS_M   = WG_M / SG_TILE_M;
  constexpr int SGS_N   = WG_N / SG_TILE_N;
  constexpr int WG_SIZE = (SGS_M * SGS_N) * 16;
  constexpr int STAGES  = 2;                       // double-buffer

  const int grid_m = (M + WG_M - 1) / WG_M;
  const int grid_n = (N + WG_N - 1) / WG_N;

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<s8, 1> slm_a(STAGES * WG_M * K_STEP, cgh);
    sycl::local_accessor<s8, 1> slm_b(STAGES * K_STEP * WG_N, cgh);   // must hold VNNI bytes (CAVEAT 2)

    cgh.parallel_for<draft_int8_gemm_prefill_kernel<WG_M,WG_N,K_STEP,SG_TILE_M,SG_TILE_N>>(
        sycl::nd_range<2>(sycl::range<2>(grid_m, grid_n * WG_SIZE), sycl::range<2>(1, WG_SIZE)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
          auto sg = it.get_sub_group();
          const int wg_m0 = it.get_group(0) * WG_M;
          const int wg_n0 = it.get_group(1) * WG_N;
          const int local = it.get_local_id(1);
          const int sg_id = local / 16;
          const int sg_m_id = sg_id / SGS_N;
          const int sg_n_id = sg_id % SGS_N;

          jm::joint_matrix<sycl::sub_group, s32, jm::use::accumulator, 8, 16, jm::layout::dynamic>
              acc[SG_ROWS][SG_COLS];
#pragma unroll
          for (int rm = 0; rm < SG_ROWS; ++rm)
#pragma unroll
            for (int cn = 0; cn < SG_COLS; ++cn) jm::joint_matrix_fill(sg, acc[rm][cn], 0);

          // Cooperative gmem->SLM stage. A as plain k-major; B as ALREADY-VNNI bytes (CAVEAT 2).
          // TODO(perf): replace scalar staging with Xe 2D block loads (XE_2D_U8x32x32_LD_N for A,
          //             XE_2D_U8x32x32_LD_V for VNNI B) + prefetch; this is the main perf TODO.
          auto load_stage = [&](int stage, int k0) {
            for (int idx = local; idx < WG_M * K_STEP; idx += WG_SIZE) {
              const int mi = idx / K_STEP, kk = idx % K_STEP;
              const int gm = wg_m0 + mi, gk = k0 + kk;
              slm_a[stage*WG_M*K_STEP + mi*K_STEP + kk] = (gm<M && gk<K) ? A[gm*K + gk] : s8{0};
            }
            for (int idx = local; idx < K_STEP * WG_N; idx += WG_SIZE) {
              const int kk = idx / WG_N, nj = idx % WG_N;        // NB: index into the VNNI-packed B tile
              const int gk = k0 + kk, gn = wg_n0 + nj;
              slm_b[stage*K_STEP*WG_N + kk*WG_N + nj] = (gk<K && gn<N) ? B_vnni[gk*N + gn] : s8{0};
            }
          };

          load_stage(0, 0);
          sycl::group_barrier(it.get_group());

          for (int k0 = 0; k0 < K; k0 += K_STEP) {
            const int stage = (k0 / K_STEP) & 1;
            if (k0 + K_STEP < K) load_stage(stage ^ 1, k0 + K_STEP);   // prefetch next while computing
            sycl::group_barrier(it.get_group());
#pragma unroll
            for (int rm = 0; rm < SG_ROWS; ++rm) {
              jm::joint_matrix<sycl::sub_group, s8, jm::use::a, 8, 32, jm::layout::row_major> a_frag;
              const int a_row = sg_m_id*SG_TILE_M + rm*8;
              jm::joint_matrix_load(sg, a_frag, slm_a.get_pointer() + stage*WG_M*K_STEP + a_row*K_STEP,
                                    K_STEP);
#pragma unroll
              for (int cn = 0; cn < SG_COLS; ++cn) {
                jm::joint_matrix<sycl::sub_group, s8, jm::use::b, 32, 16, jm::layout::ext_intel_packed> b_frag;
                const int b_col = sg_n_id*SG_TILE_N + cn*16;
                // CAVEAT 2: stride for packed B is the PACKED row pitch. For a plain [K_STEP x WG_N]
                // tile this load is WRONG -- SLM must hold VNNI bytes; pass packed stride (e.g. WG_N*4
                // for a [K_STEP/4][WG_N*4] packed tile). See codex critique below.
                jm::joint_matrix_load(sg, b_frag, slm_b.get_pointer() + stage*K_STEP*WG_N + b_col, WG_N);
                acc[rm][cn] = jm::joint_matrix_mad(sg, a_frag, b_frag, acc[rm][cn]);
              }
            }
            sycl::group_barrier(it.get_group());
          }

          // Epilogue. ROBUST per codex critique: joint_matrix_store the s32 tile to scratch, then a
          // plain elementwise dequant pass with explicit (row,col). The apply-lambda (i,j) IS the
          // logical coord on the Intel overload, but store+dequant is the portable, fp16-out-correct path.
          // TODO(perf): cache B_scale per SG_N tile, vectorize, fuse bias; use a 2D block store.
#pragma unroll
          for (int rm = 0; rm < SG_ROWS; ++rm)
#pragma unroll
            for (int cn = 0; cn < SG_COLS; ++cn) {
              const int tm0 = wg_m0 + sg_m_id*SG_TILE_M + rm*8;
              const int tn0 = wg_n0 + sg_n_id*SG_TILE_N + cn*16;
              jm::joint_matrix_apply(sg, acc[rm][cn], [=](s32& v, int i, int j) {
                const int row = tm0 + i, col = tn0 + j;
                if (row < M && col < N) {
                  float o = static_cast<float>(v) * A_scale[row] * B_scale[col];
                  if (bias) o += static_cast<float>(bias[col]);
                  D[row*N + col] = static_cast<f16>(o);
                }
              });
            }
        });
  });
}

}  // namespace draft_int8_prefill
```

CAVEATS (verify on first compile -- these are the correctness risks, ranked):
1. **joint_matrix template params / packed-B enum name** -- confirm `layout::ext_intel_packed` and the
   (Group,T,use,Rows,Cols,layout) order against the installed oneAPI headers; the int8 atom is M<=8/N=16/K=32.
2. **[BIGGEST] VNNI-packed B through SLM (codex critique, VERIFIED vs the Intel matrix asciidoc).**
   `layout::ext_intel_packed` means **already VNNI-packed** -- it does NOT pack a plain k-major tile
   internally. Physical layout: `B_packed[(k/4)*stride + n*4 + (k&3)] = B_plain[k*N + n]`. For a logical
   [32,16] atom the packed tile is [8][16*4] bytes; the `stride` to `joint_matrix_load` is the **packed
   row pitch in int8** (atom-only [8][64] -> stride 64; tile [K_STEP/4][WG_N*4] -> stride WG_N*4; global
   [K/4][N*4] -> stride N*4). So: (i) the offline reorder must produce VNNI byte order, AND (ii) the SLM
   staging must preserve it (stage as [K_STEP/4][WG_N*4], not [K_STEP][WG_N]); the draft's plain-tile
   staging + stride=WG_N is WRONG and must be fixed to packed layout. Source:
   https://raw.githubusercontent.com/intel/llvm/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_intel_matrix.asciidoc
3. **group_barrier placement** -- with true async/2D-block loads the barrier scheme changes; the scalar
   version needs barrier after each stage load and after compute (as written), re-check when async lands.
4. **Accumulator register pressure** -- SG_ROWS*SG_COLS s32 [8,16] matrices live in registers; 16x32
   sub-tile = 2x2 = 4 atoms; raising SG_TILE spills. Tune against occupancy.
5. **Epilogue scaling (codex critique)** -- the Intel `joint_matrix_apply` (i,j) ARE logical coords
   (i in 0..7, j in 0..15), so per-row A_scale / per-col B_scale is correct as written. BUT the robust,
   portable, fp16-out-correct path is `joint_matrix_store` the s32 tile -> elementwise dequant pass with
   global (row,col). Prefer that for production.
6. **K must be a multiple of 32, M tile a multiple of 8, N tile a multiple of 16** -- our shapes (K in
   {5120,17408}, N in {5120,17408}) satisfy this; add a K-tail only if a non-aligned shape appears.

------------------------------------------------------------------------------------------------
### (c.2) DRAFT decode int8 GEMV -- dp4a, one sub-group/column, SLM act reuse (m=1)
------------------------------------------------------------------------------------------------

VERIFIED idiom (ggml-sycl `mul_mat_vec_q`): one sub-group (16 lanes) per output column, dp4a partials,
butterfly sub-group reduce, lane-0 write, `reqd_sub_group_size(16)`, no SLM on the weight (BW-bound).
Our additions: SLM-stage the activation once per WG (reused by all columns in the WG), consume the TG-1
column-contiguous reordered weight for coalesced loads.

```cpp
#include <sycl/sycl.hpp>
#include <cstdint>

namespace draft_int8_decode {

using i8  = std::int8_t;
using i32 = std::int32_t;
using f16 = sycl::half;

static inline i32 sign_extend_byte(i32 x) { return (x << 24) >> 24; }

// dp4a-style signed int8 dot: acc += dot4(bytes(a), bytes(b)).
// TODO(perf): try a native Xe2 dp4a intrinsic if the SYCL stack exposes one (ggml uses dpct::dp4a,
//             which lowers to a HW dp4a where available).
static inline i32 dp4a(i32 a, i32 b, i32 acc) {
  acc += sign_extend_byte((a>>0 )&0xff) * sign_extend_byte((b>>0 )&0xff);
  acc += sign_extend_byte((a>>8 )&0xff) * sign_extend_byte((b>>8 )&0xff);
  acc += sign_extend_byte((a>>16)&0xff) * sign_extend_byte((b>>16)&0xff);
  acc += sign_extend_byte((a>>24)&0xff) * sign_extend_byte((b>>24)&0xff);
  return acc;
}
static inline i32 load_i8x4(const i8* p) {   // TODO(perf): wide vec / block_load
  return (i32)(std::uint8_t)p[0] | ((i32)(std::uint8_t)p[1]<<8)
       | ((i32)(std::uint8_t)p[2]<<16) | ((i32)(std::uint8_t)p[3]<<24);
}

template <int SG_PER_WG = 8> class draft_int8_gemv_decode_kernel;

template <int SG_PER_WG = 8>
void launch_draft_int8_gemv_decode(
    sycl::queue& q,
    const i8* xq,             // [K] int8 activation (per-token, symmetric, zp=0)
    float xs,                 // scalar per-token act scale
    const i8* W_reorder,      // [N, K] column-major-over-K: W_reorder[n*K + k]  (TG-1 reorder)
    const float* W_scale,     // [N] per-channel scale
    f16* y,                   // [N]
    int K, int N) {
  constexpr int SG = 16, EPL = 4, KSTEP = SG * EPL;   // each lane does 4 k; lanes stride by 64 -> coalesced
  const int num_wg = (N + SG_PER_WG - 1) / SG_PER_WG;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<i8, 1> x_slm(sycl::range<1>(K), cgh);     // activation reused by all SGs in WG
    cgh.parallel_for<draft_int8_gemv_decode_kernel<SG_PER_WG>>(
        sycl::nd_range<2>(sycl::range<2>(num_wg*SG_PER_WG, SG), sycl::range<2>(SG_PER_WG, SG)),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
          const int sg_in_wg = it.get_local_id(0);
          const int lane = it.get_sub_group().get_local_linear_id();
          const int wg_lin = SG_PER_WG * SG;
          const int n = it.get_group(0) * SG_PER_WG + sg_in_wg;
          auto sg = it.get_sub_group();

          // KEY REUSE: stage xq[K] into SLM once; all SG_PER_WG columns reuse it.
          for (int k = it.get_local_linear_id(); k < K; k += wg_lin) x_slm[k] = xq[k];
          sycl::group_barrier(it.get_group());

          i32 acc = 0;
          if (n < N) {
            const i8* w = W_reorder + (std::size_t)n * K;
            // lane L owns k = L*4, L*4+64, ...  -> the 16 lanes read one contiguous 64-byte line/step.
            // TODO(perf): unroll; wide vec load; K-tail when K % 64 != 0; prefetch.
            for (int k = lane*EPL; k + EPL <= K; k += KSTEP)
              acc = dp4a(load_i8x4(&x_slm[k]), load_i8x4(&w[k]), acc);
          }
          acc = sycl::reduce_over_group(sg, acc, sycl::plus<i32>());   // or butterfly permute_sub_group_by_xor
          if (n < N && lane == 0) y[n] = (f16)((float)acc * xs * W_scale[n]);   // deferred dequant, once
        });
  });
}

}  // namespace draft_int8_decode
```

CAVEATS (verify on first compile):
1. **lane/k coalescing** -- `n = group(0)*SG_PER_WG + local_id(0)` assumes dim-1 (size 16) is the
   fast-varying lane dim so each dim-0 slice is one sub-group; print `sg.get_group_linear_id()` to confirm
   on icpx 2025.x (same CAVEAT as the int4 GEMV in `06`). The lane*4 + stride-64 mapping is the whole
   coalescing argument -- verify the generated loads are 64-byte lines.
2. **reduce_over_group on int32** -- well-defined at SG=16; if it miscompiles, use the ggml butterfly
   fallback `for (m=8;m>0;m>>=1) acc += permute_sub_group_by_xor(sg, acc, m)`.
3. **SLM activation staging barrier** -- the group_barrier before any lane reads x_slm is mandatory;
   omitting it silently corrupts.
4. **dp4a sign-extension** -- the explicit `(x<<24)>>24` byte extraction must stay signed after the
   compiler optimizes; unit-test against a numpy int8 dot.
5. **K-tail** -- our shapes K in {5120,17408} are multiples of 64, so the simple loop is exact; add a tail
   if a non-multiple-of-64 K appears.
6. **TG-1 reorder ABI** -- the kernel REQUIRES `W_reorder[n*K+k]`; ship the host reorder helper + a
   `register_fake` shape so it slots into `process_weights_after_loading` (mirror the int4 plan in `06`).

================================================================================================
## (d) Verified source-pointer table (URLs fetched 2026-06-20)
================================================================================================

VERIFIED = read directly. Each line = the idiom we took -> the source.

```
WHAT WE TOOK                                  | SOURCE (URL)
----------------------------------------------+-------------------------------------------------------
[OUR CODE, on the box, VERIFIED]
  explicit-strided weight md (the gap)        | vllm-xpu-kernels csrc/xpu/onednn/onednn_ext.h:795
  int8 GEMM wrapper + s8_s8 joint dtype       | .../onednn/int8_gemm_w8a8.h ; onednn_ext.h (s8_s8_f16/bf16)
  fused per-token quant (SG=32, note Xe2=16)  | .../sycl/dynamic_per_token_int8_quant.cpp
  oneDNN v3.9.1 bundled (B5 moot)             | image /opt/intel/oneapi/2025.3 dnnl_version.h
  vLLM kernel class + apply_weights           | contrib/vllm_int8_xpu/xpu_int8.py
[PREFILL -- DPAS GEMM]
  int8 DPAS atom XE_8x16x32_S32S8S8S32_TT     | intel/sycl-tla include/cute/arch/mma_xe_legacy.hpp
   (M<=8,N=16,K=32, s8s8->s32); M=1..8 family |   + include/cute/atom/mma_traits_xe_legacy.hpp
  joint_matrix int8 combo on bmg_g21;         | intel/llvm sycl_ext_oneapi_matrix.asciidoc
   use::a row_major, use::b ext_intel_packed, |   + sycl_ext_intel_matrix.asciidoc (VNNI-B example,
   C/D accumulator dynamic; mad order (M,K,N) |     ext_intel_packed = already-packed rule, stride=packed pitch)
  TileShape Shape<_256,_256,_32>; 2D-block    | intel/sycl-tla examples/02_bmg_gemm_mixed_dtype/*
   load atoms XE_2D_U8x32x32_LD_V (VNNI B) /  |   include/cute/atom/copy_traits_xe_legacy.hpp
   _LD_N (A). (no pure-s8 example ships)      |   (CAVEAT: shipped "s8" examples upconvert to bf16 MMA)
  CTA 128x128x64 / warp 64x64x64 / 3 smem     | NVIDIA/cutlass media/docs/cpp/efficient_gemm.md
   stages / int32 accum / fused dequant       |   + examples/13_two_tensor_op_fusion (s8 sm80, Stages=3)
   epilogue (tile hierarchy + pipeline princ) |   + media/docs/cpp/{pipeline,gemm_api_3x}.md
  offline weight PREPACK into mma-ready layout | IST-DASLab/marlin README ; vllm machete
   (dequant straight into tensor-core order);  |   csrc/.../machete/machete_prepacked_layout.cuh
   4-stage async pipe; small-M tall-skinny    |   csrc/.../marlin/{marlin.cuh,marlin.cu,marlin_template.h}
   tile + stream-K + atomic reduce (decode)    |   (repo moved to csrc/libtorch_stable/quantization/)
[PREFILL -- oneDNN library win]
  format_tag::any + cached weights_desc()      | uxlfoundation/oneDNN doc/usage_models/inference.md
   reorder is THE recommended path             |   + dev_guide_matmul.html (less-info = less-perf rule)
  IPEX int8 does exactly this (the recipe)     | intel-extension-for-pytorch xpu-main
                                               |   csrc/gpu/oneDNN/QMatmul.h (L222-335)
  JIT gemm: packed B -> 1D addr, prefetch      | uxlfoundation/oneDNN
   gated on !isPacked(B) (plain B = worse path)|   src/gpu/intel/gemm/jit/generator/strategy.cpp
  hgemm_policy shape dispatch (small/large M)  | IPEX xpu-main csrc/gpu/aten/operators/xetla/
                                               |   kernels/GEMM/hgemm_policy_xehpc.cpp + hgemm_policy.h
  XeTLA int8 in-reg quant->DPAS->s32->dequant; | arXiv:2508.06753v2 (Pushing the Envelope, BMG int8
   reuse BB register 8x; K-splitting small-M   |   233 TOPS roofline; GEMV within 5-10% of roofline)
[DECODE -- dp4a GEMV]
  one sub-group/col, dp4a partials, butterfly  | ggml-org/llama.cpp ggml/src/ggml-sycl/mmvq.cpp
   reduce (mask=8,4,2,1 permute_sub_group_xor),|   + vecdotq.hpp (vec_dot_q8_0_q8_1, ggml_sycl_dp4a)
   lane-0 write, reqd_sub_group_size(16)       |   + common.hpp ; CMakeLists GGML_SYCL_WARP_SIZE=16
  ggml MMQ int8 GEMM (dp4a, single-buffer) =   | ggml/src/ggml-sycl/mmq.cpp (the dp4a baseline our
   the baseline our DPAS prefill must beat     |   DPAS + double-buffer prefill should beat)
  ggml uses dp4a NOT DPAS (BW-bound on Arc;    | ggml-org/llama.cpp discussions/12570
   maintainers deprioritized joint_matrix)     |
  column-contiguous weight reorder (int4 ->    | (our) docs/kernel/06_sycl_int4_gemv_design.md (d)
   int8 analog) for coalesced GEMV loads       |   ; ggml q4_0 reorder PR ggml-org/llama.cpp#12035
```

================================================================================================
## (e) The concrete first experiment for the lead
================================================================================================

**EXPERIMENT 1 (do this first, ~1 day, GPU via `scripts/gpu-run`): measure, then prototype
`format_tag::any` on the W8A8 prefill GEMM.** Highest EV: a library-side win that may close most of the
prefill gap with near-zero risk, gated by a measurement that tells us if a hand kernel is even needed.

Step 1 (free, ~30 min GPU): run the microbench Run-1 + Run-2 (section a.4) -- `ONEDNN_VERBOSE=2` on
`int8_gemm_w8a8` at the three real shapes, m in {1,512,4096}. Record: impl string, whether a WEIGHTS
reorder line repeats per-call, and the prefill XMX util%. This alone answers "is oneDNN leaving XMX on the
table for us?" -- the standing open risk since `01`.

Step 2 (~1 day CPU edit + minutes build): in `int8_gemm_w8a8.h` build the weight md with
`memory::format_tag::any` (not explicit strides); query `matmul_pd.weights_desc()`; add a one-time s8
weight reorder in `XPUInt8ScaledMMLinearKernel.process_weights_after_loading` into that blocked format;
cache it. Copy the structure from IPEX `QMatmul.h` (L222-335). Rebuild via `scripts/44_build_int8_kernel.sh`.

Step 3 (~30 min GPU): re-run Run-2 + a full-model prefill bench (`46_bench_prefill.sh`). Compare XMX util%
and t/s vs the baseline. **Verify correctness first** (gemm err vs `x_q.float() @ w_q.float() * scales`,
the check in `01` 3d) -- a wrong reorder is silent.

DECISION GATE:
- If `format_tag::any` lifts prefill >=1.2x and XMX util is now >60% -> ship it; the hand DPAS GEMM (PP-2)
  is LOW priority. Move to the decode track (TG-0 ggml ceiling bench -> TG-1 reorder + TG-2 dp4a GEMV).
- If XMX util stays <40% even with `format_tag::any` -> oneDNN is genuinely leaving the array idle for our
  shapes -> the hand DPAS GEMM (PP-2, skeleton c.1) is justified; start it.

This sequences the whole plan: cheap measurement -> the one-day library win -> the data to decide if/where
hand kernels pay. Everything downstream (PP-2 hand GEMM, TG-1/TG-2 decode GEMV) is gated by what
Experiment 1 measures.

================================================================================================
## Status / open items
================================================================================================
- [ ] (LEAD, GPU) Experiment 1, steps 1-3 (microbench + format_tag::any prototype). -> JOURNAL.md table.
- [ ] (LEAD, GPU) TG-0: ggml q8_0 GEMV BW ceiling on B70 (06(e) recipe, q4_0->q8_0). -> JOURNAL.md.
- [ ] (CPU) If PP-2 greenlit: host VNNI-pack helper for B + numpy reference to unit-test the c.1 skeleton.
- [ ] (CPU) If TG-2 greenlit: host column-reorder helper (TG-1) + numpy GEMV reference for c.2.
- [ ] Fix the c.1 skeleton's SLM-B staging to true VNNI-packed layout (CAVEAT 2) before any compile.
- [ ] (later) PP-4 tile-shape dispatch; TG-3 FULL graph capture (already tracked in 04 A2).

VERIFIED vs PROPOSED ledger:
- VERIFIED: our explicit-strided weight md (the gap); oneDNN v3.9.1; the int8 DPAS atom shape + VNNI-B
  rule (sycl-tla + spec, 2 sources); the ext_intel_packed = already-packed rule + packed stride (codex
  vs Intel asciidoc); ggml dp4a GEMV reduce idiom + SG=16; IPEX format_tag::any recipe; CUTLASS s8 tile
  numbers; oneDNN format_tag::any guidance.
- PROPOSED: the SIZE of the prefill gain from format_tag::any (1.1x-1.5x -- measure it); the decode 67%->90%
  from reorder+dp4a; that oneDNN m=1 lands on jit:gemm:any for int8 (confirmed for the int4 sibling, infer
  for int8 -- Run-1 verifies); both kernel skeletons (drafted, NOT compiled).
```
