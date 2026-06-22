# 06 - Purpose-built SYCL int4 GEMV for B70 decode (Lever C, design + bench plan)

> **[!] SUPERSEDED (2026-06-22) -- DO NOT BUILD this custom SYCL GEMV; it is futile.** kernel/04 step 4 + microbench
> kernel/19 measured oneDNN's int4 GEMV already at/above llama.cpp's BW efficiency, so a hand-written GEMV has no
> headroom. The decode speedup came from PIECEWISE graph capture (dispatch overhead was the bottleneck, not the GEMV).
> Kept as a design reference only.

Concrete, implementable design for a custom int4 W4A8 GEMV at decode (m=1) on the Intel Arc Pro B70
(Battlemage / Xe2, sub-group size **16**, 608 GB/s peak BW). This is the "Lever C / C2" follow-on to
`docs/kernel/04_decode_optimization.md`: replace the oneDNN `grouped_micro_gemm` GEMV-emulation
(stuck at 52-64% of peak BW) with a hand-written sub-group GEMV that targets near-peak BW.

It also specifies the **ladder step 4** llama.cpp bench (a BW-ceiling reference) to run FIRST.

Scope: this doc is RESEARCH + a DRAFT skeleton. The kernel is NOT compiled/tested. GPU runs are the
lead's job (serialize through `scripts/gpu-run`); this doc is the recipe.

Conventions: VERIFIED = fetched from upstream source (URLs in section f). PROPOSAL = our design choice.

--------------------------------------------------------------------------------------------------
## (a) Bandwidth-bound roofline for our decode shapes
--------------------------------------------------------------------------------------------------

PROPOSAL/derivation. At m=1 a GEMV is pure bandwidth: the dominant traffic is reading the int4 weight
matrix once. Everything else (the int8 activation vector, the f16 group scales, the f16 output) is
small. So:

    t_min = weight_bytes / peak_BW
    weight_bytes ~= K * N * 0.5          (int4 = half a byte/element)
    peak_BW = 608 GB/s (B70)

Per-shape roofline (the m=1 shapes from `w4a8/20_microbench_w4a8_decode.sh`). Added the f16 group
scales [K/128, N]*2B and the int8 act [K]*1B and f16 out [N]*2B to be exact; they are <1% of traffic.

    shape (K x N)        Wbytes(int4)  scales   act    out     total      t_min@608   floor t/s-ish
    -------------------  ------------  -------  -----  ------  ---------   ---------   -------------
    4096  x 11008        22.54 MB      0.70 MB  4 KB   22 KB   23.27 MB    38.3 us
    5120  x 17408        44.56 MB      1.39 MB  5 KB   35 KB   45.99 MB    75.6 us
    17408 x 5120         44.56 MB      1.39 MB  17 KB  10 KB   45.98 MB    75.6 us
    5120  x 5120         13.11 MB      0.41 MB  5 KB   10 KB   13.53 MB    22.3 us

Reading the BW formula straight from the microbench (lines 51-54):

    wbytes = k*n*0.5;  gbps = (wbytes + m*k + m*n*2)/dt/1e9

So the microbench's reported GB/s is exactly "useful weight+act+out bytes / time". At 52-64% of 608
that is ~316-389 GB/s effective. The roofline says the SAME byte count should move at ~580-600 GB/s
(>95% of peak is realistic for a clean GEMV; ggml hits high-90s on similar Q4 kernels). 

    TARGET: lift effective BW from 316-389 GB/s (52-64%) to >=550 GB/s (>=90%).
    That is a 1.4x-1.7x decode-GEMM speedup, purely from removing the GEMM-emulation overhead
    (systolic array ~1/16 utilized at m=1 -> a vectorized-FMA GEMV that is not DPAS-bound).

Why oneDNN leaves it on the table (VERIFIED, from doc 04 + IPEX/oneDNN pointers): `grouped_micro_gemm`
split-K-emulates the GEMV across the XMX/DPAS systolic array, which needs an M tile to fill; at M=1 it
runs ~1/16 lanes and pays scheduling/zp-correction overhead. A GEMV does NOT need DPAS at all -- a
sub-group of 16 doing vectorized int8 FMA saturates the read port.

--------------------------------------------------------------------------------------------------
## (b) Algorithm (one sub-group per output row) + ASCII data-flow
--------------------------------------------------------------------------------------------------

PROPOSAL, synthesized from the three verified kernels (ggml-sycl mmvq/dmmv, AWQ gemv_cuda, sycl-tla
bmg mixed-dtype). Core choices and WHY:

  1. ONE SUB-GROUP (16 lanes) PER OUTPUT ROW n. (VERIFIED idiom: ggml `mul_mat_vec_q` maps one row to
     a sub-group of WARP_SIZE; AWQ maps one warp per output row.) On Xe2 SG=16, so 16 lanes split K.
     Pack SG_PER_WG=8 sub-groups per work-group (=128 work-items) so a work-group emits 8 rows.

  2. SPLIT-K ACROSS THE 16 LANES. Lane L owns the packed words striding by 16 within each group.
     Each int32 packed word = 8 consecutive-k nibbles for column n (our ABI). Lane reads the word,
     unpacks 8 nibbles in registers, dots them against the matching 8 int8 activations.

  3. DEFERRED ZERO-POINT (symmetric int4, zp=8). (VERIFIED: ggml q4_0 `vec_dot_q4_0_q8_1_impl`
     accumulates raw nibble dot then subtracts `8 * sum(act)` once.) We do NOT subtract 8 per element.
     Per group g we keep two int accumulators:
         raw_dot += nibble * xq[k]      (nibble in [0,15])
         act_sum += xq[k]
     centered_dot = raw_dot - 8 * act_sum     (this is sum((nibble-8)*x) = sum(w_true * x))

  4. GROUP-WISE DEQUANT. group_size G=128 along K. When a group's 128 k-elements finish, fold the f16
     group scale: y_acc += ws[g,n] * centered_dot. (raw_dot/act_sum are int32 -> exact, no fp drift
     within a group.) ws indexed [g*N + n].

  5. SUB-GROUP REDUCTION. After the k-loop each lane holds a partial y_acc. Reduce across 16 lanes:
         y_acc = sycl::reduce_over_group(sg, y_acc, sycl::plus<float>());
     (VERIFIED alt idiom: ggml uses a shuffle_xor tree `for(mask=SG/2; mask>0; mask>>=1)
      tmp += permute_sub_group_by_xor(sg, tmp, mask)` -- equivalent, kept as fallback.)

  6. PER-TOKEN ACT SCALE AT THE VERY END. xs is a single f16 (m=1). Lane 0:  y[n] = (f16)(y_acc * xs).
     (Applying it once at the end, not per-element, mirrors ggml folding `d4 * (sumi*ds.x - 8*ds.y)`.)

ASCII data-flow (one work-group = 8 sub-groups = 8 output rows):

    work-group  (local range {SG_PER_WG=8, 16})
    +-----------------------------------------------------------------------+
    | sub-group 0 -> row n0  | sub-group 1 -> row n1 | ... | sub-group 7    |
    |  lanes 0..15           |                       |                      |
    +-----------------------------------------------------------------------+
                |
                v   (one sub-group, row n; 16 lanes split K)
    K dimension, packed as int32 words [K/8] for column n  (current ABI: stride N between words)
    word index:  0    1    2   ...                          (8 nibbles each)
    lane 0 -> words 0,16,32...    lane 1 -> words 1,17...   ...  lane 15 -> 15,31...
      |                                                                |
      | per word: word -> 8 nibbles (vi = (word>>4j)&0xF)  +  8 int8 act
      |           raw_dot += vi*xq ;  act_sum += xq                    |
      v   (per group of 128 k done)                                    v
    centered = raw_dot - 8*act_sum ;  y_acc += ws[g,n]*centered    (float, per lane)
      |                                                                |
      +--------------------------+ reduce_over_group(sg, +) +----------+
                                 v
                       lane 0:  y[n] = f16(y_acc * xs)

K-loop / unpack / dequant / reduce are steps 2-5 above. The int4->int8 unpack is inline in registers
(no SLM round-trip for weights), exactly like ggml's `vi0 = (v>>0)&0x0F0F0F0F; vi1 = (v>>4)&0x0F0F0F0F`.

--------------------------------------------------------------------------------------------------
## (c) DRAFT SYCL kernel skeleton (NOT compiled/tested -- design review only)
--------------------------------------------------------------------------------------------------

Drafted via `codex exec`, then row-index mapping corrected (see CAVEAT after the code). This is a
SKELETON: the TODOs (vectorized dp4a, double-buffered SLM acts, reordered layout) are the perf work.

```cpp
#include <sycl/sycl.hpp>
#include <cstdint>

namespace draft_int4_gemv {

using half = sycl::half;

static constexpr int SG = 16;                 // Xe2 sub-group size
static constexpr int G  = 128;                // group_size along K
static constexpr int NIBBLES_PER_WORD = 8;    // int4 packed 8-per-int32
static constexpr int SG_PER_WG = 8;           // sub-groups (=rows) per work-group

/*
  Current ABI layout (matches w4a8/20_microbench): W_packed[(k/8)*N + n]
    int32 word holds the 8 nibbles for k..k+7 at output column n.  Logical [K/8, N].
  TODO reordered layout (section d) for coalesced sub-group loads:
    W_reorder[n*(K/8) + kw]  -> the 16 lanes of a sub-group read 16 CONTIGUOUS words
    of one column, one cache line, instead of striding by N (the current killer).
*/

struct Int4GemvB70Kernel {
  const int32_t *W_packed;  // [K/8, N], K-major ABI
  const half    *ws;        // [K/G, N] group scales
  const int8_t  *xq;        // [K] per-token int8 activations
  const half    *xs;        // scalar per-token act scale, xs[0]  (m=1)
  half          *y;         // [N] output
  int K, N;

  [[sycl::reqd_sub_group_size(SG)]]
  void operator()(sycl::nd_item<2> item) const {
    const sycl::sub_group sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();         // 0..15

    // CAVEAT-FIX: each of the SG_PER_WG sub-groups owns a distinct row.
    // With local range {SG_PER_WG, 16} and sub_group size 16, dim-1 is fast-varying,
    // so get_local_id(0) IS the sub-group index within the work-group.
    const int n = item.get_group(0) * SG_PER_WG + (int)item.get_local_id(0);
    if (n >= N) return;

    float y_acc = 0.0f;
    const int groups          = K / G;
    const int words_per_group = G / NIBBLES_PER_WORD;   // 16 words per group

    for (int g = 0; g < groups; ++g) {
      int raw_dot = 0;   // sum(nibble * x), nibble unsigned [0,15]
      int act_sum = 0;   // sum(x)         -> deferred symmetric -8 centering
      const int gw0 = g * words_per_group;

      // 16 lanes each take 1 word of this group's 16 words (no stride needed: words==SG)
      for (int gw = lane; gw < words_per_group; gw += SG) {
        const int kw = gw0 + gw;
        const int32_t word = W_packed[kw * N + n];      // strided by N (current ABI; see (d))
        #pragma unroll
        for (int j = 0; j < NIBBLES_PER_WORD; ++j) {
          const int k  = kw * NIBBLES_PER_WORD + j;
          const int vi = (word >> (4 * j)) & 0xF;       // raw nibble [0,15]
          const int xi = (int)xq[k];
          raw_dot += vi * xi;
          act_sum += xi;
        }
      }

      // deferred zero-point:  sum((nibble-8)*x) = raw_dot - 8*act_sum
      const int   centered = raw_dot - 8 * act_sum;
      const float wsc      = (float)ws[g * N + n];
      y_acc += wsc * (float)centered;
    }
    // TODO: K-tail when K % G != 0 (our shapes are all G-aligned, so OK for now).

    y_acc = sycl::reduce_over_group(sg, y_acc, sycl::plus<float>());
    // Fallback shuffle-tree (ggml idiom):
    //   for (int m=SG/2; m>0; m>>=1) y_acc += sycl::shift_group_left(sg, y_acc, m);

    if (lane == 0) y[n] = (half)(y_acc * (float)xs[0]);

    // TODO(perf): vectorize the inner dot to dp4a-style 4-wide int8 (sycl::vec<int8,4>),
    //             unpack two int32 words at once (vi0=(w>>0)&0x0F0F0F0F, vi1=(w>>4)&...).
    // TODO(perf): double-buffer xq into SLM (local_accessor) once per work-group; all 8
    //             rows of the WG reuse the SAME activation vector -> 8x act-read amortization.
    // TODO(perf): switch W_packed -> reordered layout (section d) for coalesced loads.
  }
};

inline void launch_int4_gemv_b70_draft(
    sycl::queue &q, const int32_t *W_packed, const half *ws,
    const int8_t *xq, const half *xs, half *y, int K, int N) {
  const sycl::range<2> local{SG_PER_WG, SG};
  const sycl::range<2> global{
      (size_t)(((N + SG_PER_WG - 1) / SG_PER_WG) * SG_PER_WG), SG};
  q.parallel_for(sycl::nd_range<2>(global, local),
                 Int4GemvB70Kernel{W_packed, ws, xq, xs, y, K, N});
}

} // namespace draft_int4_gemv
```

CAVEAT (correctness, must verify on first compile):
- Row mapping `n = get_group(0)*SG_PER_WG + get_local_id(0)` assumes dim-1 (size 16) is the
  fast-varying lane dim so each dim-0 slice is one sub-group. VERIFIED reasoning via codex; CONFIRM
  against the actual SYCL sub-group layout on icpx 2025.x before trusting results (print
  `sg.get_group_linear_id()` in a unit test).
- `reduce_over_group` over a sub-group is well-defined in SYCL2020; if the toolchain miscompiles it
  for f16-accumulate, use the shuffle-tree fallback (the ggml-proven path).
- The skeleton's inner loop is SCALAR (1 nibble * 1 act at a time). That is correct but slow; the
  dp4a TODO is what reaches the roofline. Land correctness first, then vectorize.

--------------------------------------------------------------------------------------------------
## (d) Weight-layout reorder for coalesced int4 loads
--------------------------------------------------------------------------------------------------

PROPOSAL (the single highest-leverage perf item, echoing doc 04 lever B4 + ggml PR #12035 + sycl-tla
`XE_2D_U4x32x16_LD_T`).

THE PROBLEM with the current ABI `W_packed[(k/8)*N + n]`: within a sub-group the 16 lanes read words
for the SAME column n but DIFFERENT k -> addresses differ by `N * sizeof(int32)` (e.g. 11008*4 =
44 KB apart). That is 16 separate cache lines per word-row -> uncoalesced, the likely cause of the
52-64% ceiling.

REORDERED LAYOUT (offline, once, in process_weights):

    W_reorder[n * (K/8) + kw]      // column-major over the packed-word axis

Now the 16 lanes of sub-group(row n) read words kw=0..15 of ONE column = 16 contiguous int32 = one
64-byte cache line, fully coalesced. (VERIFIED analog: ggml's q4_0 "reorder" path -- PR #12035 --
splits the block into a contiguous nibble array + a separate contiguous scale array for exactly this
coalescing reason; sycl-tla loads u4 weights ColumnMajor via `XE_2D_U4x32x16_LD_T`, a 2D sub-byte
block copy.)

Also reorder the scales to match: `ws_reorder[n * (K/G) + g]` so the per-group scale read is contiguous
per row too.

Reorder is a pure index permutation of existing data (no requant), done CPU-side at load. Cost: one
pass over the weight tensor at model load; zero per-decode cost. Provide a tiny host helper +
a `register_fake`-friendly shape so it slots into our existing W4A8 process_weights.

Sketch (host, ASCII):

    for n in 0..N:
      for kw in 0..K/8:
        W_reorder[n*(K/8) + kw] = W_packed[kw*N + n]   // transpose the packed-word grid

Kernel change: `W_packed[kw*N + n]` -> `W_reorder[n*(K/8) + kw]`. (One line.)

--------------------------------------------------------------------------------------------------
## (e) PLAN: benchmark llama.cpp ggml-sycl int4 GEMV on the B70 (ladder step 4)
--------------------------------------------------------------------------------------------------

GOAL (from doc 04, C1): a real int4 GEMV BW ceiling on THIS card, BEFORE we write our own -- tells us
how much BW is actually on the table vs our 52-64%. ggml-sycl `mul_mat_vec_q` (q4_0) is the closest
existing reference (one sub-group per row, dp4a, sub-group reduce; WARP_SIZE=16 for Intel -- VERIFIED).

NOTE: the actual GPU run is the LEAD's job -- serialize through `scripts/gpu-run` (CLAUDE.md). This is
the recipe; do not run it from an agent. Use a q4_0 GGUF (ggml's int4 path that most matches W4A8).

Recipe (run inside an oneAPI/icpx env or the SYCL container; B70 = level_zero:0):

    # 0. one-time: confirm the card is seen and SG=16
    source /opt/intel/oneapi/setvars.sh
    sycl-ls                       # expect [level_zero:gpu] Intel Arc Pro B70 (Battlemage)

    # 1. build llama.cpp SYCL backend (F16 accumulate, AOT for bmg optional)
    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
    cmake -B build -G Ninja \
      -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release
      # optional AOT for Battlemage: -DGGML_SYCL_DEVICE_ARCH=bmg  (else JIT at first run)
    cmake --build build -j

    # 2. get a q4_0 model whose K,N match our decode shapes (a 14B q4_0 GGUF; q4_0 == ggml int4).
    #    (Qwen3-14B q4_0 lines up with our 5120/17408 shapes; any 14B q4_0 gives the BW ceiling.)

    # 3. DECODE bench: -n (token-gen) is the m=1 GEMV regime; -p 1 minimizes prefill.
    #    GATE THE GPU RUN: scripts/gpu-run '<the llama-bench line>'
    ONEAPI_DEVICE_SELECTOR=level_zero:0 \
      ./build/bin/llama-bench -m <qwen3-14b-q4_0.gguf> -ngl 99 -p 1 -n 128 -r 5

    # 4. confirm the kernel actually used (the q4_0 GEMV, not a fallback):
    ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DEBUG=1 \
      ./build/bin/llama-bench -m <...q4_0.gguf> -ngl 99 -p 1 -n 32 -r 1 2>&1 | grep -i "mul_mat_vec\|reorder\|q4_0"

    # 5. (optional) raw GEMV BW, no full model: llama.cpp test-backend-ops perf for MUL_MAT at m=1
    ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops perf -o MUL_MAT

INTERPRETATION:
  - llama-bench reports tg (tokens/sec). Convert to effective weight-BW for a comparable layer:
    BW_eff ~= (sum of int4 weight bytes read per token) * tg.  For a 14B q4_0 that is ~7.5 GB/token of
    weights -> BW_eff(GB/s) ~= 7.5 * tg. If ggml hits ~85-95% of 608 (~520-580 GB/s), that is our
    ceiling and proves the 52-64% oneDNN number is pure kernel overhead, justifying C2.
  - If ggml ALSO sits near 52-64%, the ceiling is lower than hoped (driver/HW limit) -> re-scope C2.
  - Also try the reorder path (PR #12035, may be a build flag / model already reordered) -- it is the
    direct analog of our section (d) and its delta quantifies the coalescing win.

DELIVERABLE of step 4: a single line in JOURNAL.md -- "ggml q4_0 decode = X t/s = Y% of 608 GB/s on
B70" -- the number our custom GEMV must beat (and the oneDNN 52-64% must be lifted toward).

--------------------------------------------------------------------------------------------------
## (f) Verified source-pointer table (URLs fetched 2026-06-20)
--------------------------------------------------------------------------------------------------

VERIFIED = source file fetched and the cited idiom read directly.

  what we took                       | file (VERIFIED URL)
  -----------------------------------+--------------------------------------------------------------
  one sub-group per row; sub-group   | ggml-sycl mmvq.cpp  `mul_mat_vec_q`, reduction
   shuffle_xor reduce; row mapping   |   for(mask=WARP_SIZE/2..) permute_sub_group_by_xor
                                     | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-sycl/mmvq.cpp
  q4_0 nibble unpack (&0x0F0F0F0F,   | ggml-sycl vecdotq.hpp  `vec_dot_q4_0_q8_1_impl`
   >>4); dp4a int8 dot; DEFERRED -8  |   sumi=dp4a(vi0,u0); ... d4*(sumi*ds.x - 8*vdr/QI4_0*ds.y)
   centering via act partial-sum     | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-sycl/vecdotq.hpp
  dequantize_q4_0 ( (n-8)*d ) and    | ggml-sycl dequantize.hpp  `dequantize_q4_0[_reorder]`
   the REORDER layout (sep nibble +  |   v.x=vui&0xF; v.x=(v.x-8)*d ; reorder: separate qs[] + d[]
   sep scale arrays)                 | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-sycl/dequantize.hpp
  nd_range {1,MMV_Y,WARP_SIZE},      | ggml-sycl dmmv.cpp  `dequantize_mul_mat_vec`, reqd_sub_group_size
   registers-only accumulate         | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-sycl/dmmv.cpp
  WARP_SIZE = 16 for Intel (Xe2)     | ggml-sycl CMakeLists.txt  add_compile_definitions(GGML_SYCL_WARP_SIZE=16)
  GGML_SYCL_MMV_Y=1, DMMV_X=32       | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-sycl/CMakeLists.txt
                                     | (defaults) ggml/src/ggml-sycl/common.hpp
  q4_0 reorder coalescing path       | ggml-org/llama.cpp PR #12035 (reorder)
                                     | https://github.com/ggml-org/llama.cpp/pull/12035
  one warp/output row, dequant+FMA,  | AWQ gemv_cuda.cu  `gemv_kernel_g64/g128`, warp_reduce_sum
   group-wise scale index, acts in   |   w_fp=(packed&0xF); deq=scale*(w_fp - zero); __shfl_down_sync
   registers (algorithm template)    | https://raw.githubusercontent.com/mit-han-lab/llm-awq/main/awq/kernels/csrc/quantization/gemv_cuda.cu
  Xe2 int8 DPAS atom + u4 ColMajor   | sycl-tla 02_bmg_gemm_f16_u4_s8.cpp
   2D sub-byte load (reorder analog);|   MMA XE_8x16x32_S32S8S8S32_TT ; GmemCopyB XE_2D_U4x32x16_LD_T
   TileShape<32,64,32>; group dequant|   ElementInputB=uint4_t, ElementScale=half_t, scale_k=ceil(k,g)
                                     | https://raw.githubusercontent.com/intel/sycl-tla/main/examples/02_bmg_gemm_mixed_dtype/02_bmg_gemm_f16_u4_s8.cpp
  B70 llama.cpp SYCL build/bench     | -DGGML_SYCL=ON -DGGML_SYCL_F16=ON ; ONEAPI_DEVICE_SELECTOR=level_zero:0
   flags (icpx 2025.3.x)             | https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/methodology.md
  Battlemage Q4 perf context (issue) | https://github.com/ggml-org/llama.cpp/issues/21517

OUR ABI / shapes (in-repo, VERIFIED):
  - W4A8 call signature + int4-packed layout + BW formula:
      /home/hotschmoe/github/b70_ai_things/w4a8/20_microbench_w4a8_decode.sh  (lines 31-54)
  - oneDNN W8A8 header style (scale masks, no-zp symmetric path we mirror):
      /home/hotschmoe/github/b70_ai_things/contrib/vllm_int8_xpu/int8_gemm_w8a8.h
  - Lever C context + verified pointers we extended:
      /home/hotschmoe/github/b70_ai_things/docs/kernel/04_decode_optimization.md (LEVER C, lines 115-146)

--------------------------------------------------------------------------------------------------
## Status / next actions
--------------------------------------------------------------------------------------------------
- [ ] (LEAD, GPU via scripts/gpu-run) Run section (e) -- the ggml q4_0 BW ceiling. Log to JOURNAL.md.
- [ ] (CPU) Stand up the host reorder helper (section d) + a numpy reference for the GEMV to unit-test
      the skeleton's correctness (deferred-zp + group dequant) before any GPU compile.
- [ ] (GPU, later) Compile the (c) skeleton, confirm the row-mapping CAVEAT, land correctness, THEN
      do the dp4a + SLM + reorder TODOs and re-microbench vs the 52-64% baseline -> target >=90%.
