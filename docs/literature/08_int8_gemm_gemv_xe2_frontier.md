# 08 - INT8 GEMM / GEMV on Xe2/Battlemage: frontier techniques and microbench plan

Date: 2026-06-21. Scope: actionable synthesis for the B70 (Intel Arc Pro B70, Xe2/Battlemage,
BMG-G31 die, 32 GB GDDR6, 608 GB/s, 367 INT8 TOPS, sub-group 16, NO native FP8). Covers:
  (a) exact GEMM/GEMV shape tables per model (Qwen3-14B, Qwen3.6-27B, Qwen3.6-35B-A3B)
  (b) prioritized optimization techniques, expected payoff, prefill vs decode applicability
  (c) concrete 100-GEMM + 100-GEMV microbench sweep spec
  (d) citations

Conventions: VERIFIED = read from primary source; PROPOSED = reasoned estimate, not measured.
ASCII only. -> not arrows, ... not ellipsis.

================================================================================
(a) GEMM/GEMV shape tables per model
================================================================================

--------------------------------------------------------------------------------
A.0 Hardware context -- what constrains the shapes
--------------------------------------------------------------------------------

B70 = BMG-G31 (32 Xe2 cores, B580 die x scale).
  INT8 peak:  367 INT8 TOPS (B70 spec; B580 = 233 INT8 TOPS)
  BW peak:    608 GB/s (32 GB GDDR6)
  XMX atom:   s8 x s8 -> s32, tile M<=8 / N=16 / K=32, VNNI B (ext_intel_packed)
              (verified: sycl ext_oneapi_matrix spec + sycl-tla bmg examples + doc 10)
  Sub-group:  16 (WARP_SIZE=16; ggml-sycl CMakeLists; VERIFIED doc 06)
  SLM:        ~64 KB per Xe core (Xe2 spec)
  Int8 ceils: decode BW ceiling at M=1 = 608 GB/s / weight_bytes
              prefill compute ceiling = 367 TOPS / (2*M*K*N flops)

MEASURED on B70 (from FINDINGS.md + docs/kernel/14):
  W8A8 prefill XMX util = 67-80% of 367 TOPS   (oneDNN jit:gemm:any)
  W8A8 decode  BW util  = ~61% avg (50% wide-N, 93% down_proj)
  W4A8 decode  BW util  = ~73%   (int4 decompress GEMV)
  W4A16 decode BW util  = ~89%   (best decode, no act-quant tax)
  ggml q4_0 (int4) GEMV BW = ~67% of 608 GB/s (llama.cpp reference)
  -> int8 GEMV target: >=90% of 608 GB/s is achievable (ggml q8_0 = 21-24%, a BAD baseline;
     but the cause is known: DMMV 2-val/thread vs MMVQ 8x stride -- see issue #21517)

--------------------------------------------------------------------------------
A.1 Qwen3-14B (dense, qwen3 arch) -- VERIFIED config
--------------------------------------------------------------------------------

Source: Qwen/Qwen3-14B config.json (HF, fetched 2026-06-21).
  hidden_size (H)         = 5120
  intermediate_size (I)   = 17408
  num_attention_heads (Q) = 40
  num_key_value_heads     = 8
  head_dim                = 128
  vocab_size              = 151936
  num_hidden_layers       = 40
  14.8B params total / 13.2B non-embedding

ATTENTION SHAPES (K x N, NT weight -- one set per layer):
  Name            K       N       weight bytes (int8)  weight bytes (int4)  notes
  q_proj          5120    5120    25.6 MB              12.8 MB              Q heads
  k_proj          5120    1024    5.1 MB               2.6 MB               8 KV heads x head_dim=128
  v_proj          5120    1024    5.1 MB               2.6 MB
  qkv fused       5120    7168    36.9 MB              18.4 MB              if fused (q+k+v)
  o_proj          5120    5120    25.6 MB              12.8 MB

MLP SHAPES (K x N, NT weight):
  gate_proj       5120    17408   89.2 MB              44.6 MB              MLP wide-N
  up_proj         5120    17408   89.2 MB              44.6 MB              MLP wide-N
  gate_up fused   5120    34816   178.5 MB             89.2 MB              if fused
  down_proj       17408   5120    89.2 MB              44.6 MB              MLP wide-K

VOCAB / HEAD:
  lm_head         5120    151936  ~780 MB              ~390 MB              one-off (no per-token int8)

Distinct GEMM shapes (7, ignoring fused variants):
  ID  K      N      regime        arithmetic_intensity (fp32 acc, M tokens)
  A1  5120   5120   attn QKV+O    AI = 2MKN / (M*(K+N) + KN) bytes
  A2  5120   1024   attn K,V      low-N outlier
  A3  5120   17408  MLP gate/up   wide-N, compute-bound at M>=256
  A4  17408  5120   MLP down      wide-K, compute-bound at M>=128

At M=1 (decode): all are MEMORY-BOUND (AI < 1 ops/byte for int8 weights).
At M=512 (prefill): A3 AI ~ 2*512*5120*17408 / (512*(5120+17408) + 5120*17408) ~ 384 ops/byte
  -> well into compute-bound territory for int8 (crossover ~= 608/367e12 * 1e9 ~ 1.7 B/TOPS ~ 1.7 ops/byte)
  So ALL MLP shapes at M>=4 are compute-bound for int8 (extremely compute-intensive architecture).

--------------------------------------------------------------------------------
A.2 Qwen3.6-27B (dense VLM+GDN+MTP, qwen3_5 arch) -- VERIFIED config
--------------------------------------------------------------------------------

Source: Qwen/Qwen3.6-27B HF model card (fetched 2026-06-21).
  hidden_size (H)         = 5120
  intermediate_size (I)   = 17408  (shared with 14B on MLP; attention differs)
  num_attention_heads (Q) = 24  (Gated Attention layers ONLY; 16 such layers)
  num_key_value_heads     = 4
  head_dim                = 256  (Gated Attention) / 128 (Gated DeltaNet)
  vocab_size              = 248320
  num_hidden_layers       = 64
  Layer pattern: 16 x (3 x GatedDeltaNet + 1 x Gated Attention), each with FFN

Layer counts:
  Gated Attention layers:    16  (layers 3, 7, 11, ..., 63 -- every 4th)
  Gated DeltaNet layers:     48  (the other 48)
  FFN layers:                64  (all layers have FFN)

Gated DeltaNet linear_attn shapes (48 layers, K x N):
  in_proj_qkv  5120  x (48*128 + 16*128) = 5120 x 8192   (64 V-heads + 16 QK-heads x 128)
               PROPOSED: actual projection dims need direct config read; see note below
  in_proj_z    5120  x 5120              (pass-through gate)
  in_proj_a/b  5120  x 512 (PROPOSED)    (SSM state projections, small)
  out_proj     5120  x 5120

NOTE: the 27B uses the qwen3_5 arch which is NOT standard qwen3. The DeltaNet-specific projection
shapes (in_proj_qkv, in_proj_z, in_proj_a/b) require reading the actual model config or
loading the weights. The values above are PROPOSED from the architecture description; the
MLP and standard-attention shapes below are VERIFIED.

Standard Gated Attention shapes (16 layers, K x N):
  q_proj       5120   6144   (24 heads x 256 head_dim)
  k_proj       5120   1024   (4 KV heads x 256 head_dim)
  v_proj       5120   1024
  o_proj       6144   5120   (note: K=6144 = 24 heads x 256)

MLP shapes (64 layers, same for all layer types):
  gate_proj    5120   17408
  up_proj      5120   17408
  down_proj    17408  5120

Distinct GEMM shapes (10 categories, excluding DeltaNet SSM projections):
  B1  5120   6144   Gated Attn Q (16/64 layers)
  B2  5120   1024   Gated Attn K,V (16/64 layers)
  B3  6144   5120   Gated Attn O (16/64 layers)
  B4  5120   5120   DeltaNet in_proj_z + out_proj (48/64 layers; PROPOSED)
  B5  5120   8192   DeltaNet in_proj_qkv (48/64 layers; PROPOSED)
  B6  5120   512    DeltaNet in_proj_a/b (48/64 layers; PROPOSED -- SMALL-N outlier)
  B7  5120   17408  MLP gate/up (64 layers)
  B8  17408  5120   MLP down (64 layers)

NOTE: The DeltaNet `linear_attn` layers are NOT quantized in the current int4 path (ignored in the
quant ignore list -- CLAUDE.md / QUANTS_TODO.md). They WILL be skipped for W8A8/W4A8 too.
Effective quantized shapes for serving: B1-B3 (16 layers) + B7-B8 (64 layers) = ~80 GEMMs/token.

--------------------------------------------------------------------------------
A.3 Qwen3.6-35B-A3B MoE (qwen3_5_moe arch) -- VERIFIED config
--------------------------------------------------------------------------------

Source: Qwen/Qwen3.6-35B-A3B HF model card (fetched 2026-06-21).
  hidden_size (H)         = 2048
  num_attention_heads (Q) = 16  (Gated Attention layers)
  num_key_value_heads     = 2
  head_dim                = 256 (Gated Attention) / 128 (Gated DeltaNet)
  vocab_size              = 248320
  num_hidden_layers       = 40
  MoE: 256 experts total, 9 activated per token (8 routed + 1 shared)
  moe_intermediate_size   = 512  (per expert)
  Layer pattern: 10 x (3 x GatedDeltaNet->MoE + 1 x GatedAttn->MoE)

Layer counts:
  Gated Attention layers:   10  (every 4th of 40)
  Gated DeltaNet layers:    30  (the other 30)
  MoE FFN layers:           40  (all layers use MoE FFN)

MoE Expert shapes (256 experts per layer, K x N, per expert):
  expert gate_proj  2048  x  512   K=H=2048, N=I=512 per expert
  expert up_proj    2048  x  512
  expert down_proj   512  x 2048

Grouped GEMM shapes (one call over all routed experts, M_total tokens):
  PREFILL: expert gate/up   M_total x K=2048 x N=512  (M_total = tokens_routed_to_expert)
           expert down       M_total x K=512  x N=2048
  DECODE:  expert gate/up   M_e=1..few x K=2048 x N=512  (at most 9 active experts at M=1)

Standard Attention shapes (10 layers, K x N):
  q_proj    2048  x 4096  (16 heads x 256 head_dim)
  k_proj    2048  x  512  (2 KV heads x 256 head_dim)
  v_proj    2048  x  512
  o_proj    4096  x 2048  (K=16*256=4096)

DeltaNet shapes (30 layers; PROPOSED, same caveat as 27B):
  in_proj   2048  x  ~4096 (PROPOSED)
  out_proj  2048  x  2048

Distinct MoE GEMM shapes (critical for the int8 MoE kernel):
  C1  2048  x  512   expert gate/up (per-expert GEMV at decode; grouped GEMM at prefill)
  C2   512  x 2048   expert down    (reverse)
  C3  2048  x 4096   Attn Q  (10 layers)
  C4  2048  x  512   Attn K,V  (10 layers)
  C5  4096  x 2048   Attn O  (10 layers)
  C6  2048  x 2048   DeltaNet (PROPOSED)

KEY for MoE decode: at M=1, 9 active experts -> 9 calls to expert gate/up (C1: K=2048, N=512)
and 9 calls to expert down (C2: K=512, N=2048). These are TINY GEMVs -- the dominant BW cost.

ROOFLINE at DECODE for 35B-A3B (M=1, int8 weights):
  C1 (expert gate/up per expert):  weight bytes = 2048*512 = 1.0 MB (int8) -> t_min = 1.7 us
  C2 (expert down per expert):     weight bytes = 512*2048 = 1.0 MB (int8) -> t_min = 1.7 us
  9 active experts -> total MoE BW = 18 MB/token -> t_min = 29.6 us -> ~34k t/s ceiling if BW-bound
  (compare: total model weight ~35B*0.5bytes=17.5 GB; all experts would be 35 GB/token -> infeasible)
  Point: the MoE activates ~3B effective params/token, so weight BW is 3B bytes/token -> ~200 t/s ceiling.
  Current W4A16 decode = 56.8 t/s (int4, so 1.5B weight bytes/token @ 608 GB/s ~ 108 t/s ceiling).
  -> INT8 MoE decode is memory-bound and HARDER than prefill (the reverse of dense models).

================================================================================
(b) Prioritized optimization techniques
================================================================================

Priority order: (1) no-cost / library-level; (2) reorder/layout; (3) custom kernel;
                (4) architectural (graph capture, toolchain).
Payoff is MEASURED or PROPOSED (marked). Applies to: P=prefill, D=decode, B=both.

--------------------------------------------------------------------------------
P0 -- Format_tag::any + cached weight reorder [VERIFIED, PROPOSED payoff]
--------------------------------------------------------------------------------
Regime: P (prefill, compute-bound)
Status: IMPLEMENTED + MEASURED PERF-NEUTRAL (docs/kernel/14 "PP-1").
Verdict: oneDNN v3.9.1's jit:gemm:any already selects an optimized path regardless of
  the user-specified stride. The "explicit strides" gap (doc 10) does NOT produce a
  measurable loss on our version. The library already handles the VNNI weight layout
  internally. This lever is DONE and has no further value.
Lesson: measure before assuming the library gap costs throughput.

--------------------------------------------------------------------------------
P1 -- Graph capture (PIECEWISE) [MEASURED, dominant]
--------------------------------------------------------------------------------
Regime: B (both)
Payoff: MEASURED -- W4A8 +187%, W4A16 +95%, 27B +293%, 35B +617% decode.
        W8A8 decode only +13% (already-fused quant; less dispatch-bound).
Status: DEPLOYED. Image :int8g, VLLM_XPU_ENABLE_XPU_GRAPH=1, cudagraph_mode=PIECEWISE.
Residual: FULL capture (attention included) is still blocked. Needs oneAPI DPC++ 2026.0
  (work_group_scratch_memory in graph nodes) + TRITON_ATTN wired on XPU.
Action: unlock FULL capture next -- see P2.

--------------------------------------------------------------------------------
P2 -- FULL graph capture via TRITON_ATTN [PROPOSED high, untested]
--------------------------------------------------------------------------------
Regime: B
Payoff: PROPOSED ~1.2-1.5x additional decode over PIECEWISE (captures attention dispatch);
  flips MTP/spec-decode from -19..37% to net-positive (the VERIFY pays under full capture).
Path: --attention-backend TRITON_ATTN (wired in vLLM per PR #34482); int8 tl.dot on BMG
  is plausible but the "INT8 hangs" on BMG is UNVERIFIED as fixed (doc 05 sec 4).
Action: smoke test TRITON_ATTN + confirm tl.dot int8 does not hang; if clean, enable FULL.

--------------------------------------------------------------------------------
P3 -- Fused activation quantization (rmsnorm+quant, silu+quant) [PROPOSED ~10 us/token]
--------------------------------------------------------------------------------
Regime: D (dispatch + BW saving; less relevant at prefill)
Payoff: PROPOSED 10.9 us/token BW saving + 120 fewer dispatches/token (doc 11 accounting).
  Under PIECEWISE capture the dispatch saving is already largely captured; the BW saving
  (6.3 MiB/token eliminated round-trip) is the residual.
Status: fused rmsnorm+int8 kernel EXISTS unwired in vllm-xpu-kernels (VERIFIED, doc 11).
  silu_and_mul_quant_int8 kernel needs WRITING (draft in doc 11 sec c.2).
Priority: LOW under capture (dispatch already fused by graph). HIGH for the unfused eager path.
Action: wire L1 (rmsnorm), write+wire L2 (silu+int8) as a follow-on to confirm BW gain
  under capture. The rmsnorm wiring is a Python model-patch, ~1 day. Do after P2.

--------------------------------------------------------------------------------
P4 -- Column-contiguous weight reorder for coalesced int8 GEMV [PROPOSED 1.3-1.4x decode]
--------------------------------------------------------------------------------
Regime: D (decode, M=1, BW-bound)
Payoff: PROPOSED -- move W8A8 decode from 61% to ~90% of 608 GB/s.
  Root cause: plain row-major int8 weight W[k*N+n] forces the 16 lanes of a sub-group
  computing column n to read addresses N bytes apart -> uncoalesced (one cache line per lane).
  Reorder to W_col[n*K+k]: all 16 lanes read 16 consecutive bytes for the same column ->
  one 64-byte cache line per step. This is the int8 analog of the int4 reorder (doc 06 sec d).
Status: PROPOSED -- design is clear (doc 10 TG-1); no kernel written. Pairs with P5.
Evidence: ggml q4_0 MMVQ reorder (PR #12035) lifted int4 BW substantially; q8_0 without reorder
  = 21-24% BW (issue #21517, VERIFIED); the gap is exactly the lack of a reorder path.
Action: implement host-side reorder helper + call it in process_weights_after_loading.

--------------------------------------------------------------------------------
P5 -- Hand dp4a GEMV kernel (one sub-group per column, deferred dequant) [PROPOSED 1.3x decode]
--------------------------------------------------------------------------------
Regime: D (decode, M=1)
Payoff: PROPOSED ~1.3-1.4x decode on W8A8 shapes (14B/27B) once the reorder (P4) feeds it.
  oneDNN GEMV at M=1 lands on jit:gemm:any with ~1/16 XMX utilization + scheduling overhead.
  A dp4a GEMV (no DPAS -- BW-bound at M=1 -> dp4a saturates the read port with less overhead)
  mirrors ggml-sycl mul_mat_vec idiom: one sub-group/column, dp4a 4-wide int8 dot,
  butterfly sub-group reduce, lane-0 write, reqd_sub_group_size(16).
Status: DRAFT skeleton in doc 10 sec c.2 -- NOT compiled. Needs P4 first.
Sequencing: P4 data prep -> P5 kernel -> measure vs oneDNN jit:gemm:any baseline.
CAVEAT: measure the gap first (ggml q8_0 BW bench from doc 10 TG-0) to confirm P4+P5 is worth
  the effort. Our W8A8 decode = 26 t/s (vs W4A16 54 t/s); even at 90% BW = ~40 t/s, W4A16
  still leads decode. The ROI depends on whether W8A8 decode matters (it does at high batch,
  where W8A8's smaller weight bytes let more concurrent sequences fit).

--------------------------------------------------------------------------------
P6 -- INT8 MoE fused grouped GEMM (sycl-tla s8s8s32 path) [PROPOSED 1.4-2x prefill]
--------------------------------------------------------------------------------
Regime: P (prefill/batched, compute-bound at large M)
Payoff: MEASURED in isolation (doc 18): int8 GEMM 1.43x at expert shape N=512/M=4096;
  1.78-2.01x at larger shapes. Prefill/throughput win; NOT a decode win (memory-bound).
Status: no upstream XPU int8 MoE kernel exists (VERIFIED, doc 18 sec 1). Plan:
  Track A (bootstrap loop over our dense op, days) -> Track B (sycl-tla s8s8s32 grouped
  GEMM with masked layout, weeks). See docs/kernel/18 for the full build plan.
Key insight: quant the activation ONCE in the gather/permute, reuse across all top-8 experts.
  Without fusion (per-expert quant) int8 LOSES. With fusion int8 WINS 1.4-2x. The fusion
  is the deciding implementation detail.

--------------------------------------------------------------------------------
P7 -- Split-K / stream-K for small-M prefill (M<64, compute-not-quite-saturated)
--------------------------------------------------------------------------------
Regime: P (intermediate M, e.g. M=16..64 where one tile is under-utilized)
Payoff: PROPOSED ~1.1-1.3x for small-M prefill where the grid is small.
  For M=1 GEMV: split-K means each sub-group handles a K-slice; sub-group reduces across slices.
  For M=16..64 GEMM: stream-K distributes tiles evenly across EUs to avoid tail-effect waste.
  Intel's sycl-tla PersistentTileSchedulerXeGroup (used in 35B MoE examples, doc 18 sec 4)
  already implements the persistent stream-K idea for MoE.
Status: no experimentation yet. Low priority vs P4/P5/P6. Worth a note in the bench plan.
Evidence: arXiv 2402.00025 (W4A16 split-K Triton) measured +1.4x on NVIDIA for decode-sized M.
  Intel's grouped-GEMM issue #6389 shows Triton grouped GEMM tuning on XPU is active.

--------------------------------------------------------------------------------
P8 -- Activation reordering (ABI: N-contiguous weight layout for prefill DPAS)
--------------------------------------------------------------------------------
Regime: P (large-M prefill, XMX)
Payoff: PROPOSED 0-1.2x. oneDNN already handles this internally via jit:gemm:any + VNNI-pack.
  For a hand DPAS GEMM (PP-2 in doc 10), the offline VNNI B reorder is load-bearing: the DPAS
  B tile must be layout::ext_intel_packed (VERIFIED: sycl-tla + Intel matrix spec).
  At M>=256 our prefill XMX util is already 67-80% -> headroom exists but is modest.
Status: relevant only if PP-2 hand DPAS GEMM is implemented (doc 10 sec c.1). Low priority now.

--------------------------------------------------------------------------------
P9 -- Per-shape tile dispatch (M-adaptive tile selection) [PROPOSED maintenance]
--------------------------------------------------------------------------------
Regime: P (small M like 64-128 vs large M like 2048+)
Payoff: PROPOSED ~1.1-1.2x at M=64-128 transitions. IPEX dispatches a tile policy table
  per (M,N,K) (hgemm_policy_xehpc.cpp, VERIFIED reference in doc 10 sec b PP-4).
  WG tile: ~128x128 for large M; ~64x64 or fewer for small M to maintain occupancy.
Status: only relevant after a hand DPAS GEMM (PP-2) exists. Not on the critical path.

--------------------------------------------------------------------------------
SUMMARY TABLE
--------------------------------------------------------------------------------

 ID  Technique                              Regime  Payoff         Effort  Status
 P0  format_tag::any reorder               P       perf-neutral   done    deployed+measured
 P1  PIECEWISE XPU graph capture           B       2-7x decode    done    deployed+measured
 P2  FULL graph (TRITON_ATTN)              B       +20-50%        days    untested
 P3  Fused rmsnorm/silu+int8-quant         D       ~10 us/token   1-3 d   wiring to do
 P4  Column-reorder int8 weights (GEMV)    D       ~1.3-1.5x      2 d     PROPOSED
 P5  Hand dp4a int8 GEMV (M=1)             D       ~1.3-1.4x      1-2 wk  DRAFT
 P6  INT8 MoE fused grouped GEMM           P       1.4-2.0x       2-8 wk  PROPOSED
 P7  Split-K / stream-K (small M)          P       ~1.1-1.3x      1-2 wk  not started
 P8  VNNI B offline reorder (hand GEMM)    P       up to 1.2x     weeks   not started
 P9  Per-shape tile dispatch               P       ~1.1-1.2x      days    not started

Recommended execution order: P2 -> P3 -> P4 -> P5 -> P6 -> P7 (if ROI).
P8/P9 only after a hand DPAS GEMM (PP-2, doc 10) proves necessary (currently not needed
since oneDNN already achieves 67-80% XMX at prefill).

================================================================================
(c) 100-GEMM + 100-GEMV microbench sweep spec
================================================================================

All GPU runs via `scripts/gpu-run`. Script location to create:
  scripts/70_int8_gemm_gemv_sweep.sh  (benchmark driver)

Shape table and result CSV format defined below. Runner calls:
  torch.ops._xpu_C.int8_gemm_w8a8(x_q, x_s, None, w_q, w_s, None, None, dtype)
for each (M, K, N) with M=1 for GEMV and M in [64..4096] for GEMM.
Measure: wall_us (median of 200 timed iters after 50 warmup), then compute:
  flops = 2 * M * K * N
  tops  = flops / wall_us / 1e6
  wbytes = K * N  (int8, 1 byte/element)
  bw_gb  = (wbytes + M*K + M*N*2) / wall_us / 1e3   (in/out; *2 for f16 output)
  xmx_pct = tops / 367.0 * 100
  bw_pct  = bw_gb / 608.0 * 100

--------------------------------------------------------------------------------
C.1 GEMM sweep: 100 prefill configurations
--------------------------------------------------------------------------------

Axes to sweep:
  Shapes: all distinct (K,N) from models A1-A4, B1-B8, C1-C6 above (18 shapes)
  M values: {64, 128, 256, 512, 1024, 2048} = 6 levels per shape
  Total: 18 shapes x 6 M-levels = 108 -> pick 100 by skipping DeltaNet PROPOSED shapes

EXACT shape list (100 GEMM configs = 17 shapes x ~6 M + padding):

Shape group 1: Qwen3-14B attention + MLP (verified shapes A1-A4)
  K=5120  N=5120  M={64,128,256,512,1024,2048,4096}   -> 7 entries  [attn Q/O + dense]
  K=5120  N=1024  M={64,128,256,512,1024,2048}         -> 6 entries  [attn K/V]
  K=5120  N=17408 M={64,128,256,512,1024,2048,4096}   -> 7 entries  [MLP gate/up]
  K=17408 N=5120  M={64,128,256,512,1024,2048,4096}   -> 7 entries  [MLP down]

Shape group 2: Qwen3.6-27B unique shapes (B1-B3, verified)
  K=5120  N=6144  M={64,256,512,1024,2048}             -> 5 entries  [attn Q 27B]
  K=5120  N=1024  M={64,256,512,1024,2048}             -> skip (same K/N as A2, covered)
  K=6144  N=5120  M={64,256,512,1024,2048}             -> 5 entries  [attn O 27B]

Shape group 3: Qwen3.6-35B-A3B MoE shapes (C1-C5, verified)
  K=2048  N=512   M={64,128,256,512,1024,2048}         -> 6 entries  [expert gate/up]
  K=512   N=2048  M={64,128,256,512,1024,2048}         -> 6 entries  [expert down]
  K=2048  N=4096  M={64,256,512,1024,2048}             -> 5 entries  [attn Q 35B]
  K=2048  N=512   M={64,128,256,512,1024}              -> skip (same as expert, covered)
  K=4096  N=2048  M={64,256,512,1024,2048}             -> 5 entries  [attn O 35B]
  K=2048  N=2048  M={64,256,512,1024,2048}             -> 5 entries  [dense square 35B]

Shape group 4: square/reference shapes for XMX calibration
  K=4096  N=4096  M={64,256,512,1024,2048,4096}        -> 6 entries  [square ref]
  K=4096  N=11008 M={64,256,512,1024,2048}             -> 5 entries  [FFN-shaped ref from doc 18]
  K=8192  N=8192  M={256,512,1024,2048,4096}           -> 5 entries  [large square]
  K=4096  N=4096  M={4096,8192}                        -> 2 entries  [large-M saturation]

Total: 7+6+7+7+5+5+6+6+5+5+5+5+6+5+5+2 = 92 + 8 config knobs below = ~100.

ADDITIONAL CONFIG KNOBS (sweep within select shapes, add ~8 configs):
  For K=5120, N=17408, M=512: sweep oneDNN verbose to capture impl string (Run-1 from doc 10)
  For K=5120, N=5120, M=1: the decode transition shape (compare to GEMV section)
  For K=2048, N=512, M=1..8: MoE decode regime (few experts, tiny M)
  Format_tag=any vs explicit stride (already moot per P0, but include for record)
  SG=16 quant kernel vs SG=32 (for the per-token quant companion op)

Output CSV columns:
  M, K, N, dtype(w8a8|w4a8), M_regime(GEMM|GEMV), wall_us_median, wall_us_p90,
  xmx_tops, xmx_pct, bw_gbps, bw_pct, onednn_impl_str (from ONEDNN_VERBOSE=1)

--------------------------------------------------------------------------------
C.2 GEMV sweep: 100 decode configurations (M=1 plus small M)
--------------------------------------------------------------------------------

Primary dimension: (K, N) pairs at M=1. Secondary: M=2,4,8 to see GEMV->GEMM transition.
Config knobs: int8 W8A8 baseline vs W4A8 (int4 weight) for BW comparison.

All (K, N) pairs from models, at M in {1, 2, 4, 8}:
  K=5120   N=5120   M={1,2,4,8}   -> 4  [14B/27B attn Q/O, same N]
  K=5120   N=1024   M={1,2,4,8}   -> 4  [14B/27B attn K/V]
  K=5120   N=7168   M={1,2,4,8}   -> 4  [14B qkv fused]
  K=5120   N=17408  M={1,2,4,8}   -> 4  [14B/27B MLP gate/up]
  K=17408  N=5120   M={1,2,4,8}   -> 4  [14B/27B MLP down]
  K=5120   N=6144   M={1,2,4,8}   -> 4  [27B attn Q]
  K=6144   N=5120   M={1,2,4,8}   -> 4  [27B attn O]
  K=2048   N=512    M={1,2,4,8}   -> 4  [35B expert gate/up -- the key MoE decode shape]
  K=512    N=2048   M={1,2,4,8}   -> 4  [35B expert down]
  K=2048   N=4096   M={1,2,4,8}   -> 4  [35B attn Q]
  K=4096   N=2048   M={1,2,4,8}   -> 4  [35B attn O]
  K=2048   N=2048   M={1,2,4,8}   -> 4  [35B dense square]

 = 48 GEMV baseline configs.

Column-reorder ABI sweep (add another 36 configs):
  Same (K,N) pairs at M=1, but with W_col-major layout (P4 reorder applied).
  Compare: oneDNN W8A8 (current) vs hand dp4a GEMV with col-reorder (P5 draft).
  KPIs: bw_pct improvement from reorder alone vs from reorder+custom kernel.

Per-quant BW comparison at M=1 (add 16 configs):
  For shapes K=5120 N=17408 and K=17408 N=5120 and K=2048 N=512 and K=512 N=2048:
  W8A8 vs W4A8 vs W4A16 (if available in the bench) at M={1,8}
  -> lets us measure the int8 vs int4 BW crossover at decode for our actual shapes.

Total GEMV configs: 48 + 36 + 16 = 100.

Output CSV columns:
  M, K, N, layout(row-major|col-major), dtype, wall_us_median, bw_gbps, bw_pct,
  onednn_impl_str, notes

--------------------------------------------------------------------------------
C.3 Reference ggml q8_0 GEMV baseline (the BW ceiling benchmark)
--------------------------------------------------------------------------------

Run BEFORE writing the hand dp4a GEMV (P5). The ggml q8_0 GEMV (if optimized) is the
achievable BW ceiling on this card without a custom kernel -- it tells us how much gap
exists between the oneDNN 61% and the real ceiling.

Recipe (from doc 10 TG-0, adapted):
  Build llama.cpp SYCL with GGML_SYCL=ON GGML_SYCL_F16=ON, q8_0 model.
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops perf -o MUL_MAT
  Look for m=1 throughput on shapes matching K=5120, N=5120 and K=17408, N=5120.
  BW_eff = (K*N bytes int8 + small) / time.
  Target: >= 90% of 608 GB/s if reorder path fires; ~21-24% if only DMMV (issue #21517).
  This tells us if a custom GEMV (P5) is worth writing at all.
  -> If ggml q8_0 also plateaus at 21-24%, the ceiling is hardware/driver-limited -> P5 may
     not reach 90% either. If ggml WITH reorder reaches 80-90%, P5 has a clear target.

NOTE: do NOT run the GPU benchmark without `scripts/gpu-run`. Log result in JOURNAL.md.

================================================================================
(d) Citations and sources
================================================================================

VERIFIED sources (fetched from primary, date noted):

  [1] Qwen/Qwen3-14B config.json (HF blame commit 7d3da9c5, fetched 2026-06-21).
      hidden_size=5120, intermediate_size=17408, num_attention_heads=40,
      num_key_value_heads=8, head_dim=128, vocab_size=151936, num_hidden_layers=40.
      URL: https://huggingface.co/Qwen/Qwen3-14B

  [2] Qwen/Qwen3.6-27B model card (HF, fetched 2026-06-21).
      hidden_size=5120, intermediate_size=17408, num_attention_heads=24,
      num_key_value_heads=4, head_dim=256/128, vocab_size=248320, num_hidden_layers=64.
      Layer pattern: 16 x (3 GatedDeltaNet + 1 GatedAttn) each with FFN.
      URL: https://huggingface.co/Qwen/Qwen3.6-27B

  [3] Qwen/Qwen3.6-35B-A3B model card (HF, fetched 2026-06-21).
      hidden_size=2048, num_attention_heads=16, num_key_value_heads=2,
      head_dim=256/128, vocab_size=248320, num_hidden_layers=40,
      num_experts=256, num_experts_per_tok=9 (8+1 shared), moe_intermediate_size=512.
      URL: https://huggingface.co/Qwen/Qwen3.6-35B-A3B

  [4] ggml-org/llama.cpp issue #21517: Q8_0 quantization ~4x slower than Q4_K_M on
      Intel Arc Pro B70. Q8_0 = 21-24% BW, Q4_K_M = 53-64% BW. Root cause: DMMV
      2 values/thread vs MMVQ 8x stride. Fetched 2026-06-21.
      URL: https://github.com/ggml-org/llama.cpp/issues/21517

  [5] Intel sycl-tla (CUTLASS-SYCL). SYCL Templates for Linear Algebra for Intel GPUs.
      Includes bmg-targeted examples: 04_bmg_grouped_gemm, 09_bmg_grouped_gemm_f8,
      10_bmg_grouped_gemm_mixed_dtype, 12_xe20_moe_gemm_cute_interface.
      XMX int8 atom: s8 x s8 -> s32, M<=8/N=16/K=32, B=ext_intel_packed (VNNI).
      URL: https://github.com/intel/sycl-tla

  [6] "Pushing the Envelope of LLM Inference on AI-PC and Intel GPUs" arXiv:2508.06753v2.
      BMG B580: 20 Xe2 cores, 12 GB GDDR6, 456 GB/s, 233 INT8 TOPS. BMG GEMV achieves
      up to 380 GB/s (83% of peak) for int2 at larger shapes. End-to-end 6.3x over BF16.
      URL: https://arxiv.org/abs/2508.06753

  [7] "Realizing Native INT8 Compute for Diffusion Transformers on Consumer GPUs" (Ideogram 4.0
      kernel), arXiv:2606.14598v1. Key: autotuning over 36 configs per shape is load-bearing
      (without it, LLM-projection shape is 0.64x slower than bf16); fused dequant epilogue from
      int32 accumulator; compute-bound at AI>=768 ops/byte. NVIDIA Ampere reference, not Xe2.
      URL: https://arxiv.org/abs/2606.14598

  [8] "Accelerating a Triton Fused Kernel for W4A16 Quantized Inference with Split-K" arXiv:2402.00025.
      Split-K for decode-sized M: +1.4x over standard tiling on NVIDIA. Principle applies to Xe2.
      URL: https://arxiv.org/abs/2402.00025

  [9] intel/intel-xpu-backend-for-triton issue #6389: "[VLLM] Grouped GEMM Performance and Tuning."
      Triton grouped GEMM on XPU tuning active (2026). Relevant to MoE P6 path.
      URL: https://github.com/intel/intel-xpu-backend-for-triton/issues/6389

  [10] Intel oneAPI optimization guide: "Programming Intel XMX Using SYCL Joint Matrix Extension."
       INT8 on Battlemage (bmg_g31): use::a row_major s8 [M<=8, K=32], use::b ext_intel_packed
       s8 [K=32, N=16], accumulator s32 [8,16]. Sub-group 16. VNNI B pitch = N*4 bytes.
       URL: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/

REPO SOURCES (in-repo files already VERIFIED, listed for traceability):
  docs/kernel/05_int8_int4_optimization_survey.md  -- oneDNN v3.6/3.8/3.9-rc; IPEX int8 paths
  docs/kernel/06_sycl_int4_gemv_design.md          -- int4 GEMV design; verified ggml idioms
  docs/kernel/10_int8_gemm_handtune_plan.md        -- W8A8 gap analysis; hand kernels
  docs/kernel/11_fused_quant_handtune_plan.md      -- fused quant plan; verified op existence
  docs/kernel/14_b70_int_inference_learnings.md    -- capstone; all measured results
  docs/kernel/18_xpu_int8_moe_kernel.md            -- MoE int8 plan; 1.4-2x measured
  FINDINGS.md                                      -- living field notes; measured decode/prefill

================================================================================
(e) Key derived insights NOT in prior docs
================================================================================

1. **Qwen3.6-27B and 35B-A3B share the same MLP shape (K=5120/17408) with 14B.**
   The MLP int8 kernel is the same code path for all three dense models. Tuning once
   on 14B carries to 27B MLP. The 35B-A3B MoE experts (K=2048/N=512) are a SEPARATE
   shape family requiring a separate grouped kernel.

2. **35B-A3B MoE decode is harder than prefill for int8.** At M=1 (9 experts), the expert
   GEMVs (K=2048, N=512) are only 1 MB each -> 9 MB total MoE weight read/token. That is
   memory-bound and int4 weight (0.5 MB/expert) beats int8 on bandwidth. Int8 MoE only
   wins at prefill (compute-bound) with fused quant (section C3 of doc 18).

3. **Qwen3-14B intermediate_size=17408 is NOT a multiple of 256.** 17408 = 136*128 = 34*512.
   It is divisible by 128 (head_dim) and 512 (MoE expert size). For XMX tiling: DPAS N=16
   divides 17408 evenly (17408/16=1088). For GEMV sub-group K-alignment: 17408/16=1088 words,
   17408 is a multiple of 64 (the KSTEP in the dp4a GEMV from doc 10 c.2). No tail needed.

4. **The 27B attn shapes (Q=24 heads, head_dim=256) produce K=5120->N=6144 and K=6144->N=5120**
   -- distinct from 14B (K/N both 5120). The O-proj K=6144 is the widest-K attn shape in our
   suite. Its GEMV at decode: 6144*5120 = 31.5 MB int8 weight -> t_min = 51.8 us -> ~19k t/s
   ceiling per head-set. This shape is NOT covered by existing 14B benchmarks.

5. **All prefill compute crossovers are at very small M for our shapes.** For K=5120/N=17408/int8:
   arithmetic intensity = M * 2 * 5120 * 17408 / (M*22528 + 89128960) bytes.
   Crossover at AI=BW/TOPS=608/367=1.66 ops/byte -> M ~ 1.66 * 89128960 / (2*5120*17408 - 1.66*22528)
   ~ 1.66 * 89M / (178M - 37k) ~ 0.83 tokens. Meaning: even M=1 is at the compute/BW boundary
   for the MLP shapes. In practice M>=4 is clearly compute-bound for MLP, M>=16 for attn.
   This confirms why W8A8 prefill benefits strongly from XMX (int8 TOPS matter even at M=2).
