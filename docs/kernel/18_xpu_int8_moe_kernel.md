# 18 -- XPU INT8 MoE kernel: state of the art + our build plan (Qwen3.6-35B-A3B)

Date: 2026-06-21. Scope: the user wants the **35B-A3B MoE served with int8-activation experts** (W8A8 or
W4A8) to hit the B70's XMX int8 systolic datapath -- "we'll build our own if we have to." This doc is the
research synthesis + the concrete build plan. Supersedes the 35B "NO-GO" verdict in
[`15_autoround_w4a8_w8a8_recipes.md`](15_autoround_w4a8_w8a8_recipes.md) sec 3: it is a NO-GO *off-the-shelf*,
but BUILDABLE. Companion: the bf16 source is now on the host (`scripts/63`); the queue is `QUANTS_TODO.md`.

================================================================================
TL;DR
================================================================================
- **Nobody ships an int8-activation fused-MoE expert kernel for Intel GPU.** Verified at every layer (vLLM,
  vllm-xpu-kernels, SGLang, Intel llm-scaler, IPEX/ipex-llm, the auto-round/INC toolchain). Every shipping
  quantized-MoE path on Battlemage keeps **activations in fp16 on XMX and quantizes weights only** (int4-W4A16
  or fp8-W8A16). int8 MoE is on NO roadmap. => to get int8-XMX MoE we BUILD it.
- **The hardware fully supports it.** Xe2/Battlemage DPAS does native `s8 x s8 -> s32` (233 INT8 TOPS on B580;
  SYCL `joint_matrix` s8->s32 at M<=8/N=16/K=32). int8 is the LOWEST precision reachable via joint_matrix
  (int4/fp8 are NOT in the joint_matrix table) -> int8 is exactly the right MoE target for Xe2.
- **We already proved the compute is correct.** Our existing dense oneDNN int8 op (`int8_gemm_w8a8`) used as a
  per-expert grouped GEMM gives `cosine 0.99992` vs bf16 and `rel_err 1.3e-3` vs an exact int8 ref, on the real
  35B-A3B expert shapes (K=2048, N=512, E=256, top-8). See `scripts/int8_moe_grouped_test.py`.
- **The naive per-expert loop is dispatch-bound, not compute-bound** -> int8 shows NO win through a 256-iter
  Python loop (eager). The win needs a FUSED grouped kernel (one launch over all experts) or graph capture.
- **Don't build from scratch.** `intel/sycl-tla` (CUTLASS-SYCL, renamed) ALREADY ships Battlemage-targeted
  grouped-GEMM + MoE examples (`04_bmg_grouped_gemm`, `12_xe20_moe_gemm_cute_interface`), and
  `vllm-xpu-kernels/csrc/xpu/grouped_gemm/` wraps them with the token-routing pipeline DONE. Their interface
  exposes only `is_B_int4`/`is_B_mxfp4` -- **W8A8 is the single missing flag.** Our job = add the `s8 x s8 -> s32`
  MMA atom + per-token/per-channel scale epilogue to that existing grouped collective.
- **[!] The ROI is NARROW -- this is a payoff question, not a feasibility one.** Our microbench shows int8 is
  SLOWER than bf16 at the A3B's tiny N=512 expert shape (memory-bound, not compute-bound -> no XMX win + quant
  overhead). int8-act only pays off COMPUTE-bound (dense W8A8 won prefill +50-60%, but A3B experts are too small).
  => int8 won't help the 35B DECODE path (W4A16's 56.8 t/s is already the ceiling); the win is prefill/large-batch
  only. **Recommendation: build the cheap Track-A bootstrap to MEASURE end-to-end int8-vs-W4A16 before committing
  to the weeks-long fused kernel.** W4A16 stays the serving recipe until the data justifies more.

================================================================================
1. State of the ecosystem (who has int8 XPU MoE -- nobody)
================================================================================
Verified June 2026 against primary source (code-level):
- **vLLM `main`:** `fused_moe/oracle/int8.py` -> `Int8MoeBackend.TRITON` only, no platform branch -> routes XPU
  W8A8 MoE to generic `TritonExperts`. `oracle/w4a8_int8.py` -> `CPU_INT4` only -> raises
  `NotImplementedError("W4A8 Int8 MoE is only supported on CPU platforms")` off-CPU. `experts/xpu_moe.py`
  subclasses: Fp8, MxFp8, BlockFp8, **WNA16 (int4)**, MxFp4 -- **NO int8/W8A8/W4A8**. The kernel op
  `xpu_fused_moe(...)` flags: `is_fp8,is_int4,is_mxfp4,is_mxfp8,is_block_fp8` -- **no `is_int8`.**
- **Roadmap:** RFC #33214 (XPU kernel migration to vllm-xpu-kernels) MoE checklist -> int4 is the ONLY WIP
  quant-MoE item; int8 appears nowhere. Issue #224's "xpu_fused_moe for W8A8" sits under the **FP8** section
  (means fp8-W8A8, not int8 -- the conflation to avoid). Closed PR #6978 (int8 fused MoE) was CUDA, never merged.
- **Intel llm-scaler** (`intel/llm-scaler-vllm:0.14.0-b8.3.1`, targets Qwen3.5/3.6-35B-A3B): ESIMD/SYCL MoE
  kernels exist but ONLY fp8-weight/fp16-act and int4-weight/fp16-act (`moe_int4.sycl`, `fp8_moe_gemm.h`).
  `grep w8a8/w4a8` over the repo = zero. Activations run fp16 on XMX in every path.
- **IPEX / ipex-llm:** weight-only (INT4 WOQ; `sym_int8` = int8 WEIGHT only, fp16 acts -- NOT W8A8). XeTLA MoE
  GEMM is gated to Xe-HPC/PVC, not Battlemage.
- **SGLang:** Intel support is CPU (AMX); no Arc GPU int8 MoE.
- **Toolchain (auto-round / INC / llm-compressor):** can PRODUCE W8A8/W4A8 MoE checkpoints, but Intel's own
  `ai-containers/vllm/0.10.2-xpu.md` says **"W8A8 quantized models through llm_compressor are not supported yet"**
  on XPU. The only non-CUDA int8 W8A8 MoE fused kernel is **vllm-ascend PR #5718 (NPU)** -- a porting reference,
  not XPU.
=> Confirmed: no upstream effort to adopt. If we build it, it LANDS in `vllm-xpu-kernels` (SYCL/oneDNN), tracked
   via RFC #33214 / issue #224.

================================================================================
2. Hardware: Xe2/Battlemage DOES int8 (so this is worth building)
================================================================================
- Battlemage DPAS: native `s8 x s8 -> s32` (and int4/int2 x int8). B580 = **233 INT8 TOPS** (Intel spec); oneDNN/
  XeTLA int8 GEMMs reach ~90% of attainable peak (arXiv 2508.06753).
- SYCL `joint_matrix` int8: `s8/u8 A x s8/u8 B -> s32` accumulator, tile **M<=8 / N=16 / K=32** for
  `intel_gpu_bmg_g31` (= our B70). **int4 and fp8/bf8 are NOT in the joint_matrix supported-combinations table**
  -> int8 is the lowest precision practically reachable via joint_matrix on Xe2. So int8 W8A8 is the natural MoE
  target (we already chose it for the dense path -- `contrib/vllm_int8_xpu`).

================================================================================
3. Our empirical test (correctness yes; loop is dispatch-bound)  -- scripts/int8_moe_grouped_test.py
================================================================================
Reuses our EXISTING dense `torch.ops._xpu_C.int8_gemm_w8a8` as a per-expert grouped GEMM (loop over active
experts), on the real 35B-A3B shapes (K=2048, N=512, E=256, top-8), image `:int8g`, single B70, EAGER.
- **Correctness PROVEN:** kernel vs exact int8 ref `mean_rel_err = 1.3e-3`; end-to-end vs bf16 `cosine = 0.99992`
  (both DECODE T=1 and PREFILL T=256). The int8 expert GEMM math is right; int8 quant of experts barely moves the
  output.
- **Perf -- the per-expert loop is launch/dispatch-bound (eager):**
  | regime | bf16 loop | int8 loop |
  |---|---|---|
  | DECODE (1 tok, 8 active experts) | 11.6 ms | 13.8 ms |
  | PREFILL (256 tok, 256 active) | 431 ms | 448 ms |
  int8 is NOT faster -- the tiny per-expert GEMMs are swamped by ~1280 kernel launches/iter of Python+eager
  dispatch (the same eager-dispatch tax we measured serving). The int8 XMX advantage is invisible behind launch
  overhead. (Note our dense op takes a SINGLE weight [K,N] -> it cannot do per-expert weights in one call, which
  is exactly why a grouped/fused kernel is required.)
- **Raw single-GEMM at the MoE-expert shape (N=512), int8(incl. quant) vs bf16:** M=2048 bf16 0.064 ms (66.8
  GFLOP/s) / int8 0.098 ms (43.9) = **int8 0.66x**; M=8192 bf16 0.125 ms (137 GFLOP/s) / int8 0.232 ms (74) =
  **int8 0.54x**. **int8 is SLOWER here.** Both run at ~70-137 GFLOP/s = <0.1% of the 233 INT8 TOPS peak ->
  these GEMMs are NOT compute-bound; they are memory/latency-bound (N=512 is tiny), where int8's XMX TOPS
  advantage does not appear and the per-token activation quant is pure overhead.
- **The boundary (important):** int8 DOES win when COMPUTE-bound -- the dense 14B/27B W8A8 measured **+50-60%
  prefill vs fp8/bf16** (big GEMMs, hidden 5120 / intermediate ~17k). The 35B-A3B's expert intermediate is only
  N=512 -> per-expert GEMMs are too small to be compute-bound, so int8 ACTIVATIONS don't help and add quant cost.
  (A GEMM-only-at-large-N confirmation is queued; the dense +60% prefill precedent already establishes the win
  exists only in the compute-bound regime.)
- **Conclusion:** the foundation works (correct), but **int8-act helps PREFILL/large-batch (compute-bound), NOT
  DECODE (memory-bound, where the A3B small experts live)**. Decode is already best served by **W4A16** (int4
  weight = least DRAM read; the current 56.8 t/s captured path). The path to ANY int8 MoE perf is (a) a FUSED
  grouped kernel (one launch over all experts, activations pre-quantized in the permute so the quant cost is
  amortized) -- the only way to reach aggregate compute-bound at prefill; and/or (b) PIECEWISE graph capture to
  amortize launches (the int4 MoE got 7.17x from capture). The naive per-expert loop is hopeless (dispatch-bound).

================================================================================
4. The build plan -- the canonical int8-MoE pipeline, ported to Xe2
================================================================================
Every mature CUDA stack (vLLM, TensorRT-LLM, SGLang, DeepGEMM) converges on the SAME 7-stage flow. Stages 0-2,
6-7 are architecture-portable; the GEMM work (3-5) is where we add int8.

  Stage 0  ROUTE: router GEMM -> softmax -> topk_ids[T,8], topk_weights[T,8]. (portable)
  Stage 1  SORT/GROUP by expert (moe_align_block_size): histogram + prefix-sum scatter -> sorted_token_ids,
           expert_ids (per block), num_tokens_post_padded; pad each expert run to BLOCK_M, sentinel = numel.
  Stage 2  OFFSETS / problem-sizes: cumsum[e] = expert start. ONLY M varies per expert; N,K fixed (experts share
           FFN shape) -> one persistent kernel sweeps all experts. Two layouts:
             - CONTIGUOUS (m_indices/-1 sentinel, pad to BLOCK_M) -> prefill.
             - MASKED (per-expert masked_m valid counts) -> CUDA-graph/decode friendly, NO padding waste.
  Stage 3  PER-TOKEN INT8 QUANT activations, FUSED into the gather (amax+int8 emit in the scatter pass; the GEMM
           gathers row offs_token//top_k so one quantized row is shared across its top_k experts).
  Stage 4  GROUPED GEMM #1 (fused gate+up w1 [E,2I,K]): int8 x int8 -> **int32 accumulate**, then epilogue
           `acc * a_scale[token] * b_scale[expert,n]` in fp32. (plain int32 accumulate -- no fp8 fast-accum.)
  Stage 5  FUSED SiLU/SwiGLU + RE-QUANT to int8 (fresh per-token scale) before GEMM #2 (one pass).
  Stage 6  GROUPED GEMM #2 (down w2 [E,K,I]): int32 acc -> dequant; FUSE the top-k router-weight multiply into
           the epilogue (acc *= topk_weight[token]).
  Stage 7  FINALIZE: sum the top_k weighted expert outputs per token (moe_sum), +bias/+residual, fp32 accumulate.

**What to REUSE (port-and-tune, not from-scratch):**
- **`intel/sycl-tla`** (CUTLASS-SYCL, v0.9.1, validates `bmg_g31` = B70): grouped-GEMM + MoE examples already
  exist -- `examples/04_bmg_grouped_gemm`, `09_bmg_grouped_gemm_f8` (EVT dequant-scale epilogue pattern),
  `10_bmg_grouped_gemm_mixed_dtype` (has an s8-WEIGHT grouped path), `12_xe20_moe_gemm_cute_interface`
  (persistent MoE tile scheduler `PersistentTileSchedulerXeGroup`, fixed for large expert counts). Start from
  04 + 12.
- **`vllm-xpu-kernels/csrc/xpu/grouped_gemm/`**: a sycl-tla-based MoE grouped GEMM with routing DONE
  (`csrc/moe/`: moe_align_sum, moe_gather, grouped_topk, init_expert_map). Interface
  `cutlass_grouped_gemm_interface(...,is_B_int4,is_B_mxfp4)` -- **add `is_w8a8` / an s8s8s32 path here.**
- **Our dense oneDNN int8 kernel** (`contrib/vllm_int8_xpu` / `csrc/xpu/onednn/*_gemm_w8a8.h`): reuse for the
  non-MoE linears AND as the per-expert compute in a captured-loop bootstrap.

**What is HARD (port-risk):**
1. The grouped-GEMM **tile scheduler** (variable-M load balance), but `PersistentTileSchedulerXeGroup` exists.
2. **Per-expert padding at 256 experts** at decode batch (few tokens/expert) wastes FLOPs -> implement DeepGEMM's
   **MASKED layout** for decode (the single most important decision for our 256-expert/top-8 shape).
3. **Quantized-weight layouts do NOT port** (Marlin interleave / lop3 are NVIDIA warp-geometry-specific). int8
   maps naturally to DPAS (sub-group 16) -> let sycl-tla's int8 reorder handle layout; don't hand-roll Marlin.
4. **Fusion boundaries:** do the GEMMs via sycl-tla (EVT epilogue for the scale broadcast); write permute-quant
   (Stage 3), SiLU-requant (Stage 5), finalize (Stage 7) as standalone SYCL kernels (mirror SGLang's Triton
   kernels). Keep oneDNN for the dense linears; the MoE glue is custom-kernel territory.

**Reference implementations to read:** vLLM CUDA int8 fused_moe (`fused_moe.py`: plain int32 `tl.dot` +
`(offs_token//top_k)` scale indexing + epilogue dequant) = the algorithm; `cutlass_moe.py` +
`grouped_mm_c3x.cuh` + `moe_data.cu` (`get_cutlass_moe_mm_data` builds expert_offsets/problem_sizes) = the
template that maps to sycl-tla; QQQ (arXiv 2406.09904, vLLM #5218) = W4A8 design study (int8 tensor cores, int32
acc, two-level scaling -- the idea ports, the Marlin layout does not); vllm-ascend #5718 = a non-CUDA int8 W8A8
MoE port reference.

================================================================================
5. Recommended two-track execution
================================================================================
**Track A -- BOOTSTRAP (days, low risk): get int8 MoE CORRECT + decode-servable without a new SYCL kernel.**
  - Mirror the int4 INC routing patch (`contrib/vllm_moe_xpu/inc.py`) but route int8 W8A8 `RoutedExperts` to an
    `XPUExpertsInt8` that loops active experts calling our existing `int8_gemm_w8a8` (+ per-token quant), with the
    Stage 1/7 routing borrowed from the existing MoE method. Correctness already proven (sec 3).
  - Rely on **PIECEWISE graph capture** to amortize the per-expert launches (decode = few active experts). Target:
    match/beat the int4 W4A16 path (56.8 t/s captured) on int8 -- plausible at decode since few experts are active.
  - Deliverable: a serving int8 W8A8 35B-A3B (decode), even if prefill is loop-bound. Validates the whole stack.

**Track B -- THE REAL KERNEL (weeks): fused int8 grouped GEMM in sycl-tla / vllm-xpu-kernels.**
  - Add the `s8 x s8 -> s32` collective MMA atom + per-token/per-channel scale EVT epilogue to the existing Xe2
    grouped collective (start: sycl-tla `04`+`12`; land: vllm-xpu-kernels `grouped_gemm` `is_w8a8` flag).
  - Implement MASKED layout for decode + contiguous for prefill; fuse permute-quant / SiLU-requant / finalize as
    standalone SYCL kernels.
  - FIRST verify (sec 5 caveat): confirm sycl-tla provides an int8-ACCUMULATE (s8s8->s32) MMA atom for Xe2 -- the
    shipped examples exercise s8 only as a DEQUANTIZED weight feeding a float MMA, so the integer-accumulate atom
    may need adding/validating (the joint_matrix s8->s32 combo exists, so it is a kernel-plumbing task, not a HW gap).
  - oneDNN's experimental grouped GEMM (`ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`, v3.12+) is a QUICK CORRECTNESS
    BASELINE -- ragged offsets, distinct per-expert weights -- BUT its int8 grouped path outputs **bf16/f16 only,
    NO s32 accumulate**, so it can't mirror our dense s8s8s32 and likely won't hit peak. Use to sanity-check, not ship.

================================================================================
6. Verdict
================================================================================
35B-A3B int8 MoE is **BUILDABLE but the ROI is NARROW** -- and that nuance is the most important output of this
investigation. It is NOT an adopt (nobody upstream has it); Xe2 supports it; the grouped-GEMM scaffolding exists
(sycl-tla + vllm-xpu-kernels) with W8A8 the one missing flag; our compute is proven correct. **BUT** our measured
perf says int8-activations only pay off when COMPUTE-bound, and the A3B's tiny N=512 experts are memory-bound at
decode -> **int8 won't speed up the common decode path; W4A16 (56.8 t/s captured) is already the decode ceiling.**
The int8 win is a PREFILL/large-batch play only, and even there it is bounded by the small expert N.

**Recommendation (honest):**
1. **Do NOT start with the weeks-long Track B kernel.** First de-risk the premise: build **Track A** (bootstrap --
   mirror the int4 INC routing patch to an `XPUExpertsInt8` that loops active experts on our existing
   `int8_gemm_w8a8`, lean on PIECEWISE capture). It is days of work, proves the whole serving stack int8-MoE, and
   lets us MEASURE end-to-end int8-vs-W4A16 on the real 35B at both decode and prefill. If int8 doesn't beat
   W4A16 there, we have our answer cheaply and skip Track B.
2. **Only if Track A shows a real prefill/throughput win** worth the maintenance: build **Track B** (the fused
   sycl-tla s8s8s32 grouped kernel, masked layout for decode) and upstream into vllm-xpu-kernels (RFC #33214 /
   issue #224 is where it lands -- and there is genuine community value since no one has int8 XPU MoE).
3. For now, **W4A16 stays the 35B-A3B serving recipe.** The bf16 source is downloading (`scripts/63`) so we CAN
   quant to int8 once a kernel justifies it; the W8A8/W4A8 recipe slots into `QUANTS_TODO.md`.

The single most valuable finding: **this is a kernel-ROI question, not a kernel-feasibility question.** We proved
we CAN build it; the data says measure the payoff (Track A) before paying for it (Track B).

================================================================================
Sources (key)
================================================================================
- vLLM: fused_moe/oracle/{int8,w4a8_int8}.py, experts/xpu_moe.py, experts/cutlass_moe.py, fused_moe.py,
  moe_align_block_size.py, csrc .../moe/{moe_align_sum_kernels.cu,moe_data.cu}, grouped_mm_c3x.cuh; RFC #33214; PRs
  #41426 (XPU W4A16 MoE), #13972 (cutlass grouped fp8 MoE), #5218 (QQQ W4A8); issue #40675 (AutoRound).
- vllm-xpu-kernels: csrc/xpu/grouped_gemm/ (interface is_B_int4/is_B_mxfp4), csrc/xpu/onednn/*_gemm_w8a8.h,
  fused_moe_interface.py; issues #224, #141.
- intel/sycl-tla (CUTLASS-SYCL) examples 04/09/10/12; releases (v0.9.1, "MoE grouped GEMM large-expert fixes").
- oneDNN grouped GEMM: dev_guide_matmul (u8/s8 -> f32/bf16/f16 only on grouped path), v3.12 release,
  ONEDNN_EXPERIMENTAL_GROUPED_MEMORY.
- Intel llm-scaler (moe_int4.sycl, fp8_moe_gemm.h); IPEX releases; intel/ai-containers vllm/0.10.2-xpu.md
  ("W8A8 ... not supported yet"); vllm-ascend #5718 (NPU int8 W8A8 MoE).
- Xe2: Arc B580 spec (233 INT8 TOPS), sycl_ext_oneapi_matrix (s8->s32 M<=8/N=16/K=32), arXiv 2508.06753.
- Algorithm refs: DeepGEMM scheduler.cuh (contiguous/masked), TensorRT-LLM moe_gemm, SGLang ep_moe/kernels.py,
  MegaBlocks (arXiv 2211.15841), Marlin (arXiv 2408.11743), QQQ (arXiv 2406.09904).
