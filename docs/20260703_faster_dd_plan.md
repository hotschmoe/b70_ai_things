# 20260703 -- Faster daily driver: research synthesis + ranked plan

**Goal:** raise the daily driver's real-world speed (single-stream decode, concurrent decode,
long-context TTFT) on the dual-B70 box, and mature the Intel-GPU serving ecosystem with our own
contributions. Baseline (benched 2026-07-03, the live config): vLLM v0.24.0 XPU, Qwen3.6-27B
W8A8 TP=2, MTP spec=3, PIECEWISE capture, PREFIXCACHE=1, push-AR -- PP 2711 tok/s, TTFT 755 ms,
TG c1 30.0 / c4 15.4 (usage-based chat probe 44 t/s), KV 320k @ MAXLEN 253952.

Inputs: 4 research sweeps (2026-07-03): (a) spec-decode/MTP literature 2025-2026, (b) vLLM+sglang
dev branches, (c) Intel Battlemage kernel/P2P/runtime ecosystem, (d) local repo bottleneck audit.
This doc is the synthesis + ranked execution plan. JOURNAL entries per experiment as usual.

## 1. Where the time goes (from our own measurements)

- Captured decode fwd pass ~74 ms @ TP=2; GEMMs are 88-100% of the 581 GB/s roofline (do NOT
  tune the GEMM inner loops -- docs/kernel/23). Post-capture the step is LAUNCH/PYTHON-bound.
- The one capture-persistent kernel hotspot: dynamic_per_token_int8_quant, 101 us on [1,17408]
  (ideal <1 us); 35% of down_proj layer time, 10% gate_up (docs/kernel/23).
- MTP: usage accept_length 2.0 (67% draft accept) with the 1-layer NEXTN head; per-position
  accept decays 80% -> 37% within a verify window. spec=3 swept-optimal on the captured path.
- c4 per-stream halving (30 -> 15.4) = MTP amortization collapsing under batch (verify shares
  the step across streams), NOT box saturation (aggregate keeps scaling to c8).
- AR is ~5% of the captured step (push-AR 34-45 us vs oneCCL 85 us); not the bottleneck anymore.
- M=1 int8 GEMV: oneDNN jit:gemm:any = 59% BW (361/608 GB/s) vs bf16 76%; small-N (KV heads,
  N=1024) only ~1.1x over bf16 -- the "int8 GEMV trap". Blocks small-M verify economics.

Implication: single-stream upside = (1) more tokens per weight-read (better drafting/accept),
(2) kill the capture-persistent quant op, (3) shave launch/replay overhead. Concurrent upside =
load-aware speculation. TTFT upside = prefix cache (done) + speculative prefill (later).

## 2. What the ecosystem sweep found (headlines)

Literature (spec decode):
- DFlash (Z Lab, ICML26, arXiv:2602.06036): block-diffusion drafter, ONE parallel fwd pass
  drafts a 16-token block conditioned on target hidden states (KV-injection). accept len ~6.5
  vs EAGLE-3 ~4.2; ~5x lossless on CUDA. **A trained drafter for our exact model exists:
  z-lab/Qwen3.6-27B-DFlash (~2B bf16, 5 layers, SWA)**. In vLLM via the `speculators` lib.
  XPU blockers: flash_attn assumed for draft attn; vLLM issue #41190 = open crash on exactly
  Qwen3.6-hybrid-GDN + TP=2 + spec (event lifetime, both qwen3_next_mtp and dflash).
- DeepSeek DSpark + DeepSpec toolkit (2026-06-27, MIT): semi-AR drafting (parallel backbone +
  rank-256 sequential correction head) + LOAD-AWARE VERIFICATION (verify depth set per-request
  from live GPU load). +16-18% accept len over DFlash on open Qwen3 models. Training stack is
  CUDA/8-GPU -- port the *scheduler idea*, not the code. Directly addresses our c4 collapse.
- PFlash (lucebox): speculative PREFILL -- 0.6B drafter scores token importance, target only
  prefills important spans via block-sparse attention. ~10x TTFT @128K on Qwen3.6-27B (3090,
  llama.cpp). Needs a block-sparse-attn XPU kernel. Separate TTFT track, not decode.
- SuffixDecoding (NeurIPS25): training-free suffix-tree drafting from prompt+history; up to
  5.3x on agentic/repetitive workloads; runtime-only, no kernels. Stacks conceptually with MTP.
- SGLang Spec V2 overlap scheduler: +33% by removing host-device syncs in the spec loop --
  same class as our launch-bound finding.

vLLM/sglang dev branches:
- torch.xpu.XPUGraph (SYCL-Graph) is IN our torch 2.12; vLLM FULL-capture wiring for XPU is
  an OPEN gap (vllm#26970, kernels#141 roadmap). We already proved XPUGraph stable on B70 in
  sglang. This is the launch-overhead endgame and an upstreamable contribution.
- VLLM_PREFIX_CACHE_RETENTION_INTERVAL (vllm#45845, merged ~06-23): mamba/hybrid prefix-cache
  retention. Check presence in v0.24.0; extends our prefix-cache unblock.
- MRV2 async model runner (VLLM_USE_V2_MODEL_RUNNER=1, experimental): overlaps schedule N+1
  with GPU step N. Targets our launch/python-bound decode. One report says the overlap does
  not materialize on XPU -- cheap A/B anyway.
- vllm-xpu-kernels: ACTIVE, Battlemage first-class (xe_2), v0.1.11. Has int4 w4a8/w4a16
  oneDNN GEMMs (stop maintaining our own copies), fused rmsnorm (incl per-token-quant-fused),
  rope, silu, and a gdn_attn xe_2 path (open correctness bug #438). **No int8 W8A8 GEMM and
  no collectives at all** -- our two custom pieces have NO upstream equivalent. Moat; consider
  upstreaming the W8A8 op.
- sglang: official Arc B-series support in-tree now (source-install); native hybrid-GDN
  spec-decode + MambaRadixCache upstream (our hand-patches have upstream twins now); target
  HybridRadixTree V2 on any future rebase. sgl-kernel-xpu exists (FP8/INT4 only).
- ipex-llm ARCHIVED, IPEX discontinued (we are correctly torch-2.12-no-IPEX). Intel's active
  stack = intel/llm-scaler "Battlematrix" (official B70 support; FP8/INT4/MXFP4 only; same
  wedging CCL_TOPO_P2P_ACCESS=1 path; NO wedge recovery; admits "limited perf for allreduce
  with small message size" on BMG-G31). Nothing to adopt; a baseline to A/B someday.

Intel low-level:
- XeTLA is archived/PVC-only. The live base is **intel/sycl-tla** (CUTLASS-SYCL, BMG CI,
  int8/int4 mixed-precision GEMM templates, flash-attn decode w/ KV cache). No M=1 GEMV
  template, but the only maintained Xe2 tile/DPAS framework.
- arXiv:2508.06753 v2 (Intel, Jan 2026 added the Xe2 section) = the playbook for the small-M
  int8 kernel: VNNI16 pre-packed weights + rectangular sg tiles reusing each dequant weight
  register ~8x to keep DPAS fed at small M; fused act quant; within 5-15% of oneDNN pure-int8
  at large M, measured on Arc B580, shipped as a vLLM plugin. Attacks exactly our GEMV trap.
- oneDNN 3.7-3.9: no GPU GEMV improvement in-window; our M=1 finding still holds. Keep oneDNN
  for prefill/M>32.
- oneCCL 2021.16+/2022.0: **capturable SYCL-graph collectives** + "Arc Pro B-Series optimized
  scale-up, low-latency protocol" -- newer than our bundled libs. Probe as (a) FULL-capture
  unblock, (b) AR baseline. Wedge caution applies (same oneCCL layer).
- Replay overhead levers: L0 mutable command lists (ZE_experimental_mutable_command_list,
  update KV ptrs/seqlen in place, no re-record) + L0 V2 adapter (counter-based events,
  in-order immediate lists; SYCL_UR_USE_LEVEL_ZERO_V2=1). SYCL-Graph replay gave +15% on
  B580 in GROMACS. Cross-device graphs NOT supported -- TP=2 stays 2 per-card graphs + our
  own sync. NOTE conflicting signal: vllm#41663 stability workaround sets V2=0. A/B both.
- Megakernels (Hazy 78% HBM BW @bs1, 2.5x vLLM on H100; Mirage MPK compiler): NO hardware
  blocker on Xe2 (global-mem atomics for sync; 64KB SLM re-tiling needed; forward-progress
  unspecified -> spin-then-yield). But multi-month, and only removes launch cost on top of
  the M=1 kernel + capture. DEFER until Tier C.
- xe kernel 7.x has PCIe P2P DMA (SVM page-migration interconnect) -- kernel-level, not a
  userspace one-sided-write primitive; does not replace push-AR. No XeLink on B70 (so
  TORCH_LLM_ALLREDUCE unavailable); no stable large-BAR peer-write API. Push-AR stays.

## 3. Ranked plan

Scoring = expected end-to-end win x probability-on-XPU / effort.

### Tier A -- config/runtime A/Bs, hours each (THIS SESSION)
A1. L0 launch-path knobs on the live config: SYCL_UR_USE_LEVEL_ZERO_V2={1,0},
    immediate command lists, batch size 0. Measure TG c1 + TTFT. (Launch-bound decode.)
A2. MRV2 async runner if present in v0.24.0 (VLLM_USE_V2_MODEL_RUNNER=1). TG c1.
A3. Concurrent-decode fix: disable/shrink speculation under batch (vLLM spec config's
    disable_by_batch_size or equivalent on V1) -- target c4 >= 20 t/s/stream while keeping
    c1 ~30. This is the poor-man's DSpark load-aware verification.
A4. Spec re-sweep {2,3,4,5} on the CURRENT v0.24.0 captured+prefix-cache config (spec=3 was
    swept pre-v0.24.0; regime may have moved).
A5. Presence checks (grep the image): VLLM_PREFIX_CACHE_RETENTION_INTERVAL, MRV2 flag,
    speculative disable knobs, suffix-decoding/speculators availability.

### Tier B -- day-scale engineering, this week
B1. Kill the capture-persistent act-quant op: fuse per-token int8 quant into the PRODUCER
    (rmsnorm -> quant for qkv/gate_up input; silu_and_mul -> quant for down_proj input),
    NOT a standalone swap (that regressed 19% -- docs/kernel/23). Check vllm-xpu-kernels'
    per-token-quant-fused rmsnorm first; write the silu_mul+quant SYCL op if missing.
    Expected: removes ~101 us x 48 layers worth of capture-persistent overhead share.
B2. DFlash feasibility spike on XPU: pull z-lab/Qwen3.6-27B-DFlash, read the speculators
    integration, inventory the draft-attn ops (SWA, non-causal block mask, KV-injection),
    map to intel_xpu XMX attn / triton; confirm #41190 exposure at TP=2. Output = a go/no-go
    memo + effort estimate. (Ceiling: accept 2.0 -> ~6 = the single biggest decode lever.)
B3. oneCCL 2022.0 probe (SEPARATE session, wedge-careful, single A/B with reset between):
    newer BMG-aware oneCCL as AR baseline + capturable-collective test OUTSIDE serve first.
B4. Suffix decoding prototype for agentic loads (training-free; vLLM plugin land).
B5. Track vllm-xpu-kernels: try gdn_attn xe_2 (after bug #438 resolves) + fused rope/rmsnorm.

### Tier C -- the ecosystem contributions, week+ each
C1. Small-M int8 DPAS kernel on intel/sycl-tla per arXiv:2508.06753 v2 (VNNI16 + register
    reuse + fused act quant). Fixes the GEMV trap, makes MTP/DFlash verify cheap at M=2..16,
    raises c1 AND c4. Upstreamable to vllm-xpu-kernels (which has NO int8 GEMM).
C2. FULL-capture wiring for vLLM XPU (torch.xpu.XPUGraph, vllm#26970) incl. the
    gdn_attention spec-assert fix; guard with the vllm#40914 MTP+graph corruption lessons.
    Then L0 mutable-command-list route if replay still re-records.
C3. Upstream our W8A8 int8 oneDNN GEMM to vllm-xpu-kernels; write up push-AR as an RFC
    (Intel admits small-message allreduce is weak on BMG; no upstream collectives exist).
C4. PFlash-style speculative prefill on XPU (needs block-sparse attn kernel) -- TTFT track.
C5. Megakernel (persistent decode kernel, Hazy-style, SYCL/ESIMD) -- LAST, after C1+C2.

### Standing quant answer (user asked "other quant types?")
Stay W8A8-int8-first: FP8 is emulated (bf16 upconvert) on Xe2 = memory-only; W4A16 loses
quality; W4A8 stays the secondary kernel path; nothing new in the sweep dethrones int8 XMX.
The sweep CONFIRMS the W8A8 bet: neither Intel (llm-scaler/sgl-kernel-xpu: FP8/INT4/MXFP4
only) nor upstream vLLM ships an int8-activation XMX path. Our kernels are the differentiator.

## 4. Session log (2026-07-03)

- [x] Research sweeps + this plan.
- [x] A5 presence checks: ALL of MRV2 / retention-interval / dynamic-SD / suffix / DFLASH are
      in-tree in v0.24.0 (dflash.py + qwen3_dflash.py + drafter arch registered).
- [x] A1+A2 LANDED: L0-V2 (+3.8%) + MRV2 (+6.7%) stack to TG c1 30.24 -> 33.60 (+11%),
      c4 15.89, gate 24/24 -> serve.sh defaults (opt-outs B70_L0V2=0 / B70_MRV2=0). JOURNAL.
- [x] A3 dynamic SD: FAIL on this stack (0-depth toggle collapses c4 to 12.4). Closed.
- [x] A4 spec re-sweep: SKIPPED (regime unchanged; spec3 optimum stands).
- [x] B2 DFlash spike: GO -- first coherent DFlash serve on XPU, TP=2, zero code changes;
      19.06 t/s at spec=15 (< MTP 30.24 at spike settings); spec=7 crashed at init.
      Memo vllm/DFLASH_XPU.md; drafter at models/files/qwen3.6-27b/dflash-draft.
- [ ] NEXT SESSION: DFlash accept telemetry on real coding workload + drafter W8A8 quant (memo
      follow-ups 1-2); B1 fused act-quant prologue; A5 leftover: suffix decoding +
      VLLM_PREFIX_CACHE_RETENTION_INTERVAL functional test.
