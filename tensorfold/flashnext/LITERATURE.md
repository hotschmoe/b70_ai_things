# LITERATURE.md - ideas for the Flash-Next two-card concurrency and prefill targets

Author model ID: claude-opus-5-5 (Claude Opus 5.5). Three read-only survey subagents, also claude-opus-5-5,
covered (1) Strata 0.2.16 source, (2) arXiv 2023-2026, and (3) local reference repos plus upstream code.
Date: 2026-11-21. This is research only: no GPU was touched and no engine code was edited.

Tags:
- [M]: measured, from B/ledger/LEDGER-train1-imm.{md,json} (build 3edbc4d) or BYTES-hc2 (build e12b9610).
- [I]: inferred or arithmetic.
- [L]: claimed by a paper or repo, usually on other hardware.

The arXiv IDs come from the survey agent, which fetched each abs page. I re-fetched eight of them myself: 2610.13778,
2611.04055, 2511.02237, 2501.08192, 2610.34440, 2608.30320, 2610.38981 and 2512.09277. Many of the papers are from
2026; they were read at abstract level only.

## 0. Budget arithmetic (what the targets demand of one decode window)

Baseline is train1-imm 3edbc4d. Window times are the markers-only lane-round p50; tok/s are the quick bench.

| Load | Window ms | Tokens/window | Bench tok/s | Target | Window needed at today's tokens |
|---|---:|---:|---:|---:|---|
| C1 | 27.4 [M] | 2.03-2.27 [M] | 82.6 | 100 | ~22.7 ms (-4.7) |
| C4 | 39.4 [M] | 6.85 [M] | 151 (markers 174) | 300 | ~20 ms markers (-19), or more tokens |
| C8 | ~62 [I, from 192 agg, 13.7 tok and the C4 bench/markers ratio 0.87] | ~13.7 [I] | 192 | 500 | ~24-27 ms |
| prefill 2k | 4.07 s per 2048 tokens [M] | - | 503 | 1500 | 1.37 s per 2048 tokens |

Where a C1 window goes (card 0, 27.4-31.0 ms traced) [M]:
- Kernels: 21.1 ms over 1750 launches.
- Idle: launch gaps 3.1, host CPU 3.1, unknown 2.6, tracer 0.9, peer wait 0.17.

The same kernels joined against the bytes model, C1 / C4 per card [M time, I bytes]:

| Kernel group | C1 measured ms | C1 ideal ms | C4 measured ms | C4 ideal ms |
|---|---:|---:|---:|---:|
| Routed experts (fn_q8_up_*, fn_q8_down_*) | 4.1 | 3.0 | 8.3 | 5.7 |
| HC readout (down + upmix + writeback_norm) | 4.54 | 2.4 | 5.81 | 2.3 |
| All-reduces (tp2_allreduce_f32) | 2.18 | ~0 | 2.48 | ~0 |
| Router + top-k + compact | 1.72 | 0.23 | 1.97 | 0.22 |
| Small dense / glue (see note) | ~5.1 | ~1.2 | ~6.0 | ~1.5 |
| Q6_K / Q5_K dense | 2.1 | 1.4 | 2.9 | 1.4 |

Efficiency and launch counts behind that table:
- Routed experts: 72-91% of the bandwidth roofline.
- HC readout: 41-67%. fn_hc_down at C4 is 41%, because it re-reads its weights once per 4-row grid row.
- Router: 106 calls at 24-35%.
- Small dense / glue covers IQ4_XS, IQ3_S, IQ4_NL, BF16 k2560, Q4_K, q8_quant, gdn_front/rows, attention, scatter and combine. These kernels sit at 5-45%. 1036 launches are shorter than 10 us.
- Q6_K / Q5_K dense: 46-75%.

What follows from that:
1. The C1 gap to 100 is about the size of HC plus glue plus the all-reduce excess. Experts are not the C1 problem.
2. C4 300 needs about half of today's window, so nearly every lever below has to land, plus more tokens per window.
3. At C8 the expert union dominates the bytes. R = 16 rows touch E(16) = 69.3 of 256 experts per rank per layer
   [I, uniform routing], which is 5.85 GB per card per window (9.6 ms at 608 GB/s). At R = 16 the dp4a path also turns
   ALU-heavy: 58 G dp4a per card per window is 5.1 ms of pure dot ALU before any IQ decode [I: 4096 lanes x 2.8 GHz].
   Strata independently found its expert kernels "limited by IQ decode arithmetic per entry, not bytes"
   (docs/20261016_night_lab.md:87-89).
4. Prefill: at chunk 512, a 16-row-tile expert path would stream about 187 GB of expert weights per card per chunk
   (32 tiles x 69 experts x 48 layers x 1.76 MB). An expert-sorted grouped GEMM reads each expert once, 21.6 GB [I].
   The ~9x difference is the shape of the prefill gap. The prefill ledger needs to be redone on aa66709 to confirm
   how the merged PF path actually tiles.

Expert union per rank, E(R) = 256 * (1 - (1 - 10/512)^R) [I]. Expert MB per card is E x 48 layers x 1.76 MB.

| R | 1 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E(R) | 5.0 | 19.4 | 37.4 | 54.0 | 69.3 | 96.5 | 119.8 |
| Marginal experts per added row | 5.0 | 4.7 | 4.4 | 4.0 | 3.7 | 3.2 | 2.7 |
| Expert MB per card | 420 | 1640 | 3160 | 4560 | 5850 | 8160 | 10120 |

The marginal byte cost of an extra verify row falls with R. That is the MoE-specific reason to draft deeper at
C4/C8 once kernels hold their bandwidth at R = 16-32 (MoESD 2505.19645, 2610.38981, 2610.34440).

## 1. Ranked top 15

Effects are per card per window unless they are written as a percentage of tok/s. All are [I] estimates from the
measured ledger and the bytes model. They are not additive: overlaps are noted in the item notes.

| # | Idea | Owner | Cost | C1 | C4 | C8 | Prefill |
|---|---|---|---|---|---|---|---|
| 1 | HC readout: Q8_0 weights + writeback/norm fused into the down prologue + split across cards + one weight read per window | kernel | L-M | -2.5..-3.4 ms (+10-14%) | -3.7..-4.7 ms (+10-13%) | -5..-6 ms | HC is 48% of prefill kernels: same Q8 + DPAS |
| 2 | Launch-tax removal: horizontal fusion of same-input projections + vertical fusion of glue ops (1750 -> <900 launches) | kernel+runtime | M | -2.5..-4 ms | -3.5..-5 ms | -4..-6 ms | small |
| 3 | Prefill: expert-sorted grouped DPAS GEMM, HC/dense on nGEN DPAS, larger chunks, half-chunk overlap, fp16 wire | prefill | H | - | - | (cold TTFT) | 503 -> 1500-2500 |
| 4 | Device-initiated all-reduce: peer store in the producer epilogue, flag-in-data, reduce in the consumer prologue, L2 weight prefetch during the wait | runtime+kernel | M | -1.3..-1.9 ms | -1.5..-2.2 ms | -1.7..-2.5 ms | -5..-10% |
| 5 | Multi-row DPAS decode (R >= 8) + cross-stream expert runs (one read per distinct expert per window) | kernel | H | 0 | -0.5..-1 ms | -3..-5 ms; enables depth 2-3 | shares kernels with #3 |
| 6 | Draft acceptance parity with Strata (matched p0 2.24 -> 2.50 tokens/window) | model/runtime | L-M | +10-12% | +4% | +4% | - |
| 7 | Host-free async window pipeline: device lane table, device accept, prepare N+1 while N runs | runtime | M | -1.5..-2.5 ms | -2..-3 ms + bench/markers gap 13% -> ~5% | same | - |
| 8 | MoE front end in one kernel: router GEMV (64 B loads) + top-k + compact + plan | kernel | L-M | -1.2 ms | -1.4 ms | -1.5 ms | some |
| 9 | Batch- and MoE-aware adaptive draft depth per stream (confidence-gated, measured cost table) | runtime | L-M | +0-5% | +7-16% | +10-15% | - |
| 10 | GDN verify with replay records; commit folded into the next verify | kernel | M | -0.1..-0.2 ms | -0.4..-0.8 ms | -0.8..-1.5 ms | - |
| 11 | Activated-expert balance across cards: profile placement + hot-expert replicas | runtime | M | -0.3..-0.5 ms | -0.4..-0.7 ms | -0.5..-0.8 ms | small |
| 12 | Dense GEMV retune: load-time SoA repack (scale-planar, 64 B lane loads) + per-(type, rows) tile table | kernel | M | -0.5..-0.8 ms | -1..-1.5 ms | -1.5..-2 ms | - |
| 13 | SuffixDecoding-style model-free drafts beside MTP | runtime | M | 0 on the bench; large on agent loops | small | small | - |
| 14 | Data-parallel MTP drafter replica per card (no cross-card hops in draft steps) | runtime | M | -0.3..-1 ms | -0.5..-1.5 ms | -1..-2 ms | - |
| 15 | (LOSSY, opt-in) batch-aware expert sharing / selection (OEA, XShare, BASE, Lynx) | kernel+model | M | 0 | -5..-10% expert bytes | -25..-35% expert bytes, about +10-15% tok/s | - |

### Item notes: sources, mechanism, arithmetic

**1. HC readout.**
- Sources:
  - Strata `STRATA_HC_Q8=1` (on in the DD config): engine/sycl/src/.../native_dense.cpp:35-81 builds the Q8_0 copies at
    load; fused_gr.dp.cpp:2128-2520 has gr_q8_down / gr_q8_up, with the contract in fused_gr.hpp:1-16. Paths are under
    /mnt/vm_8tb/b70/build/strata-0216/src.
  - vllm-xpu-kernels csrc/xpu/mhc/xe_2/mhc_fused_post_pre.cpp.
  - mHC https://arxiv.org/abs/2512.24880 and SimpleHC https://arxiv.org/abs/2611.05984: read the 4-stream residual once
    per boundary.
  - TensorFold docs/receipts/flashnext-readout-rows (tiled readout above 128 rows).
- Mechanism:
  - 13.2 MB of bf16 per sublayer becomes about 7.0 MB as Q8_0: int8 row-major quants plus planar fp16 scales; inject
    stays bf16.
  - The pending writeback (R' = R + bo * 2 * sigmoid(inj / hc)) and the RMS go into the down kernel's SLM prologue, as
    in Strata gr_q8_down. That removes fn_hc_writeback_norm: 106 launches, 0.96 ms [M].
  - The 4-row grid re-read in fn_hc_down_bf16_1 is fixed, so all rows share one weight pass (BYTES-hc2 finding 5).
  - Worker HS then splits the readout across ranks, riding an existing all-reduce.
- Arithmetic (C1):
  - Down + up is 3.58 ms [M]. At the same efficiency and 7.0/13.2 of the bytes that is about 1.9 ms.
  - Removing writeback_norm saves 0.96 ms, minus about 0.1 for the prologue.
  - The split halves the rest, to about 1.0-1.2 ms.
  - Net -2.5 (Q8 + fusion) to -3.4 ms (with the split).
- Arithmetic (C4):
  - 5.81 ms falls to about 1.1-2.1 ms, i.e. -3.7..-4.7.
- Cautions:
  - Strata's Q8 HC kernels run at 39% of roofline. Copy its byte layout, not its tiling: our bf16 kernels are already
    at 63-67%.
  - Quality gate: teacher top-1 and KL against the bf16 path. Strata serves Q8 HC as the daily driver, which is
    encouraging but is not our gate.

**2. Launch tax.**
- Sources:
  - QuixiCore-XPU perf/findings.md: fusion wins scale with launches removed (5 to 1 gave 3.4-4.3x; 2 to 1 gave
    1.5-1.7x).
  - Strata night lab: per-kernel ramp/drain about 1.5-2 us, eager tiny launch 3.3 us.
  - llama.cpp ggml-sycl mmvq.cpp `has_fusion` (gate+up through a second weight pointer).
  - vLLM merged QKV / gate_up linears.
  - Megakernel line, for the principle: Hazy "No Bubbles" https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles,
    MPK https://arxiv.org/abs/2512.22219, MegaOps https://arxiv.org/abs/2611.16248, and "Understanding GPU
    Megakernels" https://arxiv.org/abs/2611.00495 (PDL-style separate kernels come within 5.7% of a megakernel).
- Mechanism:
  - Horizontal: one launch per input vector for all projections reading it. GDN: qkv + z + beta + alpha. QSA: q + k + v
    + gate + indexer. A segment table carries the quant type per row range; this is the same multi-matrix GEMV idea as
    the expert grid.
  - Vertical: q8_quant into the producer epilogue (196 launches [M]); gdn_front into gdn_rows; moe_combine into the
    down epilogue; shared SwiGLU into shared up; norms into the next prologue; descriptor scatter into the consumer.
- Arithmetic (C1):
  - The glue kernels measure about 5.1 ms against about 1.2 ms ideal [M/I].
  - Launch gaps are 3.1 ms over about 1750 launches, about 1.8 us each [M].
  - Halving the launch count and recovering half of the small-kernel tax gives -2.5..-4 ms.
- Arithmetic (C4): about 6.0 vs 1.5 ms, plus 3.7 ms of gaps, gives -3.5..-5 ms.
- Overlaps with #8 and #12; the router and Q6_K are excluded here.

**3. Prefill.**
- Sources:
  - Strata night_grouped_expert.inc:230-321 and prefill.cpp:430-495: experts with up to 64 rows sorted into one launch
    per projection; joint_matrix 8x16x16 fp16 DPAS after register/SLM decode; NIGHT_PREFILL_HALVES (prefill.cpp:437-440,
    5880-5894) for overlap; fp16 psum return `ep_prefill_reduce_h`. It reaches about 2200 tok/s on this box.
  - Upstream TensorFold zig/tools/xpu/w4_prompt.hpp and w4a8_prompt.hpp: nGEN 32x32 per subgroup, 64x256 per
    work-group, 146 TFLOPS fp16 sustained, about 209 TOPS int8 [L, B70].
  - llama.cpp ggml-sycl fused-gemm-esimd.cpp: dequant in registers straight into the DPAS A operand, no SLM, B packed
    VNNI.
  - QuixiInfer, same model on a 5090: 64-row expert tiles, 1519-1547 tok/s.
  - MegaBlocks https://arxiv.org/abs/2211.15841, ScatterMoE https://arxiv.org/abs/2403.08245, Layered Prefill
    https://arxiv.org/abs/2510.08055 (chunking re-reads expert weights, +39% traffic), ISO https://arxiv.org/abs/2409.11155
    (-35% prefill on PCIe 4090s).
- Arithmetic:
  - A 512-token chunk needs at least 21.6 GB of expert weights per card (35.6 ms) and about 3 TFLOP per card
    (6B active).
  - The 1500 tok/s target allows 341 ms per 512 tokens. Bandwidth and compute leave more than 5x headroom; the
    remaining gap is kernel structure.
  - Exchange: 96 x 5.2 MB fp32 per 512 rows is 0.5 GB per chunk, 36-45 ms at 11-14 GB/s. That is 11-13% of the
    target budget; fp16 halves it and the half-chunk overlap hides most of the rest.
- Do not: chunkwise GDN (Strata measured FP32 chunkwise 4.4x slower than its scan; GDN is 0.4% of prefill), or write
  host-mapped memory from kernels on the critical path (Strata kv_append, 385 ms per call).

**4. Device-initiated all-reduce.**
- Sources:
  - Strata ep_kernels.dp.cpp and ep_mem.hpp:1-54: kernel peer stores, system-scope release fence, the last work-group
    stores the flag in the same kernel, uncached reads of inbound data, device-memory epoch counters, bounded spins.
  - flash-comm https://arxiv.org/abs/2611.06634 (flag-in-data, -28.8% all-reduce latency).
  - SiFAR https://arxiv.org/abs/2607.08973 (dual buffering removes the bottom barrier).
  - tt-metal models/demos/gpt_oss/tt/fused_decode (boundary op).
  - PRESERVE https://arxiv.org/abs/2501.08192: prefetch the next weights into L2 during the all-reduce, up to 1.6x [L].
  - Hazy TP megakernel https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro.
- Mechanism:
  - The producer GEMV epilogue writes its partial straight into the peer's inbox and publishes a generation flag; no
    copy engine, no separate push command.
  - The consumer prologue (HC down / next norm) polls the flag and sums in fixed rank order, so the replicated
    streams stay bit-identical.
  - While it spins, idle work-groups touch the consumer's own weights (HC down Q8 about 3.5 MB, router 2.7 MB) to warm
    L2.
- Arithmetic (C1):
  - Today 102 hops x 21 us of AR kernel = 2.18 ms, plus 0.8-1.2 ms of device-to-peer copy time [M].
  - Floor: one-way 8-12 us (TRANSPORT-B; 10 KB push 7.9 us).
  - 21 -> 11-13 us per hop gives -0.8..-1.0 ms, plus about 0.2 ms of launch gaps.
  - The L2 warm-up hides 6-10 us of the next kernel's DRAM time per hop: -0.5..-0.9 ms.
  - Total -1.3..-1.9 ms. This is codex-arfuse's target; the prefetch half is new.
- Caveat: the B70 L2 size is unconfirmed (G21 is 18 MB). Read it with zeDeviceGetCacheProperties before sizing the
  prefetch.

**5. Multi-row DPAS decode.**
- Sources:
  - QuixiInfer kernels/gemm.cu qx_tmv: bf16 tensor-core matvec for up to 16 tokens; it wins over mgemv above about
    4 tokens.
  - llama.cpp ggml-sycl `ggml_sycl_mul_mat_vec_q_xmx` (mmvq.cpp:2123) and ggml-cuda mmq-few.cu (int8 MMA for 2-24
    columns, one column group per expert).
  - exllamav3 exl3_moe_coop_kernel.cuh: rows that picked the same expert form a "run", one MMA tile per run.
  - MARLIN https://arxiv.org/abs/2408.11743: near-ideal speedup up to batch 16-32.
  - FLUTE https://arxiv.org/abs/2407.10960: LUT/codebook dequant, and IQ grids are codebooks.
  - QServe https://arxiv.org/abs/2405.04532: dequant ALU is the batch-serving bottleneck.
- Mechanism:
  - Above about 8 rows, decode each weight tile once (IQ grid lookup into int8 or fp16 registers) and feed DPAS:
    int8 8x16x32 where weight sub-blocks are 32 (IQ3/IQ2/IQ4_NL/Q2_0), fp16 8x16x16 for Q6_K (16-element scales).
  - Group rows by distinct expert across all streams, so each expert's weights are read once per window. KF8 is doing
    the grouping half.
- Arithmetic:
  - dp4a dot cost scales with rows: R = 16 is 5.1 ms and R = 32 is 17.7 ms of pure dp4a per card per window, against
    9.6 / 16.7 ms of bytes [I].
  - DPAS at 367 TOPS makes the dots about 0.2 / 0.6 ms.
  - Without this, C8 at depth 2-3 cannot stay bandwidth-bound.
- Expected:
  - C8 experts move from about 55-65% to about 80% of roofline: -3..-5 ms.
  - C4 (R = 8) gains little.
  - C1 keeps the dp4a GEMV.

**6. Draft acceptance parity** (local engineering; no paper fixes this).
- Measured gap, matched p0 (STATUS Lead A 04:45): per-draft acceptance 0.69/0.95/0.92/0.72 for Strata against
  0.62/0.88/0.88/0.69 for us; 2.50 vs 2.24 tokens per window.
- Strata's own prompt gives 3.12 tokens per window at --spec 4 (concurrency-0216 LEDGER).
- I checked two suspects and ruled them out: Strata's per-stream pre_fc_norm_hidden (`--mtp-hnorm stream`) and
  `STRATA_MTP_CATCHUP_ALL` are both opt-in and off in hotschmoe-dd-0216.json. Both engines use one RMS over 10240
  (our flashnext.cl:58).
- Remaining diffs to bisect:
  - which window rows feed the drafter K/V after a partial reject (mtp.cpp catch-up path);
  - the exact hidden tensor handed to fc_hidden (pre or post final mixer);
  - drafter projection precision (Strata Q8_0 projections vs our bf16; the experts are the same q2_0);
  - draft-vocab argmax tie rules;
  - q8_1 activation rounding in the drafter.
- Method: dump the drafter logits for the same verified prefix on both engines and compare top-1 per depth.
- +12% tokens per window at C1 is worth about +10 tok/s at zero kernel cost.

**7. Host-free async windows.**
- Sources:
  - QuixiInfer docs/progress.md:20-160: `qx_lanes_take`, the first kernel copies the step's lane table from a mapped
    ring; free-running steps; step N+1 queued before N is read (+5% at c=16).
  - PolyGraph https://arxiv.org/abs/2610.23861: one max-shape recorded graph plus a 28-byte device plan, unused tiles
    skipped. This is the robust form of R's rec2: one recording for every bucket and depth.
  - vLLM async scheduling.
- Arithmetic:
  - Non-busy time per C1 window is 6.3 ms [M]: host CPU 3.1-3.6 + launch 3.1, part of which is shared with #2.
  - C4 bench is 13% below markers (151 vs 174).
  - Expected -1.5..-2.5 ms at C1, and the bench gap recovered toward about 5%.

**8. MoE front end.**
- Sources:
  - MonoMoE https://arxiv.org/abs/2609.04244: router, top-k, quant, both projections and the reduction in one
    persistent kernel; 1.54x over Triton grouped GEMM [L].
  - llama.cpp ggml-sycl topk-moe.cpp.
  - tt-metal router_topk.hpp.
  - llm-scaler moe_decode_gemv.h: 16 B 2D loads capped at 316 GB/s; 64 B 1D loads reach about 574 GB/s [L, Xe2].
- Arithmetic:
  - fn_router_dpas_1 + fn_route_topk + fn_desc_moe_compact = 1.72 ms at C1 and 1.97 ms at C4 [M].
  - The router is 2.7 MB bf16, 4.4 us ideal. At 70% plus a fused top-k/compact tail that is about 0.5 ms per window.
- The router must stay bf16 and identical on both cards.

**9. Adaptive draft depth.**
- Sources:
  - Cascade https://arxiv.org/abs/2506.20675: naive MoE speculation slows down up to 1.5x; utility-based K gives
    +7-14%.
  - MoESD https://arxiv.org/abs/2505.19645.
  - https://arxiv.org/abs/2610.38981: draft rows co-route; L2 absorbs part of the cost.
  - https://arxiv.org/abs/2610.34440: step time = bytes / effective bandwidth + fixed time; the best depth is a
    staircase in acceptance.
  - TurboSpec/SmartSpec https://arxiv.org/abs/2406.14066.
  - EAGLE-2 https://arxiv.org/abs/2406.16858: draft confidence tracks acceptance.
  - SGLang speculative/online_acceptance_estimator.py and adaptive_spec_params.py.
  - QuixiInfer drafter.rs `pick_width`.
- Arithmetic:
  - C4 depth 1 -> 2 adds 4 rows: +16.6 experts per rank per layer = 1.4 GB, about 3.1 ms at 75%; plus about 0.6 ms
    of per-row work and one more chained MTP step (host.fn.mtp is 3.6 ms per window today [M]).
  - Tokens per stream go 1.71 -> about 2.33 (+36%).
  - Net +7-16%, depending on how cheap the draft step becomes.
  - At C8 the marginal is lower (3.2-3.7 experts per row), so depth 2-3 wins more, but only with #5.
- Strata measured depth 2 at C4 as slightly worse (49.6 vs 51.2 per stream). Hence: gate per stream on draft
  confidence; do not raise depth for everyone.

**10. GDN replay records.**
- Sources:
  - ReplaySSM https://arxiv.org/abs/2610.13778: cache inputs, not state; about half the state traffic; 1.20-1.47x [L].
  - HybridSpec https://arxiv.org/abs/2611.04055: Qwen3.6-35B-A3B verify grows 71% from length 4 to 16, mostly GDN and
    MoE.
  - vllm-xpu-kernels gdn_attn/gated_delta_rule_recoverssm.hpp: verify never writes S; about 25 KB of records per
    token per layer.
  - QuixiInfer layer.cu:96-106 qx_gdn_recur: state in registers across the window.
  - Strata verify.cpp:1288-1356 / 2520-2560 (replay commit).
- Arithmetic:
  - -113 MB per stream per window, 0.19 ms per stream at roofline: C4 -0.75, C8 -1.5 ms.
  - Check first: the desc path's fn_desc_commit is only 0.065 ms at C4 [M], so part of the BYTES-hc2 "keep" traffic
    may already be gone.
- Do not store GDN state in low precision: https://arxiv.org/abs/2610.15139 shows the error compounding.

**11. Expert balance.**
- Sources:
  - METRO https://arxiv.org/abs/2512.09277: balance activated experts per GPU, not tokens; -11-22% decode latency vs
    EPLB [L].
  - EPLB https://github.com/deepseek-ai/EPLB.
  - Strata generate.cpp:5205-5248 (snake placement by expert profile).
- Arithmetic:
  - Under uniform routing with the contiguous 256/256 split, the busier rank holds 12.3% / 8.7% / 5.6% more distinct
    experts at R = 4 / 8 / 16 [I, simulation]. Today that cost hides inside the all-reduce kernel time.
  - 0.3-0.8 ms per window. Real routing skew can make this bigger; measure it from the router outputs.

**12. Dense GEMV retune.**
- Sources:
  - llama.cpp ggml-sycl mmvq reorder (SoA quants and scales; `rows_per_sg = 2` wins on B70 at ncols >= 3).
  - Strata native_mmvq.dp.cpp:2352-2371 KREG tile table (Q6_K R4 up to 3 columns, R2 beyond; IQ4_XS R8 at 1 column).
  - QuixiCore: GGUF 18-210 B blocks defeat coalescing.
- Q6_K c8 runs at 46% and IQ4_XS c8 at 30% [M].

**13. Model-free drafts.**
- Sources:
  - SuffixDecoding https://arxiv.org/abs/2411.04975: up to 5.3x on agentic workloads [L].
  - Strata engine/src/ngram (off in the DD config).
- Effect: zero drafter cost. It will not move the fixed-prompt bench, but the user's Pi / autoresearch loops repeat
  heavily. Run it as a second chain candidate whenever a suffix match is at least 4 tokens long.

**14. Data-parallel MTP drafter.**
- The MTP layer is about 0.82 GB (116 MB dense + 708 MB q2_0 experts), which fits on each card.
- At C >= 2 each card drafts for its own streams, with no all-reduce in the draft chain and half the rows per card.
- At C1 the gain is only the hops.
- Lead A H4 owns the lane policy; this changes placement.

**15. LOSSY expert sharing.**
- Sources:
  - OEA https://arxiv.org/abs/2511.02237: -39% / -15% MoE-layer decode latency at batch 16, Qwen3-30B / 235B, no
    significant accuracy loss [L].
  - XShare https://arxiv.org/abs/2602.07265 (+14% throughput under speculative decoding).
  - BASE https://arxiv.org/abs/2609.36222.
  - Lynx https://arxiv.org/abs/2411.08982.
- Effect: at C8 the experts are about 50-60% of the bytes, so -30% union is about +10-15% tok/s. Changes outputs:
  user decision, quality gate, default off.

## 2. Concrete sketches for the top five

Engine paths are relative to /mnt/vm_8tb/b70/build/tensorfold/fn-a3.

### 2.1 HC readout (item 1), kernel owner, about 2-3 worker-days

1. Loader:
   - In zig/src/families/flashnext/xpu_gguf_model.zig (HC upload, `.replicated`, about :141-153), quantize hc down,
     up and the final mixer to Q8_0 at load.
   - Use the ggml rounding (Strata native_dense.cpp:35-81) and lay it out as int8 quants row-major followed by fp16
     scales (fused_gr_q8.hpp).
   - Keep the inject rows bf16.
   - Behind XPU_FN_HC_Q8=1, default off until gated.
2. Kernels:
   - Add fn_hc_down_q8_R and fn_hc_upmix_q8_R to zig/kernels/xpu/flashnext/fn_dpas.cl, next to fn_hc_down_bf16_1 /
     fn_hc_upmix_bf16_1 (about :457-581).
   - Every row of the window (R <= 16) shares one weight pass: no per-4-row grid dimension.
   - The down prologue applies the pending writeback and the RMS, i.e. the body of fn_hc_writeback_norm in
     flashnext.cl, staged in SLM per 512-column slab as Strata gr_q8_down does.
   - Lanes read 16-64 B contiguous along K.
3. Dispatch: zig/src/families/flashnext/xpu_ops.zig (about :100-107) and xpu_store.zig (kernel names) choose q8 when
   the flag is set. Drop the fn_hc_writeback_norm launch.
4. HS split (worker HS, b70/fn-hs): rank r owns half of the HC output columns. The existing next all-reduce carries the
   other half, so there is no extra hop.
5. Gates:
   - Teacher top-1 / KL vs bf16 HC over 2420 positions (B3-3 harness).
   - p0 tokens; C4 streams == solo.
   - Ledger regen: HC phase ms per card at C1 and C4.
   - Target: C1 HC phase <= 2.0 ms without the split, <= 1.2 ms with it.

### 2.2 Launch-tax removal (item 2), kernel + runtime, about 3-5 worker-days, incremental

1. Inventory: from LEDGER-train1-imm.json windows.*.kernels, list every launch shorter than 10 us with its producer and
   consumer (1036 at C1 [M]). Rank fusion pairs by launches removed.
2. Horizontal:
   - In the project() path of zig/src/families/flashnext/xpu_ops.zig (linearQ per matrix; BYTES-hc2 finding 6), replace
     the per-matrix fn_q8_linear_<type>_c{N} launches with one fn_q8_multi_c{N} launch.
   - Its argument is a segment table {row_begin, row_end, type, weight ptr, out ptr} per projection sharing the input
     (GDN: qkv, z, beta, alpha; QSA: q, k, v, gate, indexer).
   - Generate the type switch per (type set) at build time in fn_window_generated.cl. Keep per-row arithmetic identical
     so outputs stay row-invariant.
3. Vertical:
   - fn_q8_quant (fn_q8_common.cl) moves into the epilogue of whichever kernel produces the activation.
   - fn_desc_gdn_front goes into fn_desc_gdn_rows (fn_window_generated.cl).
   - fn_moe_combine goes into the down-projection epilogue.
   - fn_gguf_shared_swiglu goes into shared up.
   - fn_rmsnorm_rows / fn_gdn_norm go into consumer prologues.
   - fn_desc_scatter is removed by writing outputs at their final address.
4. Runtime: the descriptor windows (XPU_FN_DESC default) already carry per-window plans. Extend the descriptor so each
   fused kernel reads its segment and row tables on the device, with no per-launch kernel-argument API calls (the KF7
   cached args are the start).
5. Gate:
   - Bitwise == unfused for every fused pair (unit tests in the tf-xpu test set), then the quick bench.
   - Metric: launches per window per card and the ledger's launch_gap.

### 2.3 Prefill (item 3), prefill owner, about 1-2 worker-weeks

1. First, regenerate the prefill critical path on aa66709: B/prefill/CRITICAL_PATH.md method, chunk 512 and 2048, per
   kernel ms and per-launch weight bytes. Confirm whether experts are read once per chunk or once per row tile; the
   187 vs 21.6 GB question.
2. Routing plan:
   - On the device, build an expert-sorted CSR per layer and rank: count, then prefix-sum, then scatter, without
     integer division (vllm-xpu moe_shared_fused_decode_xe2.cpp map build; ggml-sycl GGML_SYCL_MMID_SCHED).
   - Reuse fn_desc_moe_compact's output format, so decode and prefill share one plan type.
3. Kernel:
   - One launch per projection over all experts of a rank. The grid is (expert, N/64); surplus work-groups exit.
   - Each 32-K step decodes IQ/Q2_0 blocks once into registers, as the fp16 DPAS A operand (ggml-sycl
     fused-gemm-esimd.cpp design), and issues 8x16x16 DPAS. M tiles are 8/16/32/64 by the expert's row count.
   - Gate+up share the activation tile, with SwiGLU in the epilogue.
   - Down writes weighted rows into the combine buffer.
   - Write it in nGEN, next to upstream w4_prompt.hpp, or in OpenCL C with intel_sub_group_* DPAS builtins.
   - Test fp16 denormal handling before using the subnormal trick (JOURNAL notes an IGC 26.22 flush suspicion).
4. HC and dense at M >= 128: route through the same nGEN DPAS GEMM with Q8 HC from 2.1 converted to fp16 tiles. PF's
   d8baa8d gating by rows is the switch point.
5. Exchange:
   - Send fp16 partials with a power-of-two scale (Strata ep_prefill_reduce_h).
   - Split each chunk into halves A/B and interleave per layer, so the B half's attention/GDN runs while the A half's
     all-reduce is in flight (Strata NIGHT_PREFILL_HALVES).
   - Raise XPU_CHUNK_SOLO from 512 to 2048-4096, so each expert sees 40-80 rows per chunk.
6. Gate: first-token top-1 / KL vs the current prefill, chunk-size invariance (512 vs 2048 on the same tokens),
   prefill tok/s at 2k / 8k / 32k.

### 2.4 Device-initiated all-reduce + L2 warm-up (item 4), runtime + kernel, about 3-4 worker-days

1. Memory:
   - Per rank, an inbox ring of 2 parity slots x MAX_ROWS x 2560 fp32 plus a 64 B flag line per slot, each slot its
     own zeMemAllocDevice opened on the peer through IPC (Strata ep_mem.hpp pattern; zig/src/xpu/tp.zig already opens
     peer memory for the push transport).
   - Generation counters live in device memory and are bumped by the first kernel of a window, so the scheme also
     survives future recording.
2. Producer: the out-projection / expert-down kernels whose result feeds an all-reduce get an epilogue that stores the
   partial rows to the peer inbox (posted PCIe writes), then:
   - a system-scope release fence;
   - the last work-group, via a local atomic counter, stores flag = generation.
   There are no peer atomics on this link.
3. Consumer:
   - The next prologue (fused HC-down from 2.1, or the next norm) spins on flag == generation with uncached loads,
     then sums local + peer in fixed rank order, exactly as tp2_allreduce_f32 does in zig/kernels/xpu/tp.cl, so the
     outputs stay bitwise equal.
   - Work-groups that are not needed for the poll issue strided loads over the consumer's own weights (3.5 MB HC Q8,
     2.7 MB router) to warm L2.
   - Size this from zeDeviceGetCacheProperties.
4. Remove the separate memory_copy push and the tp2_allreduce_f32 launch: 102 + 102 commands per window [M].
5. Gate: drafted == serial, C4 == solo, a 5000-window soak at shape cycling (R5 microrepro harness), and bounded
   spins that write an error word to host-mapped memory.

### 2.5 Multi-row DPAS decode + cross-stream expert runs (item 5), kernel owner, about 1 worker-week

1. Plan: KF round 8's grouping by distinct expert (one entry list per expert per window, all streams and draft rows)
   is the input. Extend the plan with per-expert row counts, so the kernel picks GEMV (rows <= 4) or DPAS (rows > 4).
   QuixiInfer qx_tmv measured the crossover at about 4-8 rows.
2. Kernel:
   - New fn_q8_up_<type>_xmx / fn_q8_down_<type>_xmx in a new zig/kernels/xpu/flashnext/fn_moe_xmx.cl.
   - One sub-group owns 16 output rows of one expert over K.
   - Decode the IQ grid / Q2_0 values once per 32-K block into int8 VNNI registers and multiply against the q8_1
     activation rows (M = rows of that expert, padded to 8) with int8 DPAS 8x16x32.
   - Apply weight-scale times activation-scale per 32-block in fp32.
   - IQ grids are codebooks: keep the grid tables in SLM or registers (FLUTE) and compute signs with shift masks, not
     tables (Strata night_grouped_expert.inc:22-24).
3. Dense: the same path for fn_q8_linear_*_c8, with a c16 variant, so 16-row windows read the weights once, not twice
   as today's <= 8-row chunks do [M BYTES-hc2 kernel choice].
4. Row invariance: a row's bits must not depend on how many rows share the tile (TensorFold NAVIGATION section 5).
   Pick one arithmetic order for both GEMV and DPAS, or accept a documented non-bitwise mode like DPAS prefill. For
   decode, prefer invariance: always run DPAS for an expert once a window has more than 4 rows on it, independent of
   the other streams.
5. Gate: per-type fp64 parity, concurrent == solo, then C8 at depth 1 / 2 / 3 on the quick bench.

## 3. Tempting, but will not help here (or not yet)

| Idea | Why not, for this model and this box |
|---|---|
| Full megakernel / persistent whole-window kernel (Hazy, MPK, Event Tensor) | Strata built a persistent single-kernel HC with a grid barrier: slower than a kernel boundary, and cooperative launch fails inside SYCL graphs (night lab :240). There are no peer atomics across cards (cookbook ADR-0009: per-card persistent kernels only). 2611.00495 finds separate kernels with dependent launch within 5.7% of a megakernel, and 2610.35336 finds 2-4 megakernels beat one. Fusion plus host-free descriptors (#2, #7) gets most of it. |
| Expert offload / caching (MoE-Infinity, Fiddler, ProMoE, HOBBIT, Pre-gated MoE, ktransformers, QuixiInfer tiers) | All experts are resident: 23 GB per card, about 8.5 GB free. A host-RAM expert over Gen3 x16 (about 13 GB/s H2D) is 45x slower than VRAM. |
| All-reduce compression / quantized collectives (2411.09510, Flash Communication 2412.04964, 2602.21144) | Decode messages are 40-80 KB and latency-bound (one-way 8-17 us); bytes do not matter. Only prefill benefits (fp16 wire, inside #3). |
| Ladder Residual (2501.06589), Kraken (2408.07802) | Change the architecture; need retraining. |
| Two-batch overlap / NanoFlow / TokenWeave token split at decode | Each micro-batch re-reads the row-independent weights: HC, dense and head, about 2.9 GB per window, about +4.8 ms per extra pass, to hide about 2 ms of all-reduce. Use it only in prefill (#3). |
| Tree speculation at C1 (EAGLE-2 trees, SpecInfer, Sequoia, STree) | Each extra verify row costs about 4.7 distinct experts per rank (about 0.4 GB, 0.65-0.9 ms per card), and GDN needs per-node state (TreeWY 2608.20961 solves that, at a cost). Chain + confidence depth (#9) + acceptance parity (#6) first. Revisit if top-2 acceptance at depth 1 is above about 0.3. |
| Strata-style EP (all dense on card 0, experts split) | Card 1 spins 17.7 ms of a 22 ms window in Strata [M census]. Our TP2 already beats Strata at C4 / C8 (192 vs 119-137 C8 aggregate). EP has two one-way messages per layer, so it does not remove hops either. |
| PP2, DP attention, full-model DP | Weights (47 GB) do not fit one card. PP re-reads weights per micro-batch. |
| LUT-GEMM / T-MAC | CPU or compute-bound regimes. Our decode is bandwidth- plus launch-bound; IQ is already a codebook, and the useful FLUTE lesson (tables in SLM/registers) is inside #5. |
| Low-precision GDN state (bf16 / int8 / int4) | Recall loss compounds over about 1000 steps (2610.15139). Use replay records (#10) or ReplaySSM's two-precision scheme instead. |
| Re-enabling recorded windows as-is (XPU_FN_RECORD) | Measured no longer a win after evpool + device MTP: C4 -10%, C8 -15% (STATUS 04:16). It returns only as a max-shape plan-buffered window (PolyGraph, #7) or R's epoch transport. |
| More bandwidth tuning of the routed-expert kernels at C1 / C4 | Already 72-91% of roofline [M]; at most -0.5 ms. The kernel work belongs at R >= 8 (#5). |
| Chunkwise (WY) GDN prefill | Strata measured FP32 chunkwise 4.4x slower than its scan on B70; GDN is 0.4% of prefill. |
| Prefix-cache policy work (Marconi, Jenga) | Helps repeat TTFT, which is already 25 ms from the RAM tier, not decode or prefill throughput. |
| Copy-engine vs EU-store tuning for the existing push | Both land within a few us of the 8-12 us latency floor; the win is removing the command and the separate AR kernel (#4). |
| Marlin / Machete code ports | CUDA-specific; nGEN (upstream TensorFold w4_prompt) and ggml-sycl ESIMD are the Xe2 equivalents. |

## 4. Rough stacking (inferred; for planning, not a claim)

| Load | Today | Levers | Projection | Verdict |
|---|---|---|---|---|
| C1 | 27.4 ms | #1 -3, #2 -3, #4 -1.5, #8 -1.2, #7 -1.5 -> about 17-19 ms after a 30% overlap haircut | 2.27-2.5 tokens -> about 110-130 tok/s | 100 is reachable with #1, #2 and #6 alone |
| C4 | 39.4 ms | #1 -4.2, #2 -4, #4 -1.8, #8 -1.4, #7 -2, #10 -0.7, #11 -0.6, #5 -0.7 -> about 24 ms | 6.85 x 1.04 (#6) x 1.1 (#9) = 7.8 tokens -> about 325 markers, about 290-310 bench | Needs nearly all of them |
| C8 | ~62 ms [I] | Same set plus #5 -4 -> about 34-38 ms | 13.7 x 1.04 x 1.12 = 16 tokens -> about 420-470 markers, about 400-450 bench | 500 needs depth 2-3 at C8 on DPAS kernels with real (not uniform) intra-stream expert overlap, or the lossy #15 |
| Prefill | 503 tok/s | #3 | 1500 is about 15% of the bandwidth/compute bound | Gated only by kernel structure |

## 5. Measurements that would change this ranking (cheap, mostly CPU or one short lease)

1. Expert overlap among the verify rows of one stream, from router outputs: distinct experts at R = 4 / 8 / 16 vs
   uniform E(R). Real overlap above uniform makes #9 and #5 rank higher (2610.38981).
2. B70 L2 size and the CCS count (zeDeviceGetCacheProperties, zeDeviceGetCommandQueueGroupProperties). These size #4's
   prefetch and any second-queue overlap.
3. Prefill per-kernel ledger on aa66709: expert weight passes per chunk. This confirms or kills the 187 vs 21.6 GB
   arithmetic behind #3.
4. Routing skew per layer from Strata's expert profile (terminal/data/expert-profile-sycl.bin) against the contiguous
   256/256 split (#11).
5. Drafter logit diff vs Strata on identical verified prefixes (#6).

## 6. Main local sources

- Strata 0.2.16 (MIT), /mnt/vm_8tb/b70/build/strata-0216/src:
  - engine/sycl/src/kernels/cuda/{native_mmvq,iq_kernels,fused_gr,ep_kernels,verify_kernels}.dp.cpp and
    night_grouped_expert.inc
  - engine/sycl/src/core/{verify,mtp}.cpp, program/generate.cpp, prefill/prefill.cpp
  - include/strata/core/{ep_mem,mtp}.hpp, batch_depth.hpp
  - Evidence: docs/20261016_night_lab.md and strata/flash-next/concurrency-0216/LEDGER.md (in this repository).
- QuixiInfer, /mnt/cache/b70/refs/QuixiInfer: kernels/{layer.cu,mgemv.cu,gemm.cu}, crates/qx-engine/src/drafter.rs,
  docs/progress.md, perf/findings.md.
- QuixiCore-XPU, /mnt/cache/b70/refs/QuixiCore-XPU: perf/findings.md, perf/optimization_status.md, kernels/moe,
  kernels/linear_attention/qwen_gdn_decode, kernels/collectives.
- TensorFold upstream, /mnt/cache/b70/refs/TensorFold: zig/tools/xpu/w4_prompt.hpp, docs/xpu/NAVIGATION.md,
  docs/receipts/flashnext-*.
- Cookbook, /mnt/cache/b70/refs/intel-arc-pro-b70-inference-cookbook: docs/qwen38-27b/TENSORFOLD-TWO-CARDS-20261119.md,
  research/b70-docs/research/2026/b70-untried-levers-20261009/xe2-hardware.md, the ADR-0009 megakernel decision.
- Upstream code (GitHub HEAD, read via gh api):
  - llama.cpp ggml/src/ggml-sycl/{mmvq.cpp,fused-gemm-esimd.cpp,topk-moe.cpp} and ggml-cuda/{mmq-few.cu,mmid.cu}
  - vllm-project/vllm-xpu-kernels csrc/xpu/{gdn_attn,grouped_gemm/xe_2,mhc/xe_2}
  - intel/llm-scaler vllm/custom-esimd-kernels-vllm/csrc/moe_batch
  - intel/sycl-tla examples/{04,10,12}_bmg_*
  - turboderp-org/exllamav3 exllamav3_ext/quant
  - sgl-project/sglang python/sglang/srt/speculative
  - tenstorrent/tt-metal models/demos/gpt_oss/tt/fused_decode
  - HazyResearch/Megakernels
  - mirage-project/mirage include/mirage/persistent_kernel
- Campaign evidence: tensorfold/flashnext/{BRIEF,STATUS,DESIGN-A,TRANSPORT-B,JOURNAL}.md and
  /mnt/vm_8tb/b70/results/tf-flashnext/B/ledger/{BYTES-hc2,LEDGER-train1-imm}.{md,json}.
