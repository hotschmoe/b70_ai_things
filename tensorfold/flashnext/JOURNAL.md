# TensorFold Flash-Next two-card campaign journal

Format: [A]/[B] CONFIG -> COMMAND -> RESULT -> VERDICT. Append at the bottom.

## [B] 2026-11-20 06:50 UTC - Lead B start: plan and worker launch
CONFIG: Lead B = Claude Opus 5.5 (claude-opus-5-5). Worktrees from b70/fn-runtime @207aaae: fn-rt-trace, fn-rt-lat, fn-rt-means.
COMMAND: read BRIEF, AGENTS, probe README/JOURNAL, TF zig/src/xpu (ze.zig, dev.zig, rt.zig, tp.zig, tp_bench.zig, means_test.zig, gemm.cl).
RESULT (reading, inferred not measured):
- All TF XPU programs reach Level Zero through function-pointer tables (ze.Api via dlopen; context.zig Driver), so a
  tracer can wrap the tables at resolve time with zero call-site changes. Existing XPU_PROF syncs per launch (distorts);
  XPU_PROFILE sums per-kernel GPU time only. No timeline / wait attribution exists.
- means-test: rowsum_hl stores group means as fp16 hi/lo pairs scaled by ASC=256; the test's data puts lo*ASC in the
  fp16 subnormal range for |a|~1e-3, and losing lo entirely estimates to ~3e-5..6e-5 relative error, matching the
  measured 5.12e-5. Hypothesis: fp16 subnormal flush (convert, load/store or DPAS input) under our IGC 26.22. To test.
- tp.zig one-shot allreduce kernels PULL from the peer send buffer (3.24 GB/s class on this box) and sharedEvents uses
  signal/wait scope 1 (SUBDEVICE). Prior box evidence (docs/P2P_GPU.md K.3) used HOST_VISIBLE pools; scope is a
  candidate for the tp-bench GuC hang. To test with one bounded reproducer.
VERDICT: launch three workers in parallel (codex-trace M-B1, codex-lat M-B2, sonnet-means numerics); CPU parts first,
GPU after STRATA DONE, card 1 preferred for single-card work.

## [A] 2026-11-20 06:55 UTC M-A1 design + GGUF census
CONFIG: PR3603 207aaae (fn-model), GGUF ISTA GSQ-RCO IQ3_XXS shards 1+2, Strata 0216 evidence docs. CPU only.
COMMAND: header census with Strata's gguf_reader.py (copied to /mnt/vm_8tb/b70/build/tensorfold/fn-tools/census.py).
RESULT: shard 1 = 1223 tensors: routed gate/up IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S, routed down Q2_0(42)/IQ4_NL,
dense IQ3_S/IQ4_XS/Q4_K/Q5_K/Q6_K, HC/router/indexer/PLE-proj BF16, head Q5_K, embed IQ3_S; shard 2 = PLE IQ4_NL
[160 x 320,001,536] 27.5 GiB; no MTP tensors (drafter from Strata rt). TP2 per-card weights ~22.5 GiB [I], leaving
~8.5 GiB/card for KV/state; int8 KV required for 200K x 4.
PR3603 reading: TP forward queues all 48 layers without host round trips, 2 all-reduces/layer, MLX Q4 everywhere,
bf16 KV; host syncs once per verify + once per MTP draft step; card 0 pulls card 1's shard logits (peer read);
rows<=16 MoE kernels launch per (row, expert) pair (an expert shared by rows is re-read).
VERDICT: DESIGN-A.md v1 written: keep PR3603's TP2 shape (head-split GDN/QSA, expert-split MoE, replicated HC as Q8_0
copies), replace the transport with a push all-reduce (Lead B), EP fallback if hop > 40 us. Workers K (kernels) and M
(GGUF semantic map + oracle) launched.

## [A] 2026-11-20 07:09 UTC M-A2 step 1: CPU dequant reference (worker K, Codex gpt-6.1-terra xhigh)
CONFIG: fn-model b70/fn-model; commit 0be66a5 (gguf_quant.zig QType/QW/QE + dequantRow for F32 F16 BF16 Q8_0 Q4_K Q5_K Q6_K
IQ4_XS IQ4_NL IQ3_S IQ3_XXS IQ2_S IQ2_XS IQ2_XXS Q2_0(42); ggml IQ tables with MIT notice).
COMMAND: cpu-run + MemoryMax=90G: zig build test-fn-gguf-host; C oracle compiled against Strata third_party/strata-ggml
ggml-quants.c on the first 64 blocks of one real tensor per type (scripts under results/tf-flashnext/workers/K/).
RESULT: host tests pass; Zig dequant == ggml bit-exact for all 14 real types (Q8_0 synthetic: absent from shard 1).
Found: strata-gguf-py has no Q2_0 dequant (C oracle needed); TF core GGUF parser rejects IQ types + type 42 (own reader
needed); fn_moe_compact groups an expert's pairs in chunks of <= 32.
VERDICT: CPU reference accepted. Reviewed by Lead A. Kernels/parity next (K relaunched; first launch refused on TF's
upstream ATC workflow, fixed with an explicit private-fork scope note).
## [A] 2026-11-20 07:04 UTC M-A1 review (claude-fable-5-2, high effort)
RESULT: TP2 confirmed; decision rule tightened to hop <= 35 us TP2 / > 60 us EP; hop is ~20-35 us on this host (CS event
wait 17-22 us), not 8-15: ~103 hops/window = 2-3.6 ms at C1, biggest wait after kernel boundaries. Added: skeleton-window
bench (asked of Lead B), table-driven placement, M-A2 speed gate, single-op semantic tests, one-window go/no-go,
cross-card selection checksum, keepRows pointer swap, PLE vs device-drafts conflict. Review: results/tf-flashnext/workers/review_fable.md.
VERDICT: DESIGN-A.md v1.1.

## [A] 2026-11-20 07:16 UTC M-A2 step 2: GGUF semantic map + Strata oracle recipe (worker M, Sonnet 5.5, CPU only)
CONFIG: GGUF1/2 header + small byte reads, Strata 0216 source (llama fork models/qwen4exp.cpp, converter qwen4exp.py/qwen.py,
SYCL kernels), PR3603 207aaae.
RESULT: tensorfold/flashnext/GGUF-MAP.md (18-row MISMATCH/ACTION table, per-role map with file:line). Critical: GDN value
heads are TILED in the GGUF (hv = j*16 + k; PR3603 assumes grouped hv/(NV/NK)) -> per rank 3 V spans + kernel pairing
hv % NK. ssm_a = -exp(A_log) (ln roundtrip exact [V]). GGUF norms have +1 folded (ssm_norm raw); MTP sources differ
(mtp-q2_0.gguf zero-centred, Strata rt folded). PLE hash formula identical to PR3603 [V]; only the IQ4_NL table decode
changes. Stop set {248044, 248046}. Oracle: tools/strata_oracle.sh/.py drive strata-dd-0216 --serve over its stdin line
protocol (GEN with logprobs=20 + STRATA_LOGITS_DUMP), 8 fixed prompts (22-4805 tokens), plain mode (no spec/MTP, fp16 KV,
HC_Q8 off) + determinism rerun; compare = teacher-forced top-1 agreement + KL. Not run yet (needs both cards).
VERDICT: accepted as the loader contract. Worker L (Codex, worktree fn-loader, branch b70/fn-loader) launched on the
GGUF loader + transforms + BF16 HC kernels + host-staged AR test path + teacher-forced driver.

## [A] 2026-11-20 07:33 UTC Strata oracle (cards 0+1, ~6 min incl. 2 failed starts)
CONFIG: strata-dd-0216 --serve in strata-sycl-dev:020 via gpu-run (both cards), hotschmoe-dd-0216.json args in "plain" mode:
fp16 KV, STRATA_HC_Q8=0, no batch MTP, prompt cache off, max-context 16384, --spec 2 --mtp-max-t 2 (MTP rt drafter; a native
IQ pack refuses to start without --spec >= 2), STRATA_LOGITS_DUMP on; stdin line protocol GEN 256 logprobs=20, raw ids.
COMMAND: ORACLE_MODE=plain tensorfold/flashnext/tools/strata_oracle.sh run (2 Lead A fixes to worker M's script: mount the
resolved shard-2 directory (symlink into ../IQ3_S); plain mode keeps --spec 2 + MTP).
RESULT: 8 prompts (22-4805 tokens): greedy continuations 256/256/256/256/256/169(eos)/5(eos)/37(eos) tokens with top-20
logprobs per position; argmax == logprob top-1 everywhere; first-token full fp32 logits saved; determinism probe (p0 rerun):
identical 256/256. xpu-health HEALTHY before and after; container gone; leases released.
Outputs: /mnt/vm_8tb/b70/results/tf-flashnext/oracle/{oracle.json,first_logits/*.f32,engine.log,prompts.json}.
VERDICT: M-A3 oracle in hand (Strata greedy, verify windows of 2 rows; timings not speed evidence).

## [B] 2026-11-20 07:35 UTC - means-test failure root-caused: IGC 26.22 folds (float)convert_half_rte(x) -> x (all Alo = 0)
CONFIG: TF PR3603 @207aaae + fix eead3e0 (branch b70/fn-rt-means, merged ff into b70/fn-runtime). Host ocloc/IGC 26.22.38646.4,
card 1 (ZE_AFFINITY_MASK=1), GPTQ 27B (SergiioB g128) + DFlash2. Worker sonnet-means (claude-sonnet-5-5); Lead B reviewed the diff.
COMMAND: CPU emulation of the test with the exact Zig PRNG data (results/tf-flashnext/B/means/cpu/emu.py), disassembly of rowsum_hl,
minimal kernel variants (B/means/variants/v.cl), means/seam/verify/multi/greedy/tree runs old vs fixed (B/means/*.log, NOTES.md).
RESULT (measured):
- Emulation: exact 7.0e-7/3.8e-7; subnormal-lo flush 2.0e-5/2.8e-6; lo DROPPED 5.118e-5/1.856e-5 = measured 5.12e-5/1.86e-5. The asm of
  rowsum_hl computes x - x: the compiler folds fpext(fptrunc(x)) to x, so every hi/lo split's low half is exactly zero. Integer widening
  or a volatile local is not folded. 17 sites (qwen27 glue/glue_attn/glue_quant/drafter/gemm/gdn_chunk SPLIT2, gemma4 glue_g4).
  Lead B's subnormal hypothesis was wrong (refuted by the emulation).
- Fix: zig/kernels/xpu/qwen27/half_exact.h widen_half() (integer-op fp16->fp32, reviewed: normal/subnormal/inf/nan correct) at all sites.
  Means test after: 4.19e-7 / 2.80e-7 ok. Double-close segfault on the error path fixed.
- Model level: seam test 0/248320 logits differ (both builds); verify + multi-8 equal to serial (fixed). Greedy tokens identical for 38
  then diverge (which is closer to a reference: NOT verified). DFlash2 tree-15, one prompt, one run: accepted 2.94 -> 4.73 of 15,
  83.3 -> 122.5 tok/s [indicative only]. Serial decode unchanged 34.2 tok/s. Prompt 1024: 2106 -> 2010 tok/s (-4.5%).
VERDICT: real numerics bug on our toolchain, fixed and default in b70/fn-runtime. All B-side 27B baselines from now on use the fixed
build. FOR LEAD A: any hi/lo (or fp16/bf16 round-trip) split in ported or new OpenCL kernels must widen with widen_half() (or integer
ops), never (float)convert_half_rte(...); the flashnext rbf()/to_bf() helpers in pr3603 are integer-based and look safe [read only].
Pending: matched serve A/B (C1/C2/C4, DFlash2) and an NLL check old vs fixed to confirm the acceptance gain is real.

## [A] 2026-11-20 07:48 UTC Review: worker K GPU kernels (M-A2) + worker L first checkpoint
CONFIG: K = b70/fn-model aa52b8f, 126396b, a0829d2 (fn_gguf.cl generic row kernel, xpu_gguf_ops.zig linearQ/moeUpQ/moeDownQ,
parity/timing harness); card 0, xpu-health HEALTHY before/after. L = b70/fn-loader a429461, 9259134, ed20839 (CPU only).
COMMAND: workers' scripts in results/tf-flashnext/workers/K/ (run_dense/slices/timing/final.sh); L: zig build test-* host.
RESULT (K, measured): 210/210 dense cases PASS (15 types incl. type 42 Q2_0 and real MTP Q8_0, R in 1,2,3,4,8,16,37, f32
rel_l2 <= 1e-6, bf16 out ~1.6e-3 = bf16 rounding); column slices PASS; EXPERT_DOWN with GPU bf16 intermediate <= 1.3e-7;
full expert chain 14 cases above the strict 1e-5 (worst 5.14e-5), all < 1e-4. Host-synced speed (diagnostic): IQ4_XS dense
R=4 18.9 GB/s, IQ2_XS/IQ4_NL expert pair 9.3 GB/s (one generic kernel, runtime type switch, 1 subgroup per row).
RESULT (L, measured CPU): placement estimate 24,137,258,880 B per rank (22.48 GiB, computed, not VRAM); ssm_a: 1707/1728
exact, 21 off by one f32 ULP vs -exp(A_log) -> kernel flag uses stored ssm_a directly (GGUF-MAP's exact-roundtrip claim
corrected); host-staged f32 AR + non-peer open + staged logits gather compile, not GPU-tested; tiled-V and ssm_a kernel flags.
VERDICT: K ACCEPTED as the correctness reference. Expert-chain bound set to 1e-4: the isolated down passes 1e-7, so the residual
is only the bf16 moe_act intermediate straddling midpoints (same storage as PR3603/Strata). Speed is ~20-60x off target:
a ~2.9 GB/card window would take ~150 ms. New worker KF (Codex, fn-model) on per-type specialised kernels (Strata MMVQ
ideas, MIT notice), device-side timing, composed per-layer critical-path numbers. L checkpoint ACCEPTED, continuation L2
running (full loader, BF16 HC kernels, card-0 single-rank load test, teacher-forced driver). Review note on L's host AR: it
does syncAll + D2H + host sum + H2D per call (~2 host round trips per all-reduce, ~100 per forward): fine for correctness,
never for timing.

## [A] 2026-11-20 07:54 UTC Lead A launch error (corrected)
A string-replace on the worker command failed and re-ran worker K's ORIGINAL prompt (07:48-07:53) instead of KF. It made
commit 4a40a09 (re-imposed the strict 1e-5 expert-chain gate, contradicting the 07:48 decision) and two short card-0 runs
(GPU.md [A-K] 07:52-07:53, health checked). Stopped; 4a40a09 reverted locally; KF launched with the correct prompt.

## [A] 2026-11-20 08:11 UTC Review: worker L final (M-A2 loader) + K/L merge
CONFIG: b70/fn-loader 7d82bff..860b1c3 (schema validation, GRankWeights loader, BF16 HC readout kernels, load + teacher
drivers, GDN flags in partial replay); card 0 single-rank loads (each TP rank alone on card 0).
COMMAND: L's tf-xpu-fn-gguf-load-test with XPU_LOAD_RANK=0/1 under gpu-run --card 0 (L/REPORT.md for the exact lines).
RESULT (measured): rank 0 and rank 1 each allocate 24,137,258,880 B, load 8.5 s / 8.0 s (page cache warm, not cold),
Sysman free after load 9.97 GB per card; 35/35 sampled roles byte-match the transformed host upload; BF16 HC down/inject
and upmix one-row checks zero difference vs CPU reference; health HEALTHY before/after.
Lead A merge: new worktree fn-a3 (b70/fn-a3) = fn-loader 860b1c3 + fn-model babfa12, one additive conflict in
zig/build/xpu.zig (kept both blocks), commit 5ea51c6; ReleaseFast build of xpu-flashnext, xpu-fn-gguf-test,
xpu-fn-teacher, xpu-fn-gguf-load-test exit 0.
VERDICT: M-A2 DONE (loader + kernels correct on one card; ~10 GB free per card for KV/state/scratch matches DESIGN-A
section 2 within 1.5 GB). Worker I (Codex, fn-a3) launched for M-A3: wire GGUF forward, two-card host-AR first tokens,
teacher-forced compare vs the Strata oracle, per-layer CPU bisect if needed.

## [A] 2026-11-20 08:25 UTC Review: worker MT (GGUF MTP drafter loader, M-A4 prep) + worker I first pass
CONFIG: b70/fn-mtp 660cbc5 (xpu_gguf_mtp.zig loadRank + load test; source A mtp-q2_0.gguf; draft head = Q5_K rows of the
main output.weight; vocab = Strata int32 (40525) or PR3603 proposal table (79591)); card 0 sequential rank loads.
RESULT (measured): host suite PASS (31-tensor census, +1 norms bit-exact, expert planes cross-checked against Strata rt
A/B/C/D, draft rows match); rank 0 516,136,992 B (20263 draft rows), rank 1 516,135,232 B (20262), loads 0.39/0.29 s,
27/27 sampled uploads match per rank; card 0 HEALTHY before/after.
Worker I first pass (952446d): diagnostics only (logit dumps, router checksum, compare tool), GGUF forward NOT wired;
stopped after a 45 s cpu-run lease wait. Resumed with B70_CPU_LOCK_TIMEOUT=1800 and the full wiring task (I2).
VERDICT: MT ACCEPTED (MTP drafter adds ~0.5 GB/card; budget still ~9.5 GB free/card). forwardMtp wiring waits for I.

## [B] 2026-11-20 08:25 UTC - hi/lo numerics fix: matched serve A/B (OLD/NEW/OLD2) - no speed win, no measurable NLL change
CONFIG: card 1, GPTQ 27B + DFlash2, serve --parallel 4 --context 16384 --temperature 0, port 18152. OLD = pr3603 207aaae tensorfold-native
(sha256 bb433044...), NEW = fn-runtime eead3e0 (sha256 f2112262...). Worker sonnet-ab (claude-sonnet-5-5). Health pre/post HEALTHY.
COMMAND: bench_openai (256 tok x3, code/chat), bench_concurrent --levels 1,2,4 --alone --serial, accept.py, teacher-forced top-20 logprob
NLL over 160 positions of corpus.txt (no prompt_logprobs capability); clients under cpu-run --quiet. B/ab-hilo/NOTES.md.
RESULT (measured, tok/s OLD / NEW / OLD2): C1 code 141.5 / 138.9 / 141.7; C1 chat 101.5 / 88.6 / 101.6; C2 agg code/chat 234/169 /
234/147 / 228/169; C4 agg code/chat 304/230 / 308/221 / 304/235. DFlash2 tokens/round 2.88 / 2.83 / 2.88. NLL (129 shared positions)
1.98349 / 1.98223 / 1.98349 (diff 0.0013 nats, SEM 0.0021: not distinguishable). 7/8 greedy continuations identical; the chat prompt
that changed trajectory explains the C1/C2 chat drop (lower acceptance on a different path). alone==concurrent and drafted==serial in
every cell of all arms; 391 in all arms.
VERDICT: the fix is a correctness change (means test passes, low halves restored), not a speed win; the one-prompt "2.94 -> 4.73
acceptance" indication did NOT reproduce. Kept as default in b70/fn-runtime. Open: long-prompt (1024+) acceptance and prefill cost
(-4.5% at 1024 from the means worker) to re-measure later on the tracer runs; XPU_FIDELITY=1 (fp64 reference in xpu-qwen27) can
settle which build is closer to reference if it ever matters.

## [A] 2026-11-20 08:30 UTC Decision: TP2 transport confirmed on measured hop latency (DESIGN-A v1.2)
CONFIG: Lead B latbench provisional pair numbers (TRANSPORT-B.md section 3): push copy + host-scope event.
RESULT: one-way p50 12-17 us for 4-64 KB (p99 18-29 us); 256 KB 34.8 us. No graph record/replay extension; regular list
execute of 1024 kernels ~3 us host vs 2.7 us per immediate append.
VERDICT: decision rule met -> TP2 stays; EP fallback retired unless the skeleton window shows > 4 ms exposed per C1
window. M-A5 first item = pre-recorded regular command lists per window shape with device-resident per-step args.

## [A] 2026-11-20 08:40 UTC Review: worker KF first pass (GGUF kernel speed) -> redirect to a Strata MMVQ port
CONFIG: b70/fn-model b9dcbdb; card 0; per-type compile-time decode, bf16 activations, 4 output features per subgroup;
Level Zero kernel timestamps, median of 50 (no host sync per op); K's kernels selectable with FN_GGUF_REF=1.
RESULT (measured): parity matrix PASS (all 15 types, R 1..37, slices; experts bit-identical to the reference path).
R=1 device time: BF16 267 GB/s, Q6_K 132, Q8_0 83, IQ4_XS 64, IQ3_S 49, Q5_K 29, Q4_K 20, IQ3_XXS 13 GB/s; GDN rank
dense projections at R=4 ~575 us (sum of medians, sampled shapes) vs 1388 us reference; expert layer-rank up+down at C1
(40 pairs / 30 experts): 708 us (IQ2_S/Q2_0) .. 2634 us (IQ3_XXS/IQ4_NL); C4 1.3-5.3 ms. Census dense byte share:
BF16 39.8%, Q6_K 22.5%, Q5_K 15.1%, IQ4_XS 10.2%, Q4_K 7.7%, IQ3_S 3.4%.
VERDICT: accepted as an intermediate (2.4x over K, numerics unchanged) but 5-40x short: Strata's whole window is ~400 us
per layer per card. Redirected KF (resumed session): (1) build/run Strata's own mmvq/native_expert benches on card 0 for a
per-op target table, (2) port Strata's MMVQ (q8_1 activations, int8 dot, wide/rowwarp/multi variants, grouped experts)
to OpenCL C type by type in byte-share order, MIT notice kept, API stable, (3) numerics gate: match fp64
dequant(W).dequant(q8_1(x)) to 1e-5 (the oracle itself uses q8_1). Fable consulted on this decision (pending).

## [A] 2026-11-20 08:42 UTC Consult: kernel strategy (claude-fable-5-2, high effort)
Question: port Strata MMVQ (q8_1 + int8 dot) vs repack/DPAS/calling Strata SPIR-V; port risks; C1 expert path.
Answer (recorded in results/tf-flashnext/workers/consult_fable_kernels.json): right call with amendments: order by time
saved not byte share (BF16/Q6_K may already be near Strata's ~145 GB/s window average); triage Q4_K/Q5_K and
IQ2_XXS/IQ3_XXS anomalies as kernel defects (asm: spills, SIMD width, per-byte loads, private-copied tables); confirm
dp4a via cl_khr_integer_dot_product in the ISA; experts = expert-grouped list with row mask, one fused gate+up launch and
one down per layer per card, q8_1 of x once per layer, quantise SwiGLU output in the up kernel tail; bench skewed routing;
keep end-to-end token parity vs Strata (q8_1 self-consistency gate is not enough); DPAS out of scope.
Decision: adopted all amendments; KF resumed with them (08:42).

## [B] 2026-11-20 08:50 UTC - M-B1 tracer v1 on the 27B (one card) + M-B3 design accepted
CONFIG: fn-rt-trace 70332f2 (PR3603 base, no hi/lo fix), XPU_TRACE=<dir> wraps the resolved Level Zero tables (default off), kernel-
timestamp events per op, host spans, markers in runner/serve; analyzer tools/xpu_trace/analyze.py (19 Python tests). Card 1, 27B GPTQ
+ DFlash2 serve --parallel 4. Worker codex-trace (gpt-6.1-terra xhigh).
COMMAND/RESULT (measured, B/trace/NOTES.md): traced == untraced token ids (32/32); verify test equal to serial; serve identity + 391.
Decode runner overhead 17.4% (0.909 s -> 1.067 s). Serve C1 (29 windows): wall 49.1 ms, busy 39.3 ms, waiting 9.8 ms (host_cpu 9.2,
host_in_sync 0.07, launch_gap 0.35, dependency 0). C4 (37): wall 52.1, busy 41.2, waiting 10.9. 173,753 full timestamps, 0 forced
syncs, conservation exact. Top device time per C1 window: w4k1_d64 ~14.9 ms, w4k_dsplit ~10.3, w4k1_decode ~5.1, gdn_step ~3.5.
Top gap: one ~7.4 ms host_cpu gap per window between two memory copies.
VERDICT: tracer works (correctness preserved, attribution conserves). The waiting headline is NOT yet trustworthy: untraced serve
rounds are ~40.5 ms (probe), so ~8.6 ms of the traced 49.1 ms is likely tracer cost, and the 7.4 ms gap is suspected to be the
tracer's own harvest inside the wrapped synchronize. v2 requested: self-accounting 'tracer_overhead' category, background harvest,
markers-only baseline mode, named host phases. Lead B's own reading: device busy ~39 ms of ~40.5 ms untraced suggests the 27B serve
is mostly device-bound at C1; real host gaps likely ~1-2 ms per window [I, to be measured by v2].
M-B3: DESIGN-B3-draft.md (fable-b3 = claude-fable-5-2, high effort; 377 lines) reviewed by Lead B and accepted as the plan:
paged int8 KV (Strata cell: int8 + fp16 scale per 64 ch, dynamic absmax, rotation off), drop the per-token indexer key cache,
park = state slab + tail page with write-behind mirror of closed pages (O(state) park ~14 ms, 200K restore ~120 ms estimated),
park and prefix cache as one Snap object, copy-engine list per card polled between rounds (no host waits on the decode list).
B3-1 (host tier substrate + D2H/H2D probes) and B3-2 (27B park/restore, gates C.6) launched (codex-park).
