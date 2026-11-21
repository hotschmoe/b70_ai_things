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

## [A] 2026-11-20 08:58 UTC M-A3 DONE: first tokens on two cards match the Strata oracle (correctness only)
CONFIG: b70/fn-a3 4877506 + bc6b969 (worker I): GGUF TP2 forward (head-split dense, expert-split MoE, BF16 HC replicated,
host IQ3_S embedding, IQ4_NL PLE from host mmap), K's correctness kernels (bf16 activations), XPU_TP_HOST_AR=1 host-staged
f32 all-reduce, chunk 16, MTP off, BF16 head logits. Oracle = Strata plain mode (07:33 entry).
COMMAND: gpu-run (both cards) tf-xpu-fn-teacher --gguf SHARD1 --ple-gguf SHARD2 oracle-eight.json eight-run{1,2}.json;
tools/fn_gguf_oracle_compare.py (worker I/report.md has the full lines).
RESULT (measured): first-token argmax 8/8 match; first-token KL max 0.00657 (p6), cosine min 0.975 (p2); teacher-forced
top-1 agreement 1470/1491 = 98.59%, every disagreement inside an oracle top1-top2 margin <= 0.496 nat (none > 1.0);
top-20 KL weighted mean 0.00133; determinism: all 6.44 GB of raw logits bitwise identical across two runs (6486 rows);
router selection hash identical on both ranks and both runs for every layer/position; p0 serial vs 8-row verify with
perfect and wrong drafts (partial acceptance) == serial; VRAM after full init ~9.66 GB free per card; xpu-health
HEALTHY before/after; no hang. Lead A re-checked first-token argmax/KL for p0/p2/p6 from the raw f32 files with numpy:
identical to the worker's numbers.
Diagnostic only (host AR, correctness kernels, audits on): ~33-37 ms per teacher-forced input token in 16-row chunks;
serial decode ~129 ms/step (Strata C1 ~7.4 ms/token).
VERDICT: M-A3 gates PASS (first token, KL, agreement, margin, determinism, cross-card checksum). Integrated: fn-a3 +
b70/fn-mtp merged (b187103), builds; b70/flashnext-2card (fn-int) fast-forwarded to b187103. Note: worker I left ~12 GB
of raw logit dumps in results/tf-flashnext/workers/I/ (evidence; can be pruned to first/p0 later).

## [B] 2026-11-20 10:10 UTC - M-B2 platform latency constants + cross-card transport verdicts (codex-lat done, merged)
CONFIG: tf-xpu-latbench (b70/fn-rt-lat 0696ede, merged into b70/fn-runtime 3507845), TF Zig + Level Zero, both cards, one shared
context, cpu-run --quiet, health pre/post per case. Worker codex-lat: gpt-6.1-terra xhigh (one resume ran gpt-6.1-sol by default
for ~1 min before Lead B stopped it and re-pinned -m gpt-6.1-terra; codex-trace's 3rd resume runs gpt-6.1-sol - recorded).
COMMAND: see B/latbench/NOTES.md per case (single, single-clock, single-extra, single-flag, pair push/push-ping/stage*/pull,
repro-tp, repro-reuse default/workaround, skeleton R=4/8, chains fresh/reuse/uncached/uncached+barrier, peer/host flag spin).
RESULT (measured): constants table in TRANSPORT-B.md 3/4/4b/4c. Headlines: host append 2.67 us vs dependent device gap 0.94 us;
push one-way 12-17 us at 4-64 KB; skeleton 48-layer window exposed exchange cost 9.4 us (R=4) / 12.6 us (R=8) per exchange at
p50 (window p99 +9-10 ms); fresh-inbox chains 10 KB 8.9/49.0 us, 160 KB 20.7/48.0 us p50/p99, 0 mismatches in 4800 hops each.
Hazards: copy-engine PULL -> BCS CAT error + reset (card 1); tp.zig-style host-reset event reuse -> DEVICE_LOST on both cards
(default), passes with NEOReadDebugKeys=1 EnableImplicitConvertionToCounterBasedEvents=0; reused 160 KB inbox without a receiver
barrier -> stale data; in-kernel flag spins (peer or host-USM flag) -> timeouts + stale data. Every hang recovered by xe-reset +
health. D2H 4.5 GB/s vs H2D 13.3-13.8 GB/s on BOTH cards and BOTH engines (park worker, 1 GiB).
VERDICT: TP2 confirmed on Lead A's rule, with a transport rule: push, fresh events, receiver-side barrier (or fresh slot) per
exchange, NEO workaround env in every two-card process. Stock tp.zig is unsafe on this driver as shipped. The hop p99 tail
(~50 us, window p99 +30%) is the open risk for TP2 decode; trace it in the real model.

## [A] 2026-11-20 10:45 UTC M-A4 DONE: GGUF MTP drafting at batch, multi-stream, drafted==serial, concurrent==solo
CONFIG: b70/fn-a3 4910d83 + bfb427f (worker I): GGUF MTP (mtp-q2_0.gguf, BF16 dense + Q2_0 experts, +1 norms, draft head =
Q5_K rows of the main head for the Strata 40525-id draft vocab), batched drafting across lanes (xpu_mtp_batch.forward),
verify widths T=2/4, confidence 0, chunk 64, XPU_TP_HOST_AR=1, correctness kernels; PLE IQ4_NL from host mmap.
COMMAND: tf-xpu-fn-a4 suites (worker I/report_a4.md; run_a4_solo_remaining.sh, run_a4_multi_suite.sh); summarize_a4.py.
RESULT (measured): drafted==serial: all 8 prompts serial == T2 == T4 token-exact (256 tokens or EOS). concurrent==solo:
28/28 stream results token-exact across C2, C4 (p0-p3), C4qsa (p4-p7 incl. the 4805-token QSA prompt), C4stagger (joins at
rounds 1/2, p6 EOS early). Acceptance T2 677/890 = 76.1% (0.757/window), T4 985/1744 = 56.5% (1.68/window); per prompt T2
tracks Strata spec-2 closely (p0 112/142 vs 104/151, p1 123/132 vs 124/131, p2 120/134 vs 122/133, p4 121/133 vs 122/134).
p5: our greedy run reaches 256 tokens while Strata hit EOS at 169 (teacher-forced p5 agreement was 165/169, misses inside
small margins) - a trajectory difference, not a gate. PLE: table never on device (allocations unchanged before/after; host
mmap 28.8 GB), selected bf16 rows uploaded (5120 B/token/rank).
Diagnostic critical path (C1, T2): 157 ms/window, of which producer wait 131 ms (GPU kernels), host-AR transport ~2.2 ms,
~404 blocking host syncs per window, PLE host gather 0.3-2.1 ms (cold page faults, uncontrolled) exposed at layer 1;
C4 T2 331 ms/window, PLE gather ~4.9 ms. xpu-health HEALTHY before/after, no hang.
VERDICT: M-A4 gates PASS. fn-int (b70/flashnext-2card) fast-forwarded to bfb427f. The window is GPU-kernel bound first
(KF port in flight), then host-synchronised (~400 syncs: needs the push transport + recorded lists), then PLE (prewarm +
early gather). M-A5 starts.

## [B] 2026-11-20 11:05 UTC - M-B1 DONE (one card): tracer v2, overhead 0.3-0.5%; 27B serve is 97% device compute
CONFIG: b70/fn-rt-trace 6e7d817 (measured binaries 3bf353f), merged into b70/fn-runtime 8228f2e (ze.zig conflict resolved by Lead B;
80/80 build steps, 22/22 Zig tests, analyzer tests pass). Binary producer rings, background harvest thread (timestamp query, event
reset after list sync, output), tracer_overhead self-accounting, XPU_TRACE_MODE=markers baseline, named host phases. Card 1,
27B GPTQ + DFlash2 serve, CPU quiet lease. Worker codex-trace: gpt-6.1-terra xhigh, last resume ran gpt-6.1-sol (Codex default on
resume; Lead B now pins -m on every resume).
COMMAND: B/trace/async/run_one.sh full | markers; async/analyze_run.py (B/trace/async/controlled/RESULTS.md).
RESULT (measured): overhead full/markers window p50: C1 40.188/40.033 ms (+0.39%), repeat +0.47%, C4 identical-prompt control
42.988/42.881 (+0.25%); 369-388 ns tracer host time per append. C1 (29 windows) mean: wall 40.19 ms, busy 39.31, WAIT 0.88
(host_cpu 0.245 = tree_plan 0.161 + draft_submit 0.067 + ..., host_in_sync 0.225, launch_gap 0.341, tracer 0.013, unknown 0.055).
Critical path: device compute 97.1%, device copy 0.7%, launch gaps 0.9%, host 1.2%. Top kernels per window: w4k1_d64 14.9 ms,
w4k_dsplit 10.3, w4k1_decode 5.1, gdn_step 3.5. Token ids identical traced/untraced; verify == serial; serve gates + health pass.
VERDICT: M-B1 met on one card. v1's 9.8 ms "waiting" was the tracer itself (Lead B's 08:50 suspicion confirmed). The single-card
27B path is device-kernel bound; host and launch cost ~2%. Next: the same tracer on Lead A's two-card Flash-Next path with cross-card
wait attribution (codex-trace, b70/fn-rt-trace2) - that is where waiting-for-data lives.

## [A] 2026-11-20 12:05 UTC Review: worker KF round 2 - Strata MMVQ port matches Strata per op (M-A5 kernels)
CONFIG: b70/fn-model 8fd616a (OpenCL C ports of Strata's q8_1 MMVQ: KREG dense for Q4_K/Q5_K/Q6_K/IQ4_XS/IQ3_S, BF16 KREG,
typed row-reuse kernels for the rest, grouped experts GU_LANES=8 with fused SwiGLU + exact ggml q8_1 tail, Q2_0 vector
decode; SIMD16 pinned, dp4a confirmed in ISA, no spills in selected variants; MIT notice kept). q8_1 default, FN_GGUF_ACT=bf16
and FN_GGUF_REF=1 selectable. Card 0, kernel timestamps, hot cache. Strata reference = Strata's own kernels built from
strata-0216 in strata-sycl-dev:020 with the qualified env, same real tensors and routing plans.
RESULT (measured, per-op, card 0): parity: all 15 types match fp64 dequant(W).dequant(q8_1(x)) to <= 1.4e-7 (f32 out);
q8_1 vs raw-activation error ~0.4% dense, ~1% expert chain (the oracle itself runs q8_1). GDN rank 0 dense at R=4 (real
row spans, incl. quantisation) 50.5 us vs Strata 64.5 us. C1 expert layer-rank (40 pairs / 30 experts, incl. input quant)
125.8-205.3 us vs Strata 141.0-199.0 us (0.69x-1.05x of Strata per type pair). BF16 dense R=1/2/4 ~597/594/577 GB/s;
Q4_K/Q5_K R=4 597/634 GB/s logical (hot cache). Known gaps: Q6_K R=4 411 vs 559 GB/s, IQ4_NL down 82.5 vs 72 us, expert
input quant 4.3 us (dedup scan) vs 2 us. Before: K kernels 708-2634 us per C1 expert layer-rank.
VERDICT: ACCEPTED as the M-A5 kernel baseline (5-20x faster than K, at Strata parity per op). Open contract item before it
becomes the default in the model: ROW-INVARIANT BITS (DESIGN-A v1.3) - RPW/NCOLS differ by R, so serial (R=1) vs verify
(R=2/4) and solo vs packed concurrent rows may round differently; the model-level drafted==serial and concurrent==solo
gates on the merged tree decide. Model-level quality (teacher-forced vs oracle) also re-run with q8_1 on.

## [B] 2026-11-20 13:20 UTC - Two-card latency profile of Lead A's Flash-Next TP2 path (tracer on fn-int, host AR)
CONFIG: b70/fn-rt-trace2 d1a8a93 (= b70/flashnext-2card bfb427f + tracer v2 merge 6bb4403 + Flash-Next markers 1999909/d1a8a93: per
window, per layer x {attention, gdn, moe, hc, ple, mtp, verify, exchange}); A4 engine harness, TP2 host all-reduce (XPU_TP_HOST_AR=1),
correctness/K kernels (pre-KF), MTP T2, 32 output tokens, XPU_A4_PROFILE=0, NEO workaround env. Worker codex-trace (gpt-6.1-terra).
COMMAND: B/trace2/run_case.sh per case (markers pre, full, markers post) under cpu-run --quiet + gpu-run both cards + health;
tools/xpu_trace/analyze.py --window a4_window. Clock check: 32-window fresh-event ping, 64/64 peer edges matched and ordered.
RESULT (measured; diagnostic, pre-optimization kernels): overhead full/markers 1.3-1.6% (C1), 2.0-3.6% (C4). Token arrays identical
across modes; T2/T4 == serial; C4 streams == solo; health pass.
  C1 T2 window 159.0 ms: per card busy 142.8 / 142.6, waiting 16.3 / 16.5 (peer wait 7.6 / 6.3, host CPU 3.2 / 5.0, launch gaps 2.2 /
  2.4); 402 host syncs per window; critical path 93% device compute. Top kernel fn_gguf_linear 83 ms/window on card 0, moe_up 27 ms.
  C4 T2 window 336.9 ms: card 0 peer wait 28.1 ms (waits for card 1 at every moe_combine -> exchange, 1.7-2.2 ms each), card 1 host
  CPU 22.4 ms (host.fn.gdn 9.0, attention 3.1, exchange 2.5, moe 2.4): the single host thread appends card 1's per-row GDN/attention
  work late, card 1 starts late, card 0 waits for it. Critical path: device 89%, host CPU 6%, peer exposure 5% (overlay).
VERDICT: Today the two-card window is GPU-kernel bound (KF port is the lever, as Lead A says). The first waiting-for-data effect is
already visible at C4: ~28 ms/window of card-0 idle caused by card-1 HOST submission lag (not PCIe), i.e. the host-append limit
from latbench (2.7 us/append) times card 1's per-row launches. Recorded lists (M-B2c, device-reset events: 10,000 replays x 96-192
hops clean, ~6.5 us host submit per window) remove it; per-row GDN launches should also be batched. The push transport is not yet
in this trace (fn-int was at bfb427f); rerun when Lead A merges it.

## [A] 2026-11-20 13:25 UTC M-A5 step 1: TP2 push transport integrated (worker I), all gates re-pass
CONFIG: b70/fn-a3 24f8093..d806c0f = Lead B's push all-reduce (b70/fn-runtime 611566d tp.zig/ze.zig + rank-ordered +0.0 sum)
behind unchanged TpComm; main logits and (batched) MTP draft-head gathers now push rank 1 -> rank 0 inbox with fresh events
and receiver barrier; no peer pull, no D2H->H2D relay, no zeEventHostReset; NEOReadDebugKeys=1
EnableImplicitConvertionToCounterBasedEvents=0; XPU_TP_HOST_AR=1 kept as fallback. K correctness kernels still in this tree.
RESULT (measured): teacher p0/p2/p6/p7: 1.35e9 compared f32 logits bitwise identical to the host-AR M-A3 run (agreement
547/554, KL max 0.0066); serial == T2 == T4 == host-AR sequences on all 8 prompts; C4 T2, C4qsa T4, C4stagger T2 and a
3-run single-process C4 T4 soak all == solo; fallback regression OK; health before/after both cards, no hang.
Diagnostic (instrumented): blocking host syncs per window 404 -> 13 (T2) / 430 -> 25 (T4); exposed AR 1.5-2.8 ms per
window; fresh-event retirement 5.3 ms (T2) / 9.2 ms (T4) per window = new host cost on the critical path; host append
13.9 ms per C1 T2 window; window wall unchanged (~160 ms C1 T2) because the K kernels dominate (GPU backlog ~128 ms).
VERDICT: transport step PASSED. Next: merge KF kernels (8fd616a + round-3 fix) and P's device PLE into fn-a3, re-run all
gates, then measure the real window. Event retirement (5-9 ms) and host append (~14 ms) are the next host items: both go
away with recorded lists + counter-based events (Lead B M-B2c).

## [B] 2026-11-20 13:25 UTC - M-B2c DONE: recorded-list (replayable) TP2 all-reduce with device-side event reset
CONFIG: b70/fn-rt-pushar c9c9efe merged into b70/fn-runtime; tp_replay_bench: one regular command list per card recording N chained
FP32 all-reduces with a producer kernel per hop and a verify kernel per hop, replayed 10,000 times; ordinal-0 async queue per card;
default driver env. Worker codex-pushar (gpt-6.1-terra xhigh).
COMMAND: B/pushar/replay/run-case.sh <policy> <bytes> <hops> 10000 <depth> (cpu-run --quiet, gpu-run both, health pre/post).
RESULT (measured): counter-based NON_IMMEDIATE -> sentinel mismatch at replay 67; IMMEDIATE -> 0x78000004 during setup; dropped.
device_reset: 4/4 shapes x 10,000 replays all verified; window p50/p99 1.416/1.432 ms (10 KB x 96), 2.823/2.844 (10 KB x 192),
2.896/2.921 (160 KB x 96), 5.778/5.830 (160 KB x 192); host submit 5.6-6.1 us p50 per window; double submission with two
recordings 10,000 clean, p99 5.831 ms. Host unit tests (slot schedule, events consumed once) pass.
VERDICT: A replayable cross-card exchange exists and is stable on this driver: ~14.7 us per 10 KB hop including verify, with a flat
tail, and the host cost of a 96-hop window drops from ~2.5 ms (immediate) to ~6 us. This unblocks Lead A's M-A5 recording; contract
in TRANSPORT-B 4e.

## [A] 2026-11-20 13:35 UTC Review: worker KF round 3 - row-invariance contract proven per op; merged into fn-a3
CONFIG: b70/fn-model 8db568f (test first) + e5eaf28 (fix). tf-xpu-fn-gguf-test bitwise mode: 37 rows (ties, signed zeros,
random), each alone vs every position of R in {2..9,16,37} under two permutations; experts with real weights, shuffled
pair order and group splits, max_pairs = exact pair count, canary-filled outputs; modes default(q8_1)/bf16/ref. Card 0.
RESULT (measured): on 8fd616a the tight test FAILS: 77.1M differing elements, all 5 expert pairs, all modes - cause is a host
launch bound that dropped expert groups under finer splits (not reduction-order drift). After the bound fix: 36,084 position
cases / 933,498,900 comparisons, zero differing bits; fp64 parity both modes passes; C1/C4 expert latency within +0.2..+1.0%;
GDN R=4 50.42 us; health OK.
VERDICT: ACCEPTED. Good catch by the stronger test: the round-2 kernels would have silently dropped experts for some
concurrent packings. Merged into b70/fn-a3 as 10d5d6e (one conflict in xpu_store.zig resolved by Lead A: kept I's profile
counters and KF's profile_events/LocalBytes launch); merged tree builds. Model-level gates on it go to worker I next.

## [A] 2026-11-20 14:05 UTC Review: worker P - PLE gather moved to the device (host USM table); merged into fn-a3
CONFIG: b70/fn-ple 8ee60cc..409bfe7 (base bfb427f). One 28,800,138,240-byte zeMemAllocHost holding the IQ4_NL PLE table,
populated at load and read by both cards; per-rank kernel hashes device token IDs + device two-token history (speculative
history per sequence, committed on accept, snapshot/restore) and writes the existing BF16 layout. XPU_PLE_DEVICE default on
for GGUF, =0 host fallback, XPU_PLE_CHECK=1 bitwise debug compare. Host AR in P's runs (pre-push base).
RESULT (measured): bitwise PLE equal to the host gather on both ranks for all 8 prompts incl. drafted rows; p0/p5 serial ==
T2 == T4 == I's sequences; C4 T2 == solo; default/fallback smokes OK; health pre/post OK. Diagnostic: host gather 0.39-1.64
ms per C1 window and 3.9-5.6 ms per C4 window -> device enqueue 17-18 us (C1) / 59-65 us (C4), kernel ~5 us (C1) / ~16 us
(C4) per rank; no added blocking sync. Limits for Lead B: 28.8 GB single host allocation works (under the 32 GiB pinned
cap); zexDriverImportExternalPointer on the mmap failed (OUT_OF_HOST_MEMORY, RLIMIT_MEMLOCK 8 MiB), so the table costs
28.8 GB of host RAM in addition to the page cache.
VERDICT: ACCEPTED. PLE is off the CPU and off the critical path (~5 us of GPU per window). Merged into b70/fn-a3 as adee05d
(conflict in xpu_diagnostics.zig resolved by Lead A: union of counters). Open: the RAM cost (28.8 GB pinned) competes with
KV eviction to RAM; import of the file mapping needs a higher memlock limit - Lead B item. Worker I now runs the full
model-level gates on adee05d (fast kernels + push + device PLE) with q8_1 and bf16 activations.

## [B] 2026-11-20 14:05 UTC - B3-1 + B3-2 DONE (27B, one card): host-RAM park tier; slot park/restore WITHOUT recompute, bit-identical
CONFIG: b70/fn-rt-park 0410fea merged into b70/fn-runtime 620851f (80/80 build steps, 22/22 tests). host_tier.zig (pinned 1 GiB
zeMemAllocHost slabs, 47 GiB process ceiling), park_list.zig (in-order immediate list on copy ordinal 1, fresh single-device event
per batch, zeEventQueryStatus polling, never a host wait or reset), slot_park.zig (27B: GDN recurrent + conv + KV (bf16 or int8 +
scales) + DFlash2 ring + position/last token). Default off. Card 1, 27B GPTQ + DFlash2. Worker codex-park (gpt-6.1-terra xhigh).
COMMAND: B/park/run-*.sh (cpu-run --quiet, gpu-run --card 1, health pre/post); B/park/RESULT.md + NOTES.md.
RESULT (measured): byte round trips 1 MiB-2 GiB on both cards and both engines; 10,000-batch soak; 1 GiB H2D/D2H 13.3-13.8 / 4.5-4.7
GB/s (all four card/engine combos); EU stores to host USM only 1.4 GB/s. Forced park/restore every K rounds, same and other slot,
K = 1/7/64: greedy bf16 and int8 token ids AND full-logit SHA256 identical to never-parked; int8 + DFlash2 drafted same matrix
(K=64: 512 ids, 1,000 drafted-row logit hashes identical); every saved byte verified. Park/prepare/restore (warm, 2 samples):
1K ctx 188 MB 44/23/15 ms; 8K 97/23/34 ms; 16K 699 MB 155/24/53 ms. Non-interference (T decoding while S parks/restores every
window, batches of 4 copies, fast no-zero restore alloc): T == solo for 256 tokens; T window p50 +2.13%, p99 +0.11% (gate was 2%/5%:
p50 narrowly FAILS). Controlled restore-failure drill returns frames and device allocations.
VERDICT: "Evict to RAM, restore without recompute" works and is bit-exact on the 27B (standalone hook, not yet a serve admission
policy). Park is D2H-bound (4.5 GB/s on both cards: a platform constant, not NUMA - the host is UMA) so the B3 design's write-behind
mirror (copy closed KV pages ahead of time, park = state slab + tail page) is required for O(state) parks; restore is ~13 GB/s.
The 2.1% p50 cost of concurrent copy traffic is real and should be scheduled into idle gaps (tracer can show them). Next: B3-5 on
the Flash-Next page pool (two cards).

## [A] 2026-11-20 14:10 UTC Decision: recorded decode windows (worker R), consult recorded
CONFIG: Lead B delivered M-B2c (recorded-list TP2 AR, device-reset events, 10k replays, ~15 us/hop, flat p99, ~6 us host
submit/window; counter-based events FAILED) and tracer v2 (b70/fn-rt-trace2 d1a8a93; C4 card 0 idles ~28 ms/window waiting
for card 1's late host appends). With KF kernels the GPU work per window should be ~10-15 ms, so host append (~14 ms),
event retirement (5-9 ms) and cross-card append skew become the critical path.
COMMAND: draft task for worker R reviewed by claude --model fable --effort high (responding model claude-fable-5-2) and
claude --model opus --effort high (claude-opus-5-5); answers in workers/consult_rec_{fable,opus}.json.
RESULT: both independently flagged the same top items: (1) do not fuse descriptor-driven grids with recording - first an
invariant-immediate path gated bitwise vs today, then recorded == invariant-immediate bitwise; (2) speculative commit of
GDN/conv state, KV length and PLE history must be specified and forced-reject tested; (3) MoE routing counts/grids must not
depend on padding - pad to trash KV/GDN/PLE slots via the descriptor; (4) a fixed-shape spike with a written floor
(~96 hops x 15 us transport + kernel time) as go/no-go before building buckets; (5) bucket rows only, n_seq as descriptor
data, pre-record, no reallocation; single-flight per transport channel; timed syncs + abort/re-record path; (6) full-vocab
logits push only in gate mode, device argmax/accept with host-equal tie/NaN rules; (7) soak against golden hashes from a
separate immediate run; per-layer timestamp variant to see inside recorded lists.
VERDICT: adopted all of them. Worker R (Codex gpt-6.1-terra xhigh, worktree fn-rec, branch b70/fn-rec from adee05d) runs
phases R0 merge gate + audit + spike (go/no-go), R1 invariant-immediate (XPU_FN_DESC=1, bitwise), R2 recorded
(XPU_FN_RECORD=1, default off), R3 device draft/argmax/accept with one readback per window. Prompt workers/R_prompt.md.
Worker I meanwhile runs model-level gates + window breakdown on adee05d (task I5).

## [B] 15:10 UTC - prefill baseline: chunk-invariant ~55 tok/s; trace worker narrowed

CONFIG: codex-trace2 probe bc8acee on Lead A push AR 24f8093 (pre-KF kernels), TP2, MTP on, 20000-token fresh prompt,
markers-only, explicit per-chunk sync, no HTTP.
COMMAND: B/prefill/run_case.sh <case> markers <chunk> single prompt-20000.json 1300 (metadata.json per run; gpu-run pair lease,
health pre/post).
RESULT (measured): chunk 256 / 512 / 1024 / 2048 -> 53.7 / 55.4 / 55.6 / 55.4 tok/s (372 / 361 / 360 / 361 s); first token 351
in all; health passed. A 1024-row chunk took 18.3 s and tripped the 10 s list-wait bound once (fixed: 60 s bound for the probe
only, 238e252).
VERDICT: Prefill on this path is chunk-invariant, so its cost is per-row serial work, not per-chunk launch/sync overhead
[inferred until the per-op trace attributes it]. ~40x short of Strata (~2200 tok/s); cold C4 TTFT cannot approach Strata's
17/39/61/84 s by scheduling alone - the prefill kernels must batch rows. Lead B decision: stopped the remaining 20K full-trace
runs (about 25 min of both-card time on pre-KF kernels) and narrowed the worker to CPU analysis of the captured 256-row trace
plus one <= 5 min run on fn-a3 (KF kernels), output B/prefill/CRITICAL_PATH.md. Lead A: this is in your kernel domain (KF);
the attribution will name the per-row loops.

## [B] 15:40 UTC - PLE host memory answer for Lead A

CONFIG/COMMAND: see TRANSPORT-B 4f; probe b60f30b merged into b70/fn-runtime; worker codex-pushar on gpt-6.1-terra xhigh.
RESULT (measured): anonymous import 4/64 MiB OK; file-mapping import 0x70000002 on host and with --ulimit memlock=-1;
zeMemAllocHost fill 1.86 GB/s buffered+DONTNEED, 3.85 GB/s O_DIRECT (4 GiB slice, card 1, health passed).
VERDICT: one zeMemAllocHost copy filled by O_DIRECT + DONTNEED is the recommended PLE load; memlock is ruled out.

## [B] 15:50 UTC - prefill attribution; KF integration prefill 362 tok/s; DPAS prefill started

CONFIG: (a) captured 20K full trace, chunk 256, pre-KF 238e252 (bfb427f + push); (b) same lease, 2048 ids (first 2048 of the
20K stream), chunk 512, markers-only: bfb427f baseline vs fn-a3 adee05d integration (q8_1, device PLE) merged at fn-rt-prefill3
6d7adb2. Model load excluded.
COMMAND: tools/xpu_trace/prefill_summary.py over runs/single-256-full/trace (cpu-run); runs per B/prefill/NOTES.md (gpu-run
both cards, 194 s lease, health pre/post passed).
RESULT (measured): (a) 79 chunks, 8.86M timed ops, 0 dropped. Device busy by class, card 0/1: dense GEMM 60.0/60.5%, experts
24.7/27.6%, HC 5.9/6.0%, AR+peer 5.4/1.8%, MTP absorb 1.9%, QSA 0.8%, GDN 0.4%. 29.1 kernel launches and 192.4 copy appends
per row per card (project() issues one memcpy per row per projection). Tracing added 13.8% wall. (b) 55.79 -> 362.20 tok/s
(6.49x), first token 9687 in both.
VERDICT: Prefill is weight-sweep bound per 16-row tile (dense dominates); KF kernels gave 6.5x, ~6x more is needed for
Strata's ~2200. Cross-card critical path unresolved (end-only events). Next: Lead B DPAS prefill GEMM, standalone and default
off (proposal to Lead A in STATUS). The per-row projection memcpy (192/row) is a cheap fix in Lead A's forward.
Workers: codex-trace -> DPAS prefill (b70/fn-rt-prefill3); codex-pushar -> B3-6 host admission (b70/fn-rt-admit, CPU only).

## [A] 2026-11-20 16:35 UTC M-A5 step 2: fast kernels + push + device PLE pass every model gate; critical path measured
CONFIG: b70/fn-a3 46afffb (= adee05d + I's streaming digests/unprofiled gates + per-card kernel timestamp profiler). Push
TP2, device PLE, FN_GGUF_ACT=q8_1 default (bf16 run as comparison). Same model/MTP/draft-vocab identity as M-A3/M-A4.
Report: workers/I/report_a5k.md.
RESULT (measured, gates profiling off): teacher p0/p2/p6/p7: q8_1 agreement 547/554 (98.74%), first-token KL max 0.0080,
max miss margin 0.18 nat; bf16 547/554, KL max 0.0057, margin 0.13 nat; both deterministic (SHA256 over all 5.4 GB of
logits, two processes), router consumption identical across ranks. serial == T2 == T4 on all 8 prompts; C4 T2, C4qsa T4,
C4stagger T2, 3-run C4 T4 soak all == solo; host-PLE fallback OK; host tests 28/28; health OK.
Diagnostic (profiled; NOT speed claims): ms/window C1 serial 47.3, C1 T2 55.2, C1 T4 66.5, C4 T2 80.0, C4 T4 116.2.
Card 0 C1 T2: kernels 31.7 ms, of which HC 20.1 ms (fn_hc_down_bf16_1 16.7 ms = 96 calls x ~170 us; fn_hc_upmix 2.5 ms,
grows to 11 ms at 16 rows), dense 5.0 ms, experts ~3 ms; submission gap 16.8 ms (host append of ~1720 kernels per card
falls behind; largest single gap 13.3 ms before fn_logits_part); exposed AR 1.65 ms; event retirement 5.2 ms. Prefill p7
(4805 tok) 287 tok/s (decode-type kernels, no XMX prefill yet).
VERDICT: M-A5 correctness on the fast path PASSED; q8_1 stays the default (same agreement, matches the oracle's own
activation path; bf16 slightly lower KL - kept selectable). fn-int (b70/flashnext-2card) fast-forwarded to 46afffb.
Critical path now = (1) HC kernels: fn_hc_down is one thread per output column looping K=10240 serially (324 work-items,
6 work-groups) - ~170 us for 6.6 MB of weights where ~12 us is the bandwidth floor; upmix reloads weights per row;
(2) host submission lag (~17 ms) and event churn - worker R's recorded windows (R0 spike GO: 32.3 ms recorded vs 31.8 ms
kernel+transport floor, host submit 13.8 ms -> 39 us, bitwise logits). Next: worker H = HC kernels; R continues R1-R3.

## [B] 16:35 UTC - DPAS prefill first cut (standalone Q6_K): correct, 1.8-2.1x at M=512 (gate 3x missed); B3-6 host prototype

CONFIG: (a) b70/fn-rt-prefill3 7f4efc6/fef67a4, card 1, real Q6_K weights, synthetic BF16 activations, FP32 out; candidate =
Q6_K -> FP16 unpack + BF16->FP16 convert + f16 DPAS (sub_group 16), baseline = current Q8_1 path. (b) b70/fn-rt-admit 853d273,
CPU only.
COMMAND: (a) gpu-run --card 1 ... B/prefill/dpas/lease_inner.sh (health pre/post passed); (b) zig test xpu_resident_test.zig;
resident-sim 40 120 750 0 2000 2 (B/admit/NOTES.md).
RESULT (measured, device timestamps): (a) 2560x5120: M=512 2105 -> 1146 us (1.84x), M=2048 8881 -> 2425 us (3.66x);
3072x2560: M=512 1346 -> 652 us (2.07x), M=2048 5600 -> 1498 us (3.74x). Unpack is 59-63% of candidate time at M=512; DPAS-only
~34 TFLOP/s. fp64 rel L2 2.0e-4; bitwise row invariance M=64..2048; full-output repeatable. (b) 18/18 host tests incl. 4 seeds x
10k random steps (no overcommit). Sim [I]: int8 C4x200K 0.25% lost capacity; C8/C16 x 200K 15.5% lost, worst gap 5.3/11.1 s;
C16 x 16K 0.1%.
VERDICT: (a) the path is sound; the unpack kernel (one work-item per weight, byte gathers) and the 16-column tiles with per-lane
B gathers and split-K are the measured losses -> optimization round authorized (VNNI B layout, bigger tiles, no split-K,
shared activation convert, fused dequant stretch). (b) accepted as a host prototype; found a fairness flaw (least-remaining
protection starves equal streams: duty 15-99.9% at C16x200K) - fix + write-behind mirror model in progress. The 15.5% loss at
200K is the full-copy park (1.6 GB at 4.5 GB/s): the write-behind mirror is what makes C8/C16 x 200K cheap [I].
Note: Lead A fn-int advanced to 46afffb (all M-A5 correctness gates); kv8/park2 branches are on the older fn-int and will
need a merge before acceptance.

## [B] 16:40 UTC - B3-6 policy fix + write-behind mirror model (CPU only)

CONFIG: b70/fn-rt-admit e0db34d (74eaac2 protection only at <= protect_tokens, default one quantum of tokens; a3e1ea5 mirror
traffic serialized with parks on the modeled D2H queue).
COMMAND: zig test xpu_resident_test.zig + sim matrix (B/admit/NOTES.md), cpu-run.
RESULT: 24/24 policy + 6/6 sim/layout tests. [I] C16 x 200K int8, 2 s quantum: mirror off 20.6% lost capacity, duty 18.6-20.6%,
worst gap 8.8 s; mirror on 9.25%, duty 21.7-23.3%, worst gap 7.1 s. Full matrix + 1/2/4 s quantum sweep in NOTES.md.
VERDICT: accepted (host prototype; merges with B3-3 later). Fairness flaw fixed (duty spread within 10% of mean). With the
mirror, the remaining swap cost is the 1.6 GB H2D restore (~120 ms per swap) [I]; C16 x 200K is time-sliced at ~22% duty per
stream by construction - concurrency beyond 4 x 200K needs smaller per-stream KV (int8 already) or more VRAM, not scheduling.

## [A] 2026-11-20 16:55 UTC Review: worker KF round 4 - Q6_K / IQ4_NL / shared q8_1 activations
CONFIG: b70/fn-model bb7f449 (Q6_K SSA aligned loads, 128 GRF, RPW4), 0118f23 (IQ4_NL register LUT, 2 output rows, 16
cyclic group slots, exact 4B-aligned load8), 4bafdb4 (ops.prepareInput / linearQPrepared / moeUpQPrepared; call pattern in
GGUF_Q8.md). Card 0, device timestamps, same Strata-kernel comparison method as round 2.
RESULT (measured, per op): Q6_K dense R=4 52.5 -> 32.0 us (409.6 -> 672 GB/s logical; Strata 38.2 us); IQ4_NL C1 expert
down 82.5 -> 70.5-71.0 us (Strata 72.2-72.7); expert input quantisation 4.3 -> 1.7-2.1 us (Strata 2.0-2.1). Bitwise
suite 83,214 cases / 2.36e9 comparisons zero differing bits; fp64 parity both modes; prepared == unprepared bitwise; health.
Projection sums per layer-rank (C1 R=4): GDN 50.3 us vs Strata 64.4; QSA 45.9 vs 46.2; GDN layer-0 dense+experts 237 vs
Strata 284 us.
VERDICT: ACCEPTED. All per-op gaps KF was asked to close are closed; our projection kernels now match or beat Strata's.
Not yet in fn-a3 (numerics of IQ4_NL association changed -> needs the model gates; will merge together with worker H's
HC kernels and gate once). Prepared-input wiring waits for worker R's forward rewrite. Round 5: C4 IQ3_S/IQ2_S up, decode
GDN/QSA non-projection kernels (gdn_rows, attn_sparse, ...), window-invariant grids. Prefill DPAS is Lead B's prototype
(B/prefill/DPAS_PLAN.md) - not duplicated.

## [B] 17:58 UTC - user speed directive applied to Lead B workers
Stopped and redirected codex-kv8 and codex-park to wrap-up: merge fn-int 46afffb, one < 3 min smoke each, HANDOFF.md for Lead A;
park2 to report unprofiled park/restore ms at 1K/8K/32K. Full suites only at milestone merges. DPAS worker continues (expert
timing), then wires DPAS prefill on fn-int behind a flag for a real prefill tok/s number. Lead A owns the serve (fn-serve).

## [A] 2026-11-20 17:56 UTC Direction change (user via orchestrator 17:25): serve numbers first, faster loop
- Worker S (fn-serve from fn-int 46afffb): tensorfold-native xpu-native OpenAI serve on :18150 (research ID, not
  hotschmoe-dd), unprofiled C1/C2/C4/C8 tok/s, TTFT, ITL, prefill 512/2k/8k; reusable S/serve_bench.sh for re-runs per merge.
- Worker I6: smoke gate I/smoke.sh (<3 min GPU: identity, greedy token hashes 3 prompts, serial==T2, C2==solo, health),
  then merge KF round 4 into fn-a3 -> smoke -> fast-forward fn-int; full gate suite only at milestone merges.
- H (HC kernels), R (recorded windows), KF5 (decode kernels) continue; each merges into fn-int on smoke pass.

## [A] 2026-11-20 18:30 UTC First serve numbers (worker S)
CONFIG: tensorfold-native serve, TP2 GGUF IQ3_XXS + MTP (adaptive, caps T2/T4), 8 lanes, ctx 16384, q8_1, push TP2, device
PLE, unprofiled; fn-serve = fn-int 46afffb + server wiring. COMMAND: workers/S/serve_bench.sh fn-serve first.
RESULT: C1 42.4 (T2) / 45.5 (T4) tok/s; C2 66.5 agg; C4 89.9 agg; C8 111.8 agg (T2); TTFT C1 107 ms; prefill 309/299/286
tok/s at 512/2k/8k. p0 serve tokens == solo serial exactly; /v1/models identity tf-flashnext-iq3xxs-tp2-mtp; coherent chat.
VERDICT: baseline for the merge train (HC, recorded windows, KF4/5). 3.3x below Strata C1; prefill 7x below.

## [O] 2026-11-20 18:35 UTC - Orchestrator NUMA probe (host memory layout and host copy rates)
CONFIG: ASRock X399 Professional Gaming, BIOS P4.05, Linux sees 1 NUMA node. Card 0 root port pci0000:00 (die 0), card 1
pci0000:40 (die 1), both Gen3 x16. Probes in /mnt/vm_8tb/b70/results/tf-flashnext/orch-numa/.
COMMAND: per-cache-line DRAM latency (clflush + timed load) for the same 200,000 random lines of one 8 GiB buffer from core 0
(die 0) and core 8 (die 1); tf-xpu-latbench single on card 0 and card 1, each pinned to cpus 0-7 and to 8-15.
RESULT: every line is ~80-100 ns from one die and ~150-160 ns from the other, split 96,303 / 94,781 (perfectly
anti-correlated; mean 129 vs 135 ns) -> BIOS memory interleave across dies (UMA). Copies (64 MiB, p50): H2D 13.0-13.6 GB/s
and D2H 4.5-4.7 GB/s on BOTH cards, from either die, on both copy ordinals.
VERDICT: the D2H ~4.5 GB/s limit is not a placement/NUMA effect we can fix by pinning (identical everywhere). Park/eviction
designs should assume ~4.5 GB/s D2H per card (write-behind mirror stays the plan). Cross-card push is device-to-device and
does not touch host DRAM. No BIOS change needed for the current campaign.

## [B] 18:47 UTC - B3-3 handed to Lead A for merge
kv8 36738f3 (fn-int 46afffb merged): off == fn-int hashes, bf16page == off (3 x 32); teacher int8 dNLL +0.0001, top-1 99.0%;
int8 7,696 / bf16page 14,144 B per token slot per card. Open: int8 C2 smoke, int8 chunk 512 != 64. B/kv8/HANDOFF.md.

## [A] 2026-11-20 19:10 UTC Serve round 2 (worker S, fn-serve 3649184)
RESULT: MTP depth policy (cap 3 at 1 lane, 1 at >=2): C1 45.8, C2 66.8, C4 89.2, C8 111.1 agg tok/s; ITL p50/p99 C1 16.8/50.8
ms, C4 34.6/91.7 ms; no-MTP C1 30.3 (33 ms/token), C4 79.8 agg; prefill 2k 298 tok/s. p0 == serial in all rows, health OK.
Re-run: workers/S/serve_bench.sh <worktree> <tag> -> S/summary.tsv.

## [B] 19:25 UTC - kv8 int8 smoke: C2 == solo; chunk-512 divergence points at router kernel selection (>16 rows), not KV append.

## [A] 2026-11-20 19:42 UTC Worker R: R0 GO, R1 PASS (fn-rec 11fecc6); R2 started
RESULT: descriptor-driven immediate path (XPU_FN_DESC=1) == merged immediate bitwise: 140 raw-state/padding/commit cases
(incl. 0-accepted-draft commit), 4 teacher hashes, 8x serial/T2/T4, C4 T2/qsa/stagger, 3-run soak; health OK. Baseline moved
with KF4 merge (IQ4_NL/Q6_K reduction order): oracle 545/554, KL max 0.0081 - approved by Lead A as the new reference.
VERDICT: R2 (recorded) next, light per-iteration gates, then serve_bench on fn-rec with XPU_FN_RECORD=1.

## [B] 20:16 UTC - B3-5 headline: park/restore without recompute on two-card Flash-Next
CONFIG: b70/fn-rt-park2 6b6af10 (fn-int 46afffb + kv8), push, int8 KV, bf16 act, prealloc; COMMAND: run-headline-*.sh (B/park2).
RESULT (1 sample, per card): 1K 66 MB park 17.1 / restore 6.1 ms; 8K 121 MB 29.9 / 10.2 ms; 32K 310 MB 75.6 / 24.9 ms. K=7 T2
parked == unparked (push). VERDICT: user's eviction-to-RAM primitive works on the real path; serve wiring next (fn-rt-evict).
DPAS prefill smoke passed (off == fn-int; on KL <= 0.0047, top-1 equal, chunk-invariant bitwise); perf pending.

## [A] 2026-11-20 20:20 UTC Worker H: HC kernels PASS; merged
RESULT: fn-hc ee86b66: hc_down split-K/row-reuse (R1 ~190 -> 16.6 us, R4 152 -> 13.5 us, R16 32.5 us), upmix R16 101 -> 74 us
(targets partly missed: R2 down 20.4 us, upmix R16/R1 4.4x); bitwise row-invariance + fp64 pass; model gates: teacher,
8-prompt serial==T2==T4, C4 T2/T4 == solo all pass. Profiled C1 T2 kernels/card 31.7 -> 17.4 ms (HC 20.1 -> 5.8).
VERDICT: merged into fn-a3 (b329393, with KF4 + serve wiring); smoke + serve_bench next.

## [A] 2026-11-20 21:20 UTC Serve numbers #2 after HC + KF4 merge
CONFIG: fn-int 186329a (KF4 + HC ee86b66 + serve wiring), serve_bench int-hc3, unprofiled, smoke-hc PASS (rebased hashes),
bench p0 baseline recorded at int-hc2 (identical in int-hc3). Bench fixes: record mode + EOS-before-256 allowed (8524431..186329a).
RESULT: C1 65.5 / C2 92.0 / C4 110.5 / C8 128.9 agg tok/s (MTP policy); no-MTP C1 54.2, C4 110.9; ITL p50 C1 11.6 ms;
prefill 386/379/367 tok/s at 512/2k/8k. C1 MTP window 35 ms vs 18.4 ms per no-MTP token.
VERDICT: +43% C1, +16% C8, +27% prefill vs serve #1. MTP draft steps are now the C1 bottleneck (host-driven draft loop).

## [B] 21:30 UTC - DPAS dense on fn-int: +3.6-4.4% prefill; HC is 48% of prefill kernel time
fn-rt-dpas, XPU_FN_DPAS=1: 2048 ids 362 -> 378 tok/s, 20K 350 -> 363 (unprofiled, both cards). 128-row profile: HC down+up
149 of 310 ms kernel time per card; Q6 dense now 20 ms. Next prefill lever is HC (Lead A H) at large M. Eviction-in-serve:
C4 capped smoke found a fill-retry stall (lane_host_fill zero-progress loop), fix eb61491, re-running.

## [B] 21:55 UTC - critical-path ledger started (user 21:50)
codex-trace redirected from DPAS to the ledger on fn-hc e12b9610 + tracer v2 (b70/fn-rt-ledger, serve :18154; C1/C4 windows,
per-kernel/gap/movement accounting, top-10 with owner). Parallel Claude Opus agent (CPU only) builds the bytes-needed model per
kernel from the GGUF inventory (B/ledger/BYTES-hc2.*). Eviction-in-serve: C4 capped smoke PASS (2 parks / 2 restores, no
prefill replay); bench found a lease-return bug on stream finish (LaneMemory), fix in progress. C2 uncapped int8 48.8 agg.

## [B] 22:05 UTC - bytes-needed model for fn-hc e12b9610 (CPU, [I])
CONFIG: Claude worker (claude-opus-5-5), code + GGUF inventory, serve MTP policy (lane=3, C1 4-row verify, C4 8-row).
RESULT [I]: per card per window C1 5.26 GB (8.65 ms ideal) vs 32.7 ms measured; C4 7.37 GB (12.1 ms) vs ~55 ms. Experts 1.64/3.16 GB,
HC readout 1.33 GB (replicated, row-independent), dense 0.95 GB. Waste: hc_down weight re-read per 4 rows, triple quantize,
768-1536 memcpy appends/window. VERDICT: windows run at 22-26% of the byte roofline; join with the measured ledger next.
B/ledger/BYTES-hc2.md.

## [A] 2026-11-20 22:08 UTC KF round 5 accepted; kv8 + hc2 merged
- KF5 (fn-model 5b62ca4..6cb4f56): GDN R4 recurrence 8.0 -> 5.0 us, KV4805 R4 attention 65.9 -> 49.4 us, C4 expert up -2..5%;
  projected saving 0.19 ms C1T2 / 1.26 ms C4T4 per card-window (bitwise + fp64 + health pass). Not yet merged.
- fn-a3/fn-int 45685f4 = kv8 (36738f3, default off; smoke-kv8off hashes identical) + hc2 (9fb00f5; tokens identical per H).
- KF6: prepared q8_1 inputs + removal of per-row project() memcpy appends (Lead B BYTES-hc2 items) on b70/fn-prep from fn-rec.

## [A] 2026-11-20 22:22 UTC R2 recorded serve (fn-rec 99bda20, no hc2) and combined build
RESULT (R/serve/r2-results.json, worker R): policy C1 79.9 / C2 101.5 / C4 134.0 / C8 160.5 agg tok/s, ITL p50 C1 9.4 ms,
all speculative token references matched; no-MTP C1 61.5. no-MTP C4 then HUNG: "FN_RECORD abort bucket=4 rank=0 ZeFailed"
(Level Zero 0x1); xe-reset rebind recovered, post-health HEALTHY. R is retrying with per-hop timestamps.
VERDICT: R2 unqualified (no-MTP C4 hang). Combined fn-a3 99e31ff (hc2 + kv8 off + R2) is building; bench defaults to the
no-MTP C1 control only under XPU_FN_RECORD (18be9ac, tools only) so the policy ladder is measured without the known hang.

## [B] 22:42 UTC - KV eviction to RAM working on the real two-card serve
CONFIG: b70/fn-rt-evict 4275ddc (fn-int 8524431 + kv8 + park2 + admit), int8 KV, XPU_FN_EVICT=1, cap 2 streams, :18153.
RESULT (1 sample, unprofiled): C2 uncapped 48.8 agg; C4 capped 43.4 agg, 44 parks/44 restores, outputs == solo, no prefill replay;
park p50/p99 43.7/585.9 ms, restore 37.1/42.6 ms; C4 worst TTFT 1.77 s (aged admission; was 15.6 s). VERDICT: user goal
"eviction to system RAM restored without recompute" demonstrated on the serve; next p99 park, mirror, idle-Snap prompt cache.

## [A] 2026-11-20 22:57 UTC Serve numbers #3: combined hc2 + R2 recorded windows
CONFIG: fn-a3 99e31ff binary (merge fn-rec 99bda20 + hc2 9fb00f5 + kv8 off), HEAD 18be9ac (bench tools only), smoke-comb PASS (hashes equal),
serve_bench comb-r2 (XPU_FN_RECORD=1, no-MTP C4 control skipped) and comb-imm (off), back to back.
RESULT: RECORD C1 86.4 / C2 108.1 / C4 143.2 / C8 183.4 agg tok/s, ITL p50 C1 8.9 ms, no-MTP C1 66.1, prefill 2k 428;
off C1 69.3 / C2 98.0 / C4 115.2 / C8 134.3, no-MTP C4 115.8. p0 exact, health OK both runs.
VERDICT: recorded windows +25% C1, +37% C8. Not default until R fixes the recorded no-MTP C4 hang.

## [A] 2026-11-20 23:40 UTC R2 NO-GO (stop gate); new path = protocol model + microrepro
RESULT: R retry 7660564 hung again at no-MTP C4 (card 0 in layer-1 MoE AR peer-ready, card 1 at the next attention AR); KF prep-rec
hung at bucket 8 C2. Recovery by rebind each time, post-health HEALTHY. Recorded upside measured: comb-r2 C1 86.4 vs 69.3 immediate.
VERDICT: recorded serve path stopped; R5 does a CPU event-lifecycle model + Fable/Opus consult + device microrepro before any serve.

## [A] 2026-11-20 23:46 UTC LEDGER-hc2 owners accepted
DECISION: from Lead B's traced ledger (cards wait on host: host CPU 11-14 ms/40 ms C1 window; peer wait <= 0.23 ms):
host.fn.mtp -> H3 (re-scoped: device-side draft chain, no inter-depth readback); host.fn.gdn -> KF8 (auto-queued after KF7);
launch_gap -> R5 (recorded windows, after hang fix). R5 restarted with orchestrator H1 (ABA stale ready across replays) and
H2 (rank bucket/AR-count mismatch, add always-on host assert) as first checks.

## [B] 23:58 UTC - LEDGER-hc2 delivered; R5 hang review; three workers launched
RESULT: LEDGER-hc2 (fn-rt-ledger c2a4854 = hc2 + tracer): markers-only overhead ~0 (C1 33.0 vs 33.0 ms off), full trace +20%/+28%.
Per card per traced window: host CPU 11-19 ms, launch 3-7 ms, peer wait <= 0.5 ms; cards wait on the HOST, not on the peer.
Chain attribution 85-96% unknown (end-only edges) -> ledger v2 asked. R5 review (TRANSPORT-B 4g): ABA and bucket mismatch excluded
by construction; immediate lists not drained at execute() entry = real hole; abort-time event dump + parity entries proposed.
Workers: vplan (codex-kv8), arfuse (codex-pushar), ledger v2 (codex-trace), evict round 2 (codex-park, running).

## [B] 00:30 UTC - ledger v2 (int-da583ba)
RESULT: timelines conserve wall; unknown idle < 5%/card; markers overhead +0.8%/+0.4%. Host-bound: per card per C1 window 12-13.5 ms
host CPU vs 22-24.5 ms busy; untraced MTP 9.2 ms, GDN host 6.9 ms (C4 14.0), verify span 23 ms is inclusive of submit.
VERDICT: the critical path is host submission (per-layer launches + MTP host steps); peer waits are negligible. B/ledger/LEDGER-int-da583ba.md.

## [B] 01:25 UTC - host split: event-pool churn found
RESULT (API-traced, +25% overhead): host time per card per C1 window is mostly L0 API (10.7-12.1 ms); Zig code 1.5-1.9 ms. TpComm.syncAll
recreates all MAX_PIPE_STEPS*4 events per sync: 8,192 events/window C1, ~8 ms traced API time. VERDICT: fix (b70/fn-rt-evpool) in progress.

## [B] 01:55 UTC - eviction round 2: mirror + RAM prompt cache
CONFIG: b70/fn-rt-evict a139dab (+ fn-int bf8415b), int8 KV, XPU_FN_EVICT=1 XPU_FN_MIRROR=1, :18153, single samples.
RESULT: park 14.4/14.4/14.6 ms at 1K/8K/32K with mirror (32.0/30.0/76.3 off); park p99 586->118 ms (frame prealloc); ITL p50 cost <0.3%;
repeat-prompt TTFT 25.4 ms via RAM-tier restore (no recompute), tokens identical; post-merge C4 smoke == solo. VERDICT: ready to merge.
Also: verify_plan exclusive 0.06 us (item closed, the marker was inclusive).

## [A] 2026-11-21 02:12 UTC Serve row #4: int-prep (hc2 + KF5 + KF6 prepared inputs)
CONFIG: fn-a3 bf8415b, RECORD/DESC off, smoke PASS (hashes equal). RESULT: C1 71.7 / C2 101.4 / C4 123.4 / C8 146.2 agg, prefill 2k 484.
VERDICT: +7% C4, +9% C8, +13% prefill vs comb-imm. Bench pre-wait bug (rg missing in subshell) fixed in 2f6b5eb.

## [B] 02:16 UTC - evpool shipped to integration
CONFIG: fn-rt-evpool 3404c06 on da583ba, XPU_TP_EVPOOL=1 vs 0, markers-only, same lease, tokens identical. Integration branch b70/fn-rt-evpool-int 740aca0 (bf8415b + 4 commits).
RESULT: C1 lane-round p50 33.17->28.01 ms, 71.1->82.2 tok/s; C4 50.25->43.36 ms, 58.4->61.6 tok/s; zero spare waits.
VERDICT: largest single host fix so far; handed to Lead A for fn-int merge + serve_bench.

## [B] 03:42 UTC - correction
The 02:16 entry "evpool shipped to integration" means handed to Lead A (branch b70/fn-rt-evpool-int 740aca0), not merged. It is not in
fn-a3 6d92e0f; it goes in the next merge train. Also 03:28: PRIORITY.hold gate added to bin/gpu-run + bin/cpu-run (uncommitted; tested).

## [O] 2026-11-21 03:45 UTC - USER directive: quick loop
GPU claims <= 5 min (most 1-3), per-merge gate = 2 prompts x 32 tok hash + C2==solo, short bench C1/C4/C8 128 tok + prefill 2k (~3 min), health only per session, no soaks/suites until near targets; merge on pass, revert failures out of the train.


## [B] 04:03 UTC - C8/C16 with a RAM tier
CONFIG: fn-rt-evict 2c5d885 (+6d92e0f), int8 KV, slot release on park; 2K prompts, 256 tok, single runs.
RESULT: C16/P16 78.9 agg; C16 on 2 physical slots (q4) 53.5 agg with 283 parks, outputs == solo for the checked subset; TTFT p50 77 s at C16.
VERDICT: logical > physical slots works; cold TTFT is prefill-bound. Ready to merge.

## [A] 2026-11-21 04:30 UTC Merge train 1 (decision build)
CONFIG: fn-a3 3edbc4d = DESC default (6d92e0f) + H3 device MTP chain (85feb5b) + B evpool + R5 F1/drain; quick bench (128 tok, C1/C4/C8, prefill 2k).
RESULT: immediate C1 82.6 / C4 151.4 / C8 191.6 / prefill 503 (p0 exact, streams == solo, health OK).
RECORD=1: load fails (ZeFailed after evpool init). RECORD=1 + XPU_TP_EVPOOL=0: C1 83.7 / C4 136.5 / C8 163.8.
VERDICT: targets C1 >= 110 / C8 >= 220 not met. Recording no longer pays once evpool + device MTP land; default stays immediate.

## [B] 05:00 UTC - decision ledger train1-imm 3edbc4d
CONFIG: fn-rt-ledger-t1 a923db7 (= 3edbc4d + tracer), C1/C4 ~64 tok greedy, markers + full, one 246 s lease, health OK.
RESULT: C1 traced 31.0 ms/window, busy 21.1/20.9 ms (68%) per card, host 3.1-3.6, launch 3.1-3.2, peer <= 0.17; markers-only 27.4 ms for
2.03 tok (C4 39.4 ms for 6.85 tok). VERDICT: GPU-busy-bound; zero-host bound ~97 tok/s at C1 -> kernels + tokens/window are the path; stay on fork.

## [O] 2026-11-21 04:35 UTC - USER targets set: C1 100 / C2 150 / C4 300 / C8 500 agg decode; prefill >= 1500 single. See BRIEF.md (roofline check: needs ~38/52/57% of byte roofline).

## [A] 2026-11-21 05:30 UTC Serve row #7: PF prefill DPAS merged
CONFIG: fn-a3 aa66709 (train1 + PF d8baa8d: DPAS HC at rows >= 64, dense/routed experts at rows >= 256), XPU_CHUNK_SOLO 512 / 2048, quick bench.
RESULT: 512: C1 83.1 / C4 151.1 / C8 191.3, prefill 2k 1024. 2048: C1 82.7 / C4 150.6 / C8 147.9, prefill 1147. p0 exact, health OK.
VERDICT: prefill 2.0x (503 -> 1024) with decode unchanged; chunk 512 default (2048 hurts C8). Targets: C1 83% / C4 50% / C8 38% / PF 68%.

## [O] 2026-11-21 05:32 UTC - DECISION: stay on the fork
LEDGER-train1-imm: C1 device-busy dominant (21 of 31 ms traced; host CPU ~3, launch ~3, peer ~0.1). Kernels ~2.4x byte ideal. Priorities: AR fold into consumer prologue (3.8 ms/window both cards), fused HC chain (4.5 ms/card), fused sibling projections, router occupancy; RECORD x evpool deferred.

## [B] 06:10 UTC - routing checks (3edbc4d)
RESULT: distinct experts per card per layer: C1 13.0 (uniform 19.4), C4 30.2 (37.4), C8 distinct prompts 54.6 (69.3). Card imbalance per window
1.5-1.9% mean; per layer at C8 8.3% mean / 17.5% p90; static rebalance ~0.07 ms/window. VERDICT: grouped-by-expert decode worth ~20-33% of expert
bytes beyond uniform math; expert rebalancing dropped.

## [B] 06:15 UTC - arfuse A/B
CONFIG: fn-rt-arfuse b3e7e8e, XPU_FN_AR_FUSE=0/1, pinned 16-31, one lease. RESULT: AR launches 204->12/round C1, window flat (33.27->33.13 ms),
C4/C8 +0.25/+1.3 ms (noise level). VERDICT: AR cost is push-arrival latency, not the reduce kernel; next = producer-epilogue peer stores microbench.

## [B] 06:50 UTC - lane policy sweep
RESULT (3edbc4d, 128 tok, single): C2 d1/d2/d3 116.3/116.3/118.2; C4 151.4/140.5/136.5; C8 191.9/165.7/157.4. VERDICT: depth 1 stays at C4/C8;
C2 depth 3 within noise; MTP depth not a C4/C8 lever.

## [A] 2026-11-21 07:18 UTC -- serve rows int-pf2, int-mtp2
CONFIG: fn-a3 2133d91 (+PF 7973882: dense prefill GEMM retile, pair-major expert gather) and e9a37df (+H4 156e010 multi-lane device
MTP, +HS cbb488c writeback_norm2 / down-once R5-16). Quick bench (FN_BENCH_QUICK=1, 128 tok, C1/C4/C8, prefill 2k, chunk solo 512).
COMMAND: B70_AGENT=A-S S_SKIP_BUILD=1 bash tools/flashnext_serve_bench.sh fn-a3 <tag> (PRIORITY.hold held).
RESULT: int-pf2   C1 83.4 / C4 151.0 / C8 187.8 / PF 1351 (target C1 83% C4 50% C8 38% PF 90%)
        int-mtp2  C1 92.2 / C4 149.5 / C8 187.7 / PF 1362 (target C1 92% C4 50% C8 38% PF 91%); ITL p50 C1 8.1 ms.
        p0 equal to solo at C4/C8, oracle_equal true, health true. Results workers/S/int-pf2-results.json, int-mtp2-results.json.
VERDICT: merged; fn-int ff to e9a37df. Prefill +170% since bf8415b (483 -> 1362). C1 +9 from multi-lane device MTP + HS norm.
C4/C8 flat: the multi-row decode path (grouped experts / DPAS at R>=8) is the gap; AU audit (expert up grids in __constant,
raw 110 B blocks) -> LY worker R1/R2.

## [B] 07:42 UTC - OOO dispatch (card 0) + milestone boundary
RESULT: sibling kernels per layer, OOO replay vs immediate: C1 -4.7% (35.4->33.7 us), C4 -11.9% (41.5->35.5 us); ~0.1/0.3 ms per window.
VERDICT: below threshold; do not wire. Lead B stops at the M-B3 boundary: eviction/RAM tier/prompt cache/kv8/evpool delivered; arfuse peer-store paused for card 1.

## [A] 2026-11-21 07:53 UTC -- card-0-only window; merges queued; MTP parity status
CONFIG: [INK] holds card 1 (orchestrator 07:25). Lead A workers on card 0 only; one-card merge gate = bitwise unit/op test +
per-kernel us R=1/4/8/16 on real GGUF tensors.
COMMAND: merged PF 4974075 (conflict in xpu_store.zig: additive fields hc_once + pf_exp, kept both) and KF 23fb20b into fn-a3;
restored tools/flashnext_serve_bench.sh from before the KF merge; zig build xpu-fn-a4 xpu-flashnext xpu-native (cpu-run, -j4).
RESULT: fn-a3 113ee5c builds. Two-card quick gates were passed on the worker branches before 07:25 (PF smoke-runs/pf-4974075,
KF round8/fusion_gate/kf8-fusion-gate-quick.json). R: fp32 draft logits leave depth-1 accept counts identical (p0 94/152,
p1 122/137, p2 120/138, p3 100/141); TF misses Strata-accepted drafts with main-model target p median 0.95 (confident);
code read shows no structural drafter mismatch. AU audit: all quantized weights raw GGUF blocks; expert-up grids in __constant.
VERDICT: serve rows for 113ee5c and the XPU_CHUNK_SOLO=1024 arm queued for card 1. Drafter gap = numeric suspects, to be bisected
with the dump + FP64 reference in one two-card lease. LY (layout repack) and H5 (single-card window harness) started.

## [A] 2026-11-21 08:20 UTC -- R: drafter FP64 reference validated (CPU only)
CONFIG: R/mtp/validate_ref.py; TF's tools/zig/qwen4_reference.py (binary64) with real Flash-Next MTP BF16 tensors, random source
streams, 3 cells at pos 0-2; compares every intermediate with R/mtp/mtp_ref.py.
RESULT: rel err 1.5e-3 .. 9.4e-3 at all HC/attention/writeback stages (oracle bf16-node floor); negative controls fail loudly
(no Gemma +1: 0.76; rope pos off by one: 1.45). MoE/head code-matched to fn_moe_head.cl / fn_dpas.cl.
VERDICT: reference trusted. R parked; run_dump.sh (bf16 vs XPU_FN_DRAFT_RES_F32 arms, 2 x ~3 min two-card) waits for card 1.

## [A] 2026-11-21 09:00 UTC -- one-card slice gate on integration fn-a3 988effc
CONFIG: fn-a3 988effc = e9a37df + PF 4974075 + KF 23fb20b + H 7aa50080 (slice harness) + HS ddc31a9 (HC_Q8 off) + PF 17d2d7bf
(PF_ATTN/PF_GDN off). Slice = layers 2 (GDN) + 3 (QSA), real rank-0 TP2 weights, local peer buffer, prefix 256, lanes 1, card 0.
COMMAND: gpu-run --card 0 env ZE_AFFINITY_MASK=0 timeout 170 bash workers/S/slice-988effc/session.sh (tf-xpu-fn-slice MAIN PLE 3 256 1).
RESULT: hashes == H reference (b59c8b.., bf6ff7.., a2f6ea.., debff3..) in all 4 mode/order combos at R=1/4/8/16; health pre/post OK.
Kernels/slice 56 (R16 60), was 68/84 before KF fusion. Window median us R1/4/8/16: immediate in-order 603/796/979/1549;
immediate OOO 605/662/906/1488; recorded in-order 519/682/908/1522; recorded OOO 463/623/854/1434.
VERDICT: integration default path bitwise; recorded+OOO is -23% (R1) / -7% (R16) on the slice vs serving's immediate in-order ->
H round 6 makes it serving-viable. Serve rows queued for card 1.

## [A] 2026-11-21 09:58 UTC -- integration fn-a3 1c15dcdf, one-card gate
CONFIG: fn-a3 1c15dcdf = 988effc + PF 4035abb2 (ATTNS, off) + LY 8930f006 (expert split layout + SLM grids, on, bitwise) + H e0fa1982
(XPU_FN_OOO, off; RECORD+evpool cleanup-slot fix). Card 0, slice MAIN PLE 3 256 1.
COMMAND: gpu-run --card 0 ... workers/S/slice-1c15dcd/session.sh (also slice-233608f after the LY merge).
RESULT: hashes == reference in all 4 modes at R1/4/8/16; health OK. Window medians us R1/4/8/16: immediate in-order 641/693/933/1552,
immediate OOO 555/653/878/1459, recorded OOO 502/671/864/1386. LY fixture (iq3 types, real weights): iq3_xxs up R8 252 -> 184 us,
iq3_s R8 253 -> 198 us; slice layers use iq2_* (smallest gainers). H: immediate OOO gain is gap reduction, not overlap (overlap 0 with fusion).
VERDICT: merged; serve qualification queued for card 1. Host not quiet: slice numbers are diagnostic.

## [A] 2026-11-21 10:33 UTC -- integration fn-a3 d97096cb (DENSE_DPAS default on, LY down/MTP split, H fused recording)
CONFIG: fn-a3 d97096cb = 1c15dcd + PF a91c2670 (XPU_FN_DENSE_DPAS, default ON via 3299501d: bitwise == KREG) + LY 922bedd7 (routed
down + MTP q2_0 split layout, load fix) + H 4376fe4d (recording captures fused/packed paths). Card 0 slice MAIN PLE 3 256 1.
COMMAND: gpu-run --card 0 ... workers/S/slice-3299501d/session.sh, then slice-d97096cb/session.sh.
RESULT: hashes == reference (b59c8b/bf6ff7/a2f6ea/debff3) in all modes at both commits; health OK; launches 56 (R1-8) / 73 (R16).
Slice window medians us R1/4/8/16: recorded OOO 465/615/822/1308 (988effc: 463/623/854/1434); immediate in-order 676/697/928/1432.
VERDICT: merged; default path still bitwise. Serve rows pending card 1 (H/r7/card1-validate.sh first).

## [A] 2026-11-21 11:22 UTC -- fn-a3 fc200f43 (PF DPAS v2), one-card gate
CONFIG: fn-a3 78e52926 (H slice TAIL harness, ff) + PF 15c82f77 (dense DPAS v2, bitwise; LM/draft head q5_k from R=2). Card 0 slice.
COMMAND: gpu-run --card 0 ... workers/S/slice-fc200f43/session.sh.
RESULT: hashes == reference all modes R1/4/8/16; health OK (card 0 only probed). recorded OOO 462/623/824/1298 us; immediate in-order
662/697/1079/1440 (R8 noisy). PF op bench: LM head R16 208.9 (KREG) -> 96.1 us DPAS, ideal 58.7.
VERDICT: merged; default bitwise. Serve rows still pending card 1.

## [A] 2026-11-21 12:13 UTC -- fn-a3 e8d534c0 (bf16 weight-stationary, batched keep default on)
CONFIG: fn-a3 = fc200f43 + LY 42662d33 (timing rotate) + PF 5601aefb (fn_bf16t, on) + H 9d9eef8e (keep batch) + e8d534c0 (KEEP_BATCH default 1).
COMMAND: gpu-run --card 0 ... workers/S/slice-e8d534c0/session.sh (layer slice MAIN PLE 3 256 1; tail slice XPU_SLICE_TAIL=1 MAIN PLE 2 256 16).
RESULT: layer hashes == reference; tail SLICE_BITWISE PASS (32 samples), SLICE_LAUNCHES PASS (109/109/112/128), TAIL_KEEP_BATCH PASS
(rows 1/4/8/16 x lanes 1/4/8/16 + odd lanes, mixed windows, reversed lanes); health OK. H probe: keep C4 1.13 -> 0.52 ms, C8 2.26 -> 0.90 ms.
VERDICT: merged, default on (bitwise). Two-card serve check pending card 1.

## [A] 2026-11-21 13:26 UTC -- fn-a3 c2ae97b8: card-0 merges while [INK] holds card 1
CONFIG: fn-a3 c2ae97b8 = b4dce2a1 + H dd5aeb7c (XPU_MTP_HC_Q8 off) + LY 4f0878e4 (dense IQ3_S row-split ON, bitwise; XPU_MTP_Q8ACT off)
+ HS af8c82b9 (XPU_ATTN_MERGE_WIDE off) + LY 9ed8a9a4 (staged parallel split repack). KF cafb9ec6 not merged (13-file conflict).
COMMAND: gpu-run --card 0 ... workers/S/slice-c2ae97b8/session.sh (layer slice + TAIL slice).
RESULT: layer hashes == reference; SLICE_BITWISE, SLICE_LAUNCHES (109/109/110/122), TAIL_KEEP_BATCH PASS; health OK. Layer slice
recorded OOO us R1/4/8/16 470/624/836/1290. LY load: rank-0 load 11.6 -> 5.1 s (raw 7.8).
VERDICT: merged; default path bitwise vs reference since e9a37df. H prepared H/r11/card1-queue.sh (whole two-card backlog, ordered,
bounded leases, per-step PASS criteria in PASS-GATES.txt) -- run first when card 1 returns.

### [A] 2026-11-21 14:44 UTC -- merges 38e2d99c + 6823d59d; C1 window ranking becomes the plan
CONFIG: fn-a3 merges of LY 38e2d99c (IQ3_S rs-only) and PF 6823d59d (LM-head DPAS with 1024 persistent WGs). Card 0 only ([INK] holds card 1).
COMMAND: the mergeGate one-card gate, i.e. workers/S/slice-<head>/session.sh (layer slice R1/4/8/16 + TAIL, xpu-health --card 0).
RESULT:
- 06dfcb1d: recOOO [449.3, 602.6, 822.5, 1227.3]. Hashes b59c8b/bf6ff7/a2f6ea/debff3. BITWISE / LAUNCHES (109,109,110,122) / KEEP_BATCH PASS. HEALTHY.
- 3fe36cf3: recOOO [455.4, 607.7, 822.6, 1236.7]. Same hashes; all PASS; HEALTHY.
- LY claims -65 MB/card and attn_q R1 18.3 -> 13.6 us.
- PF C1 ranking (workers/PF/c1-0.log; window = 36 GDN + 12 QSA at R4 + tail + 3 R1 draft steps):
  - kernel sum 15.6 ms, plus 0.34 ms partial keep;
  - about 5.9 ms over the per-rank byte ideal at 608 GB/s;
  - top excess: fused_project_c4 +954, hc_down +879, hc_upmix +782, ups_iq2_xs +530, moe_compact +479, router_shared +401, q4_k ssm_out +308, route_topk +279, hc_writeback_norm2 +244, iq4_nl shared down +188 us.
VERDICT:
- Both merges are accepted (bitwise, default-on). fn-int stays at e9a37df until card 1 returns.
- PF's non-bitwise q5_k finish (40-60 us at R16) is declined as too small for a KL gate.
- The C1 window excess is spread over HC, dense projections and routing glue. Assigned PF (dense 1/7/10), HS (HC), KF r11 (routing glue) and LY (ups_iq2_xs).

### [A] 2026-11-21 15:07 UTC -- merge LY 32ca6f77 (IQ4_XS R=2 RP8)
CONFIG: fn-a3 merge of b70/fn-ly 32ca6f77. Card 0.
COMMAND: mergeGate one-card session (slice + TAIL, xpu-health --card 0).
RESULT:
- recOOO [451.2, 599.9, 819.4, 1229.7] {1: {'b59c8b'}, 4: {'bf6ff7'}, 8: {'a2f6ea'}, 16: {'debff3'}}
  SLICE_BITWISE PASS samples=32 rows=1,4,8,16 event_pool_capacity=1024 event_pool_reused=true lanes=16 transport_pool=true
  SLICE_LAUNCHES PASS rows=1,4,8,16 counts=109,109,110,122
  TAIL_KEEP_BATCH PASS rows=1,4,8,16 lanes=1,4,8,16 odd_lanes=3,7,9,15 mixed_windows=true reversed_lanes=true banks=both scratch_canaries=true
  xpu-health: HEALTHY (cards 0)
- LY's measure-first, all bitwise:
  - IQ4_XS row split gains only 3-7% at R1-4, +15% at small N, and spills at c8: rejected.
  - Arithmetic codebook regresses project_fused at R1/R4 by 7%/13.5%: rejected.
  - RP8 at R=2 for N>512 wins 8-18%: merged.
VERDICT: accepted, bitwise default-on. It helps the C2 (R=2 verify) shapes.

### [A] 2026-11-21 15:17 UTC -- merges LY 32ca6f77 (fast-forward) + HS 84ffb7f0 (HC block reads)
CONFIG: fn-a3 at 32ca6f77, then merge of HS 84ffb7f0 (XPU_HC_BLOCK=1 default). Card 0.
COMMAND: mergeGate one-card session for each.
RESULT:
- 32ca6f77: recOOO [451.2, 599.9, 819.4, 1229.7]; all PASS; HEALTHY.
- 72d50dd0: recOOO [447.3, 595.2, 794.8, 1195.8] {1: {'b59c8b'}, 4: {'bf6ff7'}, 8: {'a2f6ea'}, 16: {'debff3'}}
  SLICE_BITWISE PASS samples=32 rows=1,4,8,16 event_pool_capacity=1024 event_pool_reused=true lanes=16 transport_pool=true
  SLICE_LAUNCHES PASS rows=1,4,8,16 counts=109,109,110,122
  TAIL_KEEP_BATCH PASS rows=1,4,8,16 lanes=1,4,8,16 odd_lanes=3,7,9,15 mixed_windows=true reversed_lanes=true banks=both scratch_canaries=true
  xpu-health: HEALTHY (cards 0)
- HS: HC_COLD_BITWISE at R=1..16 on 2 real weight sets; 154 TAIL hash lines identical.
- HS diagnosis: at R1, HC sits at the HARDWARE rule-7 small-matrix floor (14.2 us for 6.6 MB). PF's 608 GB/s ideal omitted the ~3 us ramp, so the real HC excess was only at R>=4 (narrow loads).
VERDICT: both accepted, bitwise default-on. The C1 HC excess in PF's ranking is mostly the launch floor; further HC gains need overlap or fusion.

### [A] 2026-11-21 15:34 UTC -- merge PF dc26dccd (fused_project h2)
CONFIG: fn-a3 merge of b70/fn-pf dc26dccd (default on). Card 0.
COMMAND: mergeGate one-card session.
RESULT:
- recOOO [447.8, 586.2, 797.9, 1193.2] {1: {'b59c8b'}, 4: {'bf6ff7'}, 8: {'a2f6ea'}, 16: {'debff3'}}
  SLICE_BITWISE PASS samples=32 rows=1,4,8,16 event_pool_capacity=1024 event_pool_reused=true lanes=16 transport_pool=true
  SLICE_LAUNCHES PASS rows=1,4,8,16 counts=109,109,110,122
  TAIL_KEEP_BATCH PASS rows=1,4,8,16 lanes=1,4,8,16 odd_lanes=3,7,9,15 mixed_windows=true reversed_lanes=true banks=both scratch_canaries=true
  xpu-health: HEALTHY (cards 0)
- PF: R4 layers 2+3 70.9 -> 65.8 us (interleaved A/B, 2 reps).
- With fusion off, iq4_xs (15.7 MB) takes 53.6 us vs 26 ideal, so the excess is iq4_xs weight streaming.
- DPAS t8 is worse at N=5120 (160 tiles).
VERDICT:
- Accepted, bitwise default-on.
- The IQ4_XS row split was already measured by LY and rejected: decode-limited, not load-limited.
- PF moves to the q4_k ssm_out two-quad form.

### [A] 2026-11-21 15:51 UTC -- HS norm2+down fusion: bitwise but slower (recorded, not merged); L2 ceiling
CONFIG: HS 743e11d7 (b70/fn-hs-hcn, XPU_HC_FUSE=1 default off): norm2 + grid barrier + b2/b4 down in one kernel. Card 0.
COMMAND: hc-cold bitwise + HC_HOT, slice p256 recorded OOO.
RESULT:
- Bitwise PASS; launches 56 -> 52.
- In-slice us at R1/R4/R8/R16: separate 22.6/20.8/21.7/32.0 vs fused 30.65/32.45/35.83/48.80. The grid barrier (~2.2 us) and device fences (~5 us) cost more than the launch saves.
- L2-hot HC: down R1 3.65 vs 14.48 cold, upmix 5.73 vs 13.02.
VERDICT:
- The fusion is rejected: grid barriers are not a win on this hardware at these sizes. This adds to the HARDWARE rules.
- Approved next: an XPU_HC_PREFETCH prototype, a read-only L2 warm of the next HC weights in the combine/AR/writeback span. It is judged on the slice window.

### [A] 2026-11-21 16:03 UTC -- HS HC L2 prefetch: lands in L2, window flat (not merged)
CONFIG: HS 6373c2c7 (b70/fn-hs-pf), XPU_HC_PREFETCH=C, either folded into norm2 or as an OOO sibling. Card 0 slice p256.
COMMAND: slice recorded OOO, reps 1-2, flag off vs C2/C3/C5.
RESULT:
- Hashes unchanged.
- Window off 447.9/590.3/801.2/1211.0 vs fold C2 440.9/594.6/807.4/1204.1 us, against about +-5 noise.
- Down at R1 falls 15.8 -> 5.6 us, but norm2 and the prefetch absorb the same DRAM time.
VERDICT:
- Rejected for the one-card slice: DRAM time is conserved, and only the ~5 us of idle DRAM under norm2 is reclaimable.
- The real lever is the two-card AR peer wait. An XPU_HC_PREFETCH_AR variant will be prepared and queued for card 1.

### [A] 2026-11-21 16:14 UTC -- merge LY 517d7c47 (IQ2 hw half->float)
CONFIG: fn-a3 merge of b70/fn-ly 517d7c47. Card 0.
COMMAND: mergeGate one-card session. The first attempt aborted before the GPU claim (a lead-side REPL variable clobbered the GPU.md path); it was rerun.
RESULT:
- recOOO [444.5, 588.0, 794.9, 1187.5] {1: {'b59c8b'}, 4: {'bf6ff7'}, 8: {'a2f6ea'}, 16: {'debff3'}}
  SLICE_BITWISE PASS samples=32 rows=1,4,8,16 event_pool_capacity=1024 event_pool_reused=true lanes=16 transport_pool=true
  SLICE_LAUNCHES PASS rows=1,4,8,16 counts=109,109,110,122
  TAIL_KEEP_BATCH PASS rows=1,4,8,16 lanes=1,4,8,16 odd_lanes=3,7,9,15 mixed_windows=true reversed_lanes=true banks=both scratch_canaries=true
  xpu-health: HEALTHY (cards 0)
- LY ups_iq2_xs R4 probes: no-decode 64.7 vs base 78 us. Every bitwise restructure is slower.
VERDICT:
- Accepted (bitwise, default on).
- The next ups gain needs a reduction-order change. Approved as a flagged default-off variant, gated on max-ulp plus slice argmax now and KL on card 1 later.

### [A] 2026-11-21 16:22 UTC -- merge PF 2b0de0ed (Q4_K persistent groups at R3-4)
CONFIG: fn-a3 merge of b70/fn-pf 2b0de0ed. Card 0.
COMMAND: mergeGate one-card session.
RESULT:
- recOOO [455.2, 594.7, 803.4, 1220.1] {1: {'b59c8b'}, 4: {'bf6ff7'}, 8: {'a2f6ea'}, 16: {'debff3'}}
  SLICE_BITWISE PASS samples=32 rows=1,4,8,16 event_pool_capacity=1024 event_pool_reused=true lanes=16 transport_pool=true
  SLICE_LAUNCHES PASS rows=1,4,8,16 counts=109,109,110,122
  TAIL_KEEP_BATCH PASS rows=1,4,8,16 lanes=1,4,8,16 odd_lanes=3,7,9,15 mixed_windows=true reversed_lanes=true banks=both scratch_canaries=true
  xpu-health: HEALTHY (cards 0)
- PF: ssm_out R4 15.6 -> 14.4 us. Group counts 128/384 are worse; the h2 form is 25 us.
- IQ4_XS LUT alternatives in fused R4, all bitwise: SLM 65.3, early loads 75.8, __constant 98.5, shuffle 176 us.
VERDICT: accepted. The SLM codebook stays the best IQ4_XS decode on Xe2; this adds to the HARDWARE rules.

### [A] 2026-11-21 16:51 UTC -- merges HS 092c82f0 (GDN rows J2, made default) + 07c01aab (HC_PREFETCH_AR, off)
CONFIG: fn-a3 merges of b70/fn-hs-gdn 092c82f0 and b70/fn-hs-ar 07c01aab, plus lead commit e8e21a4f (XPU_GDN_ROWS_J default 2). Card 0.
COMMAND: buildGate one-card session on e8e21a4f.
RESULT:
- recOOO [444.7, 583.1, 789.8, 1178.2] {1: {'b59c8b'}, 4: {'bf6ff7'}, 8: {'a2f6ea'}, 16: {'debff3'}}
  SLICE_BITWISE PASS samples=32 rows=1,4,8,16 event_pool_capacity=1024 event_pool_reused=true lanes=16 transport_pool=true
  SLICE_LAUNCHES PASS rows=1,4,8,16 counts=109,109,110,122
  TAIL_KEEP_BATCH PASS rows=1,4,8,16 lanes=1,4,8,16 odd_lanes=3,7,9,15 mixed_windows=true reversed_lanes=true banks=both scratch_canaries=true
  xpu-health: HEALTHY (cards 0)
- HS: J2 is bitwise (4 lanes-1 hashes and 162 lanes-16 TAIL lines identical). gdn_rows at R>=4 is latency-bound at 0.79 us/row on the serial recurrence; J2 overlaps two rows' chains.
- HS: gdn_rows R1 sits at the state-byte floor (3.15 MB of f32 state per layer). PF's +173 us excess assumed zero state bytes.
VERDICT:
- Accepted. J2 is default-on per the bitwise-default policy.
- PREFETCH_AR stays default off until the card-1 A/B arms (HS block in H/r11/card1-queue.sh).

### [A] 2026-11-21 17:14 UTC -- merge PF d1a19df6 (KREG persistent-group table)
CONFIG: fn-a3 merge of b70/fn-pf d1a19df6. Card 0.
COMMAND: mergeGate one-card session.
RESULT:
- recOOO [450.7, 594.3, 799.4, 1189.2] {1: {'b59c8b'}, 4: {'bf6ff7'}, 8: {'a2f6ea'}, 16: {'debff3'}}
  SLICE_BITWISE PASS samples=32 rows=1,4,8,16 event_pool_capacity=1024 event_pool_reused=true lanes=16 transport_pool=true
  SLICE_LAUNCHES PASS rows=1,4,8,16 counts=109,109,110,122
  TAIL_KEEP_BATCH PASS rows=1,4,8,16 lanes=1,4,8,16 odd_lanes=3,7,9,15 mixed_windows=true reversed_lanes=true banks=both scratch_canaries=true
  xpu-health: HEALTHY (cards 0)
- PF sweep: G in {tiles, 128, 256, 384, 512} with interleaved A/B. Wins: q4_k c1 12.7 -> 12.0, q6_ks c1/c4/c8 14.8/16.7/21.3 -> 14.0/15.6/18.0, draft head c1 77.8 -> 67.1 us.
- Kernels already at their best: the lm head and DPAS heads.
- PF saw two health-probe timeouts at 20 s, not real wedges: no xe/GuC dmesg, and a 60 s rerun passed.
VERDICT: accepted (bitwise, default on). Estimated -45 us per C1 window.

### [A] 2026-11-21 17:25 UTC -- merge KF e52617fe (route+compact fusion, q8 producers on norm/gate)
CONFIG: fn-a3 merge of b70/fn-prep e52617fe (XPU_FN_ROUTE_FUSE=1, XPU_FN_Q8_PRODUCERS=2 by default). The additive conflict in xpu_store.zig (project4h vs shared_quant fields/loads) was resolved by keeping both. Card 0.
COMMAND: buildGate one-card session on 2f022b5f.
RESULT:
- recOOO [450.2, 576.9, 779.7, 1189.0] {1: {'b59c8b'}, 4: {'bf6ff7'}, 8: {'a2f6ea'}, 16: {'debff3'}}
  SLICE_BITWISE PASS samples=32 rows=1,4,8,16 event_pool_capacity=1024 event_pool_reused=true lanes=16 transport_pool=true
  SLICE_LAUNCHES PASS rows=1,4,8,16 counts=104,104,105,117
  TAIL_KEEP_BATCH PASS rows=1,4,8,16 lanes=1,4,8,16 odd_lanes=3,7,9,15 mixed_windows=true reversed_lanes=true banks=both scratch_canaries=true
  xpu-health: HEALTHY (cards 0)
- KF matrices:
  - dense ROW_BITS 11640 cases, route/plan 661, shared 2640, HC 54, all zero differing bits;
  - route+compact R4 19.2 -> 10.2 us per layer;
  - TAIL launches 109 -> 104.
- HC/shared producer tails (mode 1) were latency-rejected and stay opt-in.
VERDICT: accepted, bitwise default-on. fn-a3 2f022b5f is the quick-bench candidate once card 1 is released.

### [A] 2026-11-21 17:46 UTC -- two-card quick serve bench after card 1 returns: C1 and prefill targets met
CONFIG: fn-a3 2f022b5f, which carries every card-0-gated merge since e9a37df: LY row-split/rs-only/IQ2 hw half, PF DENSE_DPAS/bf16w/logits/kregGroups/fused h2, H batched keep, HS HC block + GDN J2, KF route fusion. Defaults only, nothing flagged. Then 372bf1b2 = the same plus a bench-script knob adding C2 to quick mode.
COMMAND: integ('int-rf'|'int-rf2', FN_BENCH_QUICK=1): build, I/smoke.sh, tools/flashnext_serve_bench.sh under PRIORITY.hold, 128 tokens, cpu-run --quiet.
RESULT:
- int-rf: C1 109.5, C4 166.5, C8 210.1, PF2k 1567.
- int-rf2: C1 109.5, C2 139.5, C4 168.0, C8 209.0, PF2k 1560.
- Smoke hash_match true; oracle_equal true; health true.
- C2 policy ITL p50 11.1 ms. C4 min stream 42.0 tok/s, ITL p99 81 ms.
VERDICT:
- C1 (109%) and prefill (104%) meet the targets. C2 is at 93%, C4 56%, C8 42%.
- fn-int ff'd to 372bf1b2.
- Next: the card-1 flag A/Bs, then the C4/C8 dispersed-routing work.

## [O] 2026-11-21 18:15 UTC - OOM at 17:59:55 ended the campaign session; state recorded
CONFIG: host 121 GB RAM + 8 GB swap. At 17:59:55 five concurrent zig builds (~2.5 GB each) + agent processes exhausted RAM+swap (swap 7/7 full).
RESULT: global OOM killed tmux, the orchestrator session, all lead/worker agents and the user systemd manager. On manager restart (18:04:59)
the enabled units hotschmoe-dd-frontdoor and hotschmoe-dd (Strata 0.2.16) auto-started; Strata is ready at 18:0x: /v1/models via :18080 lists
hotschmoe-dd + alias, n_ctx 200000, images on. Lead A wrap-up item 4 and the 17:57 "cutover probe c200-desc" (ctx 200000 par 4, port 18151)
did not complete (no result). PRIORITY.hold is gone. All TensorFold worktrees are clean (no uncommitted tracked changes):
fn-a3 e87b5074 (newest one-card-gated integration head), fn-int 372bf1b2 (= int-rf2, best two-card-gated build), worker branches
fn-pf fee99fa3, fn-hc dd5aeb7c, fn-rec 9cd60dc6, fn-prep e52617fe, fn-ly 56ec4ac9. Cutover prep files: tensorfold/flashnext/dd/ (launcher, unit,
b1/b2 env, clamp proxy, soak monitor; none installed or started). NEXT-QUEUE.md exists (CPU agent, partial).
VERDICT: no data lost. Rule from now: at most TWO builds at a time host-wide, each under bin/cpu-run (MAX_JOBS<=4); no heavy builds while a server
holds the GPUs. Owner has not chosen A (Strata) or B (TensorFold); Strata stays up untouched until they do.

## [INT] 2026-11-21 18:32 UTC -- b70/dd-serve integration branch, host suite baseline, GPU window runner
CONFIG: TensorFold b70/dd-serve = 372bf1b2 (int-rf2) + 1a5a57d0 (tools/flashnext_serve_bench.py: FN_BENCH_PORT / FN_BENCH_MODEL,
defaults unchanged). Worktree /mnt/vm_8tb/b70/build/tensorfold/dd-serve. CPU only (Strata serves on both cards).
COMMAND: workers/INT/host_suite.sh <wt> base-372bf1b2 build (each step: cpu-run systemd-run MemoryMax=90G zig build <step> -j4,
umask 022): test-kv-pool test-kv-safety-host test-flashnext-xpu-host test-fn-gguf-host test-fn-gguf-mtp-host test-serve-flags-host
test-tool-parse-host test-tool-required-host test-tool-visibility-http-host test-runtime-api test-api-boundaries test-lane-api-host
test-server-security test-server-recovery-host test-template-limits-host test-flashnext-vision-contracts-host test-vision-cli-host
test-xpu-tp-host test-xpu-record-contract-host test-flashnext-pages-contract-host; then xpu-native ReleaseFast.
RESULT: 20/20 host steps PASS at 372bf1b2 (test-flashnext-host and test-qwen-vision-contract-host are not Linux steps; dropped).
test-serve-flags-host fails under umask 002 (config file group-writable -> refused by design); the suite sets umask 022.
xpu-native ReleaseFast rc 0 in 356 s, sha 9423485c... (int-rf2 gated binary was 7bce9c8e... built in fn-a3: the binary depends on
the worktree path, so the frozen build is gated again by G1 in the window). gpu_window.sh CPU self-test (mock /v1/models, no GPU):
SELFTEST PASS (hard failure stops the step loop, soft failure recorded, teardown + post-health always run).
VERDICT: base green. GPU-TESTS.md ordered (G1 regress at the int-rf2 config, G2 one production-config server on :18150 running
gpu_window_steps.tsv, then cutover C1-C3), estimated 31 min GPU. dd/: launcher gains TF_PORT / TF_EXTRA_ARGS and clears inherited
engine env BEFORE sourcing the env file (it used to unset XPU_FN_DESC after); tf-dd-prod.env (200000 x 4, @FREEZE placeholders);
unit points at tf-dd-prod.env; README.md runbook (freeze, window, cutover, rollback). Nothing installed or started. Waiting for
KV/SAMP/VIS/API handoffs.

## [O] 2026-11-21 18:56 UTC - Strata down; both cards to TensorFold for the cutover
CONFIG: owner decision (no users today). COMMAND: systemctl --user stop hotschmoe-dd hotschmoe-dd-soak; bin/gpu-run bin/xpu-health.
RESULT: Strata inactive, frontdoor active, no container, leases free; xpu-health HEALTHY (cards 0 1). Workers KV/SAMP/VIS/API/INT given quick GPU tests (<=5 min leases, ports 18150-18154).
VERDICT: cutover only when dd_functional.py passes through :18080 with an image; rollback = Strata unit.

## [INT] 2026-11-21 19:06 UTC -- two-card quick gate on fn-a3 e87b5074 (candidate base): correctness PASS, speed mixed -> base stays 372bf1b2
CONFIG: fn-a3 e87b5074 = 372bf1b2 (int-rf2) + a8b678ec (MTP rmsnorm_rows + draft argmax head without serial loops, same bits).
Strata down (owner, 18:55); cards 0+1 given to TensorFold.
COMMAND: cd fn-a3; B70_AGENT=INT FN_BENCH_ROOT=workers/INT/a3gate FN_BENCH_EXPECTED=workers/S/expected-current.json FN_BENCH_QUICK=1
tools/flashnext_serve_bench.sh fn-a3 a3-e87b5074 (incremental build under cpu-run; cpu-run --quiet outside one gpu-run lease of 155 s).
RESULT: oracle_equal true (p0 f8caa42cc5b4), every level stream == solo, health pre/post HEALTHY (cards 0 1).
C1 110.8 / C2 135.9 / C4 172.9 / C8 212.3 / PF2k 1551 vs int-rf2 109.5 / 139.5 / 168.0 / 209.0 / 1560 (ratios 1.012 / 0.974 / 1.029 /
1.016 / 0.994). Single run each, so +-3% is within run-to-run spread.
VERDICT: gate "speed >= int-rf2 per level" FAILS at C2 and PF2k (within noise; the change is one same-bits MTP commit worth about +1% C1).
Not merged: b70/dd-serve keeps base 372bf1b2; workers need not rebase. a8b678ec can be folded later with the next speed pass.

## 2026-11-21 [SAMP] real sampling on the Flash-Next TP2 XPU path (b70/fn-samp 28030ff0)

CONFIG: b70/fn-samp on b70/dd-serve 1a5a57d0; 4 commits (da4716cc chain, 874eac43 server fields, 0fc39fb4 shard
readback, 28030ff0 lane routing). Host finish: each rank's fn_topk64_part runs (256 x 64 per row, about 128 KB D2H
per rank per row) merged on the host, whole shard rows only when penalties apply; Strata 0.2.16 chain order and
precisions; TensorFold keyed Gumbel-max per (seed, position, id); exact-match MTP acceptance (= Strata with
STRATA_SPEC_PROB off, its production default). Greedy without penalties keeps the device argmax.
COMMAND: CPU: cpu-run + MemoryMax 24G + nice 19 + -j4: zig build test-fn-sampling-host test-penalty-request-host
test-flashnext-xpu-host test-model-capabilities-host test-logprob-offers-host test-model-discovery-host
test-capability-env-host test-limit-pins-host test-effort-capabilities-host xpu-native -Doptimize=ReleaseFast.
GPU: bin/gpu-run (cards 0+1) timeout 290 workers/SAMP/samp_session.sh (int-rf2 serve flags + --instruct-sampling card,
port 18152) running tensorfold/flashnext/dd/samp_gate.py with SAMP_P0_EXPECTED=workers/S/expected-current.json.
RESULT: CPU: all steps pass (sampling chain vs numpy fixture: 6 cases, ids/probs to 1e-12, 24 draws each exact;
chi-square at p 1e-4 for 3 settings; masks exact; merge == global sort on 2 x 124160 bf16 shards and tie-heavy
shards; rejection sampling preserves p for dense q and point-mass q; penalty request 28/28). GPU (124 s lease, sha
ddefd8fc): 12/12 gates PASS: p0 greedy 128 tokens bitwise == int-rf2 baseline, p0 at T 0.6 top_k 1 == greedy, chat
greedy repeat, top_k 1 == greedy, seed repeat, coherent at T 0.6 with MTP accept 0.419 (26/62), presence 1.5 and
repetition 1.1 replies coherent, seeds differ, drafts on == drafts off. Server log clean; xpu-health HEALTHY pre/post.
Sampled reply 63 tokens at 60.3 tok/s incl. TTFT vs greedy 71 tokens in 0.75 s (about 95 tok/s): lower MTP
acceptance plus the per-row host readback; not yet profiled.
VERDICT: PASS. Hand to INT for b70/dd-serve. Launcher: keep TF_TEMPERATURE empty, add --instruct-sampling card; the
clamp proxy is no longer needed. Follow-up (perf): measure the host-chain cost per verify window and, if it matters,
merge the runs on the device.

## [API] 2026-11-21 19:30 UTC - hotschmoe-dd API parity: audit, fixes, CPU parity harness, quick GPU gate 5/5
CONFIG: b70/dd-api (base b70/dd-serve 372bf1b2) head 32b27d57; Strata reference = the real serve/server.py of strata-0216 over a scripted MockEngine
(hotschmoe-dd-0216.json tokenizer, names, effort medium, budget 16384); TensorFold = fake_serve (real server, real Qwen3.8 tokenizer and chat template,
TF_FAKE_PROFILE=dd = the production XPU engine Info: greedy_only, no structures/penalties/logprobs, 4 lanes with a queue). Frontdoor log (paths only):
chat/completions, models, health, metrics, slots; no /v1/messages, no /v1/completions.
COMMAND: tensorfold/flashnext/api_parity/run_parity.sh (77 cases, streams and non-streams); zig build test-dd-api test-golden; GPU: bin/gpu-run
api_gpu_window.sh (port 18154, dd/api_gate.py all).
RESULT:
- Already at parity with no change: reasoning default on (effort medium, think_budget 16384 seen by the engine), reasoning_content stream and non-stream,
  qwen3-coder tool calls (typed params, parallel calls, streamed incrementally with argument deltas), tool_choice auto/none, stop, max_tokens and
  max_completion_tokens, include_usage, usage.prompt_tokens_details.cached_tokens, finish reasons, context-overflow 400 naming the context window,
  /v1/messages, queueing beyond --parallel (no reject).
- Gaps found and fixed (all default-off or additive, see PATCHES.md): per-request thinking budget names reasoning_budget_tokens / thinking_budget_tokens;
  max_tokens above the room 400 (Strata clamps) -> TF_FIT_MAX_TOKENS=1; tool_choice required/named and response_format 400 (no token grammar on XPU) ->
  TF_STRUCTURED_FALLBACK=1 (forced call opening in the prompt, JSON instruction); /slots and /props.
- Accepted differences (parity.py KNOWN): n>1 400 (Strata ignores), unknown named tool 400, a context overflow on a stream is an in-stream error event
  (upstream design), no logprobs on the XPU engine, response_format reply not validated (Strata 502s non-JSON), forced-call turns have no reasoning, /v1/models and
  /health bodies differ in shape. SAMP-owned: temperature>0 and penalties still 400 (greedy-only engine); they flip to equal with TF_FAKE_SAMPLED=1.
- Parity run with both flags: 77 cases, 0 gaps; without: 9 gaps (all the fixed ones). test-golden 55/55 equal. test-dd-api pass.
- libtfgrammar absent warning: harmless; the XPU lane backend has no masked first-token path so Info.structures stays false whatever the library. Build recipe:
  xgrammar v0.2.8 checkout + zig build grammar -Dxgrammar=DIR (no checkout on disk).
- GPU quick gate (first run 4/5: the clamp hit the engine's prompt + max_tokens + 1 check, HTTP 500; fixed by keeping one token back): second run 5/5 PASS,
  136 s lease, xpu-health HEALTHY. Real engine: write_file 36 argument deltas max gap 0.05 s, budget 48 -> 86 reasoning chars, forced/named calls and
  response_format 3/3 ok, 8 concurrent requests on 4 lanes all 200 (max 1.5 s).
VERDICT: API item passes its quick GPU gate. Cutover needs TF_FIT_MAX_TOKENS=1 TF_STRUCTURED_FALLBACK=1 in tf-dd-prod.env and SAMP merged for sampled clients.
Open: cached_tokens > 0 needs the prompt cache (KV); Strata /metrics (JSON monitor feed, polled by an external monitor) is not replicated (TF serves Prometheus text).

## 2026-11-21 [VIS] Vision in the TensorFold hotschmoe-dd serving path (strata-vision sidecar + M-RoPE splice)

CONFIG: branch b70/dd-vis (a2e63f48 vision; 2b552f0c = merge of b70/dd-kv bef2f1a8). Sidecar = Strata's own
strata-vision (strata-ab/build-vision-cpu, mmproj-Qwen3.8-Flash-Next-BF16.gguf, --threads 8 --max-tokens 300, libgomp from
strata-sycl-dev:020 at /mnt/vm_8tb/b70/build/tensorfold/vision-lib). Strata facts used: image pad 248056 between
vision_start 248053 / vision_end 248054; text cell (p,p,p), image cell j (p, p+j/nx, p+j%nx), p += max(nx,ny) after; pair i
of the 64 rotated dims takes stream i%3 (sections 11/11/10); pooled indexer keys rotate at cell 4b; rows replace the token
embedding raw; PLE n-gram hashes the pad id; MTP keeps token embeddings, same positions by cell. TF rotates the shifted
frame (Strata + shift, shift = sum(n - max(nx,ny))): relative positions identical, text after the last image at its own
cell, decode/drafts untouched; the frame is in the cache key (LaneHost.frame_key).
COMMAND: CPU: strata-vision probe on fixtures (blue 256 -> 64 tokens 8x8 x 2560, deterministic sha; 640x320 -> 200 (20x10);
1024 -> 289 (17x17); READY 2.4 s, 1.2 s/64-token image on 8 threads); zig build test-fn-mrope-host test-fn-vision-host
test-flashnext-xpu-host xpu-native (cpu-run, MemoryMax 32G, -j4). GPU (gpu-run cards 0+1, <= 5 min, port 18153):
dd-vis-run/vis_quick.sh {classic ctx16384 | XPU_FN_KV=int8 ctx65536 | XPU_FN_KV=int8 XPU_FN_DESC=1 XPU_FN_KV_PAGES=auto ctx65536}
+ vis_gate.py (text hash, image x2, text hash) + vis_extra.py (289-token image, red square left/right, two images in order,
image after a ~9K-token text prompt).
RESULT: host tests mrope 5/5, vision 9/9 (incl. a fake sh sidecar: cache hit, restart after death), flashnext host 28/28;
xpu-native builds (4 fn_*_mrope kernels in the zebin). GPU, all three modes: caps vision=true, 224x224 fixture -> "blue" x2,
49 tokens (7x7), encode 925-931 ms, image TTFT 1.06 s (0.12 s on the encoder cache hit), text greedy hash identical before and
after images (classic 0c46979a4e80, int8 bc270033aa5d). desc+paged extras all ok: green (289 tokens, TTFT 6.1 s incl. 5.8 s
encode), left/right correct, "blue, green" in order, blue after 8991-token prompt (TTFT 10.4 s). xpu-health HEALTHY pre/post
every lease. First lease at ctx 32768 classic failed at engine init with ZE out of device memory (not vision; classic bf16 KV);
re-run at 16384 passed.
VERDICT: VIS works in the serving path in the production KV mode; handed to INT (2b552f0c). Open: long-prompt-then-image TTFT
(10.4 s / 9K tokens) not yet compared with a text-only 9K prompt in desc+paged mode; JPEG EXIF turn and WebP need
libturbojpeg / libwebp on the host (else JPEG passes through unturned, WebP is refused); encodes are serial on 8 CPU threads
like Strata's.

## 2026-11-21 [VIS] Image vs text prefill cost at 9K tokens (desc+paged int8, 2b552f0c)

CONFIG: XPU_FN_KV=int8 XPU_FN_DESC=1 XPU_FN_KV_PAGES=auto, ctx 65536, port 18153, encoder cache warm.
COMMAND: gpu-run vis_quick.sh gpu5 desc + vis_long.py (8909-token text prompt vs the same + one 64-token image, x2).
RESULT: TTFT text 10.447 / 10.463 s, image 10.453 / 10.488 s; health HEALTHY pre/post.
VERDICT: the M-RoPE path adds no measurable prefill cost. The ~850 tok/s long-prompt prefill is the KV mode's own (int-rf2
quick bench prefill 1560 is a 2K row); a perf item for KV/INT, not VIS.

## 2026-11-21 [VIS] JPEG EXIF orientation and WebP parity with Strata

CONFIG: Strata 0.2.16 terminal/serve/server.py Vision.normalize: JPEG/PNG/BMP/GIF pass to strata-vision (stb_image) as
sent; a JPEG whose EXIF Orientation is not 1 and every other format (WebP, TIFF, AVIF) go through Pillow
(ImageOps.exif_transpose, alpha over white) to PNG. TF before: PNG passed, JPEG/WebP needed the pinned libturbojpeg /
libwebpdecoder (missing on host: EXIF JPEG went unturned, WebP refused), BMP/GIF refused. Fix: built the pinned releases
(libjpeg-turbo 3.1.4.1, WITH_SIMD=0; libwebp 1.6.0 = Pillow 13.0.0's) with zig cc + cmake/ninja into
/mnt/vm_8tb/b70/build/tensorfold/vision-lib (vision-codecs/build.sh, 32 s under cpu-run); 097ff79c passes BMP/GIF through.
COMMAND: zig build test-picture-host test-fn-vision-host xpu-native with TENSORFOLD_LIBTURBOJPEG/TENSORFOLD_LIBWEBP set (cpu-run);
gpu-run vis_quick.sh gpu6 desc (XPU_FN_KV=int8 XPU_FN_DESC=1 XPU_FN_KV_PAGES=auto, ctx 65536, port 18153) incl. new
vis_gate.py formats (dd/fixtures/vis_red_left_exif6.jpg stored 320x640 + EXIF 6; dd/fixtures/vis_red_right.webp).
strata-vision peak RSS via /usr/bin/time -v (two 289-token encodes).
RESULT: test-picture-host 31/31 (Pillow pixel goldens, pinned libs required), vision 10/10. GPU: vis-formats PASS: JPEG
answers "left" with a 20x10 grid (upright 640x320; unturned would be 10x20 "bottom"), WebP "right"; text/image gates
unchanged (hash bc270033aa5d, blue x2, 921 ms encode); HEALTHY pre/post. Encoder peak RSS 1.0 GiB.
VERDICT: JPEG/WebP parity-equal with Strata given the two env vars (INT: add to tf-dd-prod.env). Still not parity: TIFF/AVIF
(Strata converts via Pillow, TF refuses). dd/README.md records the encoder memory headroom (2 GiB for two encoders).

## [INT] 2026-11-21 20:36 UTC -- b70/dd-serve fb0e6006 = SAMP + API + VIS merged; host suite green; dd_functional.py PASS on :18150
CONFIG: b70/dd-serve 28030ff0 (fn-samp, fast-forward) -> dc8370e5 (merge dd-api 32b27d57; openai.zig field list = union of SAMP
penalty fields and API budget aliases + response_format fallback) -> fb0e6006 (merge dd-vis a2e63f48; xpu_forward.zig fields side by
side, xpu_flashnext.zig SAMP caps then openVision, greedy_only stays removed). Merge resolution in a scratch worktree (dd-merge),
then fast-forward. Quick env = tf-dd-prod.env with the worktree binary (sha 98b56885), context 16384, XPU_FN_KV=int8 (paged, no
desc), --instruct-sampling card --vision, TF_FIT_MAX_TOKENS=1 TF_STRUCTURED_FALLBACK=1.
COMMAND: workers/INT/host_suite2.sh (25 host steps in one zig build + xpu-native ReleaseFast, one cpu-run lease, 959 s incl. 408 s
build); workers/INT/quick_check.sh fb0e6006-a|b|c (gpu_window.sh prod with WIN_ONLY, three leases of 165 / ~200 / ~180 s, port 18150).
RESULT:
- Host: tests_rc=0 build_rc=0 (25 steps incl. test-fn-sampling-host, test-penalty-request-host, test-dd-api, test-fn-mrope-host,
  test-fn-vision-host).
- a: P0 identity PASS (hotschmoe-dd first + 2 aliases, reasoning default medium, caps vision/penalties/repetition_penalty, greedy_only
  off); S1-samp 12 gates PASS (drafts on == off); VIS-T0/I/T1 PASS (blue x2, encode 918 ms, image TTFT 1.06 s / 0.12 s cached,
  text hash bc270033aa5d unchanged). P1 bench (soft) C1 87.7 / C2 115.8 / C4 136.2 / C8 140.7 (4 lanes) / PF2k 1119 = 0.67-0.83 x
  int-rf2: expected for the legacy-window int8 paged mode (not the desc fast path); KV's desc+paged is the production mode.
- b: API1-4 PASS (reasoning 342 chars, 36 streamed arg deltas max gap 0.05 s, forced/named calls, response_format 3/3, budget 86
  chars); API5-edge FAIL only on its window >= 100000 check (quick context 16384 by design); overflow 400, clamp, finish, usage,
  queue8 parts PASS.
- c: P9 dd_functional.py direct to :18150 with the image: PASS (thinking 334 chars, tool call + round trip, 4 concurrent exact,
  vision blue, long 2048 tokens at 123.8 tok/s, code present). Server logs clean; xpu-health HEALTHY pre/post every lease.
VERDICT: SAMP + API + VIS integrated and functional at 16K. Remaining for production: KV (200K desc+paged, parking), then
dd-vis 097ff79c (formats; contains KV bef2f1a8, merged after KV's gated head), re-gate VIS + full window in desc+paged mode.
gpu_window.sh prod no longer holds cpu-run --quiet while queued for the cards (it blocked builds during other workers' leases).

## 2026-11-21 [SAMP] sampled decode profile and fix (b70/fn-samp a00ab802)

CONFIG: b70/fn-samp; int-rf2 serve flags (ctx 16384, --parallel auto, thinking off) + --instruct-sampling card, port
18152; 256-token fixed-length chat decode (ignore_eos), C1 and C4 (4 prompts), configs greedy / T 0.6 top_k 1 / T 0.6
top_p 0.95 top_k 20 / instruct card (T 0.7 top_p 0.8 top_k 20 presence 1.5). FN_SAMPLE_PROFILE=1 per-window timings.
Baseline = 28030ff0 behaviour (XPU_FN_SAMPLE_MERGE=0 XPU_FN_COUPLED=0); final = a00ab802 defaults.
COMMAND: bin/gpu-run (cards 0+1) timeout 290 workers/SAMP/samp_session.sh with SAMP_PROFILE=1 (prof_client.py);
leases prof-base, prof-new, check-new, check-final (XPU_FN_SAMPLE_CHECK=1 + samp_gate), prof-final; each <= 290 s.
RESULT (per-stream decode tok/s, base -> final): C1 greedy 101.9 -> 102.6, T 0.6 95.0 -> 97.8, top_k 1 99.1 ->
102.5, instruct card 70.3 -> 95.2. C4 greedy 45.6 -> 46.1, T 0.6 47.3 -> 47.4, instruct card 28.7 -> 43.3
(aggregate 106.6 -> 164.9). Per window at C1: verify 16.0 ms, draft 1.5 ms; sampled draw 1.47 -> 0.21-0.39 ms,
penalized draw 7.6 -> 0.86 ms (whole shard rows no longer read). MTP acceptance at T 0.6 is 0.36-0.37 against 0.38
greedy (C1), 0.63 against 0.65 (C4): sampling is not what limits acceptance. Coupled drafts (keyed Gumbel in the
MTP pick, XPU_FN_COUPLED=1) measured 0.35 at C1: no gain, shipped off. Exactness: 356K device-merged candidates and
2176 gathered penalized rows, 0 mismatches against the host references; samp_gate 12/12 PASS (p0 greedy bitwise,
drafts on == off); xpu-health HEALTHY in every lease. The earlier "60.3 vs 95" was a 63-token reply including TTFT,
not a decode rate.
VERDICT: PASS. The sampled-decode gap is closed (T 0.6 within 5% of greedy at C1, equal at C4); the instruct card
(open-webui thinking-off default) went from 0.69x to 0.93x greedy at C1 and 0.63x to 0.94x at C4. What remains is
greedy itself: C4 at ~46/stream, verify-bound (21-22 ms a window). Hand a00ab802 to INT.

## [INT] 2026-11-21 20:58 UTC -- merge fn-samp a00ab802 (device top-K per card, coupled draws off) -> dd-serve 7d08051e: PASS
CONFIG: b70/dd-serve 7d08051e = fb0e6006 + merge of b70/fn-samp a00ab802 (57d05a50 XPU_FN_COUPLED=1 default off; a00ab802 K candidates
per card, XPU_FN_SAMPLE_MERGE=0 restores the old merge). Clean merge. Quick env as fb0e6006 (16384, int8 paged, card defaults, vision).
COMMAND: workers/INT/host_suite2.sh <wt> merge-7d08051e build; workers/INT/quick_check.sh 7d08051e 16384
"P0-identity S1-samp VIS-T0 VIS-I VIS-T1 P9-functional" (one lease, :18150).
RESULT: host 25 steps tests_rc=0, build_rc=0 (367 s, sha b946865d). Lease: HEALTHY pre/post, load 77 s, P0 PASS, S1-samp PASS (all
gates), VIS-T0/I/T1 PASS (text hash bc270033aa5d unchanged), P9 dd_functional with image PASS (thinking 438 chars, tool round trip,
4 concurrent exact, blue, 2048 tokens at 122.2 tok/s), server log clean.
VERDICT: merged. Next: KV handoff, then dd-vis 097ff79c.
NOTE [INT]: S1 MTP accept at T 0.6 was 0.274 (117 drafted / 32 accepted, 72 tokens at 68.7 tok/s), above the 0.25 soft floor but below SAMP's 0.419 at 28030ff0; one sample each, flagged to SAMP.


## 2026-11-21 21:45 [KV] paged int8 descriptor windows at 200000 x 4: gates pass, prefill +38%

CONFIG: b70/dd-kv f70795f7 (bef2f1a8 desc over paged int8 + 0929b00c spare page + f70795f7 scale hoist), xpu-native
sha256 4ddf7dac; XPU_FN_KV=int8 XPU_FN_DESC=1 XPU_FN_KV_PAGES=auto XPU_FN_EVICT=1 XPU_FN_EVICT_SLOTS=auto
XPU_FN_MIRROR=1 XPU_CHUNK=XPU_CHUNK_SOLO=512; --parallel 4 --context 200000 --prompt-cache-gib 2, greedy, port 18151.
Reference: Strata 0.2.16 hotschmoe-dd-0216.json via nl_launch_0216.sh (container strata-ref, port 18151).
COMMAND: workers/KV/kv-quick.sh leases q2-q20 (gpu-run both cards, timeout 295) with tensorfold/flashnext/dd/kv_gate.py;
workers/KV/strata-ref.sh (one lease). Evidence: /mnt/vm_8tb/b70/results/tf-flashnext/workers/KV/q*/ and strata-ref/.
RESULT:
- Load 77-82 s; paged KV budget pages=1175 = 601,600 tokens/card, resident_90k=6, resident_200k=3; physical slots 3.
- Startup bug: padding lanes leased a pool page, so XPU_FN_EVICT refused (PagePoolNotIdle); fixed by 0929b00c.
- p0 C4 == solo bitwise, and == bef2f1a8 ref after f70795f7 (q12). 90K chat needle PASS, TTFT 87.2 s; repeat
  cached_tokens 89881, TTFT 0.113 s, ids equal (q13). 5 x 20K concurrent with 1200-token replies: every stream ==
  its solo bitwise, 113 parks / 112 restores, 0 errors (q18/q19). Parks cost 10-900 ms each.
- Prefill tok/s 8K/20K/32K/70K/90K: bef2f1a8 870/828/811/779/764 -> f70795f7 1170/1143/1119/1059/1031. Controls:
  int8 without eviction 8K 870 (eviction/mirror/cache cost ~0); bf16page 70K 1176. Cause: per-element software
  widen_half of the int8 group scales in paged attention. Strata cold 20K 1120, 70K 919.
- Raw /v1/completions ledger prompts (no chat template) sometimes end at once with <|im_end|> (seed 2 20K, needle
  early 70K/90K, also bf16page). Through chat, TF and Strata both answer correctly; Strata has no /v1/completions.
VERDICT: KV handoff gates PASS (90K, 4+ long streams at 200000 with park/restore bitwise, RAM prompt cache). The
early-EOS is raw-prompt model behaviour, not a KV bug. Head f70795f7 to INT. Open, perf: park thrash at 4 live
streams on 3 slots (113 parks in 88 s); prefill is still below the legacy bf16 desc path (1373 at 8K).
