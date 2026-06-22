# 09 - MTP Receptivity vs. Quantization Precision (W8A8 vs W4A16)

> **[UPDATED 2026-06-22 -> see [docs/kernel/20](../kernel/20_llm_scaler_int8_moe_and_mtp.md) sec 3].** MTP DOES work on
> 4x B70 (user ytnszmy): `vllm_xpu_kernels` v0.1.9 + the qwen3_5.py spec-wiring patch (vLLM #43565) + Half-KV ->
> num_speculative_tokens=5, mean accept 4.04 (88.9% @ spec=3). So our "MTP not viable / -19%" was a STACK gap (missing
> #43565 + the kernel wheel), NOT a B70 limit and NOT a quant-precision question. The receptivity analysis below stands
> as theory; the practical path is the doc-20 recipe (re-test on llm-scaler:0.14.0-b8.3.1).

**Created:** 2026-06-21
**Owner:** b70 research
**Status:** RESEARCH SYNTHESIS -- no B70 measurements yet
**Cross-refs:**
  - MTP_TODO.md (the experiment plan; phases A/B/C)
  - docs/kernel/12_mtp_specdecode_plan.md (full blocker chain + playbooks)
  - docs/kernel/17_spec_decode_modal_learnings.md (Modal post analysis + batch u-shape)
  - docs/literature/07_w8a8_int8_recovery.md (accuracy recovery levers)

ASCII only. No emoji.

================================================================================
1. THE HYPOTHESIS
================================================================================
The project lead's question:

  Is W8A8 (int8 weights, int8 activations) MORE RECEPTIVE to MTP / speculative
  decoding than W4A16 (int4 weights, fp16 activations)?

Hypothesis: speculative-decoding acceptance depends on the draft distribution
matching the target (verifier) distribution. Heavier weight quantization (int4)
likely degrades the verifier distribution more than lighter quantization (int8),
reducing accepted-tokens-per-step and erasing the spec speedup, while int8 weights
keep the verifier distribution closer to BF16.

Our context (MTP head is ALWAYS kept BF16 per MTP_TODO Playbook B ignore-list):
  - BF16 body     + BF16 head  -> draft/target alignment: exact match (baseline)
  - W8A8 body     + BF16 head  -> draft/target alignment: slight body mismatch
  - W4A16 body    + BF16 head  -> draft/target alignment: larger body mismatch
  - W4A8 body     + BF16 head  -> draft/target alignment: int4 weights + int8 acts

The BF16 head drafts tokens using the shared embeddings/weights; the body (main
model forward) verifies. When the body's distribution shifts under quantization,
the BF16 head's draft proposals become misaligned -> lower acceptance.

Our SINGLE existing B70 MTP datapoint (from MTP_TODO / JOURNAL 2026-06-21):
  Qwen3.6-27B-int4-AutoRound, PIECEWISE-only:
    N=1: accept 86.9% (first token), mean accepted 1.85, but decode -19%
    N=3: mean accepted 2.86, decode -37%
  The -19% / -37% is a SERVING-GRAPH problem (eager-attention verify overhead,
  PIECEWISE does not capture attention), NOT an acceptance-rate problem. The
  acceptance RATE is actually high (86.9% @ N=1) -- the graph issue eats the win.

  Reference BF16 4-card (real, private source, project lead verifies via primary
  contact): Qwen3.6-27B BF16, 4x B70 TP=4, spec=5, accept length 4.04, 88.9%@3,
  decode 54.2 t/s. This is BF16-body + presumably BF16/GPTQ-int4 head.

NOTE on the Lorbus int4 checkpoint: its MTP body (mtp.layers.0.*) is int4-quantized
by AutoRound (only mtp.fc stays BF16). This is a CONFOUND: the deployed head is
ITSELF int4, not a clean BF16-head + int4-body split. A proper BF16-head + int4-body
quant (the MTP_TODO ignore-list approach: `re:.*mtp.*`) would give a purer test.


================================================================================
2. WHAT THE LITERATURE SAYS
================================================================================

--------------------------------------------------------------------------------
2.1 CORE MECHANISM: acceptance is governed by draft-target distribution divergence
--------------------------------------------------------------------------------
Speculative decoding acceptance at each position is determined by the ratio
  min(1, p_target(x) / p_draft(x))
averaged over the draft distribution. This equals (1 - total_variation_distance)
between draft and target token distributions [standard spec-decode theory, Leviathan
et al. 2023; Chen et al. 2023]. Any perturbation to EITHER the draft distribution
OR the target (verifier) distribution that increases their divergence lowers
acceptance. Weight quantization of the body perturbs the TARGET distribution.

Key bound (from the literature): acceptance rate can be bounded from below by
  1 - (1/2) * KL(p_draft || p_target)
(approximately; exact bound uses TV distance). Larger KL -> lower floor on acceptance.

--------------------------------------------------------------------------------
2.2 "Speculative Decoding Meets Quantization" (arXiv:2505.22179, May 2025)
--------------------------------------------------------------------------------
The most directly relevant paper. Uses EAGLE-2 as the speculative method; keeps
the draft model in FP16; varies the TARGET model's quantization.

KEY FINDING: "Quantization has minimal impact on average accepted length."
  - FP16 body: average accepted length (tau) ~4.2 tokens (Llama-3-8B, A100)
  - W8A8 body: ~4.2 (essentially identical to FP16)
  - W4A16 body: ~4.2 (essentially identical to FP16)
  - W4A8-QQQ body: ~4.0 (slight drop)

INTERPRETATION: when the DRAFT HEAD stays in FP16, quantizing the body (verifier)
to W8A8 or W4A16 causes MINIMAL acceptance degradation -- both are "nearly no
degradation" vs FP16. W4A8 (both int4 weights AND int8 activations) shows a small
further drop.

IMPORTANT CAVEAT: the paper finds that the speedup-limiting factor is NOT acceptance
rate but rather the verification-to-decoding TIME RATIO. Because quantized bodies
run verification steps FASTER, a quantized body's speedup per accepted token is
BETTER than FP16's even with the same acceptance rate.

DRAFT MODEL QUANTIZATION (separate finding): "GPTQ quantization to the DRAFT model
leads to substantial degradation of the acceptance rate." Draft precision is the
critical axis; target/body precision is secondary. Draft model stays FP16 precisely
because the lm_head and softmax (the distribution-producing parts) are not
quantizable without harming draft quality.

Link: https://arxiv.org/html/2505.22179v1

--------------------------------------------------------------------------------
2.3 QSpec: Speculative Decoding with Complementary Quantization (arXiv:2410.11305)
--------------------------------------------------------------------------------
Proposes complementary schemes: target (verifier) model can be quantized to int4
for memory efficiency; draft model maintained at higher precision. Finding: draft
precision is the dominant factor in acceptance quality. The target model's int4
quantization (W4A16-style) is acceptable because even though the verifier's
distribution shifts, the rejection-sampling math only cares about the ratio at the
TOP token -- and int4 rarely flips the argmax at standard temperatures.

Key insight: "W8A8 quantization preserves the relative logit rankings extremely well,
and as long as the quantization process does not flip the top-1 prediction, the
verification logic ensures generation quality remains indistinguishable from the
full-precision counterpart."

Link: https://arxiv.org/pdf/2410.11305

--------------------------------------------------------------------------------
2.4 ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts
    (arXiv:2503.13565, March 2025)
--------------------------------------------------------------------------------
Uses MXFP4 (4-bit float) models as plug-and-play draft models for BF16 targets.
Achieves up to 2.72x speedup over BF16 baseline. The key design: the 4-bit model
IS the draft; the target remains at higher precision. Does not directly measure int8
vs int4 as TARGET. Confirms that even an aggressive int4-float draft can maintain
useful acceptance when the target is at BF16.

Note: this is the opposite of our scenario (int4 draft, BF16 target) vs our
(BF16 head draft, int8/int4 body target) -- but confirms the asymmetry:
high-quality target + lower-precision draft is viable; the inverse is riskier.

Link: https://arxiv.org/pdf/2503.13565

--------------------------------------------------------------------------------
2.5 Quasar: Quantized Self-Speculative Acceleration (arXiv:2603.01399)
--------------------------------------------------------------------------------
Self-speculative decoding where draft and target come from different precision
views of the SAME model. Key finding: more aggressive quantization (particularly
int4) produces lower acceptance rates compared to int8 or fp16 variants.
"Weight quantization introduces a measurable divergence between draft and target
model distributions; lower precision increases the mismatch, reducing efficiency."
However, the paper's measured acceptance at int4 is still usefully high (~35-40%
in their mobile-weight scenario; note: they use irreversible quantization noise
as a ceiling rather than a controllable variable).

Link: https://arxiv.org/pdf/2603.01399

--------------------------------------------------------------------------------
2.6 Community measurements: Qwen3.6 + MTP at Q8 (zolotukhin.ai, 2026-05-08)
--------------------------------------------------------------------------------
Qwen3.6-27B Q8 (8-bit weights), DGX Spark / GB10 (CUDA), vLLM/llama.cpp:
  - gamma=2: speedup 2.40x, acceptance rate 0.83
  - gamma=3: speedup 2.24x (more draft overhead), acceptance rate 0.72

FP8 (Qwen3.6-35B-A3B-FP8, docai.hu GB10 bench):
  - accept @ pos-0: 81.57%, pos-1: 63.48%, mean accept length 1.45
  - throughput: +24.2% concurrent, -7.2% long-context

The Q8 numbers (0.83 acceptance at gamma=2) are the best published public data for
a quantized Qwen3.6 with native MTP. No direct Q4 vs Q8 comparison on the same
hardware is published.

Links:
  - https://zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/
  - https://docai.hu/en/blog/qwen36-mtp-gb10

--------------------------------------------------------------------------------
2.7 Self-speculative and MTP-specific findings
--------------------------------------------------------------------------------
FastMTP (arXiv:2509.18362): the key challenge in native MTP heads is that the
persistent gap between MTP head accuracy and the main head limits acceptance.
Acceptance at position k decays approximately as r^k where r is the per-position
acceptance rate. FastMTP improves acceptance from 70% -> 81% at pos-0, 11% -> 56%
at pos-1 by improving drafter training, not by changing body quantization.

DeepSeek-V3 (public benchmark): built-in MTP achieves >80% acceptance rate, ~1.8x
speedup in generation throughput (BF16 or FP8 body, int4 not reported with MTP).

vLLM PR #35387 (community report): Qwen3-Next-80B-A3B (GDN/hybrid), 4xH100 TP=4,
MTP N=2 -> +76.5% latency penalty. Root cause: a per-step CPU sync for the GDN
mamba_postprocess -- NOT a quantization issue. This is the SAME GDN-specific fixed
overhead that explains our B70 -19%.

LK Losses (arXiv:2602.23881): training the draft head to directly maximize acceptance
rate can further close the distribution gap on top of body quantization.

--------------------------------------------------------------------------------
2.8 The one controlled study: mobile (irreversibly quantized) case
--------------------------------------------------------------------------------
A mobile-weight int4 study in Quasar (arXiv:2603.01399) found top-1 acceptance at
~35% for int4-mobile weights vs theoretical ~80%+ for full-precision weights. This
is an extreme case (irreversible/lossy int4); standard GPTQ int4 is NOT as severe.
GPTQ int4 group-128 typically achieves top-1 agreement of ~88% vs BF16 on Qwen3
(from our own quant eval notes, MTP_TODO Playbook B). So the GPTQ-int4 "acceptance
ceiling" is NOT 35%; the mobile study represents a worst case, not our deployed case.

--------------------------------------------------------------------------------
2.9 What is MISSING from the literature
--------------------------------------------------------------------------------
No paper directly measures:
  - MTP acceptance rate on the SAME checkpoint at BF16 vs W8A8 vs W4A16 vs W4A8
  - The effect of quantizing the body WHILE KEEPING the MTP head in BF16 (our
    exact scenario with the proper ignore-list quant)
  - Acceptance degradation on a GDN/hybrid-attention model under body quantization

The closest analog is 2505.22179 (EAGLE-2, FP16 draft head, quantized target),
which shows minimal acceptance drop W8A8 -> W4A16 on dense Llama models. The
GDN-hybrid caveat is that our model has a harder per-step overhead floor that
makes small acceptance changes LESS material than on a pure-attention model.


================================================================================
3. PREDICTED ORDERING: BF16 >= W8A8 >= W4A16 (>= W4A8)
================================================================================

Based on theory + the literature above:

  BF16 body >= W8A8 body >= W4A16 body (>= W4A8 body)

for mean accepted tokens per step, given BF16 MTP head throughout.

REASONING FOR EACH TIER:

--- BF16 (reference ceiling) ---
Body distribution = exactly what the BF16 head was trained against. Zero
distribution mismatch from quantization. Our reference: accept 4.04 @ spec=5,
88.9% @3 (4-card TP=4 benchmark -- BF16 body, vLLM #43565). On a per-card basis
this is the acceptance ceiling.

--- W8A8 (expected: near-BF16, ~3.7-4.0 mean accept) ---
Int8 symmetric per-channel weights + dynamic int8 activations. Key properties:
(a) SmoothQuant on the 16 full-attn layers recovers activation fidelity (our recipe);
(b) GPTQ weights -> ~+2.7pt top-1 agreement over RTN; per arXiv:2505.22179, W8A8
body retains ~4.2/4.2 average accepted length vs FP16 (no degradation within noise).
(c) W8A8 rarely flips the top-token argmax (the critical check for acceptance math).
(d) The activation quantization (the novel risk at W8A8 vs W8A16) is handled by
SmoothQuant pre-scaling + per-token dynamic quant -- a well-understood, recoverable
path (docs/literature/07).

The W8A8 -> BF16 acceptance gap is expected to be within ~0.1-0.3 mean accepted
tokens, likely inside measurement noise. Prediction: W8A8 accept ~= BF16 accept
for practical purposes.

--- W4A16 (expected: modest drop, ~3.3-3.8 mean accept) ---
Int4 group-128 GPTQ weights, FP16 activations. Key properties:
(a) No activation quantization -> no activation-fidelity loss; ONLY the weight error.
(b) GPTQ group-128 lifts quality (+4.2pt top-1 agreement over RTN, per our eval).
(c) BUT: int4 weight error is 2-4x larger than int8 weight error per bit-flip
    probability; logit distributions show more variance at the tails.
(d) The literature (arXiv:2505.22179) finds W4A16 also shows "nearly no degradation"
    vs FP16 in acceptance on dense Llama models -- BUT this uses EAGLE-2 with a
    separately-trained FP16 draft head whose distribution was not co-trained with the
    quantized body. Our MTP head WAS co-trained with BF16, then deployed against a
    W4A16 body -> larger mismatch than EAGLE-2's experiment suggests.
(e) Qwen3.6 top-1 agreement at W4A16 GPTQ: ~0.883 vs BF16. At ~88% agreement, the
    body occasionally produces different top tokens than BF16 -> the BF16 MTP head
    proposes tokens that the quantized body would not have generated -> rejection.
(f) The MTP_TODO expected range: ~3.0-3.5 mean accept (vs 4.04 BF16).

Prediction: W4A16 accept likely in the range 3.2-3.7, with the precise value
depending on the prompt distribution and temperature.

--- W4A8 (expected: further drop, ~2.8-3.4 mean accept) ---
Int4 weights + int8 activations. Compounds BOTH weight and activation errors.
(a) Selective SmoothQuant recovers the activation-fidelity half (same as W8A8).
(b) But the int4 weight error is still present, similar to W4A16 on the weight side.
(c) The distribution mismatch is expected to be >= W4A16 because of the activation
    quantization stack on top of the weight error.
(d) Per our own quant eval notes (MTP_TODO Playbook B): RTN ~0.822 -> GPTQ +
    SmoothQuant lifts toward ~0.86+. This is below W4A16's ~0.883 and W8A8's ~0.908.
(e) HOWEVER: for SINGLE-CARD 27B serving, W4A8 is the only int8-systolic int4-weight
    option (W8A8 needs TP=2 for VRAM headroom). The acceptance drop must be measured
    against the card-fit benefit.

Note: W4A8 is NOT in the original question (W8A8 vs W4A16), but is a planned config
(MTP_TODO Phase B2) and is included here for completeness.

--- SUMMARY TABLE (all predicted, UNMEASURED on B70) ---

  Scheme   | Body top-1 agree | Predicted mean accept | Predicted accept@3 | Confidence
  ---------|------------------|-----------------------|--------------------|------------
  BF16     | 1.000 (baseline) | ~4.0 (measured 4xB70)| ~89% (measured)    | MEASURED
  W8A8     | ~0.908 (eval'd)  | ~3.7-4.0             | ~85-89%            | HIGH (lit. supports)
  W4A16    | ~0.883 (eval'd)  | ~3.2-3.7             | ~75-85%            | MEDIUM
  W4A8     | ~0.860+ (est.)   | ~2.8-3.4             | ~68-80%            | LOW (no lit. analog)

"HIGH confidence" = predicted by both theory and direct literature analog.
"MEDIUM" = predicted by theory; literature analog exists but is an imperfect match.
"LOW" = predicted by theory only; compound error, no literature analog for W4A8+GDN.

ALL predictions are HYPOTHESES. They must be measured. MTP_TODO Phases A and B
exist precisely to turn these predictions into measurements.


================================================================================
4. WHY THE DISTRIBUTION-MISMATCH ARGUMENT IS PARTIALLY WEAKENED ON OUR STACK
================================================================================
The hypothesis (heavier quant -> more mismatch -> lower acceptance) is SOUND in
theory and supported by the literature. But two structural factors REDUCE the
practical sensitivity on our stack:

(A) GRAPH CAPTURE IS THE DOMINANT CONSTRAINT, NOT ACCEPTANCE RATE.
    On the B70 today (PIECEWISE, no FULL capture), the speculative verify runs with
    eager attention. A -19% on the already-high-accept (86.9%) int4 run shows that
    the overhead of eager-attention verify + fixed spec machinery (~2.28x per step at
    N=1) dwarfs the acceptance-rate difference between schemes. Even at 4.04 accepted
    tokens (BF16), the pay-per-step cost would still be a net negative under
    PIECEWISE. The ordering BF16 > W8A8 > W4A16 is REAL, but it becomes the SECOND-
    ORDER question only after FULL graph capture is unblocked. Per docs/kernel/12 +
    JOURNAL 2026-06-21, FULL+spec+GDN is currently blocked by a vLLM-XPU GDN spec-
    metadata bug (`spec_query_start_loc must have size [num_spec_decodes+1]`).

(B) MTP HEAD IS BF16 -- the draft-distribution side is FIXED regardless of body quant.
    The quantization-acceptance risk in our setup is purely the TARGET/VERIFIER side
    (the body's logit distribution), not the draft side. The literature (arXiv:2505.22179)
    shows this direction (high-precision draft, quantized verifier) has MINIMAL
    acceptance degradation at W8A8 and small degradation at W4A16 -- supporting the
    hypothesis that W8A8 is "close to BF16" and W4A16 shows a modest drop.
    The literature agrees: the acceptance-critical path is the DRAFT precision, not
    the verifier precision. Our BF16 head is the right choice.

(C) GDN-HYBRID PER-STEP OVERHEAD SETS A FLOOR INDEPENDENT OF ACCEPTANCE.
    The Qwen3.6 GDN architecture carries a per-step CPU sync (vLLM #35387: device->
    host copy before mamba_postprocess). This fixed cost hits all precision levels
    equally. Even if W8A8 achieves ~4.0 mean accept vs W4A16's ~3.4, the net speedup
    difference depends on whether the fixed per-step overhead dominates.
    Approximate: if FULL capture gives us speedup_factor = accept_len / (1 + overhead_frac),
    a 0.6 token accept difference at overhead_frac=0.15 gives:
      W8A8:  4.0 / 1.15 ~= 3.48x
      W4A16: 3.4 / 1.15 ~= 2.96x
    That is a ~15% speedup difference. Still meaningful, but not decisive.


================================================================================
5. EXPERIMENT DESIGN (MEASURABLE ON OUR STACK)
================================================================================
Objective: isolate acceptance-rate effect of body quantization from the
graph-capture effect, by holding all non-quant variables CONSTANT.

PREREQUISITE: FULL graph capture must be working on the GDN 27B before this
experiment has clean results. Under PIECEWISE, acceptance differences are swamped
by the graph-overhead floor. Track the GDN spec-metadata bug fix before running.
IF the bug is not yet fixed, run on the 14B (no GDN) as a proxy: A1/A2/A3 in
MTP_TODO Phase A (BF16 -> W8A8 -> W4A8 on the dense 14B, which does have FULL
capture available via TRITON_ATTN per JOURNAL 2026-06-21).

--- STEP 0: Confirm which 14B variant has MTP head ---
grep mtp.layers /mnt/vm_8tb/b70/models/<14B-checkpoint>/model.safetensors.index.json
If absent, skip to STEP 3 (27B only) or use 0.6B draft-model proxy for the 14B.

--- STEP 1: Establish BF16 baseline acceptance ---
Serve Qwen3.6-27B BF16 (or the candidate quant's GPTQ-calibrated reference),
VLLM_ATTENTION_BACKEND=TRITON_ATTN (for FULL capture, when GDN bug is fixed),
cudagraph_mode=FULL_DECODE_ONLY:
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":5}'
Measure per run (fixed prompt set, batch=1, >=3 seeds, discard warmup):
  - mean accepted tokens per step (mean_acc_len)
  - per-position acceptance rate (pos-0, pos-1, pos-2, pos-3, pos-4)
  - decode t/s MTP on
  - decode t/s MTP off (same config, no --speculative-config)
  - MTP speedup = MTP_on / MTP_off
Record: served_model_id, VRAM, KV dtype, spec_toks, temperature.

--- STEP 2: Replicate with W8A8 checkpoint (all else equal) ---
Same command, same prompts, same temperature, same spec_toks.
Target checkpoint: Qwen3.6-27B-W8A8-gptq-sq (with BF16 mtp.* in ignore-list).
Record same metrics. Diff: mean_acc_len(W8A8) - mean_acc_len(BF16) is the
W8A8 acceptance cost.

--- STEP 3: Replicate with W4A16 checkpoint ---
Same command, same prompts.
Target checkpoint: Qwen3.6-27B-W4A16-gptq (with BF16 mtp.* in ignore-list).
Record same metrics. Diff: mean_acc_len(W4A16) - mean_acc_len(BF16).

--- STEP 4 (optional, Phase B2): Replicate with W4A8 checkpoint ---
Same command. Diff: mean_acc_len(W4A8) - mean_acc_len(W4A16).

--- CONTROLS AND CONSTANTS (do not change between steps 1-4) ---
  - num_speculative_tokens: sweep {1, 3, 5}; report all three per scheme
  - temperature: 0 (greedy, maximizes acceptance) for the headline; also run
    temp=0.6 (production-like) to measure temperature sensitivity
  - prompt set: FIXED (same 20-50 prompts across all steps; mix of code, chat,
    math -- acceptance is workload-dependent)
  - context: FIXED (e.g., 2048-4096 tokens; do not change between runs)
  - KV dtype: fp16 (not half-KV / KV-quant; isolate body quant from KV effects)
  - batch: 1 (single-stream; MTP is a low-concurrency lever per docs/kernel/17 T7)
  - image + kernels: SAME across all schemes (fixes the graph-capture quality)
  - MTP ignore-list: `re:.*mtp.*` in every quant -> ensures head is BF16 in all cases

--- METRICS TO REPORT (one row per run in MTP_TODO logging table) ---
  mean_acc_len    : primary metric for this experiment (the acceptance question)
  accept@3        : P(at least 3 tokens accepted per step)
  dec MTP-on      : decode t/s with spec enabled
  dec MTP-off     : decode t/s without spec (same config, no --speculative-config)
  MTP_x           : speedup = dec_on / dec_off
  top-1 agreement : measured at eval time (cross-check vs the accept number)

--- WHAT SUCCESS LOOKS LIKE ---
  If the hypothesis is confirmed:
    mean_acc_len: BF16 (~4.0) > W8A8 (~3.8) >> W4A16 (~3.4) > W4A8 (~3.0)
    MTP_x follows the same ordering

  If the literature (2505.22179) analog holds on our stack:
    W8A8 accept ~= BF16 accept (within noise)
    W4A16 accept slightly lower but still viable
    In which case the FORMAT CHOICE is driven by t/s-per-token math, not
    acceptance, and the card-fit constraint (W4A16 fits 1 card; W8A8 needs TP=2)
    becomes the deciding axis.

  If W4A16 acceptance collapses (<2.5 mean accept):
    W4A16 is out for MTP; default to W8A8 (TP=2 when 2nd card arrives) or W4A8
    (single-card, if accuracy gate is met).

--- HARNESS INTEGRATION (our existing scripts) ---
  Primary: scripts/35_sweep_bench.sh or the decode-probe / perf_probe harness.
  The MTP_TODO logging table (JOURNAL.md mirror) is the output format.
  Verify served_model_id encodes quant scheme per CLAUDE.md model-check rule.

--- WHAT NOT TO MEASURE HERE ---
  - Prefill / TTFT: not affected by MTP accept; separate axis
  - Batch > 1: spec-decode payoff is batch-1 only per docs/kernel/17 T7
  - Different temperatures WITHOUT also measuring t/s: acc_len alone at varied temp
    tells you nothing unless you also get the speedup


================================================================================
6. PRACTICAL CONSEQUENCE FOR THE FORMAT DECISION
================================================================================
For our B70 roadmap (MTP_TODO):

  (a) W8A8 + MTP (needs 2 cards) is expected to get ~BF16-level acceptance and
      the best accuracy. It is the "safest" MTP format. The constraint is VRAM
      (27B W8A8 ~33 GB -> TP=2), and TP=2 disables graph capture on XPU (#34482)
      -> MTP at TP=2 is net-negative (measured: -50% at TP=2 per community data).
      CONCLUSION: W8A8 27B + MTP requires a different config strategy (e.g., using
      W8A8 for 2-card throughput WITHOUT MTP, and MTP on single-card formats only).

  (b) W4A16 + MTP (fits 1 card, FULL capture available) is the single-card MTP
      candidate with the simplest accuracy story (no act quant). Expected acceptance
      slightly below BF16 but likely still yielding >2x MTP speedup. The 4304-dim
      XPUwNa16 kernel gap (MTP_TODO risk) must be verified first.

  (c) W4A8 + MTP (fits 1 card, int8 systolic for verify) is the highest-intensity
      single-card option. Verification runs the int8 GEMM kernel on both draft and
      verify passes -> fewer memory bandwidth cycles per token. But the lowest
      expected acceptance. The net answer depends on whether the int8 systolic
      advantage on verify outweighs the accept-length hit.

  FORMAT STRATEGY (updated by this research):
    Measure acceptance BEFORE committing to format. Run Phase A (14B BF16->W8A8->
    W4A8) to get the acceptance curve on the cheaper test vehicle. If W8A8 acceptance
    barely differs from BF16 (as the literature suggests), and W4A16 drops only ~5-10%,
    BOTH single-card formats are viable and the decision is driven by VRAM + t/s math,
    not acceptance. If W4A8 drops >20% from BF16, steer toward W4A16 for single-card MTP.


================================================================================
7. KEY SOURCES
================================================================================
[LIT-A]  "Speculative Decoding Meets Quantization: Compatibility Evaluation and
          Hierarchical Framework Design" (arXiv:2505.22179, May 2025)
          URL: https://arxiv.org/html/2505.22179v1
          Finding: W8A8 and W4A16 TARGET quantization causes minimal acceptance drop
          vs FP16 when draft model stays FP16; GPTQ on DRAFT model degrades acceptance.

[LIT-B]  "QSpec: Speculative Decoding with Complementary Quantization Schemes"
          (arXiv:2410.11305)
          URL: https://arxiv.org/pdf/2410.11305
          Finding: draft precision is the critical axis; target int4 is viable because
          W8A8 preserves relative logit rankings.

[LIT-C]  "ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts"
          (arXiv:2503.13565, March 2025)
          URL: https://arxiv.org/pdf/2503.13565
          Finding: MXFP4 draft model achieves 2.72x over BF16 baseline (BF16 target);
          confirms 4-bit draft usable at right design point.

[LIT-D]  "Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via
          Memory-Efficient Verification" (arXiv:2603.01399)
          URL: https://arxiv.org/pdf/2603.01399
          Finding: heavier quantization (int4) produces lower acceptance than int8/fp16;
          int4 mobile weights set a 35% acceptance ceiling (worst case; not GPTQ int4).

[LIT-E]  "QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache"
          (arXiv:2502.10424, Feb 2025)
          URL: https://arxiv.org/html/2502.10424v1
          Finding: quantization preserves information better than sparsity; INT8/INT4
          KV hierarchies sustain >90% acceptance rate vs INT4 sparse methods.

[LIT-F]  "FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction"
          (arXiv:2509.18362)
          URL: https://arxiv.org/html/2509.18362v1
          Finding: acceptance at native MTP pos-0 ~70-81%; pos-1 ~11-56%; improving
          the HEAD's training (not body quantization) is the primary acceptance lever.

[COM-A]  zolotukhin.ai, "MTP Speculative Decoding for Qwen3", 2026-05-08
          URL: https://zolotukhin.ai/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/
          Data: Qwen3.6-27B Q8, DGX Spark, gamma=2 -> 2.40x speedup, 0.83 accept.

[COM-B]  docai.hu, "Qwen3.6 MTP on GB10: +24.2% throughput", 2026
          URL: https://docai.hu/en/blog/qwen36-mtp-gb10
          Data: FP8 model, pos-0 accept 81.57%, mean acc_len 1.45, 72.53% overall.

[REPO-A] vLLM issue #35387: GDN/Mamba per-step CPU sync tax with MTP.
          Context: confirms our -19% is GDN-specific fixed overhead, not a pure
          acceptance-rate phenomenon.

[REPO-B] Our own JOURNAL 2026-06-21: N=1 MTP accept 86.9% on int4 27B under
          PIECEWISE, but -19% due to eager-attention verify overhead.
