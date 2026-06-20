# 17 - Modal "Speculation is all you need" -- learnings mapped onto the B70

Source: Modal blog, "Speculative decoding is all you need" (https://modal.com/blog/spec-is-all-u-need),
read 2026-06-21. This doc (1) lists the post's actionable techniques/claims, (2) gives a CONCRETE
apply/blocked/worth-trying verdict for our SINGLE Arc Pro B70 case -- separately for the Qwen3.6-27B
(Dense-ish hybrid GDN) and the Qwen3.6-35B-A3B (MoE, also GDN-hybrid) -- (3) a prioritized
"what to try NOW vs gated" list, and (4) the one insight from the post we were missing.

Cross-refs: docs/kernel/12 (MTP/spec-decode plan + the blocker chain), docs/kernel/14 (campaign
capstone). Latest evidence: JOURNAL 2026-06-21 (FULL capture WORKS on 14B-W4A8 via TRITON_ATTN, but
FULL+spec+GDN hits `spec_query_start_loc must have size [num_spec_decodes+1]`). ASCII only.

================================================================================
TL;DR
================================================================================
- The post's thesis -- "speculative decoding is the only engine optimization that matters at high
  interactivity" -- is a SINGLE-STREAM / low-batch claim. Our B70's real product is HIGH BATCH
  (1 card = ~412 t/s @ 8 users, ~1286 @ 32; doc 14). Spec-decode is a single-stream lever; it is
  LEAST valuable exactly where the B70 is strong, and most valuable where we measured it net-NEGATIVE.
- The post's core math (speedup ~= acceptance length) is the same Amdahl framing doc 12 already uses;
  it CONFIRMS our verdict, it does not change it. Our problem was never the drafter (86.9% accept,
  acc_len 2.86 @ N=3 -- excellent "legs"); it is the eager-attention VERIFY tax under PIECEWISE.
- The one genuinely NEW, B70-relevant claim: spec-decode speedup is NON-MONOTONIC (u-shaped) vs batch
  size for MoE models but monotonic for dense. That is a direct steer for the 35B-A3B MoE: there may
  be a low-batch window where spec-decode pays AND a high-batch window where it does NOT -- so the
  35B's spec-decode decision is batch-dependent in a way the 27B's is not.
- Net effect on our roadmap: the post REINFORCES docs 12/14. The single highest lever remains
  capturing the VERIFY attention into a FULL graph -- and as of 2026-06-21 that path is unblocked on
  pure-attention models (14B) but BLOCKED on our GDN flagships by a vLLM-XPU GDN-spec-capture bug.

================================================================================
1. THE POST'S KEY TECHNIQUES / CLAIMS (with citations)
================================================================================
All quotes from the Modal post unless noted.

T1. SCOPE. Spec-decode "losslessly accelerates the 'decode' phase of LLM inference." It does NOT
    touch prefill. (post)

T2. DRAFTER CHOICE. Prefer modern speculators that "piggy-back on the target model's past
    computations" -- MTP, EAGLE-3, DFlash -- because they "use speculator models that ... are
    light-weight, result in high acceptance lengths, and are relatively easy to train." n-gram and
    separate small NN drafters are the older, weaker baseline. (post)
    Hard constraint: "speculators must run faster than the target, which generally means they are
    smaller. Smaller models struggle to saturate the memory and arithmetic bandwidth" -- i.e. on big
    fast HW a tiny drafter can be HARD to make faster than the target. (post)

T3. EAGLE / EAGLE-3. Named once as a contemporary speculator family; the post gives NO mechanism,
    acceptance numbers, or EAGLE-vs-EAGLE3 detail. (post -- thin here; our EAGLE3 detail comes from
    doc 12 H.3 / the Ex0bit PRISM head, not Modal.)

T4. TREE vs CHAIN DRAFTING. NOT discussed in the post (confirmed on re-fetch). So the post offers no
    guidance on tree drafting; our chain-accept assumptions (MTP autoregressive, EAGLE3 chain
    tau~2.2) come from doc 12, not Modal.

T5. ACCEPTANCE-LENGTH MATH. Toy model: "we just get: speedup == acc_len." Benchmark table (post):
        acc_len 1 -> 75 t/s (1x);  2 -> 140 (1.86x);  4 -> 268 (3.57x);  8 -> 422 (5.62x).
    Caveat: "the speedups are consistently overestimated" by the linear model because the draft
    forward has real cost; a roofline model corrects it downward. (post)
    Drafter cost model: "the speculator's forward pass latency as a fixed percentage of the target
    model -- something like 5% to 20% seems common"; for autoregressive drafters paid "once per draft
    token," for block drafters "once per block." (post)

T6. FINE-TUNING THE DRAFTER. "We have seen fine-tuned models increase acceptance lengths from a
    baseline of 3 to over 9. That's the difference between a 25% speedup and a 3x speedup." (post)
    => acceptance length is the dominant knob; training the drafter on production traffic is the win.

T7. BATCH-SIZE / THROUGHPUT INTERACTION (the new bit). Roofline "correctly predicts that speculator
    speedup is non-monotonic ('u-shaped') as a function of batch size for mixture-of-experts models,
    like DeepSeek-V4 Pro, but monotonic for dense models, like Qwen 3.5 27B." (post)
    The post does NOT give the mechanism. (Our reading, INFERRED -- flagged as such in section 2: a
    dense decode step is weight-bandwidth-bound, so adding a few verify tokens is ~free until you hit
    the compute roofline -> speedup falls off monotonically as batch grows into the compute-bound
    regime. An MoE step at low batch activates few experts and is bandwidth-bound (spec-decode helps),
    but as batch grows the union of routed experts widens -> more weight bytes loaded per step and a
    different roofline crossover -> the u-shape. This matches our own doc-12 H.1 [NEG] thc1006 result:
    Qwen3.6-35B-A3B draft/ngram went NEGATIVE "root cause = MoE expert-load union on verify.")
    The post also models high concurrency: "short sequence length (~4k tokens/sequence) and high
    concurrency (batches of 32 sequences)," capping draft length at "16, whichever is lower," with
    predicted speedups "approximately 20% for typical MTP drafters versus 3x for well-trained block
    drafters." (post) -- i.e. even at batch 32 a GOOD block drafter still pays, but a plain MTP head's
    payoff shrinks a lot at high batch.

T8. DYNAMIC / ADAPTIVE SPECULATION LENGTH. Flagged as a future trend ("adaptive speculation,"
    "iterative speculator training and distillation"), with a cautionary tale: a naive adaptive system
    "was triggering retrains every time the user base shifted, twice per day" -> they shipped two
    fixed "regional speculators" instead. (post) No implementation detail; references Together's prior
    work without specifics.

T9. WHEN IT HELPS vs HURTS. The post is bullish and lists only one explicit caveat -- the drafter
    must "run faster than the target." It does NOT enumerate the failure modes we hit (eager-attn
    verify, MoE expert-union, distributed/TP collectives). The "u-shaped MoE" claim (T7) is the
    closest the post comes to "spec-decode can hurt." (post)

T10. SERVING-FRAMEWORK SPECIFICS.
     - SGLang: Modal "worked closely with SGLang"; undocumented bench flag SGLANG_SIMULATE_ACC_LEN
       "mocks the acceptance behavior by just accepting generated tokens up to a certain length" --
       lets you measure the SPEEDUP CEILING for a given acc_len WITHOUT training a drafter. (post)
     - vLLM: credited with closing "the gap with proprietary engines" on spec-decode. (post)
     - TensorRT-LLM: not mentioned.

T11. DFlash. A BLOCK drafter (drafts a block per forward, "once per block") that "achieves a much
     higher acceptance length"; gives "an additional 5 - 20% speedup on a wide variety of workloads"
     over existing baselines; its "KV injection technique allows for deeper, smarter drafters" and
     needed "a fused kernel for the KV injection step" in SGLang. (post)

T12. HEADLINE NUMBER. Qwen 3.5 122B-A10B (MoE): "over 1000 tok/s at concurrency 1 on a B200 node" vs
     ~250 t/s without speculation (~4x). (post) -- a CONCURRENCY-1, B200, MoE result.

T13. MEMORY OVERHEAD of the drafter: NOT quantified in the post. (Ours: doc 12 -- MTP head ~1/40 of a
     main forward, fits inside the served checkpoint; EAGLE3 PRISM head = 1 Llama layer + fc, also
     tiny. Memory is not our spec-decode bottleneck on a 32 GB card.)

T14. THE FLYWHEEL (closing reco): generic models -> self-host pre-trained speculators -> train custom
     speculators on production data -> distill the target -> repeat. (post)

================================================================================
2. PER-TECHNIQUE VERDICT FOR THE B70 (27B Dense and 35B MoE called out separately)
================================================================================
Legend: APPLIES / BLOCKED (by what) / WORTH TRYING (how). Evidence cites docs 12, 14, and
JOURNAL/FINDINGS. "GDN-spec-FULL bug" = `spec_query_start_loc must have size [num_spec_decodes+1]`
in vllm/v1/attention/backends/gdn_attn.py (JOURNAL 2026-06-21).

--- T1 (decode-only scope) -----------------------------------------------------
APPLIES, both models. Confirms doc 14: prefill is a separate axis. Relevant because TRITON_ATTN
(the FULL-capture backend) HALVES our prefill (4953 -> ~2480 t/s on 14B; JOURNAL 2026-06-21). So
turning on the FULL-capture path to win decode COSTS prefill -- a real tradeoff for any serving
config, both models.

--- T2 (drafter choice; "must be faster than target") -------------------------
27B: APPLIES, already satisfied. Our drafters are the right modern kind: native Qwen3.6 MTP head
  (single-layer, ~1/40 of a forward) and the Ex0bit PRISM EAGLE3 head (1 Llama layer). Both are far
  faster than the 27B target and accept well (MTP 86.9% @ N=1). The post's "smaller models struggle to
  saturate bandwidth" warning is NOT our problem -- our drafter is plenty fast; our problem is the
  VERIFY, not the draft (doc 12 G.1).
35B-A3B: APPLIES with a twist. The MoE target's PER-STEP active params are small (A3B), so the
  target itself is already fast -- which makes the "drafter must be faster than target" bar HARDER and
  the relative payoff smaller (T7). The MTP head still applies if the 35B ships one; otherwise EAGLE3.

--- T3/T4 (EAGLE3 detail; tree vs chain) --------------------------------------
Post is THIN -> no new info. Our EAGLE3 path (doc 12 H.3): Ex0bit/Qwen3.6-27B-PRISM-EAGLE3 is a real
drafter, Intel lists EAGLE/EAGLE3 as supported on Arc. VERDICT, both models: EAGLE3 is WORTH TRYING as
the vendor-blessed spec method, BUT it does NOT structurally avoid the verify penalty -- it verifies
through the SAME single GDN forward, so it hits the SAME eager-attn tax under PIECEWISE and the SAME
GDN-spec-FULL bug under FULL. So EAGLE3 is a drafter swap, not a fix for our actual bottleneck.

--- T5 (speedup ~= acc_len, minus draft overhead) -----------------------------
APPLIES as the framing; CONFIRMS doc 12. The post's caveat ("consistently overestimated" by draft
overhead) is precisely our measured reality, only worse: on the B70 the overhead is not just the draft
forward but the EAGER-ATTENTION VERIFY + fixed per-spec-step machinery (doc 12 G.1). Our N=1 MTP paid
2.28x target latency for 1.85x tokens => -19% (doc 12 G.1). So acc_len 1.85 did NOT give ~1.85x; it
gave 0.81x. The post's roofline correction is directionally right but understates the B70 penalty
because Modal's drafter cost model (5-20% of target) does not include an uncaptured-verify tax.
USEFUL TOOL: the SGLang SGLANG_SIMULATE_ACC_LEN flag idea -- we can do the equivalent on vLLM by
reasoning from measured acc_len, but a "mock-accept" probe would let us measure the B70 SPEEDUP
CEILING for a given acc_len independent of the drafter (see section 4).

--- T6 (fine-tune drafter; acc_len 3->9) --------------------------------------
Both models: NOT our lever right now -- and the post's own table proves why. Even acc_len ~3 (our
measured 2.86 @ N=3) SHOULD give a big win per the toy model; we get -37% instead. Raising acc_len to
9 would do NOTHING while the verify runs eager: 9 accepted tokens still cost 9 eager-attn verify
positions per step. FIX THE VERIFY FIRST (FULL capture), THEN fine-tuning the drafter becomes the next
lever. Filed for the post-FULL phase, both models.

--- T7 (batch u-shape: MoE non-monotonic, dense monotonic) -- THE KEY NEW STEER -
27B (dense-ish hybrid GDN): MONOTONIC per the post. Spec-decode payoff is highest at batch 1 and
  falls as batch grows -- which is bad for us because the B70's product is high batch (doc 14). So even
  if we unblock FULL+spec on the 27B, the win is a LOW-BATCH/single-stream win that fades exactly where
  we serve most users. VERDICT: spec-decode on the 27B is an INTERACTIVE-LATENCY feature (N<=4 users),
  not a throughput feature. Do not expect it to help the 32-user server.
35B-A3B (MoE): U-SHAPED per the post -> spec-decode may pay at LOW batch, NOT pay in a middle band,
  and possibly pay again at very high batch. This is actionable: the 35B's spec-decode decision must be
  measured PER BATCH SIZE, not as a single on/off. Our own corroborating evidence (doc 12 H.1 thc1006:
  35B-A3B spec went -11% even at 100% accept, "MoE expert-load union on verify") says the MIDDLE of the
  u-curve is where expert-union kills it. VERDICT: if we ever run 35B spec-decode, target only the
  bottom-left of the u (concurrency 1-2); disable it in the mid-batch band. NB this is moot today --
  35B is W4A16-int4 only on XPU (no int8-MoE expert kernel; doc 14/15), and FULL+spec+GDN is bugged.

--- T8 (adaptive speculation length) ------------------------------------------
Both models: NOT NOW. Premature -- adaptive draft-length is a refinement on top of a working
net-positive spec path, which we do not have on GDN. The post's own cautionary tale (retrain
thrashing) argues for FIXED config. We already use the fixed N=1 latency default (doc 12 H.1).
Parked until FULL+spec+GDN works.

--- T9 (when it helps vs hurts) -----------------------------------------------
The post underspecifies the failure modes; OUR docs are the authority here. B70 failure modes the post
omits, both models: (i) eager-attention VERIFY under PIECEWISE (doc 12 G.1) -- the dominant -19%;
(ii) GDN/Mamba per-step CPU-sync tax (vLLM #35387) specific to this hybrid arch; (iii) TP2/distributed
collectives (doc 12 C) -- so the 2-card config is NOT a spec-decode home. The post's u-shape (T7) is
the only failure mode it shares with us, and only for MoE.

--- T10 (SGLang / vLLM specifics) ---------------------------------------------
We are vLLM-XPU, not SGLang. The Ex0bit EAGLE3 head "ships SGLang tooling" (per our context) but we
serve on vLLM. The actionable SGLang item is the SGLANG_SIMULATE_ACC_LEN IDEA (mock-accept to get the
ceiling) -- transplantable as a probe (section 4), not a flag we have on vLLM-XPU.

--- T11 (DFlash block drafter + KV injection) ---------------------------------
Both models: NOT AVAILABLE as a turnkey path on our stack today. DFlash is the post's strongest lever
(block drafter, "once per block" cost, +5-20% over baselines, beats native MTP 1.5x per LMSYS;
doc 12 H.1). It is also the one whose ARCHITECTURE best fits our problem: a BLOCK drafter pays its
verify ONCE PER BLOCK, not once per token, which amortizes the fixed per-spec-step machinery that is
half our -19% (doc 12 G.1). BUT: (a) DFlash is SGLang-first and needs a trained DFlash head for
Qwen3.6 (none on host); (b) it still verifies through the GDN target -> same FULL/GDN-spec wall;
(c) it needs the fused KV-injection kernel (CUDA/SGLang). VERDICT: WATCH-LIST, both models -- the most
promising drafter family if/when a Qwen3.6 DFlash head exists and vLLM-XPU wires it, but not testable
now. The block-drafter amortization argument is the most useful design idea to carry forward.

--- T12 (1000 t/s @ concurrency 1, B200, MoE) ---------------------------------
APPLIES as proof-of-ceiling, NOT as a B70 target. It is concurrency-1 (the regime the B70 cares least
about), B200 (FP8 ALU + huge HBM bandwidth we do not have), and a 122B-A10B MoE. Useful as the "what
good looks like" datapoint, not a number we will approach on a 608 GB/s int8 card.

--- T13 (drafter memory overhead) ---------------------------------------------
APPLIES, NON-ISSUE both models. MTP/EAGLE3 heads are tiny and fit on the 32 GB card alongside the int4
27B (~18 GB) with room for KV. Memory is not the spec-decode constraint here (doc 12 B); compute/
launch overhead is.

--- T14 (the flywheel) --------------------------------------------------------
Aspirational, both models. Our flywheel is gated at step 1 (self-host a pre-trained speculator) by the
GDN-spec-FULL bug. No point in steps 2-4 (train/distill on production data) until a pre-trained MTP/
EAGLE3 head nets positive on the B70.

================================================================================
3. PRIORITIZED "WHAT TO TRY ON THE B70" (NOW vs GATED)
================================================================================
TESTABLE NOW (one card, behind scripts/gpu-run):
  N1. [27B] EAGLE3 (Ex0bit PRISM) under PIECEWISE + TRITON-shim, N=2. The one spec method Intel claims
      is "supported on Arc." Honest prior: still PIECEWISE-eager-verify-limited (same wall as MTP), so
      expect net-flat-to-negative -- but it EXERCISES the vendor-blessed path and may carry less
      GDN-specific overhead than the qwen3_5_mtp head. Low cost, decides MTP-vs-EAGLE3 for THIS model.
      (doc 12 H.3 RANK 2.)
  N2. [27B] PROFILE one N=1 MTP spec step on the current stack and look for the vLLM #35387
      GDN/Mamba CPU-sync (device->host copy of num_accepted_tokens before mamba_postprocess). This is
      the highest-information probe short of FULL: if that host sync is present, FULL capture ALONE
      will not fully fix the 27B (it is a sync, not an attention launch) -> tells us whether to keep
      betting on MTP for the GDN 27B or pivot. (doc 12 G.3 RANK 3 / H.3 RANK 3.)
  N3. [any] MOCK-ACCEPT CEILING probe (the SGLANG_SIMULATE_ACC_LEN idea, T10). Construct a vLLM run
      that mocks high acc_len (or compute from measured) to nail the B70 SPEEDUP CEILING for a given
      acc_len, decoupled from the verify tax -- confirms how much FULL capture could possibly recover
      before we invest in the GDN fix.
  N4. [14B dense, W4A8-gptq] Already DONE (JOURNAL 2026-06-21): FULL capture via TRITON_ATTN works,
      decode +8.5%, prefill -50%. This is the spec-decode plumbing PROOF on a non-GDN target. Next: run
      a REAL draft model (Qwen3-0.6B) on the 14B under FULL+TRITON_ATTN -- the 14B has NO GDN, so it
      AVOIDS the spec-capture bug and would give us the first net-positive spec-decode number on the
      B70 (the only architecture where FULL+spec is unblocked today). (doc 12 RANK 2.)

GATED ON THE vLLM GDN-SPEC-CAPTURE FIX (the `spec_query_start_loc` bug):
  G1. [27B] FULL_DECODE_ONLY (or FULL_AND_PIECEWISE) + TRITON_ATTN + qwen3_5_mtp N=1. THE definitive
      MTP-on-B70 test: capturing the VERIFY is the only thing that removes the -19% (doc 12 G.4, H.2).
      Blocked TODAY by the GDN-spec-FULL bug (JOURNAL 2026-06-21), NOT by config. This is the single
      highest remaining lever for the 27B. Needs an upstream vLLM-XPU GDN spec-metadata fix
      (gdn_attn.py spec_query_start_loc sizing) -- deep, not a flag.
  G2. [27B] EAGLE3 + FULL -- same gate (verifies through GDN -> same bug).
  G3. [35B-A3B] spec-decode at all -- doubly gated: (a) the GDN-spec-FULL bug AND (b) no int8-MoE
      expert kernel on XPU (35B is W4A16-int4 only; doc 14/15). Even after both, the T7 u-shape says
      only concurrency 1-2 would pay. Lowest priority.

GATED ON DUAL-CARD (and even then, NOT a spec-decode home):
  D1. TP2 27B-W8A8 for CAPACITY/throughput WITHOUT spec-decode -- TP2 disables graph capture (#34482)
      and adds per-step host-staged collectives -> spec-decode at TP2 is net-negative (doc 12 C, H.1
      dredyson -50%). The 2nd card is for VRAM/throughput, never for spec-decode latency.
  D2. Dual-card GPTQ/AutoRound calibration (W4A8 + W8A8) -- enables better single-card quants to serve
      with capture ON, but is orthogonal to spec-decode.

WATCH-LIST (no action; track upstream):
  W1. A Qwen3.6 DFlash head + vLLM-XPU KV-injection support (T11) -- the block-drafter amortization is
      the most promising future fix for our fixed-per-step overhead.
  W2. oneAPI DPC++ 2026.0 -> flash-attn FULL capture (no TRITON_ATTN, so NO prefill penalty) -- a
      cleaner FULL path than TRITON_ATTN, but still hits the same GDN-spec bug for the flagships.

================================================================================
4. THE INSIGHT WE WERE MISSING
================================================================================
The MoE batch-size U-SHAPE (T7). Docs 12/14 reasoned about spec-decode almost entirely at single
stream / batch 1, and treated the 35B-A3B like the 27B for spec-decode purposes. The post's roofline
result -- spec-decode speedup is NON-MONOTONIC (u-shaped) vs batch for MoE but MONOTONIC for dense --
means the 35B's spec-decode payoff is a DIFFERENT SHAPE from the 27B's, with a mid-batch DEAD ZONE
where the routed-expert union inflates the verify (corroborated by our own doc-12 H.1 thc1006 -11% @
100% accept). Practical consequence we had not stated: any future 35B spec-decode must be gated by
batch size (enable only at the u-curve extremes, esp. concurrency 1-2), whereas the 27B's spec-decode
simply decays monotonically with batch and is purely an interactive-latency feature. This refines
doc 12 section C (which lumped both flagships) and doc 14's "what's left" MoE line.

Secondary, smaller insights worth keeping:
  - BLOCK DRAFTERS (DFlash) pay verify "once per block" not per token -- the cleanest architectural
    answer to our fixed-per-spec-step overhead (doc 12 G.1), if a Qwen3.6 DFlash head ever lands.
  - The SGLANG_SIMULATE_ACC_LEN MOCK-ACCEPT trick -- measure the speedup CEILING for an acc_len without
    a trained drafter; a cheap probe we had not considered (N3 above).
  - Everything else in the post (speedup~=acc_len, fine-tune to raise acc_len, drafter-must-be-faster)
    we had already internalized in docs 12/14; the post CONFIRMS rather than corrects them. Crucially,
    the post's headline ("the only optimization that matters at HIGH INTERACTIVITY") is a single-stream
    framing -- it does NOT apply to the B70's high-batch product, which is the opposite regime.
