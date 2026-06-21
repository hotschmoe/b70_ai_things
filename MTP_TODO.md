# MTP_TODO.md — Multi-Token Prediction as the primary decode-speed lever

**Created:** 2026-06-20
**Owner:** b70 team
**Status:** PLAN — not started
**Related:** [`docs/literature/07_w8a8_int8_recovery.md`](docs/literature/07_w8a8_int8_recovery.md) · [`JOURNAL.md`](JOURNAL.md) (MTP entries) · [`scripts/49_quantize_27b_w8a8.sh`](scripts/49_quantize_27b_w8a8.sh) · `contrib/vllm_int8_xpu/`

---

## Why this is the priority (the reframe)

**[UPDATED 2026-06-21] The 4xB70 TP=4 MTP result IS real -- the project owner knows the author personally** (primary source; it is simply not publicly documented, which is why a web search cannot find it -- do not treat "not public" as "not real"). Take the headline as a real 4-CARD datapoint: Qwen3.6-27B **BF16**, TP=4, **decode ~54.2 t/s, prefill ~2100, accept ~4.04 @ spec=5** -> ~2.9x. **HIGH-VALUE ACTION: get the exact repro from the source** -- image tag (llm-scaler-vllm version), `vllm_xpu_kernels` version, the `--speculative-config` JSON + `num_speculative_tokens`, vLLM commit/PR, **whether FULL graph capture was used**, the checkpoint (MTP-head precision), KV dtype, single-stream-vs-aggregate framing, and the **4-card interconnect** (PCIe gen / switch / platform / any P2P).

**The tension to resolve (this is why the recipe matters):** their 4xB70 result CONTRADICTS our own single-card B70 MTP, which is currently NET-NEGATIVE -- **25.5 t/s = -19% vs 31.4 MTP-off** (PIECEWISE, 86.9% first-token accept; the verify runs attention EAGER x(K+1)). So their config achieves what ours does not -- the exact recipe is the unlock. Most likely differences to probe: **FULL graph capture** (vs our PIECEWISE), a **torch-2.11 image with PR #43565 + vllm_xpu_kernels v0.1.10** (an ABI split currently blocks one wheel being both torch-2.10-safe AND spec-capable on the stock image), and/or a faster 4-card interconnect. **For OUR 2-card rig specifically:** MTP does NOT need TP, and TP>1 HURTS it on a no-P2P link (TP=2 = 0.53x), so our payoff path is **MTP per data-parallel replica** once FULL capture makes MTP net-positive on a single card.

The strategic consequence:

- **MTP is a ~3–4× multiplier.** Weight-format choice (W4A8 vs W8A8) is only ~1.2–1.5×.
- **MTP stacks orthogonally with format.** At spec=5 single-stream the verify pass is still *bandwidth-bound* (M≈6, intensity ~6 ops/byte ≪ B70's ~800 ops/byte int8 ridge), so MTP just cuts the number of bandwidth-bound passes by ~L×. It does NOT make decode compute-bound, so the weight-byte advantage of smaller formats *persists* under MTP.

**Therefore: get MTP working first, on the simplest format, then layer quant on top.** The format question is second-order and can be measured *after* the MTP pipeline is proven. Do not let the W4A8-vs-W8A8 decision block the MTP prize.

**The key per-format unknown is MTP acceptance.** The MTP head stays BF16 (required) and was trained against a full-precision body; quantizing the body may lower accept length. Expected ordering: **BF16 (4.04, proven) ≥ W8A8 (near-lossless → likely holds ~3.7–4.0) ≥ W4A8 (int4 drift → may drop to ~3.0–3.5)**. All hypotheses — *measure per format, that's the point of this plan.*

---

## Reference "known-good" config to reproduce (from the 4×B70 bench)

| Ingredient | Value | Notes |
|---|---|---|
| Image | `intel/llm-scaler-vllm:0.14.0-b8.3` | Intel's container; has gdn_attention for DeltaNet |
| XPU kernels | `vllm_xpu_kernels v0.1.9` wheel | spec-decode enablement |
| Spec wiring | `qwen3_5.py` patch — **vLLM #43565** | the spec-decode wiring patch |
| KV | **Half-KV** | needed to fit 256K context |
| Spec config | `num_speculative_tokens=5` | got accept length 4.04, 88.9% @ spec=3 |
| Engine | vLLM-XPU, TP=4 (their setup) | ours: TP=1 for 14B, TP=1 for 27B-int4, TP=2 for 27B-W8A8 |

**Integration note (our hard part):** our W8A8 fastpath is the *custom* `contrib/vllm_int8_xpu` oneDNN s8s8s32 kernel, NOT stock. And our `:int8` image historically **lacked `gdn_attention`** (27B first-token crash). So the real engineering is **one image carrying all of: (a) our W8A8 int8 GEMM registration, (b) `gdn_attention` GDN kernel, (c) the #43565 spec-wiring + Half-KV, (d) a compatible `vllm_xpu_kernels`.** That integration — not the GEMM-format choice — is where the time goes.

---

## Playbook A — How to get GOOD MTP (the knobs that move accept length × speedup)

The goal is **max end-to-end tok/s**, which is `base_decode × (mean_accept_length / verify_overhead)`. Maximize accept length, minimize verify overhead. In priority order:

1. **Kill eager-attention verify overhead — this is what murdered our earlier MTP.** Our prior run was net-negative because attention ran eager during verify (the −7% even with PIECEWISE). MTP's verify is a fixed-shape repeated forward → it *loves* graph capture.
   - Get **PIECEWISE graph capture** working with MTP first (`VLLM_XPU_ENABLE_XPU_GRAPH=1`, `cudagraph_mode=PIECEWISE`, the `:int8g`-style image with fake-kernel registrations). PIECEWISE alone gave us +16.7% decode (23.3→27.2).
   - Pursue **FULL** capture if the Intel SYCL-Graph `work_group_scratch_memory` limitation is liftable — note `torch.xpu.XPUGraph` now exists upstream (PyTorch 2.11, per doc 06); vLLM-XPU just hasn't wired it. FULL is likely what flips spec-decode from −7% to strongly positive.
2. **Sweep `num_speculative_tokens`.** The bench used 5 (→ accept 4.04). Higher spec = more drafted tokens/pass but lower per-position acceptance and higher verify cost. Sweep spec ∈ {2,3,4,5,6,8}; **pick the value that maximizes tok/s, not accept length** (they peak at different points). Log both per spec value.
3. **Measure at greedy (temp=0) for the headline, then map the temperature→acceptance curve.** Acceptance is maximal when distributions are peaked; higher temperature lowers accept length. Report max speedup at greedy; note the production-temp number separately.
4. **Keep the MTP head in BF16** (required — doc 04). It must be in the ignore-list (`re:.*mtp.*`) for every quant scheme; quantizing it kills drafting.
5. **Isolate Half-KV's effect on acceptance.** Half-KV / quantized KV perturbs the verify distribution and *can* depress accept length. Measure accept with full-precision KV vs Half-KV to confirm the context-fitting trick isn't silently costing speedup.
6. **The body's quant is the acceptance risk, not the head's.** A BF16 head drafting for a quantized body drifts: BF16 (4.04) ≥ W8A8 (likely holds) ≥ W4A8 (may drop). This is the single number Phase A/B exists to measure.
7. **Measurement hygiene:** discard warmup tokens (compile/graph capture), fixed prompt set + fixed context, batch-1, report median accept length and median tok/s over ≥3 runs.
8. **MTP is a low-concurrency/latency lever.** At batch-1 it shines (bandwidth-bound); as batch grows the verify goes compute-bound and MTP's relative win shrinks. Don't expect the 3–4× to survive into high-concurrency batched serving — there, KV capacity (favoring smaller formats) matters more.

---

## Playbook B — Quant procedures for RECOVERABLE quants (W8A8 / W4A8 / W4A16)

Produced via `scripts/49` (SCHEME-parametrized) in the Intel llmcompressor container. "Recoverable" = go hard on the activation+weight recovery levers from [`docs/literature/07`](docs/literature/07_w8a8_int8_recovery.md) so the quant clears the accuracy gate *before* it earns MTP effort.

### Common to all schemes
- **Ignore-list (name-robust):** `lm_head re:.*linear_attn.* re:.*visual.* re:.*mtp.*` — DeltaNet, vision tower, MTP head, lm_head stay BF16. Keep the **parent-module regex** (`re:.*linear_attn.*`); never enumerate leaf names (`in_proj_qkvz`/`in_proj_ba`) → avoids the vLLM #40252 silent-zeroing trap.
- **Calibration:** 512 samples final (128 was within noise + 3× faster → use 128 to iterate), `SEQLEN=2048`, **data matched to the model** (chat/instruction data for instruct models, not random tokens).
- **GPTQ flag:** `actorder=None` (act-reordering triggers an XPU gather device-lost on our stack).
- **Validate before MTP:** eval top-1 agreement / ppl / gsm8k vs the BF16 baseline. A quant that misses the accuracy gate is out regardless of MTP speed.

### W8A8 — int8 w × int8 a (best accuracy; 27B needs 2 cards)
1. **Selective SmoothQuant (THE fix for the hybrid).** Build SmoothQuant mappings only where pairing is clean — the **16 full-attn layers** (`input_layernorm → {q,k,v}`, `v_proj → o_proj`) + the **64 MLP layers** (`post_attention_layernorm → {gate,up}`). **Skip DeltaNet `linear_attn`.** This sidesteps the `ValueError: got [all 64 input_layernorm]` that currently forces `SMOOTHQUANT=0`. (smoothing_strength ~0.8; tune per-node alpha if needed.)
2. **GPTQ weights, symmetric per-channel int8** (NOT group-128 — that's an int4 tool; per-channel is standard + cheaper at int8).
3. **Per-token dynamic int8 activations** at runtime (our kernel's path).
4. *Optional rescue:* hold early+late `down_proj` at W8A16 (GLU spike site).
- Expected: ~0.908 top-1 agreement (GPTQ), essentially W8A16-class quality.

### W4A8 — int4 w × int8 a (single-card 27B; hits int8 systolic) ← the recovery-critical one
1. **Selective SmoothQuant** — same mapping as W8A8 (recovers the int8-activation half).
2. **GPTQ weights, int4, group size 128** (int4 weights need finer granularity than per-channel; this is where group-128 earns its keep).
3. **Per-token dynamic int8 activations.**
4. **`down_proj` higher-precision carve-out matters MORE here** — int4 weights amplify the GLU/Super-Weight sensitivity. Try early+late `down_proj` at W4A16 or W8A16.
- Expected: RTN ~0.822 → GPTQ + SmoothQuant lifts toward ~0.86+. **This is the scheme where the recovery work decides viability** — if it can't clear the gate near W8A8, W4A8 is out.

### W4A16 — int4 w × fp16 a (single-card 27B; simplest accuracy, no int systolic)
1. **No SmoothQuant** — weight-only, so it's a no-op (`scripts/49` auto-sets `SQD=0` for `*A16`).
2. **GPTQ (or AWQ) weights, int4, group size 128.**
3. **fp16 activations** (no activation quant → no activation-fidelity loss; the only error is int4 weights).
- Expected: ~0.883 agreement (GPTQ); GPTQ lift +4.2 over RTN here — int4 weight error is real and GPTQ recovers it.
- **Caveat:** the 4304-dim `XPUwNa16` (multiple-of-32) kernel-coverage gap blocked our W4A16 27B once — verify the layer dims actually serve before counting on this scheme.

### Recovery levers, ranked (apply in this order until the gate clears)
1. **Selective SmoothQuant** on the 16 full-attn layers — the activation-fidelity lever (recovers the ~−10pt W8A16→W8A8 drop).
2. **GPTQ weights** — recovers int4/int8 weight error; lift scales with quant error (measured: W8A8 +2.7, W4A16 +4.2).
3. **`down_proj` at higher precision** (early+late layers) — the GLU/Super-Weight spike site.
4. **Better calibration** — 512 samples, chat-domain data.
5. **KL-sensitivity layer ranking** (arXiv 2604.13440, forward-only) — pick *which* layers to hold high-precision by measurement instead of guessing.

---

## Success criteria

- **Primary:** ≥ **3× decode speedup** from MTP (target 3–4×) on each qualified config, measured as `decode_tok/s(MTP on) / decode_tok/s(MTP off)` at batch-1 single-stream.
- **Secondary:** mean accept length logged per (model, scheme); accuracy within gate (see Phase B) for the quantized 27B configs.
- Every run logged in the table below — no run counts unless it's logged.

---

## Phase 0 — Prerequisites (do before anything)

- [ ] **Confirm Qwen3-14B has an MTP head** (`mtp.*` weights in the checkpoint). The proven bench was the **27B**; MTP shipped with the Qwen3-Next/3.6 family, and our 14B test vehicle is the *dense* Qwen3-14B which **may not have MTP**. Resolve one of:
  - (a) 14B checkpoint *does* have `mtp.*` → proceed with Phase A as written.
  - (b) 14B lacks MTP → qualify the *pipeline plumbing* on 14B via **ngram / draft-model spec-decode** (proves the spec loop, image, kernels), and accept that true MTP-acceptance numbers only come from the 27B (Phase B).
  - (c) substitute a 14B-class model that *does* ship MTP as the test vehicle.
- [ ] Stand up the reference image + `vllm_xpu_kernels v0.1.9` + #43565 patch + Half-KV; confirm it boots and serves *something* on one B70.
- [ ] Pin the logging template (below) and a repeatable bench harness (reuse `perf_probe` / existing eval harness; fixed prompt set, fixed context, batch-1).

---

## Phase A — Qualify the MTP pipeline on Qwen3-14B (fast iteration, fits one card)

Rationale: the 14B is the clean, fast test vehicle (dense, fits one card, no 2-card TP complexity). Walk BF16 → W8A8 → W4A8 so each step adds exactly one variable.

- [ ] **A1 — BF16 14B + MTP.** Qualify the wiring with *zero* quant complexity. This isolates "does our MTP spec loop work" from "does quant break it." Log decode (MTP on/off), accept length, prefill, VRAM.
  - Gate to pass: ≥3× MTP speedup, accept length in a sane range (≥3). If this fails, the spec wiring/image is wrong — fix before touching quant.
- [ ] **A2 — W8A8 14B + MTP.** Our custom int8 kernel + spec wiring (the integration). Question: **does accept length hold vs BF16?** (expected: ~holds, int8 near-lossless). Log everything + top-1 agreement / gsm8k vs BF16.
- [ ] **A3 — W4A8 14B + MTP.** Adds int4 weights. Question: **does accept length hold at int4, or drop?** Log everything + accuracy. This is the first real datapoint on the W4A8-acceptance risk.

**Phase A exit:** we know (i) the MTP pipeline works on our stack, (ii) how much acceptance each format costs, (iii) the per-format MTP'd decode rate — i.e. enough to make the 27B format decision on evidence, not speculation.

---

## Phase B — Good 27B quant for single-card MTP: W4A16 + W4A8 (fit on ONE card)

Rationale: W4A16 (~25 GB) and W4A8 (~17 GB) both **fit one B70** → we can do this *now*, before a second card arrives. "Good quant" = go hard on recovery first (per doc 07): GPTQ weights + **selective SmoothQuant on the 16 full-attn layers**, DeltaNet `linear_attn` kept BF16, lm_head/vision/mtp ignored.

- [ ] **B0 — Produce the good quants** (no MTP yet): `scripts/49` SCHEME=W4A16 and SCHEME=W4A8, with GPTQ + selective-SmoothQuant. Eval accuracy vs the BF16 and W8A8 27B baselines (top-1 agreement, ppl, gsm8k).
  - **Accuracy gate:** W4A16 and W4A8 must land "close to W8A8/BF16" (within our noise band on gsm8k; agreement gap acceptable). If a scheme misses the gate, it's out for the 27B headline regardless of speed.
- [ ] **B1 — W4A16 27B + MTP.** Fits one card; simplest accuracy story. Caveat: fp16 verify (no int8 systolic), so it's the *weakest* MTP-compute fit — but the cheapest to stand up. Log MTP metrics + accuracy.
- [ ] **B2 — W4A8 27B + MTP.** Fits one card; int8 systolic verify. The interesting one — does accept length hold at int4 weights on the *27B* (which is more quant-robust than the 14B)? Log MTP metrics + accuracy.

**Phase B exit:** a *serving-ready, MTP-accelerated 27B that runs on a single B70*, with W4A16 vs W4A8 decided on measured accuracy + accept length + decode rate.

---

## Phase C — Headline 27B W8A8 + MTP (DEFERRED — needs the 2nd card)

Rationale: 27B W8A8 ≈ 33 GB weights → **does not fit one 32 GB card**; needs TP=2 for VRAM headroom (and leaves ~37 GB KV across two cards — generous for long context). This is the production headline target, but it's blocked on hardware.

- [ ] **C1 — (when 2nd card lands) W8A8 27B + MTP, TP=2.** Best accuracy of the int-fastpath schemes + best expected accept length + native int8 prefill. Expected to be the production default. Log MTP metrics + accuracy + TP=2 PCIe overhead.

---

## Logging template (fill one row per run; newest at bottom)

Capture in this table (and mirror notable runs into `JOURNAL.md`):

| Date | Model | Scheme | Image / kernels / patch | TP | Ctx / KV | spec_toks | **accept len** | accept% @3 | **dec MTP-on** | dec MTP-off | **MTP ×** | prefill | TTFT | VRAM | acc (agree/gsm8k) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | Qwen3-14B | BF16 | — | 1 | — | 5 | — | — | — | — | — | — | — | — | — | A1 |
| — | Qwen3-14B | W8A8 | — | 1 | — | 5 | — | — | — | — | — | — | — | — | — | A2 |
| — | Qwen3-14B | W4A8 | — | 1 | — | 5 | — | — | — | — | — | — | — | — | — | A3 |
| — | Qwen3.6-27B | W4A16 | — | 1 | — | 5 | — | — | — | — | — | — | — | — | — | B1 |
| — | Qwen3.6-27B | W4A8 | — | 1 | — | 5 | — | — | — | — | — | — | — | — | — | B2 |
| — | Qwen3.6-27B | W8A8 | — | 2 | — | 5 | — | — | — | — | — | — | — | — | — | C1 (2-card) |

**Reference (REAL -- private source; project owner knows the author; not publicly documented):** Qwen3.6-27B BF16, **4x B70 TP=4**, spec=5 -> accept 4.04, dec 54.2, prefill 2100. **GET THE EXACT REPRO** (image/kernels/`--speculative-config`/FULL-capture?/4-card interconnect) -- their 4-card config beats our single-card MTP (currently **-19%: 25.5 vs 31.4, PIECEWISE**), so the recipe is the unlock. (Public datapoints for sanity only: Puget 4xB70 TP=4 27B-dense 13.1 t/s/1u no-MTP; vLLM PR #43565 MTP on B60/Qwen3-Next-80B/spec=2.)

---

## Open risks / watch-items

- **14B MTP-head existence** (Phase 0) — the whole 14B plan assumes it; resolve first.
- **Acceptance decay with quant** — the W4A8 thesis lives or dies on whether accept length holds; A3 + B2 are the deciding measurements.
- **Image integration** — our custom W8A8 kernel + `gdn_attention` + #43565 spec-wiring + compatible `vllm_xpu_kernels` in one image is the real engineering; budget for it.
- **27B int4 kernel-coverage gaps** — W4A16 already hit the 4304-dim `XPUwNa16` (multiple-of-32) wall; W4A8 may hit similar. Verify dims before assuming the quant serves.
- **Half-KV / KV-quant interaction with MTP** — confirm Half-KV doesn't depress accept length.
- **DeltaNet ignore-list correctness** — keep `re:.*linear_attn.*` (parent-module regex, name-robust); don't regress to enumerating leaf names (`in_proj_qkvz`/`in_proj_ba`) or you risk silent layer-zeroing (vLLM #40252).
