# 07 — W8A8 INT8 Accuracy Recovery on Intel Arc Pro B70 (Battlemage) — Research Survey

**Date:** 2026-06-19
**Goal:** Survey the *actual* research and runnable options for recovering the quality/accuracy of our **W8A8 INT8** fast path (NOT FP8) on the B70, focused on the Qwen3.6-27B DeltaNet hybrid. Answer three questions definitively: (1) is our fast path INT8 W8A8 or FP8; (2) is FP8 a W8A8 or a W8A16 scheme; (3) what recovers the ~10-point activation-quant fidelity loss we measured.
**Scope:** Synthesis of 5 parallel research sweeps (rotation methods, SmoothQuant family + weight-side, Intel B70 toolchain, bleeding-edge 2025–2026 papers, DeltaNet/SSM angle) cross-checked against our own eval campaign.
**Builds on:** [`05_w8a8_recipe.md`](./05_w8a8_recipe.md) (the W8A8/FP8 dispatch gap — note: that doc predates our custom `contrib/vllm_int8_xpu` oneDNN kernel, so its "no INT8 W8A8 on XPU" TL;DR is now superseded by our own kernel) and [`06_xpu_kernel_fastpaths.md`](./06_xpu_kernel_fastpaths.md) (DPAS/XMX surfaces). Grounded against `evals/results/SUMMARY.md`, `scripts/49_quantize_27b_w8a8.sh`, and `JOURNAL.md`.

**Confidence markers:** `[WELL-SOURCED]` = 2+ primary sources / direct spec quote · `[SINGLE-SOURCE]` = one primary source · `[BODY-SOURCED]` = number lives in a paper body, not the abstract · `[UNVERIFIED]` = could not confirm, flagged honestly · `[OURS]` = measured in this repo.

---

## ⛔ TL;DR

1. **Our fast path is INT8 W8A8** — symmetric per-channel weights × per-token dynamic INT8 activations on oneDNN `s8×s8→s32` GEMM (our `contrib/vllm_int8_xpu` SYCL kernel), executing on the B70's native XMX DPAS systolic array. `[OURS]`
2. **FP8 is a W8A8 scheme, not W8A16** — 8-bit weights *and* 8-bit activations, just floating-point (E4M3/E5M2) instead of integer. The only difference from our INT8 path is how the 8 bits are spent (FP8's exponent gives wide dynamic range; INT8's uniform spacing does not). `[WELL-SOURCED]`
3. **Xe2/Battlemage XMX has NO native FP8 matrix unit.** FP8 on the B70 upconverts to bf16 for the matmul — memory savings only, no compute speedup. **INT8 W8A8 is the only real low-precision *compute* fast path on this hardware.** `[WELL-SOURCED]` This is why FP8 loses prefill to W8A8 by 1.64× in our evals.
4. **The fidelity cost is INT8 *activation* quant, not INT8 weights** — confirmed by our evals AND four independent 2024–2026 papers. The field considers W8A8 INT8 essentially *solved* for modern models (~99% of BF16) and moved on to W4A4. There is no shiny new W8A8 algorithm to chase; the leverage is in *applying the known recipe well* on our hybrid.
5. **Skip rotation (QuaRot/SpinQuant/etc.) at W8A8** — marginal at 8-bit by the papers' own admission, and no Hadamard kernel exists for Intel SYCL/XMX anyway.
6. **We are at the frontier on DeltaNet INT8** — no DeltaNet/Gated-DeltaNet-specific INT8 quant paper exists. The transferable recipe is the Mamba/SSM literature (Quamba2, Q-Mamba).

---

## 1. The hardware verdict — no native FP8 on Xe2 `[WELL-SOURCED]`

| Source | Evidence |
|---|---|
| oneDNN GPU data-types table | Xe2-HPG (Battlemage): fp8 (`f8_e4m3`/`f8_e5m2`) = `.` ("supported via conversion to a higher precision data type"); int8 (`s8`/`u8`) = `+` ("hardware-native compute support"). https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html |
| SYCL `joint_matrix` ext spec | Lists no fp8 type for any Intel arch (only uint8/sint8/fp16/bf16 for A/B operands). https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc |
| Chips and Cheese / HWCooling Xe2 deep-dives | XMX supports FP16, BF16, INT8, INT4, INT2 — **no FP8**. |
| B70 spec sheet | Headline **367 INT8 TOPS** (non-sparse); no published FP8 TOPS figure (because there is no native FP8). |

**Implication:** FP8 *runs* on B70 (oneDNN/vLLM-XPU have an optimized emulated path) but is upconverted to bf16/fp16 for compute — you get FP8's memory/bandwidth savings, **not** FP8 compute speedup. Per Intel's own Xe3 guidance, emulated-FP8 and INT8 land at "the same performance." So the literature's "FP8 is free near-lossless, no calibration" results **do not transfer to our hardware** — INT8 accuracy *recovery* is the whole game.

*Nuance:* the oneDNN table also marks Data Center GPU Max (Ponte Vecchio/Xe-HPC) fp8 as emulated, conflicting with PVC FP8 marketing. Native FP8 in oneDNN's accounting first appears at Xe3p. Doesn't change the Xe2 verdict. `[SINGLE-SOURCE]`

---

## 2. The single root cause: activation quant, not weights

Our own data nailed this before the literature did (`evals/results/SUMMARY.md`):

| Scheme | PPL | Top-1 agreement vs BF16 | Note |
|---|---|---|---|
| FP8 | 12.70 | 0.968 | reference |
| W8A16 (int8 w, **fp16 a**) | 12.76 | **0.981** | near-lossless — beats FP8 |
| **W8A8** (int8 w, **int8 a**) | 13.08 | **0.881** | the −10 pts is the activation quant |
| W4A16 | 13.55 | 0.841 | int4 weight error |

`[OURS]` W8A16→W8A8 drops agreement 0.981→0.881 (−10 pts) but GSM8K barely moves (95.3% vs FP8 96.0%) — **it flips low-confidence tokens, not answers.**

Four independent confirmations `[WELL-SOURCED]` / `[BODY-SOURCED]`:
- **"Give Me BF16 or Give Me Death" (2411.02355, ACL'25):** W8A8-INT recovers ~99.3% of BF16 across Llama-3.1 8B/70B/405B.
- **"Systematic Characterization" (2508.16712):** naive per-tensor INT8 *activation* quant breaks coding/math; the same 8-bit weights-only (W8A16) is fine. *(The widely-circulated "92% of HumanEval destroyed" figure is blog-sourced, not abstract-verified — `[UNVERIFIED]`.)*
- **"Quantization Hurts Reasoning?" (2504.04823, COLM'25, tests Qwen3-8B directly):** "lossless quantization can be achieved with W8A8 or W4A16."
- **Long-context (2505.20276, EMNLP'25):** 8-bit is safe (~0.8% drop, <5% on RULER at 128K); degradation is a sub-4-bit phenomenon.

**The field abandoned W8A8 as solved.** OmniQuant (2308.13137) verbatim "excludes W8A8 quantization since SmoothQuant can nearly achieve lossless W8A8." Nearly every post-2023 transform paper (AffineQuant, DuQuant, FlatQuant, OSTQuant) reports W4A4/W4A8, not W8A8.

---

## 3. Recovery options, ranked by leverage for our path

### Tier 1 — do these

**3.1 Fix SmoothQuant for the hybrid (highest leverage).** `[WELL-SOURCED]` + `[OURS]`
SmoothQuant (2211.10438) is *the* activation-recovery lever — a math-equivalent per-channel transform `s_j = max(|X_j|)^α / max(|W_j|)^(1−α)` that migrates outlier difficulty from activations (hard) to weights (easy), fused into preceding layers at zero runtime cost. We currently run `SMOOTHQUANT=0` because pairing breaks on the Qwen3.6 16/64 full-attention split (`scripts/49` line 104, JOURNAL 822). The fix is **selective application**: apply SmoothQuant to the layers where pairing is clean (the 16 full-attention layers' q/k/v/o + MLP up/gate/down) and skip the DeltaNet `linear_attn` layers. NNCF's design is the reference — **per-node-type alpha** (`smooth_quant_alphas` for matmul vs others) lets you tune the full-attention layers without touching the recurrent path. Our GPTQ-only run already buys +2.7 agreement (0.881→0.908); SmoothQuant on the full-attn layers is the next chunk.
- α is per-model, NOT a constant: 0.5 default; ~0.75 for outlier-heavy (GLM-130B has ~30% outlier channels); ~0.8 for big Llamas. `[WELL-SOURCED]`

**3.2 OS+ (Outlier Suppression+, 2304.09145, EMNLP'23) as the SmoothQuant alternative.** `[SINGLE-SOURCE]`
The one genuine direct successor that adds something at INT8: per-channel **shifting** (not just scaling) to handle *asymmetric* outliers that SmoothQuant's scale-only transform misses. More local than SmoothQuant → potentially better fit for a hybrid that resists clean q/k/v pairing. Repo: github.com/ModelTC/Outlier_Suppression_Plus (exports an FP model; no inference-backend integration — we'd port the transform, not a kernel).

**3.3 Targeted higher-precision layers — automate the ignore-list.** `[WELL-SOURCED]`
`down_proj` (FFN second linear) is the dominant activation-outlier site:
- SwiGLU/GLU activation spikes concentrate in `down_proj` at early+late layers (2405.14428).
- "Super Weight" (2411.07191): a single param in early `mlp.down_proj` whose removal raises PPL 3 orders of magnitude.
- Llama-3-70B is uniquely W8A8-fragile via per-channel weights; fix = <3% of layers at finer granularity (2408.15301). Qwen2/3, Mistral, Mixtral, Phi3 all robust (<1% drop).

Action: a knob to hold `down_proj` (early+late) at W8A16. Choose layers principledly with **`quantize_with_accuracy_control()` (NNCF)** — ranks layers by sensitivity and auto-reverts the worst to FP16 until a max-drop target. A systematic version of our hand-curated DeltaNet ignore-list.

**3.4 Keep RTN for weights unless measured otherwise.** `[WELL-SOURCED]` + `[OURS]`
At W8, RTN ≈ GPTQ ≈ AWQ (Llama3-8B W8: FP16 6.1 / RTN 6.2 / GPTQ 6.1 / AWQ 6.1; Qwen3-8B MMLU all within 0.2 pt). Matches our "lift scales with quant error" finding (commit 2eab596: W8A8 +2.7, W4A16 +4.2). GPTQ's small W8 lift comes with calibration-overfitting risk RTN avoids. Use GPTQ only where a real win is measured. Group-128 is an INT4 tool — at INT8 use symmetric **per-channel** weights (zero-point folds away, cheaper kernel). Calibration: 512 samples, data matched to the model (chat/instruction data for instruct models); 128 was within noise in our study and 3× faster — use 128.

### Tier 2 — situational

- **KV-cache → INT8 or FP8-E4M3:** near-lossless, frees memory for batch/throughput. Key quantized per-channel, Value per-token (Key has channel outliers). `[WELL-SOURCED]`
- **Per-tensor *static* INT8** (if we ever want max GEMM saturation over per-token dynamic): only safe after neutralizing outliers — **PrefixQuant (2410.05265)** removes token-wise outliers so static beats dynamic; **TWEO (2511.23225)** makes per-tensor static W8A8 usable but needs a *training-time* regularizer (weight-colinearity is the root cause of extreme outliers, cut from 10,000+ to <20). `[SINGLE-SOURCE]`

---

## 4. The DeltaNet / linear-attention frontier (most novel, most relevant)

**There is NO DeltaNet- or Gated-DeltaNet-specific INT8 quantization paper.** `[WELL-SOURCED]` That gap is real — we are at the frontier. The transferable body is SSM/Mamba quantization, which confirms our outlier hypothesis: **activation outliers appear in the linear-recurrent state and are the #1 INT8 obstacle there.**

| Method | arXiv | Architecture / precision | Transferable insight | Repo |
|---|---|---|---|---|
| **Quamba** | 2410.13229 | Mamba, W8A8 static per-tensor | SSMs have "massive outliers in output activations not present in self-attention"; quantize the SSM output in a Hadamard-rotated space | github.com/enyac-group/Quamba |
| **Quamba2** | 2503.22879 (ICML'25) | Mamba1+2, W8A8/W4A8/W4A16 | **Cleanest INT8-a-recurrent-state recipe: offline sort+cluster the recurrence input + per-state-group quant of input-dependent B/C.** 1.6% avg drop | github.com/enyac-group/Quamba |
| **MambaQuant** | 2501.13484 (ICLR'25) | Mamba, W8A8 | parallel scan amplifies outliers; plain Hadamard insufficient → KLT-enhanced rotation. (Notes QuaRot drops 21% on Vim-T at W8A8 without it) | link `[UNVERIFIED]` |
| **Q-Mamba** | ACL'25 Findings | Mamba1+2, W8A8/W8A8H4 | outliers in BOTH state-dim AND channel-dim → **decoupled per-axis scales** | not linked |
| **Mamba-PTQ** | 2407.12397 | Mamba diagnostic | outliers are <1% of channels but critical (removing them costs 12.6–17.5% acc); the `dt` timescale proj has almost none | — |

**DeltaNet-specific evidence (characterization, not a method):** "Dissecting Outlier Dynamics in LLM NVFP4 Pretraining" (**2602.02047**) — linear attention (incl. DeltaNet, GLA) "reduces per-tensor heavy tails but still exhibits persistent block-level spikes under block quantization"; residual outliers attributed to **gating**; **post-QK ops most quantization-sensitive.** (FP4 not INT8 — use its characterization, not its method.) `[SINGLE-SOURCE]`

**Takeaway:** Our outlier hypothesis is correct and well-supported. Most transferable recipe = **Quamba2's per-state-group scaling + offline sort/cluster of the recurrence input**, plus **Q-Mamba's decoupled state-dim/channel-dim scales**, with Hadamard on the state output as the standard outlier move. **Good news: linear/delta attention is *inherently easier* to quantize than softmax attention — so the interleaved full-attention layers, not the DeltaNet layers, are where INT8 bites hardest.** This validates keeping `linear_attn` BF16 (we already do) and spending recovery effort on the 16 full-attention layers.

**⚠️ Operational gotcha (vLLM #40252):** Qwen3-Next-style Gated DeltaNet uses *combined* tensor names **`in_proj_qkvz` / `in_proj_ba`**. Community quants used stale names (`in_proj_qkv`/`in_proj_z`); vLLM silently skipped the mismatched tensors → zeroed layers → `!!!!!` garbage output. **A silent correctness failure, not a crash.** Verify our 27B ignore-list (commit a4190ba) matches the real combined names. `[SINGLE-SOURCE]`

---

## 5. Eval methodology — upgrades worth adopting

Our top-1-agreement work is already ahead of perplexity-only shops. Two papers extend it:
- **SLQ "Statistically-Lossless Quantization" (2605.02404, Helcig/Kurtic/Alistarh — direct "BF16 or Death" successor):** distinguishes *task-lossless* (zero-shot within noise) from *distribution-lossless* (next-token distribution indistinguishable). Introduces **Expected Acceptance Rate (EAR)** = max token-agreement probability under optimal coupling. Argues zero-shot benchmarks *understate* distributional degradation. TL needs 3.3–4.7 bits/param, DL needs 5.0–6.6 (a ~1.5–2 bit gap). Repo: github.com/IST-DASLab/SLQ. `[BODY-SOURCED]`
- **"A KL Lens on Quantization" (2604.13440) — explicitly about SSM-Transformer hybrids:** a forward-only (no backprop) **KL-divergence layer-sensitivity metric** that beats MSE/SQNR for deciding which layers to keep high-precision. The principled tool to rank *which DeltaNet vs full-attention layers* belong in our ignore-list. Repo: github.com/jasonkongie/kl-ssm-quant. `[SINGLE-SOURCE]`
- Practitioner consensus (llama.cpp/Unsloth): perplexity misleads (token errors cancel); KL divergence is the "gold standard," correlated with token flip-rate.

---

## 6. What NOT to spend effort on

- **Rotation (QuaRot 2404.00456 / SpinQuant 2405.16406 / DuQuant 2406.01721 / FlatQuant 2410.09426): skip at W8A8.** `[WELL-SOURCED]` The papers say so: QuaRot INT8 Llama-2-7B = 5.47→5.50 PPL (RTN and GPTQ *identical* — rotation adds nothing at 8-bit); DuQuant refuses to test W8A8 ("lossless per SmoothQuant"); SpinQuant calls its online Hadamard "marginal" at A8. AND **no Hadamard/FWHT kernel exists for Intel SYCL/XMX/oneDNN** — CUDA-only (Dao-AILab, HadaCore) + an AMD ROCm port in vLLM. The portable fallback is an O(d²) dense matmul (llm-compressor's `QuIPModifier`), taxing the INT8 win for a marginal 8-bit benefit. Only reconsider if we push to W4A8.
- **QAT / EfficientQAT (2407.11062): overkill at W8A8** — the gap is too small to justify training. Reserve for W4A8 and below. (EfficientQAT is weight-only W2/W3/W4 anyway.) `[WELL-SOURCED]`
- **FP8 as a speed play on B70:** emulated; memory play only (see §1).
- **Switching runtime to OpenVINO/NNCF or llm-compressor:** we already built the harder thing — a native oneDNN W8A8 XMX kernel that most toolchains *lack* (llm-compressor's W8A8 output is CUDA-only; vLLM-XPU/LLM-Scaler doesn't expose INT8 W8A8 at all; Intel's own container doc says "W8A8 quantized models through llm_compressor are not supported yet"). NNCF is valuable as a **source of techniques** (per-node SmoothQuant alpha, accuracy-aware layer reversion), not a runtime to migrate to. `[WELL-SOURCED]`

---

## 7. Concrete next experiments (this repo)

1. **Selective SmoothQuant** on the 16 full-attention layers + MLPs (skip `linear_attn`), per-node alpha tuned — measure agreement lift over the GPTQ-only 0.908 baseline. (`scripts/49`)
2. **`down_proj`-at-W8A16 sweep** (early+late layers) via an ignore-list knob — cheapest likely fidelity recovery.
3. **KL-sensitivity ranking** (2604.13440 method) to replace the hand-curated ignore-list with a measured one.
4. **Add EAR / KL-divergence to the eval harness** (we already do top-1 agreement; small extension that catches what agreement misses).
5. **Verify `in_proj_qkvz` / `in_proj_ba`** names in the 27B ignore-list against the actual checkpoint (silent-zeroing guard).

---

## 8. Citations (load-bearing, with confidence)

| Topic | Paper | arXiv | Conf. |
|---|---|---|---|
| Activation smoothing (the core lever) | SmoothQuant | 2211.10438 | WELL-SOURCED |
| Per-channel shifting (asymmetric outliers) | Outlier Suppression+ | 2304.09145 | SINGLE |
| W8A8 ~99% recovery, recipe | Give Me BF16 or Give Me Death | 2411.02355 | WELL-SOURCED |
| W8A8 lossless on Qwen3/reasoning | Quantization Hurts Reasoning? | 2504.04823 | WELL/BODY |
| Activation path is the failure | Systematic Characterization | 2508.16712 | SINGLE (92% fig UNVERIFIED) |
| 8-bit safe for long context | long-context quant study | 2505.20276 | WELL-SOURCED |
| Recurrent-state INT8 recipe | Quamba2 | 2503.22879 | SINGLE |
| Recurrent-state outliers | Quamba / MambaQuant / Q-Mamba | 2410.13229 / 2501.13484 / ACL'25 | mixed |
| DeltaNet/GLA outlier characterization | Dissecting Outlier Dynamics | 2602.02047 | SINGLE |
| DeltaNet tensor-name silent-zeroing | vLLM issue #40252 | — | SINGLE |
| down_proj / GLU spikes | SwiGLU spikes / Super Weight | 2405.14428 / 2411.07191 | WELL-SOURCED |
| Llama-3-70B W8A8 fragility | per-channel uniqueness | 2408.15301 | SINGLE |
| Eval: distribution-lossless / EAR | SLQ | 2605.02404 | BODY |
| Eval: KL sensitivity on hybrids | A KL Lens on Quantization | 2604.13440 | SINGLE |
| Rotation marginal at W8A8 | QuaRot / SpinQuant / DuQuant | 2404.00456 / 2405.16406 / 2406.01721 | WELL-SOURCED |
| Per-tensor-static via training reg. | TWEO | 2511.23225 | SINGLE |
| Token-wise outlier removal | PrefixQuant | 2410.05265 | SINGLE |

**Flagged unverified (re-check before citing):** the "92% HumanEval" figure (2508.16712, blog-sourced); MambaQuant's GitHub link; several search-surfaced 2026 arXiv IDs (some real like DuQuant++ 2604.17789, but all FP4/W4A4-focused and irrelevant to W8A8); per-task numbers for 2504.04823 / 2505.11574 are paper-body-sourced (abstracts confirm only the "W8A8 ≈ lossless" direction). The §1 hardware verdict and §2 root-cause are verified against primary sources / our own evals.
