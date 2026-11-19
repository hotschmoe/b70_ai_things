# Hy4-Flash quant review for the 2x B70 box (2026-11-19)

Question: is there a Hy4-Flash quant that fits under ~54 GB of VRAM (IQ2 / Q2 class)
without losing a lot of accuracy? Research only (HF API, model cards, GitHub). No GPU
used, no weights downloaded. Items marked SELF are self-reported by the quantizer or
vendor; items marked EST are my estimates.

## 1. What Hy4-Flash is

"Hy4-Flash" = Tencent Hy4 flash, HF repo tencent/Hy4-flash (released ~2026-11-17;
Tencent Hy License, Apache-2.0 based with MaaS clause). Not "Hunyuan-4-Flash" (no such
repo). Siblings: tencent/Hy4-flash-FP8, tencent/Hy4-flash-base, older tencent/Hy4-preview.
Source: https://huggingface.co/tencent/Hy4-flash (README), https://github.com/Tencent-Hunyuan/Hy4-flash

| Property | Value |
|---|---|
| Total params | 220B = 155.5B backbone + 63.5B Engram tables + 0.43B ViT + 0.85B DFly2 drafter |
| Active per token | 11.3B text (+0.43B vision) |
| Layers | 42 (1 dense + 41 MoE), hidden 3072, vocab 120,832 |
| Experts | 512 routed per layer, 16 active, 1 shared; latent MoE (latent 1536, expert inter 1536); ~148.6B routed params |
| Attention | MLA; hybrid sparse (IndexPool, 12 layers) + sliding window 512 (30 layers), 1:3 |
| Residual | iHC (identity Hyper-Connections, 4 residual streams) |
| Context | 1M tokens; Tencent says KV is 1/24 of Hy3 and 1/6 of Hy4 |
| Engram | bigram/trigram hash embeddings at layers 1,11,21,31; 63.4B params; FP8 file 63.4 GB; a few KB read per token -> can stay on NVMe (same idea as Flash-Next PLE, 28.8 GB) |
| MTP / spec decode | built-in DFly2 5-layer drafter (0.85B), reuses target KV + sparse indices. Public HF checkpoint has no classic MTP block; Tencent GGUFs carry one |
| Vision | yes (SigLIP2 based, mmproj files exist) |
| Thinking | on / no_think; recommended temp 0.9, top_p 1.0 |

Resident-set arithmetic: GPU-resident = everything except Engram = ~155.5B params
(+ ViT/drafter). Routed experts are 148.6B of that, so expert bit-width decides the size.

## 2. Quants on HF (GPU-resident part = trunk shard only)

All GGUFs need llama.cpp PR #32517 (see section 4); the 2nd shard is the Engram table
(63.4 GB FP8 for most, 35.6 GB Q4_0/IQ4_NL in some) and is NOT counted below.

| Repo / quant | Resident GB | Engram file | Fits <54 GB? | Notes |
|---|---:|---:|---|---|
| peasantsmith/Hy4-Flash-Maya-GGUF  Maya-XS (IQ2_XXS) | 44.8 | 63.4 | YES | + mmproj 0.8, drafter 0.7 |
| AtomicChat/Hy4-flash-GGUF  AD-IQ2_S-IQ2_XXS | 49.06 | 35.64 (Q4_0) | YES | experts 2.3 bpw, rest Q8_0; +mmproj 0.97 |
| peasantsmith  Maya-S (IQ2_XXS L1-23, IQ3_XXS L24-41) | 53.0 | 63.4 | marginal (53.0 + 0.8 vision + 0.9 drafter = 54.7) | best quality in the 50 GB class |
| Xylolabs/Hy4-flash-GGUF  IQ2_M | 56.26 | 63.4 | no (+2 GB over; see KV note) | KL 0.7155 (SELF) |
| Xylolabs  Q2_K | 61.79 | 63.4 | no | KL 0.677 |
| Xylolabs  IQ3_XXS | 64.89 | 63.4 | no | KL 0.509 |
| AngelSlim (Tencent) IQ2-AH | 71.64 | 63.4 | no | the official smallest |
| Maya-M (IQ3_XXS) | 68.4 | 63.4 | no | KLD 0.046 (SELF) |
| Xylolabs Q3_K_M / IQ3_M / IQ4_XS / Q4_K_M ... Q8_0 | 72.5 to 165.5 | 63.4 | no | |
| AngelSlim Q3_K / Q4_K_M / SPARK-4Bit | 92 / 105 / ~160 | 63.4 | no | |
| nace-ai Q3_K_M / Q4_K_M | 107 / 127 total (3 shards) | IQ4_NL | no | KL 0.233 / 0.133 (SELF) |
| Baekpica Mixed-Quant (MQ-Q5 ...) | ~105 | FP8 | no | DGX Spark oriented |
| anemll/Hy4-preview-FlashMoE-STQ1_0 | 22.7 dense + ~125 sidecar | - | no | Apple SSD streaming, preview model |
| EXL3 (LibertAIDAI 2.5 bpw etc., aitamedia 3.0 bpw) | trunk ~63 (2.5 bpw) | FP8 | no | CUDA only |
| Grawnch WV-v6.4 2.50 bpw SM120 | ~53.5 + 64.7 engram | FP8 | numerically yes | Blackwell SM120 custom wheel only; unusable here |
| tencent FP8, intel-ai MXFP8-AutoRound, NVFP4, AWQ-g32, MLX 4/8 bit | 100 to 160+ | - | no | vLLM/MLX/CUDA |

No ISTA-DASLab, unsloth, bartowski, mradermacher or lmstudio-community Hy4-flash repos exist
yet (HF author search returned empty on 2026-11-19). No true "Q2_0" (ISTA-style) exists for Hy4.

Fit under <54 GB resident: Maya-XS 44.8, Atomic AD-IQ2 49.06, Maya-S 53.0 (tight).
KV note (EST): sparse MLA layers are 12 x (512+64) latents ~ 14 KB/token fp16 -> ~2.8 GB
per 200K stream, + small SWA windows, so 4 x 200K is roughly 11-12 GB. KV is not the
binding constraint; the weight budget could stretch to ~56-58 GB if the engine really
stores latent KV only (unverified). That would admit Xylolabs IQ2_M (56.3) but IQ2_M is
the worst-measured quant in its class (KL 0.7155), so not attractive.

## 3. Quality

### 3a. Quant fidelity (all SELF, scorers differ, do not compare across rows from different sources)

Xylolabs (wikitext-2, 24x512 tokens, vs BF16): Q8_0 KL 0.174 / top-1 87.9%; Q4_K_M 0.244 / 85.1%;
IQ3_XXS 0.509 / 77.2%; Q2_K 0.677 / 73.8%; IQ2_M 0.7155 / 72.9%.
Important: even Q8_0 only agrees with BF16 on 88-92% of top tokens (AtomicChat: 91.8%).
Their reading: tiny rounding changes flip which 16 of 512 experts the router picks. This
model is unusually quantization-sensitive.

AtomicChat (30-language corpus, 104x4096): Q8_0 0.136 / 91.8%; Tencent Q4_K_M 0.198 / 89.3%;
Tencent IQ2-AH (71.6 GB) 0.651 / 77.9%; AD-IQ2_S-IQ2_XXS (49.1 GB) 0.725 / 76.4%.
Says: no task benchmarks run on their file.

Project Maya (80 held-out chat convs, 110k tokens, calibrated on similar chat data incl. the
model's own thinking, so favorable to them):

| Quant | Resident GB | KLD | Top-1 same |
|---|---:|---:|---:|
| Maya-XS | 44.8 | 0.156 | 88.5% |
| Maya-S | 53.0 | 0.072 | 91.8% |
| Maya-M | 68.4 | 0.046 | 93.6% |
| Maya-L | 89.5 | 0.012 | 96.6% |
| Tencent IQ2-AH | 71.6 | 0.152 | (same scorer) |
| Tencent Q3_K | 92.0 | 0.028 | |
| Tencent Q4_K_M | 105.0 | 0.013 | |

On that scorer Maya-XS (44.8 GB) ~ matches Tencent IQ2-AH (71.6 GB), and Maya-S (53 GB) is
~2x closer than IQ2-AH. Maya uses error-feedback (GPTQ-like) rounding + per-layer bit
allocation with standard ggml types. Not independently verified; no task benchmarks published.

### 3b. Task-level loss from quantization (only data point: Tencent AngelSlim, SELF)

Hy4-flash BF16 vs Tencent GGUFs (https://huggingface.co/AngelSlim/Hy4-flash-GGUF chart):

| Bench | BF16 | Q4_K_M | Q3_K | SPARK-4Bit | IQ2-AH (71.6 GB) |
|---|---:|---:|---:|---:|---:|
| DeepSWE v1.1 | 64.6 | 62.7 | 64.3 | 64.1 | 59.9 (-4.7) |
| APEX-Agents v1.1 | 63.1 | 62.6 | 63.2 | 61.8 | 56.7 (-6.4) |
| AutomationBench | 52.2 | 51.0 | 51.9 | 51.6 | 49.0 (-3.2) |
| PPTGenBench | 54.2 | 54.1 | 53.8 | 54.1 | 52.0 (-2.2) |

So the 2-bit class costs 2-6.5 points on agentic/coding tasks at 71.6 GB. Our <54 GB options
are at or below that bit-width: Maya-XS ~ IQ2-AH-equivalent (EST -2..-6 pts), Atomic AD-IQ2
slightly worse than IQ2-AH on its own scorer (EST -4..-8 pts), Maya-S the best (EST -1.5..-4 pts,
unmeasured). No agentic task benchmark exists for any <54 GB Hy4 quant.

### 3c. Base-model quality vs Qwen3.8-Flash-Next (vendor table, SELF, Tencent harnesses, highest reasoning setting)

Source: tencent/Hy4-flash assets/benchmark.png (Flash-Next column is Tencent's own run, BF16/API).

| Benchmark | Hy4 flash | Qwen3.8-Flash-Next | Delta |
|---|---:|---:|---|
| DeepSWE v1.1 | 64.6 | 65.1 | ~tie |
| ProgramBench (almost solved) | 16.0 | 22.0 | Flash-Next +6 |
| Terminal-Bench 4.0 (Claude Code harness) | 19.2 | 29.5 | Flash-Next +10.3 |
| APEX-Agents v1.1 | 63.1 | 57.6 | Hy4 +5.5 |
| AutomationBench | 52.2 | 49.3 | Hy4 +2.9 |
| HLE-Full (tools) | 55.2 | 54.1 | Hy4 +1.1 |
| OSWorld 2.1 | 68.1 | 58.8 | Hy4 +9.3 |
| MMMU-Pro | 75.8 | 77.2 | Flash-Next +1.4 |
| Mean of the 7 "Open Agent" rows (my arithmetic) | 48.3 | 48.1 | tie |

Reading: a tie overall, but Flash-Next is clearly better on terminal/CLI and ProgramBench
(the closest proxies for agentic coding); Hy4 flash is better at computer-use / office-agent
tasks. Tencent itself says Hy4 flash is for everyday productivity and recommends full Hy4
(not Flash) for long-horizon software engineering. Public SWE-bench / LiveCodeBench numbers
for Hy4 flash: none found. Independent: Artificial Analysis has no Hy4 flash entry (full Hy4 = 35);
no llm-stats/vals.ai independent coding result found. Hy4 flash blind user eval 2.79/4 vs
Claude Haiku 5.5 2.81/4 (SELF, 259 WorkBuddy tasks).

### 3d. Qwen3.8-Flash-Next quantized reference (ISTA GSQ-RCO README, SELF, BF16 task avg 93.12 / LCB v6 87.43)

| Quant | Resident GB | PLE | Task avg | AIME25 | GPQA-D | LCB v6 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 354 total | | 93.12 | 100.00 | 91.92 | 87.43 |
| ISTA Q2_0 | 37.6 | 28.8 | 89.07 | 96.67 | 89.39 | 81.14 |
| ISTA IQ2_XS | 39.2 | 28.8 | 89.16 | 96.67 | 87.37 | 83.43 |
| ISTA IQ3_XXS (our DD) | 47.0 | 28.8 | 92.57 | 100.00 | 91.41 | 86.29 |
| ISTA IQ3_S | 54.8 | 28.8 | 93.26 | 100.00 | 92.93 | 86.86 |
| unsloth UD-Q2_0 | ~41.7+ | 28.8 | no numbers | | | |
| unsloth UD-IQ2_XS | ~47 | 28.8 | no numbers (image-only card) | | | |
| unsloth UD-Q2_K_XL | ~50.7 | 28.8 | no numbers | | | |

Flash-Next at 47.0 GB loses only ~0.55 task-avg points (LCB -1.14) vs BF16; Hy4 at ~50 GB
loses an EST 2-8 points on agentic tasks. Flash-Next is far more quantization-tolerant.

## 4. Engine support

- Strata (upstream v0.2.16, released 2026-11-18): NO Hy4-flash support. docs/MODELS.md covers
  only Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. Only Hy4 trace: open issue #6212
  "[Request] AngelSlim/Hy4-flash-GGUF" (template-incomplete, bot comment only, no maintainer
  response). The AngelSlim (Tencent) GGUF README says "Strata version coming soon", so a
  Strata target may appear but is unannounced. Strata engines are per-architecture
  (a separate dsv41 DeepSeek engine exists, with an Intel SYCL port by a community member:
  issue #5908, B70 single card, IQ1_M, experts on CPU: ~13 tok/s decode, 600 tok/s prefill at 8K).
  Hy4 needs new graph code: iHC, hybrid sparse/SWA MLA with indexer, latent MoE, Engram hash
  lookups, DFly2. Open Strata issue #5653: SYCL instruction-following gap vs CUDA on
  identical weights (2% vs 77% on a paired probe), a risk for any new SYCL arch.
- llama.cpp: arch hy_v4_flash is only in open PR ggml-org/llama.cpp#32517 (not merged,
  updated 2026-11-19, bot review lists blocking items). Tested on CUDA, Metal, CPU. PR adds
  no new ggml ops, so SYCL/Vulkan should in principle run but nobody has reported it; no
  B70/Arc report found. Our own matched test (2026-10-15) had llama.cpp SYCL at 0.39x C1 /
  0.45x C2 of Strata, so a llama.cpp route would give poor speed regardless.
- vLLM / SGLang: model cards list support (CUDA); formats are >=100 GB. No XPU report.
- Intel Arc / B70 specific Hy4-flash reports: none found.

## 5. Speed estimate on our box (EST, not measured)

Bytes per decoded token (11.3B active):
- routed experts: 16/512 x 148.6B = 4.64B params x ~2.2-2.5 bpw ~ 1.3-1.45 GB (~1.2-1.35 GiB)
  vs Flash-Next 0.89 GiB.
- non-routed (attention, shared expert, dense layer 0, latent projections, indexer, LM head)
  ~6.3-6.9B params at Q8_0 ~ 6.7-7.3 GB read every token. Flash-Next's equivalent dense part
  is much smaller (Qwen3.8-Flash-Next is a ~3B-active class model; exact figure not checked).
- total ~8 GB/token vs a Flash-Next total of very roughly 1.5-3 GB (EST).

Decode: B70 ~608 GB/s, layer split across two cards means the cards run sequentially for
one stream. Floor ~13 ms/token -> ceiling ~75 tok/s at 100% bandwidth; at the 50-60% we get
today -> ~35-45 tok/s C1 (vs 115-141 on Flash-Next), i.e. ~0.3x. Extra overhead from iHC
4-stream residuals, indexer, Engram lookups likely makes it lower. DFly2/MTP could give
1.5-2x if Strata ever implements it (not available now). C4 aggregate: experts mostly
disjoint, dense part shared across the batch, so aggregate scales better than C1,
EST 80-120 tok/s vs 188 now.
Prefill: compute scales with active params (11.3B vs ~3B class) -> EST 400-700 tok/s vs ~2000 now.
Targets (2000+ prefill, 150+ decode) would be missed by 3-5x and 3-4x.

## 6. Verdict

- Best <54 GB Hy4-flash options: Maya-S 53.0 GB (best fidelity, no task benchmarks, tight),
  Maya-XS 44.8 GB, AtomicChat AD-IQ2 49.06 GB (weakest KLD, but ships a smaller Q4_0 Engram).
  All are 2-bit-class, i.e. the regime where Tencent's own IQ2-AH lost 2-6.5 points on agentic
  tasks, on a model whose router is quantization-sensitive (Q8_0 only 88-92% top-1).
- No engine path today: Strata unsupported (request open, "coming soon" hint from Tencent),
  llama.cpp support unmerged and CUDA/Metal only, no Arc report.
- Even in the best case the base model is a tie with Flash-Next on vendor aggregate and worse
  on terminal/ProgramBench, with ~3x the per-token bytes -> slower decode and prefill.
- Recommendation: stay on Flash-Next IQ3_XXS (47.0 GB). Do not spend GPU time on Hy4-flash now.
  Revisit only if (a) Strata ships a Hy4 target (watch issue #6212 and Strata release notes),
  and (b) someone publishes SWE/terminal numbers for a <54 GB quant (Maya-S is the one to test).
  If a 2-bit alternative is wanted for headroom, ISTA Flash-Next Q2_0 / IQ2_XS (37.6 / 39.2 GB,
  89.1 task avg) is a far better-characterized 2-bit option than any Hy4 quant.

## Links

- https://huggingface.co/tencent/Hy4-flash
- https://huggingface.co/AngelSlim/Hy4-flash-GGUF
- https://huggingface.co/AtomicChat/Hy4-flash-GGUF
- https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF
- https://huggingface.co/Xylolabs/Hy4-flash-GGUF
- https://huggingface.co/nace-ai/Hy4-flash-GGUF
- https://github.com/ggml-org/llama.cpp/pull/32517
- https://github.com/Niko1221/Strata/issues/6212
- https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
