# Inkling 2 Mini vs our Qwen3.8-Flash-Next stack on 2x B70 (2026-11-20)

Author: Claude agent, model ID claude-opus-5-5 (Opus 5.5). Research only: HF API, model card,
config/tokenizer/template files, safetensors HEADERS (range requests of the JSON header only, about 150 KB;
no weight data downloaded), GitHub PRs, web search. No GPU touched.

Labels: VENDOR = Thinking Machines (TML) or Qwen self-reported; INDEP = third party (Artificial Analysis
page); MEAS = measured on this box by us (cited file); EST = my arithmetic, not measured; HDR = read from
the safetensors header or config.json of the HF repo at sha 6e7c8248.

## 0. Summary

- Inkling 2 Mini is a 26.4B-total looped MoE, released today (2026-11-20, Apache-2.0). It has 24 physical
  layers. Layers 6-17 run 3 times, so a token makes 48 layer visits. Active parameters are 3.4B on paper and
  about 5.4B "effective" per token. 128 experts, top-8. Hybrid attention: 1024-token SWA plus 4 global
  layers. Has vision and a bundled 3-layer DSpark block drafter. The shipped checkpoint is already ~4-bit:
  19.5 GB, with NVFP4 experts in L0-19, MXFP8 experts in L20-23, and BF16 for everything else.
- Quality: below Flash-Next and about equal to base Qwen3.8-27B. Independent AA Intelligence Index: Inkling
  2 Mini 20, Flash-Next 29. On the vendor cards, with Qwen3.8-27B as the common anchor, Flash-Next leads
  Inkling on DeepSWE 1.1 (58.7 vs 48.2), relative SWE-bench Multilingual, and JobBench (55.7 vs 49.6). In
  TML's own pi run (Terminal-Bench 2.1), Inkling scores 71.6 and base Qwen3.8-27B scores 73.0. The card also
  warns about reasoning loops while debugging, destructive file actions, and inaccurate self-reports in
  agent use. Reasoning cannot be fully turned off.
- Fit: one full replica fits on ONE B70. About 18.2 GB with an IQ4_NL/NVFP4-class quant, the drafter and
  vision, leaving about 10.3 GB of KV per card. That holds 2 x 200K int8 slots or 7 x 64K int8 slots per
  card, so DP=2 gives 4 x 200K across the box.
- Speed (EST, DP=2, one replica per card, mature engine assumed): C1 at short context is 70-88 tok/s
  without drafting and roughly 90-120 tok/s with DSpark. Aggregate at short context, no drafting: C4
  ~215-255, C8 ~300-335, C16 ~395-430. C1 TP=2 + DSpark has a theoretical 140-190, but needs the TP2
  collective path. Decode falls at long context: 8 full-attention visits read 3.3 GB of int8 KV per token
  at 200K, giving C1 46-53 tok/s with no drafting. Reference (MEAS): Strata Flash-Next 2 cards C1 133.5,
  C4 ~148 aggregate. TensorFold 27B + DFlash2 on one card C1 142, C2 228, C4 304, C8 265.
- Engines: none runs it on Intel today. llama.cpp is closest. Two PRs are open (#32694 non-draft, #32695
  draft), and both use only standard ggml ops, so SYCL should load it once one merges. GGUFs are ready
  (ggml-org IQ4_NL 17.63 GB, unsloth UD-Q4_K_XL 17.63 GB, dspark Q8_0 0.40 GB). vLLM PR #69028 and SGLang
  PR #51926 are NVIDIA-only (custom CUDA kernels). Strata has no Inkling work. TensorFold upstream has no
  Inkling family; the only proposals are CUDA, for Inkling 2 Small.
- Recommendation: do not switch the daily driver, and do not start a TensorFold Inkling family now. Flash-Next
  stays: better quality (AA 29 vs 20), flat long-context decode, and its TensorFold port is already in
  progress. Optional cheap probe: when llama.cpp merges Inkling 2 Mini, run evals/abfast (pi) on ONE card
  with the ggml-org IQ4_NL GGUF, during a window where the DD is down anyway. Spend engine effort only if it
  matches Flash-Next IQ3_XXS on abfast. Its real attraction is a small, one-card, high-concurrency
  secondary model, not a quality upgrade.

## 1. Model facts (HDR = config.json / safetensors header; VENDOR = card/blog)

| Property | Value |
|---|---|
| Repo / date / license | thinkingmachines/Inkling-2-Mini, created 2026-11-17, released 2026-11-20 (card), Apache-2.0, not gated |
| Architecture class | `Inkling2LoopForConditionalGeneration`, model_type inkling2_loop; "multimodal autoregressive looped transformer", sparse MoE (VENDOR, HDR) |
| Params | 26.4B total (HF safetensors count 26.39B), 3.4B active; +1.0B effective per extra loop pass -> 5.4B effective active (VENDOR) |
| Layers | 24 physical; loop core 6..17 (core_start 6, core_end 18, num_passes 3, gate_rank 64, is_varloop false) -> 6 + 3x12 + 6 = 48 visits/token (HDR) |
| Hidden / vocab | hidden 2304; vocab 201,024 padded (200,058 unpadded); untied embed + unembed, BF16, 926 MB each (HDR) |
| Attention | 24 Q heads, 8 KV heads, head_dim 128 (GQA 3:1). SWA 1024 on 20 layers (0-4, 6-10, 12-16, 18-22); global on layers 5, 11, 17, 23 (one per 6-layer block). Learned relative-position logits (d_rel 16, extent 1024) instead of RoPE in the trunk; learned attention sink; sigmoid-style output gate (gate_attn 3072x2304); short convs (kernel 4) on K and V (dynamic, input-projected, on global layers), plus attn/mlp sconv on the residual input; QK L2 clip; log-n scaling (floor 8192, alpha 0.1); value norm without gamma (HDR, HF blog) |
| Residual | "attention residuals": each layer reads a learned mix of earlier block outputs (n_layers_per_attn_res_block 6) instead of a plain running sum (HDR, blog) |
| Loop messages | passes 2-3: every core layer reads gated cross-pass messages from the previous pass (4 controllers q/k/v/mlp per core layer, rank 64 gain/forget gates, FP32 merge in the reference) (HDR, vLLM PR) |
| MoE | 128 routed experts, top-8, no shared expert, expert inter 1152 (SwiGLU), gate bias, route_scale 8, softcaps gate 4 / up 25; pruned from 384 experts in midtraining (blog) |
| Output | final logit softcap 30 |
| Context | 1,048,576 (model_max_length); vLLM single-GPU recipe uses 262,144; vLLM PR reports NIAH 550/550 4K-1M at TP4 (VENDOR) |
| KV/state | per-visit caches: "each visit keeps its own KV and short-convolution state" (HF blog, ggml-org card, vLLM PR). An opt-in shared-cache mode exists only for the variable-loop research checkpoint (TML-d24-varloop-research) |
| Drafter | mtp.safetensors 0.636 GB BF16: "DSpark" block drafter, 3 dense layers (hidden 2304, MLP 9216; layers 0,2 SWA 1024, layer 1 global), block_size 8 (7 drafted tokens), reads target hidden states at visits 5/23/41/48, plus a Markov RNN head (rank 256) for in-block dependence. vLLM default: 7 tokens, probabilistic; mean acceptance length 3.57 on GSM8K (VENDOR, vLLM PR) |
| Vision | yes: ViT 23 layers, hidden 1024, patch 20, ~0.3B (mmproj 0.64 GB BF16 / 0.35 GB Q8_0); images 40-4096 px; no audio |
| Precision | experts trained NVFP4 (QAT, W4A4: NVFP4 per-token activations, group 16, e4m3 scales, "fouroversix_mae" scale rule) in L0-19 and MXFP8 (W8A8, block 32, ue8m0) in L20-23; rest BF16 (HDR hf_quant_config.json) |
| Tokenizer | tiktoken-style BPE (tiktoken/tokenizer.model + tokenizer.json), o200k-sized vocab, role tokens `<\|message_user\|>` etc., EOS `<\|content_model_end_sampling\|>` (id 200006) |
| Sampling | generation_config temperature 1.0; unsloth suggests temp 1.0, top_p 1.0, top_k 0 |

### 1a. Exact byte layout of the shipped checkpoint (HDR, decimal MB)

| Part | Bytes | Note |
|---|---:|---|
| Routed experts L0-19 (NVFP4 packed + e4m3 group-16 scales) | 11,466 MB | 4.479 MB per expert (w13 2304x2304 + w2 2304x1152) |
| Routed experts L20-23 (MXFP8 + ue8m0 block-32 scales) | 4,204 MB | 8.211 MB per expert |
| Attention + router + misc, 24 layers, BF16 | ~1,306 MB | 54.4 MB per layer (q 3072x2304, gate 3072x2304, o 2304x3072, k/v 1024x2304, rel 384x2304, sink, router 128x2304) |
| Loop controllers (12 core layers), BF16 | 57 MB | 4.72 MB per layer |
| embed + unembed, BF16 | 1,853 MB | 926 MB each |
| Vision tower + projector, BF16 | ~630 MB | |
| Total model file | 19,526 MB | + mtp.safetensors 636 MB |

### 1b. Prompt format, thinking, tool calls (from chat_template.jinja)

- Roles are special tokens: `<|message_system|>`, `<|message_user|>`, `<|message_model|>`, `<|message_tool|>NAME`;
  content types `<|content_text|>`, `<|content_thinking|>`, `<|content_image|>`; each part ends with `<|end_message|>`;
  an assistant turn ends with `<|content_model_end_sampling|>` (EOS).
- Thinking: `reasoning_effort` none/minimal/low/medium/high/xhigh/max = 0.0/0.1/0.2/0.7/0.9/0.99/0.99, or any number in
  [0, 0.99]. The default is 0.9 (high). It is injected as a system message "Thinking effort level: X". The card says
  the minimum does NOT fully disable reasoning. Reasoning text is replayed into history as `<|content_thinking|>` when the
  client sends reasoning_content.
- Tools: declared once as `<|message_system|>tool_declare<|content_xml|>[...json specs, sorted keys...]<|end_message|>`.
  A call is `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`. Arguments must be a
  parsed object; the template raises on a JSON string ("canonicalize upstream"), so the server must parse OpenAI-style
  string arguments before rendering. Tool results are `<|message_tool|>NAME<|content_text|>...`. Parsers: vLLM
  `--tool-call-parser inkling` and `--reasoning-parser inkling`, SGLang `inkling2`, llama.cpp chat parser merged
  (#32381, plus jinja sort_keys #32366). For pi this is a new format: our engines (Strata, TensorFold) would need a
  parser, and the pi-adapted LoRA training does not transfer.

## 2. Quality vs Qwen3.8-Flash-Next and Qwen3.8-27B-pi-adapted

### 2a. Independent

| Source | Inkling 2 Mini | Qwen3.8-Flash-Next | Qwen3.8-27B |
|---|---:|---:|---:|
| Artificial Analysis Intelligence Index (INDEP, AA model pages, fetched 2026-11-20) | 20 (#2 of 16 in its class) | 29 (#10 of 33) | 20.1 (as quoted on the TML card) |
| AA output tokens used to run the index (verbosity) | 161M | 156M ("very verbose") | - |
| AA Coding/Agentic sub-indices, Terminal-Bench Hard, LCB, tau2, IFBench | not shown on the AA page for either model | | |

No independent SWE-bench, Terminal-Bench, LiveCodeBench or tool-call reliability results exist yet for Inkling 2 Mini. No
hands-on reports exist either: the model is a few hours old, and the HN thread is mostly about Inkling 2 Small. No local
tok/s figures have been published on any hardware.

### 2b. Vendor cards, using Qwen3.8-27B (in both tables) as the anchor (VENDOR, different harnesses)

| Benchmark | Inkling 2 Mini (TML) | Qwen3.8-27B (TML run) | Qwen3.8-Flash-Next (Qwen) | Qwen3.8-27B (Qwen run) | Reading |
|---|---:|---:|---:|---:|---|
| DeepSWE 1.1 | 48.2 (mini-swe-agent) | 42.2 | 58.7 (best of CC / mini-swe-agent) | 42.2 | Flash-Next +10.5 at the same anchor |
| SWE-bench Multilingual | 73.7 (bash only) | 70.0 | 81.0 (mini-SWE-agent) | 73.8 | delta vs own 27B: Inkling +3.7, Flash-Next +7.2 |
| SWE-bench Pro | 77.9 ("v2 Hard", mini-swe-agent) | 73.8 | 62.5 (refined set, Claude Code) | 61.7 | different subsets; delta vs 27B: Inkling +4.1, Flash-Next +0.8 |
| JobBench | 49.6 | 33.4 | 55.7 | 33.4 | Flash-Next +6.1 at the same anchor |
| SWE-bench Verified (bash only) | 79.2 | 77.8 | - | - | |
| Terminal-Bench 2.1 in pi v0.83.0 | 71.6 | 73.0 | - (one aggregator claims 86.1, unverified) | - | Inkling slightly BELOW base 27B in pi |
| Terminal-Bench 4.0 (mini-swe-agent) | 5.6 | 5.6 | 29.5 (Claude Code harness, Tencent's run, see docs/20261119_hy4_flash_quant_review.md) | - | harness differs; not comparable |
| MCP Atlas (tool use) | 82.7 | 77.3 | Toolathlon Verified 73.5 (27B 67.1) | | both beat their 27B on tool use |
| LongBench v2 / MRCR v2 8-needle 256-512K / GraphWalks BFS 256K | 66.9 / 46.4 / 69.7 | 52.3 / 38.0 / 50.4 | - | - | Inkling's long-context claims are strong (VENDOR) |
| LiveCodeBench v6 | - | - | 91.9 | 90.3 | no Inkling number |
| IFBench / ComplexConstraints | ComplexConstraints 33.4 (27B 30.1), MultiChallenge 50.0 | | IFBench 81.3 (27B 79.5) | | not comparable |
| AutomationBench-AA / EnterpriseOps-Gym / GDPVal-AA | 41.8 / 42.0 / 1319 | 48.2 / 44.2 / 1427 | - | - | Inkling below 27B on office agents |
| Destructive actions / misbehavior (internal, lower is better) | 10.2% / 9.8% | 35.2% / 57.7% | - | - | TML's own internal eval |

Reading: on TML's own table, Inkling 2 Mini is about Qwen3.8-27B level overall (same AA index; ahead on SWE-style coding
and long context, behind on Terminal-Bench-in-pi and office automation). On every row that shares the 27B anchor,
Qwen3.8-Flash-Next is clearly ahead of Inkling, and the independent AA index agrees (29 vs 20). Our evals weight
agentic coding in Rust/Dart/Go/Python, sysadmin, and multi-day loops under pi. For that use, the TML card's own
limitations matter: "stuck in long reasoning loops ... particularly when debugging code", "may take destructive
actions ... without confirmation, and may not report the outcome of its work accurately", weaker non-English, and
reasoning that cannot be fully disabled.

### 2c. Vs Qwen3.8-27B-pi-adapted (harness-hub LoRA; SWE-bench Pro hard-51 64.7%, Terminal-Bench 2.1 55.8% under pi, thinking off)

No matched numbers exist. TML's 73.0 for base Qwen3.8-27B in pi (TB2.1) was evidently not run under our thinking-off
protocol, since our pi-adapted number at thinking off is 55.8. Inkling has no thinking-off mode, so it cannot run that
protocol at all. Given Inkling < base 27B on TML's own pi run, I expect pi-adapted 27B >= Inkling in pi (EST, low
confidence). This needs a paired abfast run to settle.

## 3. Available ~4-bit checkpoints (HF API, 2026-11-20)

| Repo | File | Size | Fidelity (SELF) | Notes |
|---|---|---:|---|---|
| thinkingmachines/Inkling-2-Mini (original) | NVFP4 experts L0-19, MXFP8 L20-23, BF16 rest | 19.53 GB + 0.64 GB drafter | lossless (training precision) | the native "4-bit" form; an XPU engine can read NVFP4 directly (e2m1 + e4m3 group-16 scales) |
| ggml-org/Inkling-2-Mini-GGUF | IQ4_NL (imatrix; IQ4_NL experts L0-19, everything else Q8_0) | 17.63 GB | KLD 0.0095, 95.5% top-1 vs BF16 (wikitext 40x512) | Q8_0 27.82 GB (KLD 0.0030); BF16 52.2 GB; mmproj Q8_0 0.35 GB |
| unsloth/Inkling-2-Mini-GGUF | UD-Q4_K_XL / UD-Q4_K_M / UD-IQ4_XS | 17.63 / 16.61 / 15.73 GB | KLD vs UD-Q8_K_XL 0.0107 / 0.0108 / 0.0109; top-1 98.0 / 97.9 / 97.7% | also UD-IQ3_XXS 11.66 GB (KLD 0.0269), down to IQ1_S 7.76; dspark drafter Q8_0 0.40 GB and BF16 0.65 GB GGUFs |
| AtomicChat/Inkling-2-Mini-GGUF | (uploading) | - | - | built at llama.cpp PR #32695 commit 6196d98 |
| nativ-community/Inkling-2-Mini-nvfp4 | MLX, native NVFP4/MXFP8 | 19.53 GB | bit-exact repack | patched mlx-vlm; Apple only |

No GPTQ/AWQ/AutoRound/compressed-tensors int4 yet. Because the experts are already QAT NVFP4, requantization buys
little; the only real size lever is the 1.3 GB of BF16 attention and the 1.85 GB of BF16 embed/unembed.

## 4. Engine support status (2026-11-20)

| Engine | Status | What it needs here |
|---|---|---|
| llama.cpp (SYCL) | Inkling 2 (Small) is merged (#32217 + fixes #32429, #32430, #32657); chat/tool parser merged (#32381). Inkling 2 Mini loop: PR #32694 (pcuenca, open, non-draft, "reviewed offline by ngxson and CISC") and PR #32695 (danielhanchen, draft). DSpark for Inkling 2: PR #32218 open. | Wait for a merge, then build SYCL. Graph ops are standard (mul_mat/MoE, soft_max_ext, ssm_conv, get_rows, build_attn with a per-head bias tensor kq_b and sinks); no new ggml op. Risks: (1) kq_b materializes [24 heads x n_kv x n_tokens] fp32 per global visit, which is fine for decode (~19 MB at 200K) but forces small ubatches for long prompts (128K x 512 tokens = 6.4 GB); (2) SYCL flash-attn support for per-head bias masks is unverified; (3) our matched test (2026-10-15) had llama.cpp SYCL at 0.39x C1 / 0.45x C2 of Strata, and it needed GGML_SYCL_ENABLE_OPT=0 for correctness (PR #30604). Good for a quality probe, not for serving. |
| vLLM | PR #69028 (WoosukKwon, open, +3003 lines) adds Inkling 2 Mini under vllm/models/inkling2/nvidia/ with custom CUDA kernels (csrc/inkling2/*.cu, FA4, trtllm_bmm, sm90/sm120 MoE). Needs vLLM >= 0.35 nightly. Validated on GB200/B300/H100/H200/DGX Spark. | No XPU path. A port means rewriting the attention (rel-bias + sink + gate), sconv, attn-res, loop merge and NVFP4 MoE for XPU (Triton/SYCL), plus a vLLM rebase from our pinned stack. Large. |
| SGLang | PR #51926 open; images for Hopper/Blackwell, MI35x, TPU7x; `--tool-call-parser inkling2`, DSpark | No XPU. Our sglang XPU stack is far behind. |
| Strata (Niko1221) | No Inkling issue or code (search 2026-11-20). Per-architecture engines; Flash-Next and Qwen3.6-35B-A3B only (v0.2.16). | A new engine from scratch. |
| TensorFold upstream (ashhart) | No Inkling family. Open proposals for Inkling 2 Small are CUDA only (#3476 four GB10s, #3593 EXL3). | - |
| TensorFold XPU fork fn-int (/mnt/vm_8tb/b70/build/tensorfold/fn-int) | No Inkling code (grep). It has what an Inkling family would reuse: GGUF loader, XPU DPAS/q8_1 GEMV kernels incl. IQ4_NL, a one-card dense 27B path with a DFlash2 block drafter (close cousin of DSpark), DP serving, paged int8 KV, park/restore to host RAM without recompute, a push all-reduce and recorded windows for TP2. | New family "inkling2-loop" (EST 2-4 engineer-weeks with agent workers, see section 6). |
| transformers | AutoModelForMultimodalLM / AutoProcessor (HF blog); usable as a CPU/XPU-eager numerical oracle | Reference activations would differ slightly from TML's runtime, which quantizes activations to NVFP4. |

Soonest path to "it runs on a B70": llama.cpp SYCL, once #32694 or #32695 merges (days, if past Inkling 2 merges are
a guide), using the ggml-org IQ4_NL GGUF on one card. Soonest path to "fast": a TensorFold family (weeks).

## 5. Speed arithmetic on our box (all EST unless marked MEAS)

Assumptions (from the brief): per card 608 GB/s x 0.70 = 425.6 GB/s effective. Per decode step fixed cost 2-5 ms
(+1 ms for a drafter round). TP hop 12-17 us one way; I charge 25 us per exchange, 2 exchanges per layer visit,
96 per step. Card->host 4.5 GB/s, host->card 13 GB/s. Usable VRAM ~30 GB per card. Decode is bandwidth-bound;
compute is checked at the end.

### 5a. Bytes per decode step (one card holds the whole model)

Per token (one row):
- Routed experts: 44 NVFP4-layer visits x 8 x 4.479 MB = 1,576.6 MB, plus 4 MXFP8-layer visits x 8 x 8.211 MB =
  262.8 MB, total 1,839 MB. Looping multiplies this: L6-17 weights are re-read in each pass, and no cache can
  hold them (one visit is ~65 MB of weights; per-pass routing differs anyway).
- Attention+router: 48 visits x 54.4 MB BF16 = 2,611 MB as shipped; at Q8_0 48 x 28.9 = 1,387 MB; at ~4.5 bpw
  48 x 15.3 = 734 MB.
- Loop controllers (passes 2-3): 24 x 4.72 MB = 113 MB BF16. Unembed 926 MB BF16 / 492 MB Q8_0 / 380 MB Q6_K.
- KV read per stream: 8 global visits x L x 2048 elements x (2 B bf16 | 1 B int8), plus 40 SWA visits x 1024 x 2048
  x bytes. Int8: 4K ctx 151 MB; 64K 1,158 MB; 200K 3,361 MB.

| Config (C1, 4K ctx) | Bytes/step | Step (fixed 2-5 ms) | tok/s |
|---|---:|---:|---:|
| As shipped (BF16 attention + unembed, bf16 KV) | 5.79 GB | 15.6-18.6 ms | 54-64 |
| "Q": experts native NVFP4/IQ4_NL + 8-bit, attention/unembed Q8_0, int8 KV (~ggml-org IQ4_NL) | 3.98 GB | 11.4-14.4 ms | 70-88 |
| "Q4": as Q but attention ~4.5 bpw, unembed Q6_K | 3.22 GB | 9.6-12.6 ms | 80-105 |

Rows vs bytes (MoE union): distinct experts per visit for r independent rows = 128 x (1 - (1 - 8/128)^r) = 8.0 /
15.5 / 29.1 / 51.6 / 82.4 for r = 1/2/4/8/16. Draft rows of ONE stream are correlated. I use r_eff = 1 + 0.6(r-1),
which reproduces Steve's measured Flash-Next union exactly (2/3/4 rows -> 16/22/27 distinct of 512;
docs/20261119_b70_reference_refresh.md S4). A DSpark verify of 8 rows -> r_eff 5.2 -> 36.5 distinct experts per
visit (4.6x the single-row expert bytes). Each distinct expert costs 229.9 MB summed over the 48 visits. Saturation
(all 128 experts at every visit) is 29.4 GB/step, the hard ceiling for any batch.

### 5b. Memory, KV and slots (Q config)

Weights per card (DP=2, full replica): experts 15.67 + attention Q8_0 0.69 + unembed 0.49 + embed 0.49 (it could live
in host RAM) + controllers/router 0.09 = 17.43 GB; + DSpark Q8_0 0.40 + mmproj Q8_0 0.35 = 18.18 GB. Leaving 1.5 GB of
workspace gives a KV budget of ~10.3 GB per card. As shipped (19.53 + 0.64 GB): ~8.3 GB.

KV per token (per-visit caches): 8 global visits x 2 x 8 x 128 = 16,384 elements, so 32 KiB bf16 / 16 KiB int8, plus
the drafter's global ctx-KV 4 KiB bf16. Fixed per slot: SWA 40 x 1024 x 4 KiB = 168 MB bf16 (84 MB int8), conv state ~2 MB.
For comparison, the 27B is 64 KiB/token bf16 (tensorfold-probe README).

| Slot | bf16 | int8 target + bf16 drafter | all int8 | Slots/card @10.3 GB (int8) | Box DP=2 | Box TP=2 (~19.3 GB/card free) |
|---|---:|---:|---:|---:|---:|---:|
| 64K | 2.59 GB | 1.44 GB | 1.30 GB | 7 | 14 | ~29 |
| 128K | 5.01 GB | 2.78 GB | 2.51 GB | 4 | 8 | ~15 |
| 200K | 7.55 GB | 4.19 GB | 3.78 GB | 2 (bf16: 1) | 4 | ~10 |

So the goal "200K per stream at C2" fits on ONE card with int8 KV, and 4 x 200K fits the box in DP=2. Eviction: a 200K
int8 slot (~4.2 GB) parks in ~0.93 s at 4.5 GB/s and restores in ~0.32 s at 13 GB/s; 121 GB of RAM holds ~25 parked
200K slots. TP2 keeps more slots resident, but costs collectives.

### 5c. Decode estimates per card (DP=2 box = 2x the per-card aggregate)

| Per card | Bytes/step | Step | Per stream | Card aggregate |
|---|---:|---:|---:|---:|
| S=1, 4K, no draft | 3.98 GB | 11.4-14.4 ms | 70-88 | 70-88 |
| S=2, 4K | 5.86 GB | 15.8-18.8 ms | 53-63 | 107-127 |
| S=4, 4K | 9.29 GB | 23.8-26.8 ms | 37-42 | 149-168 |
| S=8, 4K | 15.07 GB | 37.4-40.4 ms | 25-27 | 198-214 |
| S=1, 64K | 4.99 GB | 13.7-16.7 ms | 60-73 | 60-73 |
| S=2, 64K | 7.87 GB | 20.5-23.5 ms | 43-49 | 85-98 |
| S=4, 64K | 13.32 GB | 33.3-36.3 ms | 28-30 | 110-120 |
| S=8, 64K (KV 10.4 GB, borderline fit) | 23.12 GB | 56.3-59.3 ms | 17-18 | 135-142 |
| S=1, 200K | 7.19 GB | 18.9-21.9 ms | 46-53 | 46-53 |
| S=2, 200K | 12.28 GB | 30.8-33.8 ms | 30-32 | 59-65 |
| S=1, 4K, DSpark k=7, tau 3.0-3.57 | 11.43 GB | 29.9-32.9 ms | 91-120 | 91-120 |
| S=1, 4K, k=3 (tau ~2.8 EST) | 7.91 GB | 21.6-24.6 ms | 114-130 | 114-130 |
| S=1, 4K, k=2 (tau ~2.4 EST) | 6.94 GB | 19.3-22.3 ms | 108-124 | 108-124 |
| S=1, 64K, k=7 | 12.44 GB | 32.2-35.2 ms | 85-111 | 85-111 |
| S=2, 4K, k=7 | 17.58 GB | 44.3-47.3 ms | 63-81 | 127-161 |
| S=4, 4K, k=7 | 25.24 GB | 62.3-65.3 ms | 46-57 | 184-229 |
| S=8, 4K, k=7 (64 rows; add ~6-12 ms compute) | 31.52 GB | 77-80 ms (+compute) | 37-46 (~33-41) | 300-371 (~265-330) |

Box summary, DP=2 (one replica per card, short context; EST ceilings for a mature engine):

| | C1 | C2 | C4 | C8 | C16 |
|---|---:|---:|---:|---:|---:|
| No draft, 4K | 70-88 | 140-176 | 214-254 | 298-336 | 396-428 |
| No draft, 64K | 60-73 | 120-146 | 170-196 | 220-240 | 270-284 (fit borderline) |
| DSpark k=7, 4K (tau 3.0-3.57) | 91-120 | 182-240 | 254-322 | 368-458 | ~530-660 (compute-limited, low confidence) |
| MEAS Strata Flash-Next, 2 cards | 133.5 | 147-158 | ~148 | capped | capped |
| MEAS TensorFold 27B + DFlash2, ONE card | 142 | 228 | 304 | 265 | - |

TP=2 (one replica over both cards, EP for the experts, 96 exchanges x 25 us = 2.4 ms/step): C1 4K with no draft is 1.99
GB/card, 9.1-12.1 ms, 83-110 tok/s. C1 4K with DSpark k=7 is 5.72 GB/card, 18.8-21.8 ms, 137-190 tok/s. C2 with DSpark
is 206-274 aggregate. This is the only configuration whose C1 model reaches the 150+ goal, and it depends on a stable
two-card collective path (TensorFold push all-reduce + recorded windows exist; vLLM TP>1 on this box has the known
oneCCL hazard).

PP=2: the loop crosses the split each pass (a split at layer 12 gives 5 hops/token, ~0.1 ms), so C1 equals the
one-card number and only memory improves. Not useful, since one card already fits. EP=2 alone (attention replicated)
gives C1 ~79, no better than DP; TP+EP is the two-card option if any.

Compute check: matmul FLOPs ~8.8 GFLOP/token (4.4B effective params). Decode at <= 16 rows is <= 0.14 TFLOP/step, i.e.
1-3 ms, so the bandwidth model holds. Only the C8-per-card x 8-row DSpark case (64 rows, ~0.56 TFLOP/step) is partly
compute-limited.

Prefill (EST): ~8.8 GFLOP/token of matmul + ~2 GFLOP/token of attention at a 32K prompt, ~10.9 GFLOP/token. At 40-70
TFLOPS effective for grouped NVFP4/IQ4_NL->bf16 DPAS tiles that gives ~3.7-6.4K tok/s per card (two independent
replicas in DP). Compare MEAS: Strata Flash-Next ~2.2K on 2 cards; TensorFold 27B 1.9-2.5K on one card; TensorFold
Flash-Next port 300 today. llama.cpp SYCL would be far lower (kq_b materialization limits ubatch at long prompts).

Caveats on these estimates: every engine on this box so far lands below a pure-bandwidth model at C>=4. Strata
Flash-Next C4 ~148 is far below its union model, and TensorFold 27B C8 265 < C4 304. Derate the C>=4 rows by 0.6-0.8
for a first port. DSpark acceptance on agentic code at temperature >0 is unmeasured (vLLM: 3.57 on GSM8K, greedy-ish).
Long agent contexts (64-200K) cut decode 15-45% vs 4K, because of the 8 full-attention visits. Flash-Next's QSA (2048-token
budget) + GDN avoids that.

## 6. What an Inkling family in our TensorFold fork would need (EST scope)

1. GGUF map from the llama.cpp converter names (inkling2 + loop tensors: loop_msg_query, *_ctrl_{down,gain,forget}),
   or read the HF NVFP4 safetensors directly (NVFP4 dequant is trivial; it keeps the trained expert values exactly).
2. Attention kernels: SWA-1024 + global paged attention with learned relative bias (16-dim rel queries -> 1025-entry
   table per head, clipped), learned sink logit, output gate, QK L2 clip, log-n scaling, value norm, K/V short convs
   (static on SWA layers, input-projected on global layers) with per-visit conv state.
3. Attention-residual reads over block outputs and the loop driver: 3 passes over L6-17, per-visit KV/conv caches
   (48 visit slots per token), cross-pass messages with 4 rank-64 controllers per core layer (FP32 merge).
4. MoE 128 x top-8 with gate bias, route scale 8, gate/up softcaps; NVFP4 (or IQ4_NL) and MXFP8 (or Q8_0) expert GEMV/DPAS
   tiles. The existing IQ4_NL/q8_1 work in fn-int helps.
5. DSpark: 3-layer block drafter fed by target visits 5/23/41/48 + Markov RNN head; close to DFlash2, but new.
6. inkling tool/reasoning parser in the serve layer; effort -> system message mapping.
7. Gates: teacher-forced top-1/KL vs a CPU/transformers or llama.cpp-CPU oracle; concurrent == solo; park/restore bitwise.
Reuse: DP serve, paged int8 KV, park/restore (B3), recorded windows, push AR for an optional TP2.
Rough effort: 2-4 weeks of the current Lead A/B agent setup, competing directly with the Flash-Next port now in M-A5.

## 7. Recommendation

1. Keep Qwen3.8-Flash-Next (ISTA GSQ-RCO IQ3_XXS) as the daily driver and keep the TensorFold Flash-Next campaign
   first. Inkling 2 Mini is roughly a Qwen3.8-27B-class model on vendor data, is clearly below Flash-Next on
   independent data (AA 20 vs 29), and carries explicit agentic-reliability warnings.
2. Cheap quality gate before any engine work: once llama.cpp merges #32694 or #32695, build SYCL. With the DD stopped,
   run evals/abfast (pi 1.2.0, 131K window) on ONE card with ggml-org IQ4_NL (17.63 GB). Speed is irrelevant to that
   probe. Proceed only if it ties or beats Flash-Next IQ3_XXS on paired tasks (same rule as the 2026-10-13 A/B brief).
3. If it passes, the right serving shape is DP=2 (one replica per card, int8 KV, 2 x 200K or 7 x 64K slots per card,
   DSpark k=2-3 for C1, k=7 at higher concurrency), built as a TensorFold family. Expected ceiling: C1 ~90-130,
   C4 ~250-320, C8 ~300-450 aggregate at short context; less at 64K+.
4. Watch: llama.cpp #32694/#32695/#32218, vLLM #69028 (for an XPU-able Triton path), any Strata/TensorFold Inkling
   issue, AA sub-index results, and the TML-d24-varloop-research checkpoint. Its 1-2 pass mode with a shared cache
   would cut bytes/step by ~25-50% and halve the global KV, at some quality cost.

## Links

- https://huggingface.co/thinkingmachines/Inkling-2-Mini (README, config.json, hf_quant_config.json, chat_template.jinja)
- https://thinkingmachines.ai/news/inkling-2-mini/ and https://huggingface.co/blog/inkling2mini
- https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini ; https://github.com/vllm-project/vllm/pull/69028
- https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini ; https://github.com/sgl-project/sglang/pull/51926
- https://github.com/ggml-org/llama.cpp/pull/32694 ; https://github.com/ggml-org/llama.cpp/pull/32695 ; https://github.com/ggml-org/llama.cpp/pull/32218
- https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF ; https://huggingface.co/unsloth/Inkling-2-Mini-GGUF ; https://unsloth.ai/docs/models/inkling-2-mini
- https://artificialanalysis.ai/models/inkling-2-mini ; https://artificialanalysis.ai/models/qwen3-8-flash-next
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next (vendor table)
- https://news.ycombinator.com/item?id=50581260
- Local: strata/flash-next/concurrency-0216/tensorfold-probe/JOURNAL.md (27B MEAS), tensorfold/flashnext/BRIEF.md and STATUS.md
  (Strata/TensorFold MEAS), docs/20261119_b70_reference_refresh.md (expert-union data), docs/20261119_hy4_flash_quant_review.md
