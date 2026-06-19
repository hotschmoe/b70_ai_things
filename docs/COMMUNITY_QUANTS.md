# Community INT8-activation quants for newer models (W8A8-INT8 / W4A8-INT8)

What exists on Hugging Face that our **INT8 W8A8 oneDNN fast path** could serve, for
newer model families (~Oct 2025 -> mid 2026). Compiled 2026-06-20 from a 4-way HF sweep;
every repo in the tables below was confirmed to exist via the HF API / `config.json`
(activation dtype read directly: INT8 = tensor type `I8` / `"type":"int"`, NOT FP8).

> We only want **INT8 activations**. The B70 (Xe2) has no native FP8, so anything with
> FP8 or FP4/NVFP4 activations is useless to us even if the weights are int. The two
> usable schemes, and the kernel each lands on:
> - **W8A8-INT8** -- 8-bit int weights, 8-bit int activations. Our custom oneDNN kernel
>   `XPUInt8ScaledMMLinearKernel` (image `vllm-xpu-env:int8`). The proven 8-bit path.
> - **W4A8-INT8** -- 4-bit int weights, 8-bit int activations. Lands on vLLM-XPU's
>   `XPUW4A8IntLinearKernel` (int4 weights + per-token int8 act, oneDNN). Win is *memory/fit*,
>   not decode speed (decode stays bandwidth-bound). End-to-end W4A8 not yet validated on our
>   box -- W8A8 is what we've run; confirm the kernel routes before relying on it.

## B70 fit math

Per card: **Arc Pro B70, 32 GB, ~30.3 GiB usable** (measured). With tensor-parallel,
weights and KV split across cards. Budget ~6 GiB/card for KV + activations + graphs.

```
                 aggregate usable   practical weight budget (on-disk safetensors)
  1x B70           ~30 GiB             <= ~24 GB   (>~24 GB only at short ctx / low concurrency)
  2x B70           ~60 GiB             <= ~50 GB
  4x B70          ~120 GiB             <= ~105 GB
  > 4x  ...........  needs 8x+ cards or CPU/disk offload
```

Rule of thumb on-disk size: W8A8-INT8 ~= 1.0-1.1 GB per 1B params; properly-packed
W4A8-INT8 ~= 0.55-0.65 GB per 1B params (see the W4A8 bloat warning below -- several
community "W4A8" repos are NOT packed and give no fit win).

---

## Try-first: the three Quark W8A8-INT8 (2x B70)

All AMD Quark 0.11.1, `ptpc_int8` recipe: **W INT8 per-channel (axis=0) symmetric static**
+ **A INT8 per-token (axis=1) symmetric dynamic**; `lm_head`/`embed_tokens`/vision-tower/MTP
kept BF16. This is exactly our oneDNN W8A8 path. **All three need 2x B70** (none clears one
card's 30.3 GiB once KV is added) -- they are the natural first dual-card workload.

| Repo | Base | Type | Size | Min VRAM | Accuracy vs BF16 |
|------|------|------|------|----------|------------------|
| nameistoken/Qwen3.6-27B-Quark-W8A8-INT8 | Qwen3.6-27B (+27-layer ViT) | dense, 64 hybrid layers (16 full + 48 linear attn) | ~30 GB | ~32 GB | GSM8K 96.74%, **0.00pp** |
| nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8 | Qwen3.6-35B-A3B (+ViT) | MoE 35B/3B, 256 exp top-8 | ~37 GB | ~40 GB | GSM8K 95.91%, **0.00pp**, Jaccard 0.977 |
| nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8 | gemma-4-31B-it (+vision) | dense | ~33 GB | ~48 GB | GSM8K-8shot 96.66%, **-0.08pp** |

Caveats: MoE router + shared-expert gates stay BF16 (35B-A3B) -- confirm kernel handles
mixed BF16-gate + INT8-expert GEMM. Each bundles a BF16 vision tower (strip it to save
~2-4 GB if text-only). 35B-A3B needs vLLM >= 0.19.2rc1 with `qwen3_5_moe` registered and
`chat_template_kwargs={"enable_thinking":false}`. Single-uploader community checkpoints --
re-run our own GSM8K/HumanEval+ before trusting the 0.00pp claims.

---

## Availability matrix (verified INT8-activation, in-scope families)

Sizes: exact where the card/API gave a file sum, else `~` estimate from param count.

### Fits 1x B70 (<= ~24 GB on disk)

| Repo | Base | Params | Scheme | Size | Tool / Uploader |
|------|------|--------|--------|------|-----------------|
| RedHatAI/Qwen3.5-4B-quantized.w8a8 | Qwen3.5-4B | 4B dense | W8A8-INT8 | 6.8 | llm-compressor / RedHatAI |
| RedHatAI/Qwen3.5-9B-quantized.w8a8 | Qwen3.5-9B | 9B dense | W8A8-INT8 | 14.0 | llm-compressor / RedHatAI |
| NotaMG/Qwen3.5-4B-W8A8-Dynamic | Qwen3.5-4B | 4B dense | W8A8-INT8 | 6.8 | compressed-tensors / NotaMG |
| nunusadmqk/gemma-4-E4B-it-W8A8-INT8-v10-datafree | gemma-4-E4B-it | ~8B dense | W8A8-INT8 | 11.9 | compressed-tensors / nunusadmqk |
| alecccdd/GLM-4.6V-Flash-W8A8-INT8 | GLM-4.6V-Flash (VL) | ~10B dense | W8A8-INT8 | 16.9 | llm-compressor / alecccdd |
| RedHatAI/Llama-Guard-4-12B-quantized.w8a8 | Llama-Guard-4-12B (Llama4) | 12B dense | W8A8-INT8 | 14.9 | llm-compressor / RedHatAI |
| RedHatAI/phi-4-quantized.w8a8 | phi-4 | 15B dense | W8A8-INT8 | ~15 | llm-compressor / RedHatAI |
| alishafique/Phi-4-reasoning-quantized.w8a8int8-llmcompressor | Phi-4-reasoning | 15B dense | W8A8-INT8 | ~15 | llm-compressor / alishafique |
| alishafique/Phi-4-reasoning-quantized.w4a8int8-llmcompressor | Phi-4-reasoning | 15B dense | **W4A8-INT8** | ~8 | llm-compressor / alishafique |
| fedora-copr/granite-4.0-h-tiny-quantized.w8a8 | granite-4.0-h-tiny | 7B MoE (Mamba+Tx) | W8A8-INT8 | ~7 | llm-compressor / fedora-copr |
| kaitchup/Olmo-3-7B-Instruct-w8a8-smoothquant | Olmo-3-7B-Instruct | 7B dense | W8A8-INT8 | ~7 | llm-compressor / kaitchup |
| kaitchup/Olmo-3-7B-Think-w8a8-smoothquant | Olmo-3-7B-Think | 7B dense | W8A8-INT8 | ~7 | llm-compressor / kaitchup |
| pytorch/gemma-3-4b-it-HQQ-INT8-INT4 | gemma-3-4b-it | 4B dense | **W4A8-INT8** | ~3 | torchao+HQQ / PyTorch |

1x-tight (~25 GB; runs on 1x only at short ctx / low concurrency, else 2x):

| Repo | Base | Params | Scheme | Size | Tool / Uploader |
|------|------|--------|--------|------|-----------------|
| RedHatAI/Devstral-Small-2507-quantized.w8a8 | Devstral-Small-2507 | 24B dense | W8A8-INT8 | 24.9 | llm-compressor / RedHatAI |
| noneUsername/Mistral-Small-3.2-24B-Instruct-hf-W8A8 | Mistral-Small-3.2-24B | 24B dense | W8A8-INT8 | ~25 | llm-compressor / noneUsername |
| jiangchengchengNLP/Mistral-Small-3.2-24B-Instruct-W8A8 | Mistral-Small-3.2-24B | 24B dense | W8A8-INT8 | ~25 | llm-compressor / jiangchengchengNLP |
| noneUsername/Magistral-Small-2506-W8A8 | Magistral-Small-2506 | 24B dense | W8A8-INT8 | ~25 | llm-compressor / noneUsername |

### Fits 2x B70 (~25-50 GB on disk)

| Repo | Base | Params | Scheme | Size | Tool / Uploader |
|------|------|--------|--------|------|-----------------|
| **nameistoken/Qwen3.6-27B-Quark-W8A8-INT8** | Qwen3.6-27B | 27B dense | W8A8-INT8 | 30.4 | Quark / nameistoken |
| **nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8** | Qwen3.6-35B-A3B | 35B/3B MoE | W8A8-INT8 | 36.7 | Quark / nameistoken |
| **nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8** | gemma-4-31B-it | 31B dense | W8A8-INT8 | 33.3 | Quark / nameistoken |
| groxaxo/Qwen3.6-27B-uncensored-heretic-v2-...-W8A8 | Qwen3.6-27B FT | 27B dense | W8A8-INT8 | 31.2 | compressed-tensors / groxaxo |
| RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8 | Qwen3-30B-A3B-Instruct-2507 | 30B/3B MoE | W8A8-INT8 | 31.2 | llm-compressor / RedHatAI |
| RedHatAI/Qwen3-30B-A3B-Thinking-2507-quantized.w8a8 | Qwen3-30B-A3B-Thinking-2507 | 30B/3B MoE | W8A8-INT8 | 31.2 | llm-compressor / RedHatAI |
| fedora-copr/granite-4.0-h-small-quantized.w8a8 | granite-4.0-h-small | 32B MoE | W8A8-INT8 | 32.3 | llm-compressor / fedora-copr |
| kaitchup/Olmo-3.1-32B-Think-w8a8-smoothquant | Olmo-3.1-32B-Think | 32B dense | W8A8-INT8 | ~32 | llm-compressor / kaitchup |
| arnepa/Llama-3_3-Nemotron-Super-49B-v1_5-W8A8-Dynamic | Nemotron-Super-49B-v1.5 | 49B dense | W8A8-INT8 | ~49 | llm-compressor / arnepa |
| lokeshe09/gemma-4-31B-it-INT4-W4A8 | gemma-4-31B-it | 31B dense | W4A8-INT8 (!) | 33.7 | compressed-tensors / lokeshe09 |
| Avesed/Qwen3.6-27B-W4A8 | Qwen3.6-27B | 27B dense | W4A8-INT8 (!) | 36.2 | compressed-tensors / Avesed |
| Avesed/Qwopus3.6-27B-v2-W4A8 | Qwopus3.6-27B-v2 (Qwen3.6 FT) | 27B dense | W4A8-INT8 (!) | 36.2 | compressed-tensors / Avesed |
| AMbaye018/sarvam-30b-gptq-w4a8-targeted | sarvam-30b (2026-03) | 32B MoE | W4A8-INT8 | 42.6 | GPTQ / AMbaye018 |

(!) **W4A8 bloat warning:** these "W4A8" repos are ~33-36 GB -- the *same* footprint as the
W8A8 versions, not the ~14-16 GB a real 4-bit pack should be. They appear stored unpacked
(int4 values in wide containers / BF16 master retained), so they give **no fit advantage**.
Verify on-disk layout before using; do not assume they drop a 27/31B model onto 1x.

### Fits 4x B70 (~50-105 GB on disk)

| Repo | Base | Params | Scheme | Size | Tool / Uploader |
|------|------|--------|--------|------|-----------------|
| RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8 | Qwen3-Next-80B-A3B-Instruct | 80B/3B MoE | W8A8-INT8 | 81.7 | llm-compressor / RedHatAI |
| inference-optimization/Qwen3-Coder-Next.w8a8 | Qwen3-Coder-Next | 80B MoE | W8A8-INT8 | 81.7 | llm-compressor / Red Hat |
| stanidiener/gpt-oss-120b-w4a8 | gpt-oss-120b | ~120B/~5B MoE | W4A8-INT8 | ~60-65 | compressed-tensors / stanidiener |
| ArliAI/GLM-4.5-Air-Derestricted-W8A8-INT8 | GLM-4.5-Air FT | 106B/12B MoE | W8A8-INT8 | ~110 (4x-tight) | llm-compressor / ArliAI |

### Exceeds 4x B70 (needs 8x+ or offload) -- listed for completeness

| Repo | Base | Params | Scheme | Size | Tool / Uploader |
|------|------|--------|--------|------|-----------------|
| ArliAI/Mistral-Medium-3.5-128B-INT8-W8A8-Dynamic | Mistral-Medium-3.5-128B (+vision) | 128B dense | W8A8-INT8 | ~128 | llm-compressor / ArliAI |
| baidu/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle | ERNIE-4.5-300B-A47B | 300B/47B MoE | W4A8-INT8 + INT8 KV (**Paddle fmt**) | ~169 | PaddleSlim / baidu |
| nameistoken/Step-3.5-Flash-Quark-W8A8-INT8 | Step-3.5-Flash | ~197B/~11B MoE | W8A8-INT8 | ~191 | Quark / nameistoken |
| nameistoken/MiniMax-M2.5-Quark-W8A8-INT8 | MiniMax-M2.5 | ~229B MoE | W8A8-INT8 | ~230 | Quark / nameistoken |
| nameistoken/MiniMax-M2.7-Quark-W8A8-INT8 | MiniMax-M2.7 | ~229B MoE | W8A8-INT8 | ~230 | Quark / nameistoken |
| RedHatAI/MiniMax-M2.5-quantized.w8a8 | MiniMax-M2.5 | ~229B MoE | W8A8-INT8 | 232.9 | llm-compressor / RedHatAI |
| RedHatAI/Qwen3-235B-A22B-Instruct-2507-quantized.w8a8 | Qwen3-235B-A22B-Instruct-2507 | 235B/22B MoE | W8A8-INT8 | 236.6 | llm-compressor / RedHatAI |
| RedHatAI/GLM-4.6-quantized.w8a8 | GLM-4.6 | 353B/~32B MoE | W8A8-INT8 | 362.7 | llm-compressor / RedHatAI |
| ArliAI/GLM-4.6-Derestricted-W8A8-INT8 | GLM-4.6 FT | 357B MoE | W8A8-INT8 | ~360 | llm-compressor / ArliAI |

---

## Traps and near-misses (do NOT use as INT8-activation)

- **FP8 activations (B70 can't run):** `amd/Kimi-K2.5-W4A8` and `amd/Kimi-K2-Thinking-W4A8`
  are W4-INT4 / **A-FP8** (the exact "W4A8" naming trap). Also FP8-act: RedHatAI's entire
  `*-FP8`/`-FP8-dynamic`/`-FP8-block` lines (Qwen3-Next, Qwen3.5/3.6, gemma-4, Llama-4,
  Magistral, MiniMax), `czhu-cohere/Qwen3-30B-A3B-quantized.w4a8-v2`,
  `dominant-strategies/quai-deepseek-v4-flash-igemm-w8a8` (quant_method fp8),
  `TMElyralab/DeepSeek-V3.1-AWQ-W4AFP8`, alishafique's `*w8a8fp8*`/`*kv8fp8*` variants.
- **FP4/NVFP4 activations:** all RedHatAI NVFP4 lines; `mistralai/Mistral-Small-4-119B-2603-NVFP4`;
  `CohereLabs/command-a-plus-05-2026-w4a4` (W4**A4**).
- **Weight-only (activations stay BF16/16-bit) despite "INT8"/"W4A8" in the name:**
  - All **cpatonn** AWQ `*-INT8-INT4` / `*-BF16-INT8` repos -> `input_activations:null`; the
    suffix is *mixed weight* bit-widths, A16. (W8A16/W4A16, not A8.)
  - All **ModelCloud** GPTQModel repos -> W4A16.
  - RedHatAI/inference-optimization `*.w4a16` (Qwen3-Next, Coder-Next, Qwen3.5, Llama-4).
  - `QuantTrio/GLM-4.6-GPTQ-Int4-Int8Mix` and any `-Int8Mix` GPTQ -> mixed *weight* bits, A16.
  - Falcon-H1/Falcon3 `*-GPTQ-Int8` -> W8A16.
- **Empty stub repos (HTTP 200 but no weights):**
  `inference-optimization/Qwen3-Next-80B-A3B-{Instruct,Thinking}-quantized.w8a8` contain only
  a README. Use RedHatAI's Qwen3-Next w8a8 instead. The `inference-optimization/Qwen3-Coder-Next.w8a8`
  and `...Qwen3-235B...Thinking...w8a8` repos should be **size-checked before download** (sibling
  stubs existed).
- **Scheme unverifiable (name says w8a8, config empty):** `KingsonHO/Qwen3.6-35B-A3B-w8a8`
  (39.8 GB) and `KingsonHO/Step-3.7-Flash-w8a8-mtp` (170.9 GB) -- real repos on newer bases,
  but no `quantization_config` to confirm INT8 vs FP8. File-level check needed before serving.

---

## Gaps: newer families with NO usable INT8-activation checkpoint

| Family | What exists instead | Verdict |
|--------|--------------------|---------|
| **Meta Llama 4** (Scout/Maverick) | FP8, FP8-block, NVFP4, w4a16 only | No INT8-act. (Only Llama-Guard-4-12B has w8a8.) |
| **DeepSeek** (V3.1 / V3.2-Exp / V4-Flash) | AWQ-W4A16, NVFP4, W4AFP8, native UE8M0-FP8 | No INT8-act anywhere. |
| **Moonshot Kimi** (K2.x / K2-Thinking / K2.5) | FP8-block, W4A16, AMD W4A8-**FP8**, native INT4-weight | No INT8-act. |
| **Tencent Hunyuan** (A13B / HY-MT) | FP8, GPTQ-Int4 (weight-only); "W8A8C8" = FP8 | No INT8-act. |
| **NVIDIA Nemotron-3 / Nemotron-Nano-2 (9B-v2)** | FP8, NVFP4, W4A16 | No INT8-act. (Only older Llama-3.3-Nemotron-Super-49B + OpenReasoning-32B have it.) |
| **Cohere Command-A / Command-R 2025** | W4A4, FP8 only | No INT8-act anywhere. |
| **Microsoft Phi-5** | (none found) | No verified quant repos. |
| **Granite-4 micro** | only h-tiny / h-small have w8a8 | Gap at the smallest size. |

---

## Recommendations: what we should quantize ourselves

Priority by value-for-effort, using our recoverable-quant (GPTQ/AutoRound) playbook and
the `ptpc_int8` recipe our oneDNN kernel already matches:

1. **W4A8-INT8 (properly packed) of Qwen3.6-27B and Gemma-4-31B -- highest value.**
   The W8A8 versions need 2x B70; a real 4-bit weight pack (~14-16 GB) drops each onto a
   **single B70**. The community W4A8 repos for these (Avesed, lokeshe09) are bloated to
   ~33-36 GB and give no fit win -- this is an open, high-impact gap directly on our
   headline models. (Prereq: confirm the kernel's int4->int8 unpack path; the win is fit,
   not speed.)

2. **Nemotron-Nano-9B-v2 W8A8-INT8** (or Nemotron-3-Nano-30B-A3B). Popular, currently
   FP8/NVFP4-only, and the 9B fits 1x cleanly. A clean, in-demand checkpoint nobody has shipped.

3. **Qwen3-Coder-Next W8A8 / W4A8** for our coding-eval line (HumanEval+/LiveCodeBench).
   First verify `inference-optimization/Qwen3-Coder-Next.w8a8` actually has weights (sibling
   stubs were empty); if it does, validate it -- else produce our own (80B/3B MoE -> 4x).

4. **Cohere Command-A INT8** -- only if we want that family; nothing INT8-act exists at all.

5. Low priority / skip: **DeepSeek, Kimi, Hunyuan** -- all exceed 4x and ship FP8-native;
   only worth it if we build a CPU/disk-offload + INT8 path.

### Quick answer: best newer-model picks per config (today, off-the-shelf)
- **1x B70:** Qwen3.5-9B w8a8 (14 GB), gemma-4-E4B w8a8 (~8B), GLM-4.6V-Flash w8a8 (VL, 10B),
  phi-4 w8a8 (15B), Granite-4.0-h-tiny, Olmo-3-7B. Tight: Mistral-Small-3.2-24B / Devstral-24B.
- **2x B70:** the three try-first (Qwen3.6-27B, Qwen3.6-35B-A3B, Gemma-4-31B), Qwen3-30B-A3B-2507,
  Granite-4.0-h-small, Olmo-3.1-32B, Nemotron-Super-49B.
- **4x B70:** Qwen3-Next-80B-A3B w8a8, gpt-oss-120b w4a8, GLM-4.5-Air (tight).
