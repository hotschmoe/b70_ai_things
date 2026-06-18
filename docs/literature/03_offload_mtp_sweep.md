# CPU/RAM Offload, MTP/Speculative Decoding, and Benchmark Sweep Methodology

**Target system:** Intel Arc Pro B70 (Battlemage / Xe2, 32 GB GDDR6 ECC, 256-bit, ~608 GB/s, 367 INT8 TOPS, **PCIe 5.0 x16**), 128 GB slow DDR4 system RAM, running **Qwen3.6-27B in Docker on Unraid**.
**Date:** June 2026. **Author:** literature review (research-agent synthesis + direct source verification).

---

## Key model facts (load-bearing for everything below)

Qwen3.6-27B is a **dense 27B** model (NOT MoE) with a hybrid **Gated DeltaNet + Gated Attention** layout across 64 layers (3 of every 4 sublayers are linear-attention DeltaNet), native **262K context** (→1M via YaRN), and **native Multi-Token-Prediction (MTP)** heads trained for speculative decoding. ([MarkTechPost](https://www.marktechpost.com/2026/04/22/alibaba-qwen-team-releases-qwen3-6-27b-a-dense-open-weight-model-outperforming-397b-moe-on-agentic-coding-benchmarks/), [vLLM recipes](https://recipes.vllm.ai/Qwen/Qwen3.6-27B))

Two consequences drive this whole report:

1. **Dense ⇒ MoE-specific offload tricks (`--n-cpu-moe`, ktransformers expert offload) do not apply to the 27B.** They apply only to its sibling **Qwen3.6-35B-A3B MoE**. For the dense 27B, offload means classic *whole-layer* offload, which is a cliff, not a slope.
2. **Q4_K_M (~15.6 GiB weights) fits comfortably in 32 GB with large KV headroom**, so for the 27B **offload is almost never necessary** — the interesting offload questions are about the 35B-A3B MoE sibling or very long context.

3. The Gated-DeltaNet ("gdn") attention path is **exactly the code path that currently breaks MTP speculative decoding on vLLM's Intel XPU backend** (see Topic B). This is the single biggest gotcha in this document.

---

# TOPIC A — CPU/RAM offload on Intel XPU

## A.1 llama.cpp partial offload (SYCL / Vulkan)

**`-ngl` (n-gpu-layers) for the dense 27B is a cliff, not a slope.** Every generated token must traverse *all* layers; any layer left on the CPU is computed at DDR4 bandwidth (~tens of GB/s, 2-channel DDR4 ≈ 50–90 GB/s) instead of GDDR6 (608 GB/s), and the per-token activation has to cross PCIe each direction. Community measurements on dense models show ~70 % of layers on GPU already loses ~40 % of throughput, and a 0/N→N/N offload swing was ~10× (e.g. 9 → 95 t/s on a CUDA box). ([Medium / Ekansh Jain](https://medium.com/@ekansh.jain2011/squeezing-every-drop-of-performance-out-of-llama-cpp-the-practitioners-guide-to-local-ai-2bcc3663f06f), [bmdpat ngl guide](https://bmdpat.com/blog/llama-cpp-n-gpu-layers-explained-2026)) **For Qwen3.6-27B the correct answer is `-ngl 99` (everything on the GPU) with a quant that leaves KV headroom — do NOT partially offload it.**

Gotcha: **`-ngl 99` does not guarantee full offload.** If VRAM is exhausted (e.g. Q8_0 + long context), llama.cpp silently falls back to running some tensors on CPU and you get mysteriously low t/s. ([bmdpat "when -ngl 99 still runs on your CPU"](https://bmdpat.com/blog/llama-cpp-ngl-cpu-fallback-fix-2026)) Always confirm actual VRAM residency.

**`--n-cpu-moe` / `--cpu-moe` — MoE only, does NOT help the dense 27B.** These flags route the **expert tensors** of MoE layers to system RAM (counting from the highest-numbered layers down). Because only ~3B of 35B params are active per token in the A3B MoE, the model streams a small fraction of expert weights over PCIe per token, which is *why MoE tolerates offload and dense models don't*. The mechanism: the token's activation vector is sent VRAM→RAM, the CPU does the expert matmul using weights resident in RAM, the result returns RAM→VRAM. ([llama.cpp MoE offload guide / HF](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide), [dev.to MoE offloading](https://dev.to/someoddcodeguy/understanding-moe-offloading-5co6)) These flags exist for `llama-cli`, `llama-server`, and `llama-bench`. **Use them only for Qwen3.6-35B-A3B**, e.g. on the B70 you'd keep attention + a few experts on GPU and push `--n-cpu-moe <K>` experts to the 128 GB DDR4 to run a larger MoE that doesn't fit in 32 GB.

**KV cache offload / `--no-kv-offload`.** By default the KV cache lives in VRAM alongside the model. `--no-kv-offload` keeps KV in system RAM, freeing VRAM for weights but making every attention step PCIe/DDR4-bound — generally a *last resort*. Far better levers to fit KV in 32 GB: (a) **KV quantization** `-ctk q8_0 -ctv q8_0` (halves KV VRAM at <0.1 % ppl), (b) **flash attention** `-fa on` (lower KV footprint), (c) shorter context. The Lemon-Gravy B70 build explicitly uses `--cache-ram 0` to *forbid* KV spill to RAM and keep it VRAM-only for speed. ([Lemon Gravy / B70 SYCL](https://lemongravy.me/articles/intel-gpu-llamacpp/), [llama.cpp #5932 KV quant](https://github.com/ggml-org/llama.cpp/discussions/5932))

## A.2 vLLM XPU CPU offload

vLLM exposes **`--cpu-offload-gb <N>`** (reserve N GiB of CPU RAM as "extra GPU memory" for weights) and **`--swap-space <N>`** (CPU RAM for KV swapping under preemption). On Intel XPU, vLLM gained Arc support in v0.6.6 (Feb 2025) and there is an env var `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1` for offloading weights to CPU before quantization. ([Intel/ipex-llm](https://github.com/intel/ipex-llm), [vLLM XPU docs](https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/)) **Caveat (flag as partially verified):** `--cpu-offload-gb` is primarily validated on CUDA; XPU offload paths are newer and the performance penalty mirrors llama.cpp's — any weights served from DDR4 over PCIe cut decode throughput proportionally to the offloaded fraction. For the dense 27B at INT4/FP8 in vLLM (fits in 32 GB), **do not set `--cpu-offload-gb`**; it is only relevant if you push to BF16 or run the 35B-A3B without enough VRAM.

## A.3 IPEX-LLM offload — **deprecated, plan around it**

IPEX-LLM (Intel's `ipex-llm`) historically provided low-bit (INT4/SYM-INT4) inference and a `low_bit` path on Arc, plus a **FlashMoE** path that ran DeepSeek-V3/R1 671B and Qwen3-MoE-235B on 1–2 Arc GPUs by offloading experts (May 2025). ([Intel/ipex-llm](https://github.com/intel/ipex-llm)) **However, the agent research found IPEX-LLM was archived in January 2026** — for new work target `intel-extension-for-pytorch` + **vLLM-XPU** or **llama.cpp SYCL** instead. Treat any IPEX-LLM-specific offload guidance as legacy.

## A.4 ktransformers — MoE expert offload, *does* support Intel Arc, but CUDA-first

ktransformers is the heterogeneous CPU+GPU MoE inference framework: it keeps hot/attention tensors on GPU and offloads **cold MoE experts to CPU**, accelerated by **Intel AMX (AMX-INT8 / AMX-BF16) and AVX512** kernels. It claims Intel Arc GPU support (tutorial dated May 2025) alongside CUDA, ROCm, and Ascend. ([kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers)) **Two caveats:** (1) its CPU-offload speed relies on **AMX**, which is a **Xeon/Sapphire-Rapids+ server feature — your Unraid host's DDR4 box almost certainly lacks AMX**, so the headline numbers (e.g. DeepSeek-R1 FP8 at 227 t/s on 8×L20 + Xeon Gold) will not transfer; (2) it is **MoE-only** — useless for the dense 27B. ktransformers is only worth considering if you want to run the **35B-A3B MoE** (or a bigger MoE) and your CPU has AMX.

## A.5 Performance penalty of PCIe/DDR4-bound offload — concrete reasoning

- **B70 link:** PCIe 5.0 x16 (≈64 GB/s) is generous; in dual-card setups it negotiates x8 per card. ([Phoronix B70 review](https://www.phoronix.com/review/intel-arc-pro-b70-linux), [PMZFX benchmarks](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks)) PCIe is rarely the bottleneck for *dense* offload; **DDR4 bandwidth is.**
- **DDR4 is the wall.** Partial dense offload speed is bound by **system-RAM bandwidth**, and DDR5 "significantly outperforms DDR4" here. A typical 2-channel DDR4 desktop is ~89 GB/s vs ~358 GB/s for 8-channel DDR5-5600 — a 4× gap that lands directly on decode t/s when any dense layers sit in RAM. ([Medium / Ekansh Jain](https://medium.com/@ekansh.jain2011/squeezing-every-drop-of-performance-out-of-llama-cpp-the-practitioners-guide-to-local-ai-2bcc3663f06f)) On *slow* DDR4 the penalty is worse.
- **Rule of thumb (flag as extrapolated, not B70-measured for dense offload):** each ~10 % of a dense model pushed to DDR4 costs roughly proportional decode throughput; pushing the majority to RAM drops you toward CPU-only speeds (single-digit t/s for a 27B).

## A.6 Decision guidance — offload vs. smaller quant

**For the dense Qwen3.6-27B: choose the quant, don't offload.** Q4_K_M (~15.6 GiB) fits in 32 GB with ~15 GB free for KV/context — there is no reason to offload. Going *up* to Q6_K/Q8_0 to "use the VRAM" is counterproductive on Battlemage because decode is memory-bound and **heavier quant = more bytes/token = slower decode** (Topic C table: Q4_K_M 20.6 t/s vs Q6_K 13.8 t/s, and Q8_0 historically 4–5× slower than Q4_K_M on B70 SYCL due to a kernel-efficiency bug, [llama.cpp #21517](https://github.com/ggml-org/llama.cpp/issues/21517)). **A smaller/faster quant that fits beats offload every time for a dense model.**

**When offload IS worth it:**
- You insist on running the **35B-A3B MoE** and it doesn't fit → use llama.cpp `--n-cpu-moe` (MoE experts stream cheaply over PCIe), or ktransformers *if your CPU has AMX*.
- You need a model genuinely larger than 32 GB (70B-class) and accept single-digit t/s — then dual B70 (64 GB) is a better answer than DDR4 offload (PMZFX ran 80B-A3B at 43 t/s on dual B70).
- Long context where **KV** (not weights) overflows → prefer KV quant / flash-attn / shorter ctx over `--no-kv-offload`.

| Situation | Recommended action | Avoid |
|---|---|---|
| Dense 27B, normal ctx | Q4_K_M, `-ngl 99`, all in VRAM | offload, Q8_0 |
| Dense 27B, very long ctx | `-fa on`, `-ctk q8_0 -ctv q8_0`, shorter ctx | `--no-kv-offload` |
| 35B-A3B MoE doesn't fit | llama.cpp `--n-cpu-moe K` to DDR4 | dense-style `-ngl` partial |
| 70B-class model | dual B70 (64 GB) tensor split | majority-to-DDR4 offload |

---

# TOPIC B — MTP and speculative decoding on Intel XPU

## B.1 llama.cpp speculative decoding & native MTP

**Native MTP landed in llama.cpp via PR #22673 (merged 2026-05-16, by am17an).** ([llama.cpp PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)) Key facts verified from the PR:

- **Flag:** `--spec-type draft-mtp --spec-draft-n-max <N>` (N = draft tokens; 2–3 recommended). MTP weights load **automatically from the same GGUF** (use the `*-MTP-GGUF` builds, e.g. `ggml-org/Qwen3.6-27B-MTP-GGUF`), into a separate lightweight context/KV cache (~10 % extra memory). **Requires `--parallel 1`** (single concurrent request). ([mer.vin MTP guide](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/))
- **Backends tested in the PR: CUDA (RTX 3090/3060, DGX Spark), Vulkan (AMD R9700, Strix Halo), Metal (with limitations). SYCL is NOT in the tested list.** This is the central open question for the B70 — see B.6.
- **Reported speedup:** ~1.85–2.5× decode, acceptance 0.72–0.82 at `n-max 3`; example RTX 3090 Q6_K **22.97 → 42.45 t/s**. **Prompt processing can regress** due to extra device↔host transfers. ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673))

Classic **draft-model speculation** is also available: `llama-speculative` / `llama-server` with `--model-draft <draft.gguf>` (`-md`), `--draft-max` / `--draft-min`, `--draft-p-min`. The draft must be **vocab-matched** to the target (e.g. a Qwen3-0.6B-class draft for Qwen3.6). **Prompt-lookup / n-gram** decoding needs no model at all (`--lookup` / ngram-cache), good for high-repetition (code-edit) prompts. ([llama.cpp spec-decode discussion #10466](https://github.com/ggml-org/llama.cpp/discussions/10466))

There is also an **EAGLE-3** PR in flight for llama.cpp ([PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)) — status unconfirmed as merged; flag as in-progress.

## B.2 Are the shipped MTP weights usable per engine?

- **llama.cpp:** YES via the `*-MTP-GGUF` repackage + `--spec-type draft-mtp` (PR #22673). The MTP head is consumed natively as a self-draft. ([mer.vin](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/))
- **vLLM:** MTP is a first-class speculative method (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`, no separate draft model), but the official doc lists **Gemma 4 and MiMo-7B** as MTP-supported — **Qwen3.6 is not listed**, and on **Intel XPU MTP currently crashes** (B.6). ([vLLM MTP docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/))
- **IPEX-LLM (legacy):** does **self-speculative** decoding (auto-creates an INT4 draft from the FP16/BF16 model, no external draft, no MTP), ~30 % latency improvement on Intel GPU/CPU. Does **not** consume Qwen's MTP heads. ([IPEX-LLM self-spec doc](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Inference/Self_Speculative_Decoding.md))

## B.3 IPEX-LLM self-speculative decoding on Arc

IPEX-LLM's self-speculative decoding accelerates the original FP16/BF16 model by ~**30 %** on Intel GPU by using an auto-generated INT4 copy as the draft — no external draft, no finetune. Verified for Qwen-family. ([IPEX-LLM self-spec doc](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Inference/Self_Speculative_Decoding.md), [example script](https://github.com/intel/ipex-llm/blob/main/python/llm/example/GPU/Speculative-Decoding/Self-Speculation/qwen/speculative.py)) **Caveat:** IPEX-LLM is archived (Jan 2026); 30 % is modest vs MTP's ~2×; only worth it if MTP paths remain broken on XPU.

## B.4 Real-world speedup numbers (with provenance)

| Source | Hardware | Model | Method | Speedup | Notes |
|---|---|---|---|---|---|
| [dasroot.net](https://dasroot.net/posts/2026/05/qwen-36-27b-52-8-tok-s-on-mi50s-mtp-turboquant/) | MI50 / general | Qwen3.6-27B | SGLang NEXTN, n_spec=3 | **1.94×** | also "3× on GTX 1080 Ti" (unverified) |
| [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) | RTX 3090 | Qwen3.6-27B Q6_K | llama.cpp MTP | **~1.85×** (22.97→42.45) | accept 0.72–0.82, n-max 3 |
| [mer.vin](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/) | RTX 3090 | 27B Q6_K | llama.cpp MTP | **~1.9×** (22.4→42.5) | |
| [MTPLX](https://github.com/youssofal/MTPLX) | Apple M-series | Qwen3.6-27B | native MTP | **2.24×** decode @ temp 0.6 | **Apple only**, not Intel |
| [thc1006 / HackMD](https://hackmd.io/@thc1006/SJly6IE6Wx) | RTX 3090 | 35B-A3B MoE | classic draft (Qwen3.5-0.8B) | **net-negative** (−3…−12 %) | MoE + small target = no win |
| draft-model coding | RTX 5000 Ada | Qwen2.5-Coder | 0.6B draft | up to **4×** high-draftability; ~1.3× low | task-dependent |

**Takeaways:** (1) The ~1.5–2× claim is well-supported for **MTP on the dense 27B** on CUDA/Apple/MI50. (2) Speculation can be **net-negative** for the 35B-A3B MoE (already cheap per token). (3) **None of the strong MTP numbers are on Intel XPU/SYCL** — Intel-specific MTP speedup is currently **unverified** (and on vLLM, broken).

## B.5 Acceptance-rate considerations

Speedup ≈ a function of **acceptance rate × draft depth − verification overhead**. Higher acceptance (draft tokens the target confirms) → more tokens per verify pass. MTP self-drafts achieve high acceptance because the draft head shares the target's representations: PR #22673 reports **0.72–0.82** at `n-max 3`. Acceptance and ideal draft depth are **task-dependent** — high-draftability (code refactor, repetitive) prompts accept long runs (up to 4× speedup); low-draftability (open-ended reasoning) accept little (~1.3× or worse). Going too wide (`n-max`/`draft-max` too high) wastes compute on rejected tokens: efficiency crossover is >32 draft tokens for a 0.5B draft, ~11 for a 3B draft. **Start at `--spec-draft-n-max 2–3` and tune by measuring acceptance.** ([spec-decode #10466](https://github.com/ggml-org/llama.cpp/discussions/10466), [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673))

## B.6 EAGLE / Medusa on Intel — and the XPU MTP blocker

- **vLLM XPU supports n-gram, EAGLE, and EAGLE3** speculative methods (per Intel's vLLM-on-Arc-Pro material); **MTP and Medusa are not listed for XPU.** ([Intel/vLLM Arc Pro blog](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b), [vLLM spec-decode docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/))
- **CONFIRMED BLOCKER:** Running vLLM-XPU with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` on a Qwen3.5/3.6-class model **crashes**: `"XPU gdn_attention does not yet support 'spec_sequence_masks'"` (fatal, at `qwen3_5.py:forward_xpu`). The **Gated-DeltaNet ("gdn") attention** path — exactly what Qwen3.6 uses — lacks the spec-decode masking on XPU. **No fix/workaround as of the issue's April 2026 state.** ([intel/llm-scaler #386](https://github.com/intel/llm-scaler/issues/386))

**Net recommendation for the throughput win on B70:**
1. **First choice — llama.cpp SYCL + native MTP** (`--spec-type draft-mtp --spec-draft-n-max 2`, `--parallel 1`, `*-MTP-GGUF`). This is the cleanest path to the ~2× decode win *if* the SYCL backend honors the MTP path. **Verify on hardware — MTP is not in PR #22673's tested-backend list, so flag SYCL-MTP as unverified and benchmark it before relying on it.**
2. **If SYCL-MTP misbehaves — classic draft model** in llama.cpp: vocab-matched Qwen3-0.6B draft via `--model-draft`, `--draft-max 4`. Works on SYCL today.
3. **vLLM on XPU — use EAGLE3 or n-gram, NOT MTP** (MTP crashes on gdn). EAGLE3 needs EAGLE weights for Qwen3.6 (availability unconfirmed — flag).
4. **IPEX-LLM self-spec (~30 %)** only as a legacy fallback.

---

# TOPIC C — Benchmark sweep methodology

> **Scope caveat:** the strongest B70-specific data comes from **one community tester** ([PMZFX/intel-arc-pro-b70-benchmarks](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks)), who self-describes the methodology as *"not scientific — no thermal controls, no ambient monitoring, no power metering."* Treat all public B70 tok/s below as **anchors, not authoritative baselines** — re-measure under your own pinned harness.

## C.1 Metrics that matter (definitions + tradeoffs)

Every metric maps to one of two phases — **prefill** (whole prompt in one forward pass; **compute-bound**) and **decode** (autoregressive; **memory-bandwidth-bound**). ([Sarathi-Serve](https://arxiv.org/html/2403.02310v1))

- **PP (prefill / prompt-processing throughput, tok/s)** — input tok/s during prefill. **Explicitly reported by llama-bench (PP column)** but is *not* a separately-named serving metric in vLLM/NVIDIA docs (surfaces via TTFT). Batching barely improves prefill (already compute-saturated). ([Anyscale metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics))
- **TTFT (time to first token)** — submit→first token; in vLLM measured from arrival, so it **includes queue wait**. NVIDIA interactive SLO ≤250 ms. ([BentoML](https://bentoml.com/llm/inference-optimization/llm-inference-metrics), [NVIDIA cost blog](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/))
- **TG (decode / generation throughput, tok/s)** — output tok/s; vLLM "Output token throughput." llama-bench **TG column**. Scales ~linearly with batching. Distinguish **per-user** vs **per-GPU** tok/s. ([vLLM CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/))
- **TPOT vs ITL** — both are per-token decode speed (equal for one request); differ only in aggregation — **TPOT is request-weighted**, **ITL is token-weighted** (longer outputs contribute more samples → better steady-stream measure). ([BentoML](https://bentoml.com/llm/inference-optimization/llm-inference-metrics))
- **Aggregate throughput / goodput** — req/s and total tok/s under concurrency; **goodput** = req/s *meeting SLOs* (the real metric). Report **p50/p90/p99**, never just means. ([GuideLLM / Red Hat](https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference))

**Core tradeoffs:** latency↔throughput Pareto frontier (tune to an SLO point); concurrency raises aggregate tok/s but lowers per-user tok/s; prefill↔decode interference causes "generation stalls" — mitigated by **chunked prefill** (vLLM/SGLang default), which trades a little TTFT for much lower tail latency. ([Sarathi-Serve / OSDI'24](https://arxiv.org/html/2403.02310v1))

## C.2 Tools and exact usage

**llama.cpp `llama-bench`** (in-process microbenchmark; runs the Cartesian product of all multi-valued flags, `-r` reps each with ±stddev; **excludes tokenization/sampling**). ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md))
Flags: `-p` prefill size (512), `-n` gen size (128), `-pg pp,tg` combined, `-d` prefill depth (measure at KV depth), `-b`/`-ub` batch/ubatch (2048/512), `-ctk`/`-ctv` KV type (f16/q8_0/q4_0/…), `-ngl` (use 99), `-fa on|off|auto`, `-ts`/`-sm` split, `-t` threads, `-r` reps (5, incl. 1 warmup; `--no-warmup`), `-o csv|json|md|sql`. Multi-value: comma-list / repeated flag / range `first-last+step` or `*mult`.
```sh
# B70 single-config (PMZFX style)
ONEAPI_DEVICE_SELECTOR=level_zero:0 llama-bench -m model.gguf -ngl 99 -p 512 -n 128
# sweep ubatch + KV quant + gen length
llama-bench -m model.gguf -ngl 99 -fa on -ub 64,128,256,512 -ctk q8_0,f16 -n 128-1024+128
# PP/TG vs context depth
llama-bench -m model.gguf -ngl 99 -fa on -p 512 -n 128 -d 0,4096,16384,32768
```

**llama.cpp serving** — `llama-bench` has **no server mode**. Measure concurrency with:
- **`llama-batched-bench`** (in-process batched decode; cols PP, TG, **S_PP**, **S_TG**): `llama-batched-bench -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,512 -ntg 128 -npl 1,2,4,8,16,32`. ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/README.md))
- **`llama-sweep-bench`** (ik_llama.cpp; PP/TG **at each KV depth**, cols N_KV/S_PP/S_TG): `llama-sweep-bench -m model.gguf -c 8704 -ub 512 [--output-format jsonl]`. ([README](https://github.com/ikawrakow/ik_llama.cpp/blob/main/examples/sweep-bench/README.md))
- **`llama-server` + k6** (`tools/server/bench`, xk6-sse for streaming): `./k6 run script.js --duration 10m --iterations 500 --vus 8` (vus = concurrency, match `--parallel`). ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/bench/README.md))
Server flags: `-np/--parallel` (slots; context divides per slot), `--cont-batching`, `--metrics`, `-c`, `-b/-ub`, `-ctk/-ctv`, `-ngl`.

**vLLM `vllm bench serve`** (online, load-controlled). Install `pip install vllm[bench]`. ([CLI docs](https://docs.vllm.ai/en/latest/benchmarking/cli/))
Key args: `--backend vllm`, `--dataset-name sharegpt|random|sonnet|hf`, `--dataset-path`, `--num-prompts`, **`--request-rate`** (arrival rate; `inf`=all at t0, finite=Poisson), **`--max-concurrency`** (in-flight ceiling), `--random-input-len`/`--random-output-len`, `--burstiness`, `--metric-percentiles 99`, `--save-result`. Standard steady-state idiom: `--request-rate inf --max-concurrency 64`. Reports req/s, output/total tok/s, Mean/Median/P99 TTFT/TPOT/ITL.
```bash
vllm serve Qwen/Qwen3.6-27B --quantization awq   # XPU server
vllm bench serve --backend vllm --model Qwen/Qwen3.6-27B \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 500 --request-rate inf --max-concurrency 32 --metric-percentiles 99
```
**vLLM `vllm bench throughput`** (offline ceiling, no HTTP; **`--num-warmups` defaults to 0** — set it!): `vllm bench throughput --model … --input-len 512 --output-len 128 --num-prompts 500`.
**vLLM sweep tooling:** `vllm bench sweep serve` / `serve_workload` / `plot` / `plot_pareto` automate server-param × workload sweeps and emit Pareto frontiers. ([sweeps doc](https://docs.vllm.ai/en/latest/benchmarking/sweeps/))

**SGLang `bench_serving`** (async HTTP load tester; same `--request-rate`/`--max-concurrency` semantics; default port 30000; **rule: `--num-prompts ≥ 5 × max-concurrency`** for steady state):
```bash
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
  --model Qwen/Qwen3.6-27B --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --random-range-ratio 0.5 \
  --num-prompts 2000 --request-rate inf --max-concurrency 32 --output-details
```
Server on Arc: `--device xpu --attention-backend intel_xpu`. Also `bench_offline_throughput`, `bench_one_batch`. ([SGLang bench_serving](https://docs.sglang.io/developer_guide/bench_serving.html), [SGLang XPU](https://sgl-project.github.io/platforms/xpu.html))

**guidellm** (production-traffic simulator, any OpenAI endpoint; per-load-level RPS/TTFT/ITL/throughput distributions). `pip install guidellm`. **Current flag is `--profile`** (`--rate-type` is legacy alias — confirm with `--help`). Profiles: `synchronous|throughput|concurrent|constant|poisson|sweep`. **`sweep`** auto-finds the frontier (sync baseline → throughput peak → interpolated points).
```bash
guidellm benchmark --target "http://localhost:8000" --model "Qwen/Qwen3.6-27B" \
  --data "kind=synthetic_text,prompt_tokens=256,output_tokens=128" \
  --profile kind=sweep,sweep_size=10 --output-dir ./gllm-out
```
([guidellm repo](https://github.com/vllm-project/guidellm), [Red Hat](https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference))

## C.3 Variables worth sweeping (with B70 anchor data)

**Qwen3.5-27B on B70, llama.cpp SYCL F16, commit `ec6f7a6a5c`** (verified verbatim, [PMZFX table](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/llm-benchmarks.md)):

| Quant | Size GiB | pp512 t/s | tg128 t/s |
|---|---|---|---|
| Q4_0 | 14.63 | 243 | **23.67** |
| Q4_K_S | 14.68 | 309 | 23.05 |
| **Q4_K_M** | 15.58 | 302 | 20.56 |
| IQ4_XS | 13.94 | 267 | 17.52 |
| Q5_K_M | 18.25 | 300 | 13.78 |
| Q6_K | 20.90 | 304 | 13.83 |

(Q8_0 omitted from current table — it predated SYCL fixes [PR #21527/#21638](https://github.com/ggml-org/llama.cpp/issues/21517); the old "776 t/s" Q8_0 figure is **stale**, do not rely on it.)

1. **Quant** — sweep Q4_0 / Q4_K_M / Q5_K_M / Q6_K (Q8_0 only if KV fits). **Q4_K_M is the sweet spot** (fits + fast + minimal quality loss). Gen ∝ bytes/token, so heavier = slower. IQ4_XS is *smaller but slower* on Battlemage (XMX prefers K-quant kernels). vLLM paths: **AWQ / GPTQ-INT4 / FP8** (no FP4 on B70). VRAM fit (27B): Q4_K_M ~16.8 GB, Q5_K_M ~19.5, Q6_K ~22.5, Q8_0 ~28.6. ([quant ppl arxiv](https://arxiv.org/html/2601.14277v1), [B70 LLM hub](https://emelia.io/hub/arc-pro-b70-local-llm))
2. **n-gpu-layers** — for 27B, keep `-ngl 99` (no spill). Partial offload = cliff (Topic A).
3. **Context** — 4K/8K/16K/32K. KV scales linearly with `ctx × concurrency`; usually caps slots before weights.
4. **Batch/ubatch** — `-b ∈ {512,1024,2048}`, `-ub ∈ {64,128,256,512}`; mainly affects prefill, often a sharp `-ub` optimum + cliff — **remeasure on B70** (the cliff evidence is AMD/aggregator, flag).
5. **Flash attention** — `-fa on`: ~1.3–2× prefill, lower KV VRAM, and **required for V-cache quant** to behave. Verify SYCL FA kernels exist. ([#5932](https://github.com/ggml-org/llama.cpp/discussions/5932))
6. **KV quant** — `f16` vs `k=q8_0/v=q8_0` vs `k=q8_0/v=q4_0`. **Q8 KV is ~free (<0.1 % ppl) — make it default.** Q4 K-cache OK (~0.4 % ppl); Q4 V-cache worse (~1.4 %). Highest-leverage knob for more context/concurrency. ([#5932](https://github.com/ggml-org/llama.cpp/discussions/5932))
7. **Concurrency** — 1/4/8/16/32/50. B70 (Qwen3.5-27B, vLLM XPU): ~13 t/s single → ~370 t/s aggregate @50 (peak ~550), ~28× — single-stream gen is the weak point. Plateau = usable ceiling (KV-VRAM-bound). ([emelia B70 hub](https://emelia.io/hub/arc-pro-b70-local-llm), single-tester — flag)
8. **Prompt distribution** — fixed-length `random` (isolates variables for clean sweeps) **and** ShareGPT (realistic headline). Always report which + the input/output length distribution.

## C.4 Isolating variables / avoiding confounds

1. **Warmup** — absorbs SYCL/oneDNN JIT, autotuning, lazy module load, page-cache warming. llama-bench: 1 warmup + 5 timed; **vLLM throughput defaults `--num-warmups 0` — set it.** Don't reuse the measured prompt as warmup (pollutes prefix cache). ([Speechmatics CUDA timings](https://blog.speechmatics.com/cuda-timings))
2. **Thermal throttling** — sustained load drops clocks 10–20 %; cold-vs-stabilized swings **12–18 %**. Run to steady state, report sustained (or both burst+sustained), log continuously: `xpu-smi dump -d 0 -m 0,1,2,3 -i 1 -n N` (util/power/freq/temp); add cooldowns. ([craftrigs methodology](https://craftrigs.com/benchmarks/llama-cpp-benchmark-methodology-reproducible/), [xpu-smi guide](https://intel.github.io/xpumanager/smi_user_guide.html))
3. **Cache effects (biggest confound)** — (a) **OS page cache:** first weight load from disk vs later from RAM (TTFT can differ massively); fix cold-vs-warm and keep identical (`sync; echo 3 > /proc/sys/vm/drop_caches` for cold). (b) **Prefix/radix caching** (on by default) makes repeated identical prompts skip prefill → inflated throughput/fake-low TTFT — **disable for clean baselines** (vLLM `--no-enable-prefix-caching`, SGLang `--disable-radix-cache`) or vary prompts. ([vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/))
4. **Seeds/determinism** — pin `seed`, greedy (temp 0). **No batch-invariant/deterministic mode is documented for Intel XPU** (vLLM `VLLM_BATCH_INVARIANT=1` needs NVIDIA CC≥8.0; SGLang deterministic ~34 % slower) — assume **bitwise reproducibility is unattainable on B70**; instead hold workload fixed and report mean±stddev / percentiles. ([Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), [vLLM batch-invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/), [SGLang deterministic](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/))
5. **Repetitions/stats** — `-r 5`+ , report mean±stddev (microbench) and p50/p90/p99 (serving). Explicit, consistent outlier rule.
6. **Pin clocks/power** — Intel Arc: `xpu-smi config -d 0 -t 0 --frequencyrange MIN,MAX` (hard-pin via MIN==MAX, e.g. `1200,1200`). **Flag: single-value pin holding under sustained thermal load is unverified on B70 — validate by logging `xpu-smi dump` MHz during long runs.** ([xpu-smi guide](https://intel.github.io/xpumanager/smi_user_guide.html))
7. **Quiesce host** — dedicate GPU (no compositor/other models), CPU governor `performance`, disable turbo, pin threads NUMA-local (`numactl`, `taskset`), stop cron/services.
8. **Reproducibility manifest** — pin & record: engine commit (PMZFX uses `ec6f7a6a5c` + icx/icpx 2025.3.3), oneAPI/driver/kernel, **Docker image by digest** (`image@sha256:…`, never `:latest`), **model SHA-256**, and full config (quant, ctx, -ngl, batch, seed, threads).

### B70 / Docker specifics (June 2026)
- Use the **`xe` kernel driver** (not i915), HWE kernel 6.17+, oneAPI 2025.3.3, Level Zero. Build SYCL with `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON`. **SYCL gen ≈ 2.2× faster than Vulkan** on B70 (229 vs 102 t/s, Qwen-1.5B). ([PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks), [SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md))
- **llama.cpp Docker:** `ghcr.io/ggml-org/llama.cpp:server-intel`, run `--device /dev/dri/renderD128 --device /dev/dri/card0`, env `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `ZES_ENABLE_SYSMAN=1`, `-ngl 99`.
- **vLLM Docker:** `intel/vllm:0.10.2-xpu` (pin digest), run `--device=/dev/dri --ipc=host`.
- **Fragility (flag):** strict ABI/version alignment across the container boundary (SYCL runtime, `_GLIBCXX_USE_CXX11_ABI`, icpx) — mismatch → cryptic oneDNN error at first matmul. Community consensus: "not production-ready OOTB." ([idfs.ai B70](https://idfs.ai/blog/six-days-with-the-intel-arc-pro-b70))

---

## Recommended first sweep matrix (run these first)

**Phase 0 — microbench (llama.cpp `llama-bench`, SYCL, `-ngl 99`, `-r 5`, log `xpu-smi dump`):**

| Variable | Values to test first |
|---|---|
| Quant | **Q4_K_M**, Q4_0, Q5_K_M, Q6_K |
| Context depth `-d` | 0, 4096, 16384, 32768 |
| ubatch `-ub` | 64, 128, 256, 512 |
| Flash attn `-fa` | on (baseline), off (delta only) |
| KV quant `-ctk/-ctv` | f16, q8_0/q8_0, q8_0/q4_0 |

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:0 llama-bench -m Qwen3.6-27B-Q4_K_M.gguf \
  -ngl 99 -fa on -ub 64,128,256,512 -ctk f16,q8_0 -ctv f16,q8_0 \
  -p 512 -n 128 -d 0,4096,16384,32768 -r 5 -o json
```

**Phase 1 — concurrency/serving (`llama-batched-bench` or `vllm bench serve`, prefix-cache OFF, warmup ON):**

| Variable | Values |
|---|---|
| Concurrency | 1, 4, 8, 16, 32, 50 |
| Prompt dist | fixed `random` 1024/256 (clean) **and** ShareGPT (headline) |
| Best-quant from Phase 0 | carry forward winner |

**Phase 2 — speculative/MTP delta (carry Phase-1 best config, `--parallel 1`):**
- llama.cpp: `--spec-type draft-mtp --spec-draft-n-max 2` vs baseline → measure decode t/s + acceptance. **Verify SYCL honors MTP (unverified).**
- Fallback: `--model-draft <Qwen3-0.6B> --draft-max 4`.
- vLLM XPU: EAGLE3 or n-gram **only** (MTP crashes on gdn).

---

## Sources

**Topic A — offload**
- Qwen3.6-27B is dense: https://www.marktechpost.com/2026/04/22/alibaba-qwen-team-releases-qwen3-6-27b-a-dense-open-weight-model-outperforming-397b-moe-on-agentic-coding-benchmarks/ · https://recipes.vllm.ai/Qwen/Qwen3.6-27B
- llama.cpp MoE offload (`--n-cpu-moe`): https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide · https://dev.to/someoddcodeguy/understanding-moe-offloading-5co6
- `-ngl` cliff / CPU fallback: https://bmdpat.com/blog/llama-cpp-n-gpu-layers-explained-2026 · https://bmdpat.com/blog/llama-cpp-ngl-cpu-fallback-fix-2026 · https://medium.com/@ekansh.jain2011/squeezing-every-drop-of-performance-out-of-llama-cpp-the-practitioners-guide-to-local-ai-2bcc3663f06f
- KV offload / KV quant: https://github.com/ggml-org/llama.cpp/discussions/5932 · https://lemongravy.me/articles/intel-gpu-llamacpp/
- vLLM XPU / cpu-offload: https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/ · https://github.com/intel/ipex-llm
- ktransformers: https://github.com/kvcache-ai/ktransformers
- B70 PCIe / dual-card: https://www.phoronix.com/review/intel-arc-pro-b70-linux · https://github.com/PMZFX/intel-arc-pro-b70-benchmarks
- Q8_0 SYCL slowdown: https://github.com/ggml-org/llama.cpp/issues/21517

**Topic B — MTP / speculative**
- llama.cpp native MTP PR #22673: https://github.com/ggml-org/llama.cpp/pull/22673 · usage: https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/
- EAGLE3 PR (in-progress): https://github.com/ggml-org/llama.cpp/pull/18039
- vLLM MTP doc: https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/ · spec-decode: https://docs.vllm.ai/en/latest/features/speculative_decoding/
- **vLLM XPU MTP crash (gdn):** https://github.com/intel/llm-scaler/issues/386
- vLLM XPU spec methods (n-gram/EAGLE/EAGLE3): https://vllm.ai/blog/2025-11-11-intel-arc-pro-b
- IPEX-LLM self-spec: https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Inference/Self_Speculative_Decoding.md · https://github.com/intel/ipex-llm/blob/main/python/llm/example/GPU/Speculative-Decoding/Self-Speculation/qwen/speculative.py
- Speedup data: https://dasroot.net/posts/2026/05/qwen-36-27b-52-8-tok-s-on-mi50s-mtp-turboquant/ · https://github.com/youssofal/MTPLX · https://hackmd.io/@thc1006/SJly6IE6Wx · https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090
- Acceptance/draft theory: https://github.com/ggml-org/llama.cpp/discussions/10466

**Topic C — benchmark methodology**
- Prefill/decode + chunked prefill (Sarathi-Serve): https://arxiv.org/html/2403.02310v1
- Metrics: https://bentoml.com/llm/inference-optimization/llm-inference-metrics · https://docs.anyscale.com/llm/serving/benchmarking/metrics · https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/ · https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/
- llama-bench: https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md · batched-bench: https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/README.md · sweep-bench: https://github.com/ikawrakow/ik_llama.cpp/blob/main/examples/sweep-bench/README.md · server bench: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/bench/README.md
- vLLM bench: https://docs.vllm.ai/en/latest/benchmarking/cli/ · sweeps: https://docs.vllm.ai/en/latest/benchmarking/sweeps/
- SGLang: https://docs.sglang.io/developer_guide/bench_serving.html · https://sgl-project.github.io/platforms/xpu.html
- guidellm: https://github.com/vllm-project/guidellm · https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference
- B70 anchor data: https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/llm-benchmarks.md · https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/methodology.md
- Concurrency scaling (single-tester): https://emelia.io/hub/arc-pro-b70-local-llm
- Determinism: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ · https://docs.vllm.ai/en/latest/features/batch_invariance/ · https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/
- Warmup/thermal/cache: https://blog.speechmatics.com/cuda-timings · https://craftrigs.com/benchmarks/llama-cpp-benchmark-methodology-reproducible/ · https://docs.vllm.ai/en/stable/design/prefix_caching/
- Clock pinning / xpu-smi: https://intel.github.io/xpumanager/smi_user_guide.html · https://www.phoronix.com/news/Intel-XPU-Manager-B65-B70
- KV quant detail: https://www.techplained.com/kv-cache-quantization
- B70 Docker/setup: https://idfs.ai/blog/six-days-with-the-intel-arc-pro-b70 · https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md · TurboQuant KV (not yet SYCL): https://github.com/ggml-org/llama.cpp/discussions/20969

---
*Quantitative claims tagged "flag"/"unverified" lack a B70-specific primary source and should be re-measured. All public B70 tok/s derive from a single community tester who self-describes their method as non-scientific.*
