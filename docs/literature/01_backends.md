# LLM Inference Backends on the Intel Arc Pro B70 (Battlemage / Xe2)

**Literature base for the B70 optimization project.**
Compiled: 2026-06-17. Hardware target: Intel Arc Pro B70 (BMG-G31, Xe2/Battlemage, 32 GB GDDR6, ~608 GB/s, 32 Xe-cores / 256 XMX engines, 367 INT8 TOPS, PCIe 5.0 x16; launched 2026-03-25 at $949; PCI ID `0xe223`).

> **Skeptical reading guide.** Most numbers below come from a small number of enthusiast benchmark repos, blog posts, and GitHub issues, not controlled studies. Where two independent sources agree I note it; where a single source makes a strong claim I flag it as **[unverified]**. Battlemage software support is *young* (the card is ~3 months old as of writing) and several backends are demonstrably immature or broken for specific quant/model combinations. Treat all tokens/sec figures as order-of-magnitude until you reproduce them on your own driver stack.

---

## 1. Executive picture

- **For a single B70 running a 7B-32B model: `llama.cpp` is the clear front-runner**, and the **SYCL** backend (built with `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON`) is the fastest path for decode. Multiple independent sources put SYCL decode at **~2.2x Vulkan** on the same hardware. ([PMZFX benchmarks](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/llm-benchmarks.md), [lemongravy](https://lemongravy.me/articles/intel-gpu-llamacpp/), [Bibek Poudel](https://bibek-poudel.medium.com/how-to-run-qwen3-6-27b-locally-on-intel-arc-pro-b70-what-actually-works-c96dec67c6f7))
- **BUT** there is a credible counter-report ([llama.cpp #22413](https://github.com/ggml-org/llama.cpp/issues/22413)) where SYCL was **3x *slower* than Vulkan** on an older DDR4 / PCIe 3.0 host. The SYCL-vs-Vulkan winner appears to be **host- and driver-sensitive**. You must benchmark both on your own box before committing.
- **The XMX INT8 fast path (367 TOPS) is largely NOT being exercised by llama.cpp today.** Decode is memory-bandwidth-bound, and the INT8 (Q8_0) path on Xe2 is actually a *regression* relative to INT4 ([#21517](https://github.com/ggml-org/llama.cpp/issues/21517)). The headline INT8 TOPS number is essentially a prefill/GEMM spec that current GGUF decode kernels don't reach.
- **vLLM-XPU / Intel LLM-Scaler** works on B70 (official support landed in `llm-scaler-vllm 0.14.0-b8.2`, April 2026) and is the right choice for an **OpenAI-compatible server, batching, and multi-GPU tensor parallelism** — but per-request decode speed is currently *worse* than llama.cpp (~14-16 t/s on 2xB70 for a 27-31B model vs 20+ t/s single-card llama.cpp).
- **IPEX-LLM is archived (read-only since 2026-01-28).** Its portable llama.cpp/Ollama zips still exist but are unmaintained; Intel folded XPU support upstream into PyTorch ≥2.8. Do not build new work on IPEX-LLM.

---

## 2. Backend comparison table

| Backend | Maturity on B70 | Best quant on B70 | Reported single-card decode (≈27-31B, 4-bit) | XMX INT8 used? | Verdict |
|---|---|---|---|---|---|
| **llama.cpp SYCL** (`GGML_SYCL`) | Working, actively optimized; some kernels still memory-bound | **Q4_K_M / Q4_K_S** (avoid Q8_0, IQ4_NL) | **~20-22 t/s** dense (Qwen 3.5/3.6-27B), **42-55 t/s** MoE (35B-A3B) | Partially in prefill (F16 GEMM); **not** in decode | **Bet here first** for single user |
| **llama.cpp Vulkan** | Working, easiest setup (no oneAPI) | Q4_K_M | **~14 t/s** (Bibek) / faster than SYCL on some hosts | No (vendor-agnostic) | Fallback / quick start; host-dependent |
| **vLLM-XPU via Intel LLM-Scaler** | Official B70 support since 0.14.0-b8.2 (Apr 2026); rough edges | AWQ INT4, AutoRound INT4, FP8 (online); GPTQ auto-detected | **~14-16 t/s** per-request on **2xB70** TP (Gemma4-31B, Qwen3.5-27B) | FP8/INT8 paths exist; AWQ-INT8 kernel **missing** | Use for **serving / batching / multi-GPU**, not single-stream speed |
| **OpenVINO GenAI / OVMS** | Immature for newest models | OpenVINO INT4 IR | — (model-load failures common) | INT8 IR possible | Watch, don't rely on yet |
| **SGLang XPU** | Source-only, Battlemage is a *roadmap target* (2026 Q2) | follows vLLM-style | not benchmarked on B70 | planned | Too early |
| **IPEX-LLM (portable zips)** | **Archived 2026-01-28**, read-only | sym_int4 / sym_int8 / fp8 | ~15-20 t/s (14B, portable) | claimed INT8 path | **Avoid for new work** |
| **Standard Ollama (upstream)** | No native Arc accel — runs on **CPU** | — | CPU speed | No | Do not use directly; use IPEX-LLM's Ollama zip or llama.cpp server |

> Decode numbers above are **TG (token generation / decode)**, the user-facing speed. Prefill (PP) is much higher (hundreds-to-thousands of t/s) and benefits more from XMX; see §5.

---

## 3. llama.cpp on Battlemage — the detail that matters

### 3.1 SYCL vs Vulkan (contradictory evidence — benchmark your own host)

| Source / host | Model | Vulkan TG | SYCL TG | Notes |
|---|---|---|---|---|
| [PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/llm-benchmarks.md) (B70) | Qwen2.5-1.5B Q4_K_M | 102 t/s | **229 t/s** | SYCL **2.24x** faster decode; PP roughly tied (~0.97x) |
| [Bibek Poudel](https://bibek-poudel.medium.com/how-to-run-qwen3-6-27b-locally-on-intel-arc-pro-b70-what-actually-works-c96dec67c6f7) (B70) | Qwen3.6-27B Q4_K_M | 14 t/s | **22 t/s** | SYCL **+52%** |
| [artificialintelligence.dk](https://artificialintelligence.dk/b70-blog-alex-v5.html) (B70) | Gemma4-26B-A4B Q4_K_XL | 41.25 t/s | **55.77 t/s** | SYCL wins |
| [llama.cpp #22413](https://github.com/ggml-org/llama.cpp/issues/22413) (B50/B70, **DDR4 / PCIe 3.0**) | (small) | **40.2 t/s** | 13.3 t/s | **Vulkan 3x faster** — SYCL near CPU speed |

**Interpretation:** On a modern host (DDR5, PCIe 5.0, recent kernel + oneAPI) SYCL decode wins clearly. On an old host (DDR4 / PCIe 3.0) SYCL can collapse to CPU-like speed — issue #22413's reporter hypothesizes SYCL/Level-Zero leans harder on host memory and PCIe bandwidth. **Action: verify your host is PCIe 5.0 / DDR5 before assuming SYCL will win.**

### 3.2 The F16-accumulation flag is the single biggest prefill lever

Building with **`-DGGML_SYCL_F16=ON`** (F16 accumulators) gave PMZFX **+139% prefill** on Qwen-27B Q8_0 (296 → 707 t/s PP512) with **~0% decode regression**. This is where Xe2's XMX engines actually get exercised on prompt-processing GEMMs. Newer PMZFX builds enable it by default. **Always build with F16 accumulation.** ([PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/llm-benchmarks.md))

### 3.3 Quant choice on Xe2 — counter-intuitive: INT8 is a trap right now

[llama.cpp #21517](https://github.com/ggml-org/llama.cpp/issues/21517) documents that on B70:

- **Q8_0 reaches only 21-24% of the ~608 GB/s bandwidth; Q4_K_M reaches 53-64%.** Result: Q8_0 decode is **~4x slower than Q4_K_M** despite holding only 1.7x more data.
- Cause is **kernel-algorithmic**, not capacity: Q8_0 uses the generic DMMV path; Q4_K_M/Q4_0/Q4_K_S use optimized MMVQ+reorder kernels. **IQ4_NL is also broken (~14% BW).** This is an **Xe2 regression** — Arc A770 (Xe1) does *not* show it.
- There has since been a **Q8_0 optimization** (referenced in [discussion #12570](https://github.com/ggml-org/llama.cpp/discussions/12570) and PMZFX's "post-fix" Q8_0 = 776 PP / 15.3 TG, ~66% BW) — a ~3.1x speedup — **but Q8_0 still trails Q4_K_M for decode.** **[Partially resolved — verify on your build.]**

**Practical rule today: prefer Q4_K_M (or Q4_K_S / Q4_0). Do not reach for Q8_0/INT8 GGUF expecting the XMX INT8 TOPS to pay off — they don't in the decode path.**

### 3.4 Flash attention

- Older guidance ([#12570](https://github.com/ggml-org/llama.cpp/discussions/12570)) says flash-attn was **not implemented for the Intel/SYCL path**.
- Newer working setups pass **`--flash-attn on`** and an env flag **`GGML_SYCL_ENABLE_FLASH_ATTN=1`** ([lemongravy](https://lemongravy.me/articles/intel-gpu-llamacpp/), [Hal9000AIML](https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-ubuntu-server)). **[Status evolving — test whether `-fa on` helps or silently no-ops on your build.]**

### 3.5 Other llama.cpp landmines

- **Multi-GPU:** row-split **segfaults** on SYCL; use **layer-split** instead. Dual-B70 layer-split scales poorly for dense models (Gemma4-31B: 20.79 → 21.53 t/s, basically flat) but enables 70B/80B that don't fit on one card. ([PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/llm-benchmarks.md))
- **First-request JIT:** SYCL pays a **~27 s one-time kernel-compile** on first inference. Set `SYCL_CACHE_PERSISTENT=1` to persist (note: Hal9000AIML actually sets it to `0` for benchmarking reproducibility — flip to `1` for production). ([lemongravy](https://lemongravy.me/articles/intel-gpu-llamacpp/))
- **4 GB allocation cap:** default Level-Zero limit blocks full 32 GB residency. Set **`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`**. ([lemongravy](https://lemongravy.me/articles/intel-gpu-llamacpp/), [Hal9000AIML](https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-ubuntu-server))
- **Immediate command lists:** `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` for direct dispatch. ([lemongravy](https://lemongravy.me/articles/intel-gpu-llamacpp/))
- **Vulkan on Windows:** VRAM fragmentation reportedly limits usable VRAM to **~21 GB of 32 GB**. ([#22413](https://github.com/ggml-org/llama.cpp/issues/22413))
- **Multi-GPU device selection:** `export GGML_VK_VISIBLE_DEVICES=1` (Vulkan) to stop vision-projector models splitting across cards. ([Bibek](https://bibek-poudel.medium.com/how-to-run-qwen3-6-27b-locally-on-intel-arc-pro-b70-what-actually-works-c96dec67c6f7))
- **Qwen3.6 hybrid (attention+recurrent) layers** break cache rollback; community patch PR #22534 prevents destructive cache clears. ([lemongravy](https://lemongravy.me/articles/intel-gpu-llamacpp/))

---

## 4. vLLM-XPU / Intel LLM-Scaler

- **Official B70 support arrived in `intel/llm-scaler-vllm 0.14.0-b8.2`** (Phoronix, April 2026). Pin an **exact beta tag** from Releases — do **not** use `latest`. B60/B70 image: `intel/llm-scaler-vllm:<VERSION>` (e.g. `0.14.0-b8.3.1`); a stable `:1.0` "PV" image also exists. ([Phoronix](https://www.phoronix.com/news/Intel-LLM-Scaler-vllm-0.14-b8.2), [llm-scaler README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md))
- **Quant formats:** online **FP8** and **INT4 (sym_int4)**; **GPTQ/AWQ auto-detected** from model config; **MXFP4** for gpt-oss only. **AWQ INT8 lacks a WNA16 kernel → fails.** AutoRound INT4 may need `--allow-deprecated-quantization`. ([artificialintelligence.dk](https://artificialintelligence.dk/b70-blog-alex-v5.html))
- **Key flags:** `-tp=<N>` (tensor parallel), **`--enforce-eager`** (required — XPU graph capture is disabled for comms ops, so CUDA-graph-style speedups don't apply), `--quantization fp8`, `--gpu-memory-util=0.9`, `--max-model-len`. Env: `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1` (avoid OOM during online quant), `ZES_ENABLE_SYSMAN=1`. Multi-GPU needs a **`--privileged`** container for oneCCL. ([llm-scaler README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md), [Hal9000AIML](https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-ubuntu-server))
- **Tensor-parallel divisibility gotcha:** TP degree must divide the attention-head count. Gemma4-31B has 32 heads → TP=2 OK, **TP=3 fails before inference**. ([artificialintelligence.dk](https://artificialintelligence.dk/b70-blog-alex-v5.html))
- **Reported throughput:** ~14-16 t/s per-request decode on **2xB70** (Gemma4-31B INT4 14.2-14.3 t/s; Qwen3.5-27B AWQ-INT4 15.65 t/s). Multi-card aggregate/batched is the selling point: Hal9000AIML claims **140 t/s on 2xB70** and **540 t/s on 4xB70** aggregate **[unverified — single repo; the 2xB70 figure was "swap-bottlenecked on 16 GB RAM"]**. ([artificialintelligence.dk](https://artificialintelligence.dk/b70-blog-alex-v5.html), [Hal9000AIML](https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-ubuntu-server))
- **GPTQ regression to know:** upstream **vLLM v0.19.0 broke GPTQ on XPU** (`AttributeError: ... 'gptq_shuffle'`) by removing the XPU branches in `gptq.py` that worked in v0.17; `gptq_marlin` also fails. Fix exists only in a community fork; no upstream timeline. **If you want GPTQ on B70 via plain vLLM, stay on v0.17.x or use LLM-Scaler's image rather than vanilla 0.19.** ([vLLM #39474](https://github.com/vllm-project/vllm/issues/39474)) — **good open issue for a knowledgeable contributor.**

---

## 5. Reference performance table (single B70, llama.cpp SYCL, Q4_K_M, F16 accum)

From [PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/llm-benchmarks.md) (commit `ec6f7a6`, 2026-04-21):

| Model | Quant | PP512 (t/s) | TG128 (t/s) | GPUs |
|---|---|---|---|---|
| Llama 3.1-8B | Q4_K_M | 2,452 | 82.6 | 1 |
| Phi-4 14B | Q4_K_M | 1,424 | 43.7 | 1 |
| Mistral Small 3.2-24B | Q4_K_M | 994 | 30.1 | 1 |
| **Qwen 3.5-27B** | **Q4_K_M** | **718** | **20.4** | **1** |
| Gemma 4 31B | Q4_K_M | 601 | 21.7 | 1 |
| Qwen 3.6-35B-A3B (MoE) | UD-Q4_K_M | 615 | **54.7** | 1 |
| Gemma 4 26B-A4B (MoE) | Q4_K_M | 1,129 | **52.6** | 1 |
| Llama 3.3-70B | Q4_K_M | 338 | 11.5 | 2 |
| Qwen3-Coder-Next 80B-A3B (MoE) | Q4_K_M | 305 | 43.4 | 2 |

Quant sweep on Qwen3.5-27B (illustrates §3.3): Q4_K_S 23.0 / Q4_0 23.7 / **Q4_K_M 20.6** / Q6_K 13.8 / **Q8_0 (pre-fix) ~4.9, (post-fix) 15.3** t/s decode.

A separate single-card SYCL result from [lemongravy](https://lemongravy.me/articles/intel-gpu-llamacpp/): **Qwen3.6-35B-A3B Q4_K_XL → PP512 977 t/s, TG128 70.5 t/s, sustained 62-64 t/s** on an i5-12400F host — higher than PMZFX's MoE numbers, plausibly newer commit + tuned env. **[single source, but config fully documented]**

**Takeaway for a 27B dense model:** budget **~20 t/s decode** at Q4_K_M single-card; a **35B-A3B-class MoE at ~50-65 t/s** is dramatically faster if a MoE model is acceptable for your use case.

---

## 6. Driver / firmware / runtime stack (version-sensitive)

Battlemage LLM performance is **highly sensitive** to the GPU compute stack. Versions seen in working/benchmark setups:

| Component | Known-good versions | Source |
|---|---|---|
| Linux kernel | **6.17+** (xe driver); Phoronix used 7.0 | Hal9000AIML; Phoronix |
| Mesa (for Vulkan) | 26.0 | Phoronix |
| Compute Runtime (NEO) | **25.40.35563.10** (lemongravy) / **26.09.37435.1** (PMZFX, Phoronix) — install from GitHub, not APT | lemongravy; PMZFX; #21517 |
| Intel Graphics Compiler (IGC) | 2.20.5 / **2.30.1** | lemongravy; PMZFX |
| Level-Zero loader | 1.28.2 | lemongravy |
| oneAPI / DPC++ | **2025.3.3** | lemongravy; #21517 |
| Distro | Ubuntu 25.10 detects B70 OOB; 24.04 needs HWE kernel; LLM-Scaler wants 24.04; build on 26.04 seen | Bibek; llm-scaler README; Phoronix |

**Sensitivities & bugs to watch:**
- The **Q8_0/IQ4_NL kernel regression** (§3.3) is tied to the SYCL kernel code, not driver — but the Q8_0 "fix" requires a recent llama.cpp commit. Pin your commit.
- **`unset OCL_ICD_VENDORS`** if IPEX-LLM-style OpenCL conflicts appear. ([IPEX-LLM BMG quickstart](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/bmg_quickstart.md))
- **Resizable BAR must be enabled in BIOS** for B-series. ([IPEX-LLM BMG quickstart](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/bmg_quickstart.md))
- Windows IPEX-LLM driver floor: **32.0.101.6449 / 32.0.101.6256+**. ([IPEX-LLM BMG quickstart](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/bmg_quickstart.md))

---

## 7. Is the XMX INT8 fast path actually used? (the core question for this project)

**Short answer: mostly no, for decode — and that's the optimization opportunity.**

- The 367 INT8 TOPS is a **matrix-engine GEMM** spec. It shows up in **prefill** when F16/INT8 GEMMs feed the XMX engines (hence the +139% prefill from `GGML_SYCL_F16`). It does **not** translate to decode, which is a tall-skinny mat-vec (GEMV) and is **memory-bandwidth-bound**, capping at ~50-65% of 608 GB/s on good kernels. ([PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/llm-benchmarks.md), [#21517](https://github.com/ggml-org/llama.cpp/issues/21517))
- llama.cpp on Xe2: **DP4A support was added but gave "negligible gains" — kernels stayed memory-bound**, and the joint_matrix/DPAS path was deemed unsuitable for current GGUF data layouts. Future work is explicitly **DPAS/DPASW targeting + quant layout redesign**. ([discussion #12570](https://github.com/ggml-org/llama.cpp/discussions/12570))
- vLLM-XPU exposes FP8/INT8 online quant and AWQ/GPTQ paths that *can* hit XMX for prefill/batched GEMM, but **AWQ-INT8 has no WNA16 kernel** and **GPTQ is broken in upstream 0.19**. So even on vLLM the INT8 XMX path is patchy. ([artificialintelligence.dk](https://artificialintelligence.dk/b70-blog-alex-v5.html), [vLLM #39474](https://github.com/vllm-project/vllm/issues/39474))

**How to confirm XMX/INT8 utilization yourself:**
1. `intel_gpu_top` / `xpu-smi`/XPU-Manager — watch the **XMX / matrix-engine** activity counter vs the EU/vector counter during prefill vs decode.
2. Compute **achieved BW** = (model bytes read per token x t/s) and compare to 608 GB/s. If decode sits at 50-65%, you're memory-bound (expected); a low number (Q8_0 ~21%) signals a bad kernel.
3. **Level-Zero / oneAPI profiling** (`ze_tracer`, VTune GPU) to see whether `dpas`/`joint_matrix` instructions are emitted.
4. Compare PP512 with `GGML_SYCL_F16` ON vs OFF — a large prefill jump = XMX is engaged for GEMM.

---

## 8. Recommended starting configurations

### 8.1 Single B70, 27B dense at 4-bit — *bet here first*
```bash
# Build (Linux, oneAPI 2025.3.3 sourced)
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON
cmake --build build -j

# Env
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1          # persist the ~27s JIT
export ZES_ENABLE_SYSMAN=1

# Run (Qwen-27B Q4_K_M)
./build/bin/llama-server -m Qwen3.5-27B-Q4_K_M.gguf \
  -ngl 99 --flash-attn on --ctx-size 8192 \
  --threads 1 --parallel 1
```
Expect ~20 t/s decode, ~700 t/s prefill. **Quant: Q4_K_M (or Q4_K_S). Do NOT use Q8_0 expecting INT8 speedup.**

### 8.2 If 8-bit is a hard requirement
- On **llama.cpp**: use a **recent commit with the Q8_0 fix** and re-measure; Q8_0 still likely trails Q4_K_M. Consider **Q6_K only if quality demands it** (slower).
- Better 8-bit story is **vLLM-XPU FP8 online quant** (`--quantization fp8`), which can engage XMX for batched GEMM — but verify per-request decode isn't worse than llama.cpp Q4_K_M for your single-stream use case.

### 8.3 OpenAI-compatible server / batching / multi-GPU
```bash
docker run --privileged --device /dev/dri \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  intel/llm-scaler-vllm:0.14.0-b8.3.1 \
  vllm serve <model-AWQ-int4> -tp 2 --enforce-eager \
  --gpu-memory-util 0.9 --max-model-len 8192
```
Pin an exact beta tag. Mind TP-divisibility (TP must divide head count).

### 8.4 Quick start, no oneAPI
llama.cpp **Vulkan** build, `-ngl 999 --jinja`. Easiest, ~30% slower decode on a modern host; may be *faster* on an old DDR4/PCIe3 host.

---

## 9. "Things to try" / open research threads for the optimization project

1. **Reproduce the SYCL-vs-Vulkan crossover** on the actual project host; characterize how much is PCIe-gen / DDR-gen dependent (test the #22413 hypothesis directly).
2. **Profile decode to confirm the memory-bound ceiling** (achieved BW vs 608 GB/s) and identify which kernels miss it (Q8_0, IQ4_NL, IQ-quants generally).
3. **Push the XMX/DPAS work**: the upstream llama.cpp need (per #12570) is **DPAS/DPASW kernels + GGUF quant layout suited to joint_matrix**. This is a concrete, high-value contribution area for INT8.
4. **Validate / extend the Q8_0 "post-fix"** — measure whether it now beats Q4_K_M anywhere and where it still lags.
5. **Test flash-attention actually engages** on the SYCL build (`-fa on` + `GGML_SYCL_ENABLE_FLASH_ATTN=1`) and quantify long-context benefit.
6. **vLLM-XPU FP8 vs llama.cpp Q4_K_M**: head-to-head on single-stream and batched throughput for the target 27B model.
7. **Sweep oneAPI / compute-runtime / IGC versions** — pin the combination that maximizes both PP and TG; document regressions.
8. **MoE alternative:** evaluate whether a **35B-A3B-class MoE at ~50-65 t/s** can replace the 27B dense target — 2.5-3x the decode speed for similar VRAM.
9. **Contribute to / track open issues:** vLLM #39474 (GPTQ XPU), llama.cpp #22413 (SYCL host-sensitivity), #21517 (Q8_0 kernel). All are live places a knowledgeable user can add data or PRs.

---

## 10. Sources

- PMZFX — Intel Arc Pro B70 benchmark repo (llama.cpp SYCL, quant sweeps, multi-GPU): https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/llm-benchmarks.md
- lemongravy — "Taming the Battlemage: 63 t/s on Qwen 3.6 with Intel SYCL" (full config, env vars, flags): https://lemongravy.me/articles/intel-gpu-llamacpp/
- Bibek Poudel (Medium) — "How to Run Qwen3.6-27B on Arc Pro B70: What Actually Works" (SYCL vs Vulkan, IPEX-LLM archived note): https://bibek-poudel.medium.com/how-to-run-qwen3-6-27b-locally-on-intel-arc-pro-b70-what-actually-works-c96dec67c6f7
- artificialintelligence.dk — "The Intel Arc Pro B70 Local LLM Landscape" (backend ranking, vLLM/OpenVINO/OpenArc/LLM-Scaler results, TP gotchas): https://artificialintelligence.dk/b70-blog-alex-v5.html
- llama.cpp Issue #22413 — "brutally bad SYCL performance on Battlemage" (Vulkan 3x faster on DDR4/PCIe3): https://github.com/ggml-org/llama.cpp/issues/22413
- llama.cpp Issue #21517 — "Q8_0 ~4x slower than Q4_K_M on B70 (kernel efficiency)": https://github.com/ggml-org/llama.cpp/issues/21517
- llama.cpp Discussion #12570 — "Current status of Intel Arc GPUs for llama.cpp" (DP4A, DPAS roadmap, Q8_0 fix): https://github.com/ggml-org/llama.cpp/discussions/12570
- llama.cpp SYCL backend docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md
- Hal9000AIML — Arc Pro B70 vLLM tensor-parallel server setup (2x/4x B70, env vars, flags): https://github.com/Hal9000AIML/arc-pro-b70-inference-setup-ubuntu-server
- Intel LLM-Scaler vLLM README (Docker tags, quant formats, flags): https://github.com/intel/llm-scaler/blob/main/vllm/README.md
- Phoronix — "Intel LLM-Scaler vllm-0.14.0-b8.2 With Official Arc Pro B70 Support": https://www.phoronix.com/news/Intel-LLM-Scaler-vllm-0.14-b8.2
- Phoronix — "Intel Arc Pro B70 Benchmarks With LLM/AI, OpenCL, OpenGL & Vulkan" (driver/kernel/runtime versions; OpenVINO + llama.cpp): https://www.phoronix.com/review/intel-arc-pro-b70-linux
- vLLM Issue #39474 — "GPTQ models fail to load on Intel XPU in v0.19.0": https://github.com/vllm-project/vllm/issues/39474
- IPEX-LLM repo (archived 2026-01-28): https://github.com/intel/ipex-llm
- IPEX-LLM Battlemage quickstart (driver floor, ResBAR, OCL_ICD_VENDORS): https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/bmg_quickstart.md
- IPEX-LLM deprecation discussion (folded into PyTorch ≥2.8): https://github.com/intel/ipex-llm/issues/13325
- SGLang XPU platform docs / Intel XPU roadmap: https://docs.sglang.io/platforms/xpu.html , https://github.com/sgl-project/sglang/issues/8309
- Intel — "Run LLMs on Intel GPUs Using llama.cpp": https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html

> **Reliability note:** PMZFX, lemongravy, Hal9000AIML, Bibek, and artificialintelligence.dk are enthusiast/individual sources; their absolute numbers vary with commit + driver + host and should be reproduced. The GitHub issues (#22413, #21517, #12570, #39474) and Phoronix/Intel docs are the most trustworthy for *behavioral* claims (which paths break, which flags matter), even if specific t/s figures need re-measurement.
