# Multi-GPU LLM Inference on Intel Arc Pro B70 (Battlemage / Xe2)

**Scope:** Planning a 2× Intel Arc Pro B70 (32 GB each, Xe2/Battlemage, BMG-G31) inference host on Unraid/Docker, with the two cards in **PCIe 3.0 x16 slots** (no Xe-Link, no NVLink-equivalent). Research date: June 2026.

**Bottom line up front:** On Intel Arc, multi-card LLM inference works today but is **immature and communication-bound**. There is **no GPU-to-GPU P2P** on these cards — every cross-card transfer round-trips through host RAM over PCIe. That makes **pipeline/layer-split parallelism the safe default** and **tensor parallelism a latency win only for MoE models or once software stabilizes**. A second B70 primarily buys you **capacity** (bigger models, longer context, or two independent model instances) far more reliably than it buys you **single-request speed**.

> Skeptical note up front: most "2× / 4× B70 = N× throughput" claims in vendor/forum posts conflate *aggregate throughput at high concurrency* with *single-request scaling*. They are not the same thing, and the difference matters enormously on slow PCIe. Numbers below are flagged by source quality.

---

## 0. Hardware reality check

| Spec | Intel Arc Pro B70 | Notes |
|---|---|---|
| Architecture | Xe2 "Battlemage", BMG-G31 die | Full big-Battlemage silicon |
| VRAM | 32 GB GDDR6, 256-bit | 608 GB/s per card |
| XMX engines | 256 | 367 TOPS INT8 |
| FP32 | ~22.9 TFLOPS | |
| Host interface | **PCIe 5.0 x16** (card capability) | TBP 230 W, MSRP $949, released 2026-03-25 |
| Inter-GPU link | **None** (no Xe-Link on Pro B-series) | All cross-card traffic over PCIe via host |

Sources: [Intel B70 product page](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html), [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/intel-arc-pro-b70-and-arc-pro-b65-gpus-bring-32gb-of-ram-to-ai-and-pro-apps-bigger-battlemage-finally-arrives-but-its-not-for-gaming), [ServeTheHome](https://www.servethehome.com/intel-announces-arc-pro-b70-and-b65-video-cards-big-battlemage-brings-big-memory-for-ai-workstations/).

### ⚠️ Your specific constraint: PCIe **3.0** x16, not 5.0

This is the single most important fact for your build. The B70 *card* is PCIe 5.0 x16, but your **slots are PCIe 3.0 x16 (~16 GB/s per direction)**. The published dual-B70 community benchmarks (PMZFX, Zing) ran on **PCIe 5.0 x8 (~32 GB/s)** — i.e. roughly **2× the inter-card bandwidth you will have**. Treat every tensor-parallelism scaling number you read elsewhere as an *optimistic ceiling*: your effective cross-card bandwidth will be ~half theirs, so TP all-reduce penalties will be **worse** on your host, and the case for pipeline/layer-split over tensor-parallel is **stronger** for you than for the sources cited here.

PCIe generation bandwidth (x16, one direction, theoretical): Gen3 ≈ 16 GB/s, Gen4 ≈ 32 GB/s, Gen5 ≈ 64 GB/s. The cited dual-card reports used Gen5 **x8** ≈ 32 GB/s ([Zing forum](https://www.zingnex.cn/en/forum/thread/intel-arc-pro-b70-gpu-llm-vllm), [PMZFX multi-gpu.md](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/multi-gpu.md)).

---

## 1. Inter-GPU communication on Intel (the crux)

**There is no peer-to-peer (P2P) DMA between Arc cards.** The most direct evidence is vLLM issue [#41663](https://github.com/vllm-project/vllm/issues/41663), reporting on **two real Arc Pro B70s**: the devices report `p2p_access:0` and **no Xe-Link**. All collective traffic for tensor parallelism therefore copies GPU buffers **to host RAM and back** over PCIe.

How Intel's stack does collectives:

- **oneCCL** (oneAPI Collective Communications Library) provides `all-reduce`/`all-gather` for TP/PP. Unlike NVIDIA NCCL (kernels run entirely on-GPU and can drive the NIC/NVLink directly), **oneCCL is host-driven**: CPU worker threads schedule and advance the collective; Level-Zero kernels do not independently initiate transfers ([arXiv "Landscape of GPU-Centric Communication"](https://arxiv.org/html/2409.09874v3)). This adds CPU/host-stack overhead on every step.
- oneCCL exposes two transport modes in the Intel vLLM/llm-scaler stack:
  - `p2p` mode — "GPU memory direct access" *when available*. On Arc cards without P2P, the win is limited.
  - `usm` mode — explicitly via host (Unified Shared Memory), used for multi-node.
  - Toggle: `ONECCL_BINDINGS_FOR_PYTORCH_ENV_MODE=p2p|usm` ([llm-scaler DeepWiki](https://deepwiki.com/intel/llm-scaler/2.5-multi-gpu-and-parallelism)).
- oneCCL's default `topo` all-reduce algorithm **copies GPU-buffer data to host** for non-topo paths; recent GPU-aware MPI work shows direct Level-Zero/IPC paths can beat oneCCL when host intervention is the limiter ([arXiv 2409.09874](https://arxiv.org/html/2409.09874v3)).

**Stability is currently poor for TP=2 on Battlemage.** vLLM [#41663](https://github.com/vllm-project/vllm/issues/41663) (Nov 2025, `intel/vllm:0.17.0-xpu`, kernel 6.17, GuC 70.44.1) documents general-protection faults + `xe` BCS engine resets during `ProcessGroupXCCL` init for TP=2. Working around it required a stack of flags:
- `CCL_ENABLE_SYCL_KERNELS=0` (disable SYCL collective kernels — the only stable path)
- `SYCL_UR_USE_LEVEL_ZERO_V2=0` (fall back to Level-Zero V1)
- `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` (avoid BCS resets)
- `CCL_ALLREDUCE=ring` avoided a `topo`-algorithm crash **but tanked performance**
- `--enforce-eager` + `VLLM_XPU_ENABLE_XPU_GRAPH=0` to dodge Dynamo/Inductor compile failures

Even with the stable workaround the result was **~362 tok/s aggregate at 50 concurrency** on Qwen3-30B-A3B (an MoE) FP8 — and the issue was **still open/unresolved**, pending re-test on Intel's exact validation BOM (Ubuntu 25.10 / specific KMD). This is the clearest signal that **dual-B70 TP in vLLM-XPU is not yet turnkey** as of early-mid 2026.

### PCIe bandwidth math for TP (why it bottlenecks)

Tensor parallelism does an **all-reduce per transformer layer, per forward pass**. For a dense ~27–70B model that is dozens of all-reduces per token. General references put PCIe TP all-reduce overhead at a level that "often negates the compute savings from adding more GPUs," vs NVLink at 600–900 GB/s ([Spheron NVLink explainer](https://www.spheron.network/blog/what-is-nvlink-gpu-interconnect-bandwidth-explained/), [willitrunai multi-GPU guide](https://willitrunai.com/blog/multi-gpu-llm-inference-guide), [Flash Communication, arXiv 2412.04964](https://arxiv.org/html/2412.04964v1)).

Rough per-token estimate (2-GPU all-reduce, hidden size *h*, dtype 2 bytes, *L* layers, ring all-reduce ≈ 2×(N−1)/N × payload):
- For a 27B-class model (*h*≈5120, *L*≈48): each all-reduce payload ≈ `h × 2 bytes` ≈ 10 KB *per token per layer*, ×48 layers ×2 (down-proj + attn out) ≈ **~1 MB of cross-card traffic per token at batch=1**, plus fixed oneCCL/host launch latency on every one of those ~96 collectives.
- The killer at low batch is **not raw bytes** (1 MB at 16 GB/s = ~60 µs) but **latency × count**: ~96 host-mediated, CPU-launched collectives per token, each with microsecond-to-tens-of-microsecond launch + host-copy overhead, serialized into the critical path. This is exactly why Intel/community guidance is `--enforce-eager` + ring-allreduce "works but is slow."
- At **high batch/concurrency** the payload grows with batch but the *per-collective fixed cost amortizes*, so aggregate throughput scales much better than single-request latency. This is the regime where the "362 tok/s @ 50 concurrency" and "140/540 tok/s" numbers live.

**Implication for PCIe 3.0:** halving the link bandwidth vs the Gen5-x8 reports roughly doubles the bytes-portion of TP cost and does nothing to help the latency-portion. TP single-request latency on your box will be **worse than the already-unimpressive cited numbers**.

---

## 2. Framework support matrix (Battlemage multi-card, mid-2026)

| Framework | TP (tensor) | PP / layer-split | Multi-card maturity on Battlemage | Notes |
|---|---|---|---|---|
| **llama.cpp (SYCL)** | ❌ Not implemented for SYCL (`row` split "under development"/crashes on master) | ✅ `--split-mode layer` (default) works | **Most stable multi-card path today** | `--tensor-split`, `--main-gpu` supported. Tensor/`row` mode SYCL = crash. ([SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md), [multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md), [PMZFX FINDINGS](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/FINDINGS.md)) |
| **vLLM-XPU / Intel vLLM (llm-scaler)** | ⚠️ TP=1/2/4 "validated" but **fragile** (see #41663) | ⚠️ PP "beta" for online serving (needs Ray) | **Promising but buggy** on B70 | Requires `--enforce-eager`, `distributed_executor_backend=mp`, `VLLM_WORKER_MULTIPROC_METHOD=spawn` for BMG. ([llm-scaler](https://deepwiki.com/intel/llm-scaler/2.5-multi-gpu-and-parallelism), [vLLM XPU docs](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/), [#41663](https://github.com/vllm-project/vllm/issues/41663)) |
| **IPEX-LLM** | ✅ TP via DeepSpeed-AutoTP (`--tensor-parallel-size`), needs `libfabric-dev` | partial | A770-era validated; **less B-series multi-card evidence** | Also wraps vLLM/llama.cpp. ([ipex-llm](https://github.com/intel/ipex-llm), [Deepspeed-AutoTP example](https://github.com/intel/ipex-llm/tree/main/python/llm/example/GPU/Deepspeed-AutoTP)) |
| **SGLang XPU** | ⏳ "enable TP for multi-ARCs" is **planned (2025 H2 roadmap)**, partially landing in LLM-Scaler-Omni 0.1.0-b5 (Jan 2026, diffusion TP) | — | **Least mature** for multi-card LLM TP | `--tp`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`. ([SGLang #8309](https://github.com/sgl-project/sglang/issues/8309), [SGLang XPU](https://sgl-project.github.io/platforms/xpu.html), [Phoronix](https://www.phoronix.com/news/Intel-LLM-Scaler-Omni-0.1.0-b5)) |

**Production-ready verdict:** *None* is "production-ready" in the NVIDIA sense. For a homelab, **llama.cpp SYCL layer-split is the most dependable multi-card path today**; **vLLM-XPU is the throughput path** once you accept the workaround flags and version-pin to Intel's validated BOM.

---

## 3. TP vs PP on slow PCIe — comparison table

| Dimension | Tensor Parallelism (TP) | Pipeline / Layer Parallelism (PP) |
|---|---|---|
| Comm per forward pass | All-reduce **every layer** (dozens/token) | **One** hidden-state handoff per stage boundary (~4–16 KB/token) |
| Sensitivity to PCIe/no-P2P | **Very high** — this is the failure mode on Arc | **Low** — tiny payloads, latency-tolerant |
| Single-request latency | Best *with NVLink*; **poor on PCIe3 + host-mediated oneCCL** | Higher latency at batch=1 (pipeline bubble: one GPU idles), but **doesn't get worse from slow link** |
| Aggregate throughput @ high concurrency | Good once fixed costs amortize (the "362 tok/s @50" regime) | Good with enough in-flight tokens to fill the pipeline |
| VRAM scaling | Splits weights **and** KV cache across cards | Splits weights/KV by layer ranges across cards |
| Battlemage software status | Fragile (vLLM #41663); not in llama.cpp SYCL | **Works today** in llama.cpp SYCL |
| Best for | MoE models (low per-token compute hides comm), high-concurrency serving | Running models too big for one card; long context; stability |
| Measured dual-B70 efficiency | n/a clean number; aggregate only | **~30–37%** of combined bandwidth for dense 70B ("446 GB/s of 1,216 GB/s") |

**Recommendation for PCIe 3.0 x16, no P2P:**
1. **Default to PP / layer-split** (llama.cpp `--split-mode layer`) for stability and because comm volume is ~3 orders of magnitude lower than TP.
2. **Use TP only for MoE models** (e.g. Qwen3-30B/35B-A3B, Qwen3-Coder-80B). MoE's ~3B *active* params per token mean compute >> comm, so PCIe overhead is "negligible" even on slow links ([PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/multi-gpu.md)). This is the one place TP earns its keep here.
3. **For interactive single-user dense-model latency: stay on ONE card** if the model fits in 32 GB. Splitting a model that fits on one card across two slow-linked cards almost always *loses* (PMZFX: dense 27B layer-split ≈ ~30% efficiency vs single-card).
4. Prefer **data parallelism** (two independent instances, one per card) when you want throughput and the model fits in 32 GB — zero inter-card comm, near-linear, "within 1% of single-card baselines."

General PP-over-PCIe guidance corroborated by [Sysart](https://sysart.consulting/insights/multi-gpu-inference-parallelism-on-premises/), [GigaGPU](https://gigagpu.com/tensor-vs-pipeline-parallelism/), [JarvisLabs](https://jarvislabs.ai/blog/scaling-llm-inference-dp-pp-tp).

---

## 4. What 2× B70 realistically buys (quantified)

All dual-card numbers below are from community benchmarks on **PCIe 5.0 x8** hosts — expect **somewhat worse** on your PCIe 3.0 x16. Source quality flagged.

### a) Bigger models (the main win) — llama.cpp SYCL layer-split, dual B70
*(Source: [PMZFX multi-gpu.md](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/multi-gpu.md) — single hobbyist repo, ⚠️ unverified by third party)*

| Model | Quant | Prompt (pp) t/s | Gen (tg) t/s | Type |
|---|---|---|---|---|
| DeepSeek-R1-70B | Q4_K_M | 336 | **11.5** | dense |
| Llama-3.3-70B | Q4_K_M | 338 | **11.5** | dense |
| Qwen3-35B-A3B | Q8_0 | 458 | **36.5** | MoE |
| Qwen3-Coder-Next-80B | Q4_K_M | 305 | **43.4** | MoE |

> Read this carefully: **a 70B dense model at ~11.5 tok/s is the headline "unlock," and it is slow.** The MoE models (35B/80B) at 36–43 tok/s on two cards at **~79 W combined** are the genuinely attractive result — and they're attractive precisely *because* MoE hides the PCIe penalty.

> ⚠️ Data anomaly: the same source's "Qwen 3.5-27B Q4_K_M: 718 t/s single → ~19.7 t/s dual" line is almost certainly a garbled extraction (718 is a prefill number, not single-GPU decode). I would **not** trust the "27B dense splits to ~19.7 tg" figure without re-running it; treat as unverified.

### b) Bigger KV cache / longer context
- 32 + 32 = **64 GB pooled VRAM** lets KV cache + weights for a 70B-4bit (~38–42 GB weights) leave room for tens of thousands of tokens of context, or run a 27B with very long context.
- PMZFX reports **context-invariant decode**: full throughput maintained even at 64K context ("you don't pay a decode tax for using long context"). Useful, but single-source.

### c) Higher concurrency / throughput (vLLM-XPU)
- Dual B70, Qwen3-30B-A3B FP8, TP=2: **~362 tok/s aggregate @ 50 concurrency** (with the stability workarounds) ([#41663](https://github.com/vllm-project/vllm/issues/41663)).
- Vendor/automation claim: **140 tok/s dual / 540 tok/s quad** B70 via vLLM TP ([Zing forum](https://www.zingnex.cn/en/forum/thread/intel-arc-pro-b70-gpu-llm-vllm)) — ⚠️ marketing-grade source; the 4-card→~3.9× jump vs 2-card looks **too good** and likely reflects a different model/concurrency point, not clean TP scaling. Flag as unverified.
- For comparison, **B60** (24 GB sibling, 456 GB/s): 4× B60 TP=4 on Qwen3-VL-30B-A3B reached **~1000 tok/s peak**, "linear" from 16→64 concurrency ([embeddedllm B60 benchmark](https://embeddedllm.com/blog/benchmarking-llm-inference-intel-arc-pro-b60)); GPT-OSS-120B MXFP4 on 4× B-series hit ~1495 tok/s @100 concurrency ([vLLM blog](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b)). Again these are **MoE + high concurrency** — TP's friendly regime.

### d) Two independent models (underrated)
- Data-parallel: run two separate models/instances, one per card, **near-zero cross-talk**, "within 1% of single-card baselines." E.g. a 9B coding model on card 1 while card 0 serves a 27B chat model. For a homelab this is often more useful than one slow split model.

---

## 5. Reported dual/Battlemage scaling efficiency (be skeptical)

- **Dense 70B layer-split: ~30–37% bandwidth efficiency** — effective ~446 GB/s of combined 1,216 GB/s; "at most 1× single-GPU speed, you're not doubling compute" ([PMZFX FINDINGS](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/FINDINGS.md)). This matches the *architectural* expectation: layer-split = sequential cards, so decode speed ≈ single-card minus handoff overhead. You add **VRAM**, not **speed**, for dense models.
- **MoE layer-split: near-1× single-card decode but with a model that wouldn't fit on one card** — the practical sweet spot.
- **TP aggregate throughput scales with concurrency**, not single-request latency. No clean, independently-verified TP-efficiency % for dual B70 exists yet; the cleanest real datapoint is the *workaround-laden* 362 tok/s@50 in #41663.
- A770 historical reality check: "TP for 70B is possible but gains are modest because the cards talk over PCIe rather than NVLink" ([Local AI Master A770](https://localaimaster.com/blog/intel-arc-a770-local-ai), [llama.cpp discussion #1923](https://github.com/ggml-org/llama.cpp/discussions/1923)).

**Verdict:** Plan for **~1.0–1.3× single-card single-request speed** from two cards on dense models (you're buying capacity), and **meaningful aggregate-throughput scaling only under concurrency, mostly for MoE**.

---

## 6. Recommended configs to benchmark when card #2 arrives

Run these in order; each isolates one variable. Use Qwen3-class models you already have.

### Config A — Baseline & the "does splitting even help?" test (llama.cpp SYCL, PP)
```bash
# Single card baseline (per card, sanity)
ONEAPI_DEVICE_SELECTOR=level_zero:0 llama-bench -m qwen3-27b-q4_k_m.gguf -ngl 99

# Dual card, layer split (default)
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
llama-cli -m qwen3-27b-q4_k_m.gguf -ngl 99 \
  --split-mode layer --tensor-split 1,1 --flash-attn -c 16384
```
Measure tg t/s vs single card. **Expectation:** dual ≤ single for a model that fits on one card. This proves the "don't split what fits" rule on *your* PCIe 3.0 host.

### Config B — The real unlock: 70B dense + long-context 27B (llama.cpp SYCL, PP)
```bash
# 70B dense that needs both cards' VRAM
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
llama-cli -m llama-3.3-70b-q4_k_m.gguf -ngl 99 \
  --split-mode layer --tensor-split 1,1 --flash-attn -c 32768
```
**Target:** confirm ~10–12 tg t/s (PMZFX got 11.5 on Gen5 x8 — expect equal-or-slightly-lower on Gen3). Also test a 27B at 64K context to validate context-invariant decode.

### Config C — The MoE + tensor-parallel throughput test (vLLM-XPU, TP=2)
Pin to Intel's validated container/BOM, then:
```bash
# Stability flags are mandatory on Battlemage today (per vLLM #41663)
export CCL_ENABLE_SYCL_KERNELS=0
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ONECCL_BINDINGS_FOR_PYTORCH_ENV_MODE=p2p   # also test =usm
vllm serve Qwen3-30B-A3B-FP8 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --enforce-eager
```
**Target:** aggregate tok/s sweep at concurrency 1, 16, 32, 50. Compare TP=2 vs running the same MoE layer-split in llama.cpp. Also try `--pipeline-parallel-size 2` (Ray) as the comm-light alternative. This is where TP *might* win — only because it's MoE.

### (Bonus) Config D — Two independent instances (data parallel)
Two `vllm serve` / `llama-server` processes, each pinned to one card (`ONEAPI_DEVICE_SELECTOR=level_zero:0` and `:1`). **Target:** ~2× aggregate throughput, ~0% cross-card penalty. Often the best homelab answer.

**What to record each run:** model+quant, split mode, `-c` context, tg/pp t/s, TTFT, TPOT, concurrency, combined watts, and whether any oneCCL/BCS crash workaround was needed. Track which `intel/vllm` tag + kernel + GuC firmware you used — version drift is a leading cause of breakage here (#41663).

---

## 7. Forward-looking: 4× B70 and PCIe 5.0

- **4× B70 (128 GB):** Unlocks 70B dense at higher concurrency and 100B+ MoE comfortably. B60/B-series data shows 4-card MoE TP hitting ~1000–1500 tok/s aggregate at high concurrency ([embeddedllm](https://embeddedllm.com/blog/benchmarking-llm-inference-intel-arc-pro-b60), [vLLM blog](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b)). **But:** 4-way TP all-reduce over host-mediated PCIe with no P2P scales *worse* per-GPU than 2-way; expect aggregate throughput gains under concurrency, not latency gains. Without P2P, 4× also stresses host RAM bandwidth and CPU (oneCCL is CPU-driven). Pipeline-parallel across 4 cards (PP=4, or PP=2×TP=2 hybrid) becomes attractive to bound comm.
- **Moving to PCIe 5.0 x16 (~64 GB/s, 4× your Gen3):** Directly relieves the *bytes* portion of TP cost and roughly halves all-reduce transfer time vs the Gen5-x8 reports (full x16 vs x8). It does **not** fix the absence of P2P or the host-driven oneCCL latency — those are architectural. So PCIe 5.0 makes TP *more viable*, especially for dense models and 4-card setups, but Arc still won't behave like NVLink. **The biggest single upgrade would be P2P/Xe-Link support landing in driver+hardware, which the Pro B-series does not have.**
- **Software trajectory:** SGLang XPU TP is "planned/landing"; vLLM-XPU TP is stabilizing but buggy on Battlemage; llama.cpp SYCL tensor/row split is "under development." Expect the multi-card story to improve materially over 2026 — re-test quarterly against Intel's validated BOM.

---

## 8. Summary recommendations

1. **Buy the 2nd B70 for capacity, not single-request speed.** Dense-model decode does **not** meaningfully scale across two slow-linked cards.
2. **Default to PP/layer-split** (llama.cpp SYCL) for stability and minimal comm. It's the only fully-working multi-card path today.
3. **Reserve TP for MoE models** (Qwen3-30B/35B-A3B, Qwen3-Coder-80B) via vLLM-XPU, accepting the #41663 workaround flags. MoE is the only regime where TP earns its keep on PCIe 3.0.
4. **Strongly consider data-parallel** (two independent instances) — near-linear throughput, zero comm risk.
5. **Expect your PCIe 3.0 results to trail the cited PCIe 5.0 x8 numbers**, especially anything TP/all-reduce heavy.
6. **Version-pin everything** to Intel's validated BOM; multi-card breakage is dominated by driver/firmware/container drift.

---

## Sources

- Intel Arc Pro B70 specs — https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html
- Tom's Hardware, B70/B65 announcement — https://www.tomshardware.com/pc-components/gpus/intel-arc-pro-b70-and-arc-pro-b65-gpus-bring-32gb-of-ram-to-ai-and-pro-apps-bigger-battlemage-finally-arrives-but-its-not-for-gaming
- ServeTheHome, B70/B65 — https://www.servethehome.com/intel-announces-arc-pro-b70-and-b65-video-cards-big-battlemage-brings-big-memory-for-ai-workstations/
- **vLLM issue #41663 — dual B70 TP=2 GP-fault / BCS reset, oneCCL/P2P, workarounds, 362 tok/s** — https://github.com/vllm-project/vllm/issues/41663
- Intel llm-scaler multi-GPU & parallelism (oneCCL p2p/usm, TP/PP/DP flags) — https://deepwiki.com/intel/llm-scaler/2.5-multi-gpu-and-parallelism
- vLLM Parallelism & Scaling docs — https://docs.vllm.ai/en/latest/serving/parallelism_scaling/
- vLLM XPU installation — https://docs.vllm.ai/en/v0.6.5/getting_started/xpu-installation.html
- vLLM blog: Arc Pro B-series serving (MXFP4, TP=4, 120B) — https://vllm.ai/blog/2025-11-11-intel-arc-pro-b
- EmbeddedLLM: B60 vLLM vs LLM-Scaler benchmark (TP=4, ~1000 tok/s) — https://embeddedllm.com/blog/benchmarking-llm-inference-intel-arc-pro-b60
- **PMZFX intel-arc-pro-b70-benchmarks — multi-gpu.md** (dual-B70 dense/MoE numbers) — https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/multi-gpu.md
- **PMZFX intel-arc-pro-b70-benchmarks — FINDINGS.md** (37% bw efficiency, TP-not-in-SYCL) — https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/FINDINGS.md
- PMZFX hardware.md — https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/hardware.md
- Zing forum: B70 cluster vLLM TP tuning (140/540 tok/s, PCIe5 x8) — https://www.zingnex.cn/en/forum/thread/intel-arc-pro-b70-gpu-llm-vllm
- llama.cpp SYCL backend (split-mode layer/none; row under dev) — https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md
- llama.cpp multi-GPU doc (layer vs row vs tensor; --tensor-split/--main-gpu) — https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- llama.cpp Arc A770 issue #7042 / discussion #1923 — https://github.com/ggml-org/llama.cpp/issues/7042 , https://github.com/ggml-org/llama.cpp/discussions/1923
- IPEX-LLM repo & DeepSpeed-AutoTP example — https://github.com/intel/ipex-llm , https://github.com/intel/ipex-llm/tree/main/python/llm/example/GPU/Deepspeed-AutoTP
- SGLang XPU backend roadmap #8309 — https://github.com/sgl-project/sglang/issues/8309
- SGLang XPU platform doc — https://sgl-project.github.io/platforms/xpu.html
- Phoronix: Intel LLM-Scaler-Omni 0.1.0-b5 (SGLang TP) — https://www.phoronix.com/news/Intel-LLM-Scaler-Omni-0.1.0-b5
- oneCCL host-driven collectives analysis — https://arxiv.org/html/2409.09874v3
- Flash Communication (TP all-reduce bottleneck) — https://arxiv.org/html/2412.04964v1
- PP vs TP over PCIe guidance — https://sysart.consulting/insights/multi-gpu-inference-parallelism-on-premises/ , https://gigagpu.com/tensor-vs-pipeline-parallelism/ , https://jarvislabs.ai/blog/scaling-llm-inference-dp-pp-tp
- NVLink vs PCIe bandwidth — https://www.spheron.network/blog/what-is-nvlink-gpu-interconnect-bandwidth-explained/ , https://willitrunai.com/blog/multi-gpu-llm-inference-guide
- Intel Arc A770 local AI (PCIe TP "modest gains") — https://localaimaster.com/blog/intel-arc-a770-local-ai
- Intel Community: GPU P2P memory copy / P2P DMA routing — https://community.intel.com/t5/GPU-Compute-Software/GPU-peer-to-peer-memory-copy/td-p/1518166

*Reliability note: the dual-B70 throughput figures derive heavily from one hobbyist benchmark repo (PMZFX), one vendor/automation forum post (Zing), and one open, unresolved vLLM bug (#41663). None is independently reproduced. Treat all dual-card numbers as directional, and re-validate on your own PCIe 3.0 host.*
