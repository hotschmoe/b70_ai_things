# Intel Arc Pro B70 — LLM Inference Findings (hobbyist field notes)

What we've actually measured running LLMs on a single **Intel Arc Pro B70** (32 GB GDDR6,
Battlemage/Xe2, ~$949), in Docker on an Unraid host. Goal: help fellow Team Blue tinkerers
skip the dead-ends. Living doc — see [RESULTS.md](RESULTS.md) for the raw number tables and
[JOURNAL.md](JOURNAL.md) for the blow-by-blow.

## TL;DR
- **The B70 is a solid single-card inference GPU for ~14B-class models.** Qwen3-14B at **FP8**
  does **~35 tok/s single-stream** and **~324 tok/s aggregate** at concurrency 32, near-lossless.
- **Best backend: upstream vLLM-XPU built from source** (`Dockerfile.xpu`). It has the real XPU
  FP8 matmul kernel and native model support. (llama.cpp SYCL also works well for standard models.)
- **FP8 is the 8-bit sweet spot.** There is **no INT8 W8A8 kernel on Battlemage** in vLLM — W8A8
  silently dequant-falls-back to FP16. Don't bother; use FP8.

## What works (single B70, 32 GB)
| Thing | Result |
|---|---|
| **Qwen3.6-27B (Gated-DeltaNet)** | **RUNS** via int4 AutoRound on vLLM **0.23.0** — 7.9 t/s, coherent. Only known single-card path. |
| Qwen3-14B **FP8** (vLLM-XPU) | **35 t/s** single / **324 t/s** @ C32, TTFT ~62 ms (w/ compile), near-lossless |
| Qwen3-14B **F16/BF16** | 18.7 t/s, but ~28 GB barely fits (tiny KV). FP8 is ~1.9x faster — just use FP8 |
| Qwen2.5-7B **Q4_K_M** (llama.cpp SYCL) | ~90 t/s decode — llama.cpp SYCL is great for standard-attention models |
| **torch.compile (inductor)** | cuts single-stream **TTFT ~6x** (1032->176 ms) + ~11% decode at low concurrency |

## What does NOT work (yet) — save yourself the time
- **INT8 W8A8:** no XPU kernel (FP8-only `scaled_mm`). Use FP8. The only int8-activation path is **W4A8**
  (`XPUW4A8IntLinearKernel`, int4 weights + int8 activations) — niche, fragile.
- **Speculative decoding (draft model):** **net-negative on B70** (3.4x slower in our test) because XPU has
  **no CUDA-graph capture**, so the extra forward passes per step cost more than the token savings.
- **Qwen3.6 (Gated-DeltaNet) FP8/8-bit:** does NOT fit a single card (28.5 GiB, no KV room) and the FP8
  DeltaNet kernel only exists in vLLM **0.23.0** (older images: ESIMD `.weight` bug / `scaled_mm` `KeyError(XPU)`).
  BUT int4 (AutoRound) on 0.23.0 **does run** (see above) — just slow. 8-bit Qwen3.6 needs a 2nd card.
- **torch.compile on vLLM 0.23.0:** broke (torch 2.11 + inductor). Run `--enforce-eager` on 0.23.0
  (its eager TTFT is already ~7x better than 0.20.2 eager anyway).
- **Gemma 4 12B:** the `gemma4_unified` multimodal checkpoint routes through vLLM's generic Transformers
  fallback, which mis-reshapes its mixed head dims (256/512) and crashes. Needs a native-resolving checkpoint.

## Practical gotchas
- **Unraid `docker.img` is 50 GB by default** — multi-GB vLLM images overflow it. Grow it (we went to 200 GB
  on the NVMe cache, non-destructively) or relocate Docker storage.
- **Keep model weights + all caches on a fast SSD via bind-mounts**, never baked into image layers.
- **Build vLLM from source** at a known-good commit (we used `c51df4300` = 0.20.2rc1.dev2) — pre-built XPU
  images lag, and the Intel `llm-scaler` image wraps an *older* upstream for dense models.
- It's a **memory-bandwidth game** at batch 1: decode t/s ≈ 608 GB/s ÷ model-bytes. Smaller quant = faster
  decode; FP8 (~1 byte/wt) ≈ 2x BF16. Compute (XMX TOPS) only matters for prefill + big batches.

## Hardware
Intel Arc Pro B70: Xe2/Battlemage, 32 GB GDDR6, 608 GB/s, 367 INT8 TOPS, PCIe 5.0 x16 (Gen3 on our
Threadripper host). `xe` kernel driver. Pass into containers with `--device /dev/dri`.

*Next up: dual-B70 (tensor/pipeline parallel), v0.23.0 vLLM comparison, and int4/W4A8.*
