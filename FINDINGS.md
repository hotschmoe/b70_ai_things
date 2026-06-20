# Intel Arc Pro B70 — LLM Inference Findings (hobbyist field notes)

What we've actually measured running LLMs on a single **Intel Arc Pro B70** (32 GB GDDR6,
Battlemage/Xe2, ~$949), in Docker on an Unraid host. Goal: help fellow Team Blue tinkerers
skip the dead-ends. Living doc — see [RESULTS.md](RESULTS.md) for the raw number tables and
[JOURNAL.md](JOURNAL.md) for the blow-by-blow.

## TL;DR
- **The B70 is a solid single-card inference GPU for ~14B-class models.** Qwen3-14B at **FP8**
  does **~35 tok/s single-stream** and **~556 tok/s aggregate** at concurrency 64, near-lossless.
  (Default `--max-num-seqs 16` caps you at ~330 — raise it for throughput.)
- **Best backend: upstream vLLM-XPU built from source** (`Dockerfile.xpu`). It has the real XPU
  FP8 matmul kernel and native model support. (llama.cpp SYCL also works well for standard models.)
- **INT8 W8A8 now WORKS on Battlemage — we wrote the kernel.** Stock vLLM had no XPU int8 kernel (hard
  `KeyError: PlatformEnum.XPU` at load). We added a native oneDNN `int8_gemm_w8a8` (s8s8s32) + a fused
  per-token int8 quant + `XPUInt8ScaledMMLinearKernel` + the registry entry. W8A8 serves, beats FP8 ~1.6x in
  prefill, ~matches decode, and composes with FP8 KV cache (2x context). FP8 still wins pure single-stream
  decode; W8A8 wins prefill/large-batch — the long-context coding-server pick. docs/kernel/02 + `contrib/vllm_int8_xpu/`.
- **PIECEWISE XPU graph capture = free +16.7% decode (2026-06-19).** vLLM 0.23.0 already wires XPU graph
  capture (`VLLM_XPU_ENABLE_XPU_GRAPH=1`); adding `register_fake` meta kernels for our 2 custom int8 ops lets
  dynamo trace through them. PIECEWISE mode (attention eager, linear/MLP captured) lifts 14B W8A8 decode
  23.33 -> **27.23 t/s**. FULL capture is blocked by Intel's SYCL Graph ext (`work_group_scratch_memory`, via
  flash-attn). Use image `vllm-xpu-env:int8g` + `cudagraph_mode=PIECEWISE`. (This w8a8 number predates the
  compile-pass fix below and is being re-confirmed on the current image.)
- **[BREAKTHROUGH 2026-06-20] PIECEWISE graph capture lifts W4A8 decode 16.79 -> 48.18 t/s (+187%, 2.87x).**
  The biggest decode win on the B70 so far. W4A8 eager is severely dispatch-bound because its per-token
  activation quant is the UNFUSED pure-PyTorch `dynamic_per_token_int8_quant_ref` (hundreds of tiny ops/token);
  graph capture fuses the whole decode step, taking decode from ~26% to ~74% of the BW ceiling (9.3 GiB @
  608 GB/s ~= 65 t/s). That is why W4A8 gains 2.87x where W8A8 (which uses a fused quant op) gained only 1.17x.
  **The real headline (corrected after capturing the rivals too): graph capture roughly DOUBLES int4 decode on
  the B70.** Apples-to-apples, PIECEWISE: **w4a16 28.04 -> 54.57 (+95%)**, w4a8 16.79 -> 48.18 (+187%), w8a8
  23.62 -> 26.68 (+13%). So **w4a16 LEADS decode (54.57), w4a8 leads prefill/TTFT** (int8-XMX) -- both 9.3 GiB,
  both ~2x'd by capture; w4a16's int4xfp16 GEMM is ~13% more BW-efficient at m=1 (no act-quant). w8a8 gains
  little (already-fused quant). W4A8 is no longer "dominated" -- it co-leads with w4a16, split by decode-vs-
  prefill. (Earlier note "w4a8 beats everything" was only true while the others were eager; corrected here.)
  Two gotchas fixed (both in `w4a8/30_serve_w4a8_graph.sh`): (1) a vLLM XPU+compile crash
  `NameError: MLARoPEKVCacheCatFusionPass` -- vLLM auto-enables CUDA-only inductor fusion passes under
  torch.compile but XPU never imports them; disable the CUDA/ROCm-only fusion flags in the serve `pass_config`.
  (2) the int4 `register_fake` is REDUNDANT -- vLLM already ships a native fake for `int4_gemm_w4a8`. Serve:
  `w4a8/30_serve_w4a8_graph.sh GRAPH=1` (image `:int8g`, dtype float16); confirm via the
  `Capturing CUDA graphs (PIECEWISE)` + `Graph capturing finished` log lines.
- **Quant quality measured (2026-06-19): the int8 ACTIVATION quant is the quality cost, not int8 weights.**
  First eval campaign, Qwen3-14B × 6 quants (see [evals/](evals/) + [evals/results/SUMMARY.md](evals/results/SUMMARY.md)):
  ppl + top-1 token-agreement vs a CPU-scored bf16 anchor + gsm8k. **W8A16 (int8 w / fp16 a) is near-lossless**
  (ppl 12.76 vs bf16 12.70, **0.981** agreement) but **has NO XPU kernel** (`XPUwNa16` is int4-only). Quantizing
  activations to int8 (W8A16→**W8A8**) drops agreement 0.981→**0.881** — yet gsm8k barely moves (95.3% vs fp8 96.0%),
  because it flips low-confidence tokens, not answers. **Weight bits dominate:** W8A8 > W4A16 on every metric.
  **Kernel takeaway:** keep optimizing **W8A8** (only path that lights the INT8 systolic datapath; ≈fp8 task
  quality, 15 GB); a W8A16 int8-weight kernel is the one gap but it's a fidelity/memory play (fp16 acts, no INT8
  matmul), not a speed one. **W4A8** (0.822 agree, gsm8k 92.7%) only when memory-bound (9.3 GB).

## What works (single B70, 32 GB)
| Thing | Result |
|---|---|
| **Qwen3.6-27B (Gated-DeltaNet)** | **RUNS** via int4 AutoRound on vLLM **0.23.0** — 7.9 t/s, coherent. Only known single-card path. |
| Qwen3-14B **FP8** (vLLM-XPU) | **35 t/s** single / **556 t/s** @ C64 (raise `--max-num-seqs`!), near-lossless |
| Qwen3-14B **F16/BF16** | 18.7 t/s, but ~28 GB barely fits (tiny KV). FP8 is ~1.9x faster — just use FP8 |
| Qwen2.5-7B **Q4_K_M** (llama.cpp SYCL) | ~90 t/s decode — llama.cpp SYCL is great for standard-attention models |
| **torch.compile (inductor)** | cuts single-stream **TTFT ~6x** (1032->176 ms) + ~11% decode at low concurrency |
| **Qwen3-14B W8A8 + PIECEWISE XPU graph** | **27.23 t/s** decode (+16.7% over eager 23.33) — image `:int8g`, `cudagraph_mode=PIECEWISE` |
| **Qwen3-14B W4A8 + PIECEWISE XPU graph** | **48.18 t/s** decode (**+187% / 2.87x** over eager 16.79) — `:int8g` + `30_serve_w4a8_graph.sh GRAPH=1`. Best prefill/TTFT (int8-XMX). |
| **Qwen3-14B W4A16 + PIECEWISE XPU graph** | **54.57 t/s** decode (**+95% / 1.95x** over eager 28.04) — **fastest decode measured**; same 9.3 GiB; no act-quant tax. `GRAPH=1`. |

## What does NOT work (yet) — save yourself the time
- **INT8 W8A8:** stock vLLM has no XPU kernel -> hard-crashes at load (`KeyError: PlatformEnum.XPU`).
  **We FIXED it (2026-06-18):** wrote a native `int8_gemm_w8a8` oneDNN kernel (s8xs8->s32) + the vLLM
  `XPUInt8ScaledMMLinearKernel` + registry entry. The W8A8 checkpoint now SERVES on the B70 and selects our
  kernel (`Selected XPUInt8ScaledMMLinearKernel`), coherent output. First INT8 W8A8 path on Battlemage. See
  docs/kernel/01 + contrib/vllm_int8_xpu/. (Perf vs FP8 in prefill/large-batch = TODO; FP8 still wins decode.)
- **W4A8-INT** (`XPUW4A8IntLinearKernel`, int4 weights + per-token int8 activations, oneDNN `int4_gemm_w4a8`):
  **[SUPERSEDED 2026-06-20 -> now in "What works"]** the "decode ~16.6 t/s, half of FP8, FP8 stays the pick"
  conclusion below was measured EAGER. With PIECEWISE graph capture W4A8 decode is **48.18 t/s** -- it now
  BEATS fp8 single-stream and is the fastest decode config measured (see the breakthrough bullet in the TL;DR).
  Original eager notes kept for the record: **Tested 2026-06-18 — it's the only path that lights the INT8-XMX
  datapath, and it serves (9.3 GB, smallest footprint, 12x concurrency). BUT single-stream decode is ~16.6 t/s,
  half of FP8** (the int4_gemm_w4a8 decode kernel is unoptimized). Head-to-head (identical harness): **FP8 wins
  per-stream at every matched concurrency (~1.7-1.8x)**; W4A8's only real edge is VRAM (9.3 vs 15.2 GB). FP8
  stays the pick. (Note from literature/06: FP8 is conversion-based on Xe2; INT is native systolic — so a real
  INT8 W8A8 kernel could beat FP8 in *prefill/large-batch*. That kernel doesn't exist upstream yet = our top
  contribution target.) [The eager-vs-capture gap was the real story: w4a8 was the most dispatch-bound config.]
- **Speculative decoding:** **net-negative on B70**, even with graph capture. ngram (prompt-lookup) on 14B
  W8A8: eager 23.33->21.51 t/s (-7.8%); WITH PIECEWISE graph 27.23->25.28 (-7%). ~16% draft acceptance. Why:
  in PIECEWISE mode **attention runs eager**, so the multi-token verify still pays full eager attention launch
  overhead x(N+1). Spec-decode needs **FULL** graph capture (attention included) to win — and FULL capture is
  blocked by the SYCL Graph `work_group_scratch_memory` limit. Parked until that's unblocked. (Earlier draft-
  model test was 3.4x slower; the no-graph-capture premise is now refined — graph IS available, just PIECEWISE.)
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
