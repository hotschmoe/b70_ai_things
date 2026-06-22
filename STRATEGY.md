# B70 Optimization Strategy & Roadmap

Living document. The "why" and "where we're going." Tactical results go in `JOURNAL.md`;
deep background goes in `docs/literature/`.

## Findings update (2026-06-17) — these reshape the plan

From the literature sweep (`docs/literature/`) + first hands-on runs:
1. **Qwen3.6-27B is DENSE with Gated-DeltaNet (linear) attention** — bleeding-edge. It currently
   **segfaults on the released llama.cpp intel image (build 9680), CPU and GPU** — needs the absolute
   latest llama.cpp for the DeltaNet operators. Action: build llama.cpp SYCL from latest source.
2. **8-bit via GGUF Q8_0 is a dead end on Xe2** (~4x slower than Q4; INT8 XMX unused in decode). For the
   8-bit goal use **vLLM-XPU FP8**. Keep **Q4_K_M as the single-card workhorse**.
3. **Offload (128 GB DDR4) rarely helps the dense 27B** — it fits in 32 GB; a smaller quant always beats
   partial offload. Offload's real use is the **35B-A3B MoE** (`--n-cpu-moe`) or 70B-class on dual-card.
4. **XMX INT8 (367 TOPS) only pays off in PREFILL**, not decode (decode is memory-bound). So PP/TTFT are
   where the card's compute shines; TG is bandwidth-bound -> MTP/spec-decode is the only lever there.
5. **MTP [UPDATED 2026-06-22 -- PROVEN on vLLM-XPU]:** vLLM-XPU 0.23.0 (`vllm-xpu-env:v0230`, #43565 native) DOES MTP
   on Gated-DeltaNet -- single-card 27B W4A16 + MTP spec=4 PIECEWISE = **1.79x (55.28 t/s vs 30.84)**, the primary
   decode lever (DENSE only; MoE MTP is flat). No llama.cpp / draft-model fallback needed. Full campaign: MTP_TODO.md.
   (The old "vLLM-XPU can't do MTP on DeltaNet" was true on pre-0.23 images.)
6. **Dual-card:** no P2P on Arc; prefer **pipeline/layer-split over tensor-parallel** on PCIe3; expect
   ~1.0-1.3x single-stream, real wins in **capacity + concurrency (esp. MoE)**. Two independent
   single-card instances may give the best homelab throughput.
7. **Strategic option to raise:** the sibling **Qwen3.6-35B-A3B MoE** decodes ~50-65 t/s (vs ~20 dense),
   scales better on 2 cards, and is the natural fit for offload+concurrency. Worth targeting alongside
   the 27B if the use-case allows. (User selected 27B; flagging as an option, not a change.)
8. **PCIe "Gen1 x1" reading on the GPU die is cosmetic** (real load bandwidth ~1+ GB/s) — not a bottleneck.

## North star

Squeeze the best possible LLM inference out of the Intel Arc Pro B70(s), quantify
exactly what each optimization and each additional card buys, share results with the
community, and contribute fixes upstream (Intel `xe` kernel driver, compute-runtime,
oneAPI, llama.cpp/vLLM/IPEX-LLM) where we hit real gaps.

Optimize for three headline metrics, separately (they trade off):
- **PP** — prefill throughput (tok/s), prompt-processing / context ingestion speed.
- **TTFT** — time to first token (latency), dominated by prefill + scheduling.
- **TG** — decode throughput (tok/s/stream), memory-bandwidth-bound for single stream.
- plus **aggregate throughput** under concurrency (tok/s across N parallel requests).

## Hardware trajectory

| Phase | Hardware | Unlocks |
|-------|----------|---------|
| Now | 1x B70 (32 GB), PCIe (verify lane width), 128 GB DDR4, TR 1950X | single-card baselines, quant sweep, MTP, offload |
| Next week | 2x B70 (PCIe 3.0 x16 each, no NVLink) | bigger models (70B@4bit), bigger KV/longer ctx, more concurrency; TP vs PP study |
| If great | 4x B70 and/or move to PCIe 5.0 x16 platform | larger TP groups, dethrottle inter-card comm |

Key reality: **inter-card link is slow (PCIe 3.0 x16 ~= 16 GB/s, likely no P2P)**. Tensor
parallel does a per-token all-reduce -> sensitive to this. Pipeline parallel moves far less
data but adds latency. We will measure TP vs PP efficiency directly rather than assume.
(See `docs/literature/02_multigpu.md`.)

## Phased plan

- **Phase 0 — Foundation [DONE].** SSH, recon, GPU passthrough validated (B70 visible via
  Level-Zero + OpenCL, 30.3 GiB usable), SSD work tree, journal. See JOURNAL baseline.
- **Phase 1 — Single-card llama.cpp baseline [IN PROGRESS].** Qwen3.6-27B Q4_K_M then Q8_0
  on the SYCL backend. Establish PP/TTFT/TG numbers, confirm full GPU offload, VRAM use.
- **Phase 2 — Backend bake-off.** Same model/quant across llama.cpp (SYCL vs Vulkan),
  IPEX-LLM, vLLM-XPU/Intel vLLM, (OpenVINO/SGLang if promising). Confirm XMX INT8 fast path.
- **Phase 3 — MTP / speculative decoding.** Use Qwen3.6-27B MTP layers / draft model to beat
  the memory-bandwidth decode ceiling. Measure acceptance rate + net TG speedup.
- **Phase 4 — Offload study.** Use 128 GB DDR4 for partial offload (bigger ctx / bigger model
  than VRAM). Quantify the PCIe/DDR4 penalty; find the break-even vs smaller quant.
- **Phase 5 — Sweeps + concurrency.** Automated variable sweep (below) + concurrency scaling
  curves (1 -> N parallel requests) for the winning backend(s). Find the Pareto front.
- **Phase 6 — Dual-card.** Drop in 2nd B70. TP vs PP, P2P check, scaling efficiency, what 2x
  buys for PP/TTFT/TG/concurrency and for running 70B-class models.
- **Phase 7 — Scale-out analysis.** Project 4x B70 and PCIe 5.0 gains; write up; upstream any
  fixes/benchmarks; share with community.

## Variable sweep design (Phase 5 core)

Sweep axes (cross product, pruned greedily — don't brute-force all combos):

| Axis | Values to try |
|------|---------------|
| Quant | Q4_K_M, Q8_0, (IPEX sym_int8 / sym_int4, AWQ/GPTQ int4 on vLLM) |
| GPU layers (`-ngl`) | full (all on GPU) vs partial offload points |
| Context length | 4k, 16k, 32k, 128k (KV pressure) |
| Batch / ubatch | llama.cpp `-b`/`-ub`; vLLM max-num-seqs / max-num-batched-tokens |
| Flash attention | on / off (`-fa`) |
| KV cache quant | f16, q8_0, q4_0 KV |
| Concurrency | 1, 2, 4, 8, 16, 32 parallel requests |
| Backend | llama.cpp-SYCL, llama.cpp-Vulkan, vLLM-XPU, IPEX-LLM |

Metrics captured per run: PP tok/s, TTFT ms, TG tok/s (single + aggregate), peak VRAM,
GPU util, power, output correctness sanity. Results -> `results/*.csv` (one row per run).

Methodology guardrails: warm up before timing; fixed seed/prompts; let the card settle
between runs (thermal); pin the B70 via `ONEAPI_DEVICE_SELECTOR=level_zero:0`; record
driver/oneAPI/image versions with every result. (Detail in `docs/literature/03_*.md`.)

## Dual-card readiness checklist (before next week)

- [ ] Confirm both PCIe slot lane widths (`lspci -vv` link cap/status) and ReBAR/AER state.
- [ ] Verify Level-Zero P2P capability between cards (or confirm host-RAM staging).
- [ ] Pick TP vs PP starting point per backend from `02_multigpu.md`.
- [ ] Pre-stage a 70B-class 4-bit model on SSD to exercise 2x VRAM.
- [ ] Power/cooling headroom check (2x 230 W TBP in the case).

## Community / upstream contribution opportunities

Track in the journal as we hit them: backend bugs, missing Battlemage kernels, perf
regressions, missing quant kernels for XMX INT8, multi-GPU comm inefficiencies, doc gaps.
Publish reproducible benchmarks (this repo's scripts + results) for the B70 community.
