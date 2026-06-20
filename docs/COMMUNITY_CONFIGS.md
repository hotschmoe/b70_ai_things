# Community / External B70 Inference Configs — Reference Scoreboard

External configs other people have reported for Intel Arc Pro B70 (Battlemage) LLM
inference, collected in one place so we can chase, reproduce, and beat them. **These are
OTHER people's numbers, not ours.** Our own measured results live in
[FINDINGS.md](../FINDINGS.md) (older raw tables in [archive/RESULTS.md](../archive/RESULTS.md)); the
blow-by-blow is in [JOURNAL.md](../JOURNAL.md); the
version/quant background is in [docs/literature/](literature/).

Legend: [TARGET] chasing this | [OK] credible/detailed | [WARN] caveat — verify before trusting | [NEG] negative result (a dead end someone already found)

---

## [TARGET] PRIMARY CHASE TARGET — Qwen3.6-27B BF16, 4x B70 TP=4, vLLM-XPU + MTP

The number we most want to reproduce and then beat. Single-stream, real model, MTP doing the work.

| Field | Value |
|---|---|
| Model | **Qwen3.6-27B** (dense, Gated-DeltaNet), BF16 weights, **float16 runtime** |
| Hardware | **4x Intel Arc Pro B70** (32 GB each, ~128 GB pooled), PCIe only, no XeLink |
| Engine | **vLLM-XPU**, image `intel/llm-scaler-vllm:0.14.0-b8.3` |
| Parallelism | **TP=4** |
| Context | 262,144 native | Batch | 1 (single-stream) |
| **Decode** | **54.2 tok/s** (with MTP) |
| **Prefill** | **~2,100 tok/s** (measured on 3–8K prompts, conservative, under disk contention) |
| Raw (no-MTP) decode | ~17 tok/s (the other reported column; MTP is doing ~3x here) |
| MTP wiring | **vllm_xpu_kernels v0.1.9** wheel + `qwen3_5.py` spec-wiring patch (**vLLM #43565**) + **Half-KV**; `num_speculative_tokens=5`, **mean accept length 4.04** (88.9% accept at spec=3) |
| Reference point | **2.9x faster than llama.cpp 27B Q8** (15.6 tok/s b1) |

**Why this is the target:** it is the proven, full-precision (BF16) 27B run with MTP unblocked
from userspace. Decode is normally bandwidth-bound, but MTP verifies ~4 tokens per weight-read
(mean accept len 4.04), multiplying the ceiling. Prefill ~2100 t/s shows the XMX compute scaling
across 4 cards.

**What it takes to reproduce (the non-obvious parts):**
- MTP on Qwen3.6 Gated-DeltaNet is **not a stock toggle** on the llm-scaler image — it crashes on
  `XPU gdn_attention does not yet support 'spec_sequence_masks'` (llm-scaler #386). The real fix
  (vLLM **#43565**, GDN-attention MTP) only landed upstream in **vLLM v0.23.0**. This run
  back-ported it via the **vllm_xpu_kernels v0.1.9** wheel + a manual `qwen3_5.py` patch.
- **Half-KV** to fit KV alongside the BF16 weights across the 4 cards.
- The MTP head must stay BF16 (don't quantize `mtp.fc`) or drafting dies.

**Our ladder toward it (single -> 4 card):** see JOURNAL "Reality check + the MTP lever". Single-card
W8A8/FP8 + MTP projected ~40–55 t/s at 8-bit accuracy; 2 cards unlock BF16; 4 cards = this full run.
On our v0.23.0 image the GDN-MTP fix is native (no patch needed) — that's our cleanest path to it.

---

## Full community scoreboard (all reported B70 configs)

Newest model-of-interest first. "Source" = where we got it (user-supplied data point unless linked).

| # | Model | HW | Engine / image | Quant | Key numbers | Flag |
|---|---|---|---|---|---|---|
| 1 | **Qwen3.6-27B** BF16 | 4x B70 TP4 | vLLM-XPU `llm-scaler 0.14.0-b8.3` + MTP | BF16/fp16 | **54.2 dec / ~2100 pre** t/s, accept 4.04 | [TARGET] |
| 2 | Qwen3.6-**35B-A3B** (MoE) | 4x B70 TP4 | vLLM `0.20.2rc1.dev2` | ~~Quark W8A8 INT8~~ -> see correction | ~~99.77 t/s~~ likely FP8/INT4 + total-vs-output mixup (deep-dive below) | [CORRECTED] |
| 3 | Qwen3.6-**35B-A3B** (MoE) | 1x B70 | **llama.cpp SYCL** (oneAPI 2026) | Q4_K_XL (~22 GB) | **75 t/s** sustained (Vulkan capped ~45) | [OK] |
| 4 | Qwen3.6-**35B-A3B** (MoE) BF16 | 4x B70 TP4 | vLLM-XPU **graph mode** (custom all-reduce) | BF16 | **~102 t/s** dec (5.8x eager), TTFT ~120 ms, ~830 t/s agg @ C100, prefill ~1130 t/s | [OK] |
| 5 | Qwen3.6-**35B-A3B** (MoE) | 1x B70 | **llama.cpp Vulkan**, Win11 native, b9553 | Q4_K_M | **~107 t/s** median (ctx 262144, KV fp16, batch 2048) | [OK] |
| 6 | Qwen3.6-27B (or 35B) | 2x B70 TP2 | vLLM `0.20.1` + MTP (patched) | INT4 AutoRound W4A16 | **slower** than 1-card MTP; latency 7.20 s, accept 62.6% | [NEG] |
| 7 | **Gemma 4 12B** (multimodal) | 4x B70 TP4 | upstream vLLM `0.20.2rc1.dev2`, piecewise cudagraph | INT4 AutoRound W4A16 | runs; prefix-cache on, `--limit-mm-per-prompt {image:4}` | [OK] |
| 8 | Qwen3.6-27B (CUDA ref) | RTX 5090 | vLLM CUDA | INT4 AutoRound + MTP | ~60 t/s vs ~30 (2x), ~85% accept (cross-arch MTP ceiling) | [WARN] |

**Credibility notes:**
- **#2 (Quark W8A8 INT8, 99.77 t/s):** [WARN] **Tested 2026-06-18 — claim does not hold as an INT8-XMX
  run.** We self-quantized a compressed-tensors W8A8 INT8 Qwen3-14B and served it on vLLM 0.23.0 (XPU):
  it **hard-crashes at model load** with `KeyError: PlatformEnum.XPU` in `choose_scaled_mm_linear_kernel`
  (the INT8 scaled-MM registry has no XPU entry — see [05_w8a8_recipe.md](literature/05_w8a8_recipe.md)
  and JOURNAL/RESULTS 06-18). **Source-confirmed on the community's exact commit:** that run used vLLM
  `0.20.2rc1.dev2+gc51df4300` = our `:tf` image. In that image, **`QuarkW8A8Int8.create_weights` calls the
  same `init_int8_linear_kernel`** as compressed-tensors W8A8, and `_POSSIBLE_INT8_KERNELS` has keys for
  **CPU/CUDA/ROCM only — no XPU**. So stock c51df4300 hits `KeyError: PlatformEnum.XPU` at load for Quark
  W8A8 INT8 too. The 99.77 t/s therefore was **not** stock INT8 W8A8 on XPU — it required a custom XPU int8
  kernel + a registry patch (`_POSSIBLE_INT8_KERNELS[XPU]=[...]`), or the Quark config resolved to FP8/W4A8,
  or it's misattributed. Writing that kernel is **our contribution target #1** (see literature/06).
- **#6 (TP2+MTP):** [NEG] Corroborated by our own reading: vLLM **disables XPU graph capture for TP2
  comm ops**, so verify+collective overhead eats the draft savings. **MTP is a single-card win, not
  dual.** Don't waste time re-deriving this.
- **#5 / #3 / #4 are all the 35B-A3B MoE** — much friendlier to the B70 than the dense 27B (more
  active-param headroom, batches better). If the use-case allows MoE, these are the high-throughput
  configs. Note the ~107 (Vulkan) and ~102 (graph-mode vLLM) are different stacks at a similar ceiling.

---

## Detailed reproducible setups (worth keeping verbatim)

### A. Single B70, Qwen3.6-35B-A3B Q4_K_XL, llama.cpp **SYCL** — 75 t/s

The headline finding: **SYCL >> Vulkan for MoE on Battlemage because of XMX.** Vulkan reports
`matrix cores: none` (no cooperative-matrix), so the MoE's pile of small expert GEMMs runs
unaccelerated (~45 t/s ceiling). Rebuilding llama.cpp with the SYCL backend on Intel oneAPI 2026
lets the XMX matrix engines do the expert matmuls -> 75 t/s sustained.

Toolchain gotchas (the genuinely useful part):
- oneAPI 2026 image is `intel/oneapi-toolkit:2026.0.0-devel-ubuntu24.04` (the old
  `oneapi-basekit` is frozen at 2025.3.1).
- The 2026 toolkit pre-runs `setvars.sh`; sourcing it again returns non-zero and kills `&&` chains.
  Guard it: `. setvars.sh --force >/dev/null 2>&1 || true`, and use `;` (not `&&`) before exec.
- `-hf` downloads need a TLS backend: add `libssl-dev` + `-DLLAMA_OPENSSL=ON`.
- `level-zero` is a virtual package — install the real providers `libze-intel-gpu1 libze1` from
  Intel's GPU apt repo (`https://repositories.intel.com/gpu/ubuntu noble unified`). Driver renamed
  from `intel-level-zero-gpu`.
- Pick the GPU explicitly: `ONEAPI_DEVICE_SELECTOR=level_zero:N` (check `sycl-ls`).
- `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` required to allocate the >4 GB model over Level-Zero.
- stdout is block-buffered to a pipe -> wrap with `stdbuf -oL -eL` for live logs.
- First request JIT-compiles ~25–30 s (one-time, not a hang).

Build flags: `cmake -B build -DGGML_NATIVE=OFF -DGGML_SYCL=ON -DBUILD_SHARED_LIBS=OFF
-DLLAMA_OPENSSL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx`.

Serve: `llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL --jinja -ngl 99 -c 32768 -fa on`,
env `ONEAPI_DEVICE_SELECTOR=level_zero:1`, `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`,
`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`, `ZES_ENABLE_SYSMAN=1`.

> Relevance to us: this is the **llama.cpp SYCL** route for the **MoE** model — complementary to our
> vLLM-XPU work on the dense 27B. If/when we run 35B-A3B, this is the single-card baseline to match.
> (Our DeltaNet 27B can't use llama.cpp SYCL yet — kernel gap, build 9680. The MoE 35B-A3B can.)

### B. 4x B70, Qwen3.6-35B-A3B BF16, vLLM **graph mode** with hand-rolled all-reduce — 102 t/s

The hard one. vLLM hard-asserts eager mode on Intel GPUs (~18 t/s decode). Un-gating XPU graph
capture (`torch.xpu.XPUGraph`) folds the whole decode step — including the per-layer TP all-reduce —
into one replayed graph. The stock 4-way oneCCL all-reduce **mis-replays inside a captured graph and
corrupts output**, so they wrote a **capture-safe all-reduce in SYCL/ESIMD**: Level-Zero IPC peer
buffers + a custom GPU barrier (PCIe device-to-device atomics were unreliable -> local-write/remote-read
with uncached ESIMD loads + explicit `system_acquire` fences for coherent cross-GPU sync).

Stack: vLLM (main) XPU backend, PyTorch 2.12+xpu, oneCCL 2021.17, Intel DPC++/SYCL 2025.3,
Level-Zero, Ubuntu 26.04, TP4, PCIe only (no XeLink).

Results: **~102 tok/s decode** single-stream (5.8x over eager), ~120 ms TTFT, output coherent.
With a batch-size capture ladder: ~830 tok/s aggregate @ C100. Prefill ~1,130 t/s. 256K context;
a maxed 128K prompt ~2-min prefill. Power 300–490 W avg across 4 cards, 585 W peak. Single-stream
ceiling looks like ~100–105 tok/s.

> Relevance to us: this is the **upstream contribution frontier** — un-gating XPU graph capture +
> a capture-safe collective is exactly the "no CUDA-graph capture on XPU" wall we keep hitting (it's
> why our draft spec-decode went 3.4x *slower*). If XPU graph capture lands upstream, our single-card
> TTFT and spec-decode math both flip. Worth tracking whoever did this.

---

## Community repo deep-dive: steveseguin/b70-optimization-lab (investigated 2026-06-18)

`github.com/steveseguin/b70-optimization-lab` — a serious, prolific 4x-B70 optimization lab. We cloned
and grepped it looking for the rumored "Quark W8A8 INT8" kernel. **It is NOT there**, and the investigation
CORRECTS our row #2 above.

**[CORRECTION] The "Quark W8A8 INT8 = 99.77 t/s" data point does not survive contact with this repo:**
- This community uses **"W8A8" to mean FP8 W8A8** (e.g. "static compressed-tensors **W8A8 FP8** safetensors").
  Their repeatedly-cited missing kernel is a **native XPU 128x128 block-FP8 W8A8 GEMM** (FP8, for the official
  Qwen3.6-FP8 block-128 checkpoint) — NOT INT8. They work around it by dequant block-FP8 -> requant to per-channel
  FP8 W8A16.
- The ~99 numbers are **MiniMax-M27 INT4-AutoRound MoE = 99.79 TOTAL tok/s but only ~33 OUTPUT tok/s** (and one
  "99.77" is a gemma4 *compile time*). No INT8-W8A8 decode run exists.
- `grep quark` = 0 real hits; no INT8 kernel, no `_POSSIBLE_INT8_KERNELS[XPU]` patch. Their Qwen 35B paths are
  **INT4-AutoRound** (preferred) and **FP8** (line: "Static FP8 TP4 remains the preferred long-context Qwen layout").
- **Conclusion [REVISED 2026-06-18 — re-opened as a chase target, NOT debunked]:** steveseguin's *public*
  repo shows FP8/INT4, but that does not rule out the 99.77 t/s "Quark W8A8 INT8" run being REAL with a
  **custom INT8 kernel** — which we have now PROVEN is possible (we wrote one; stock vLLM KeyError-crashes, so
  any real W8A8-INT8 B70 run REQUIRED a custom kernel like ours). That run was **Qwen3.6-35B-A3B (MoE, ~3B
  active) on 4x B70** — a different regime from our 14B-dense-1-card 22.6 t/s: MoE decode is fast (few active
  params) and 4 cards add throughput, so ~99 t/s output is plausible there WITH a working int8 kernel.
  **CHASE PLAN (open):** to reach parity we need (1) a **MoE int8 W8A8 kernel** (our dense int8_gemm_w8a8 does
  NOT cover the fused-MoE expert path) + (2) **multi-card** (card #2 incoming) + (3) the 35B-A3B W8A8 checkpoint.
  If steveseguin (or anyone) published an XPU int8/MoE kernel, study it for parity. Status: credible, unmatched
  yet, actively chasing.

**What the repo IS (high-value reference for our OTHER threads):**
- **Custom capture-safe all-reduce / XCCL** work on 4x B70 (lots of `benchmarks/b70_xccl_*`, `patches/...allreduce...`)
  — the dual-card collective + XPU-graph frontier (matches literature/06 contribution #3).
- **vLLM-XPU FP8 fallback patches** (`patches/vllm-...-fp8-fallbacks.patch`, `vllm-xpu-qwen-fp8-bf16-fallback`)
  — how they made Qwen3.6-FP8 load on XPU without the block-FP8 kernel. Directly relevant to our FP8 path.
- **A SECOND missing-kernel target:** native XPU **128x128 block-FP8 W8A8 GEMM** (for Qwen3.6-FP8 checkpoints).
  Sibling to our INT8 W8A8 op in the same vllm-xpu-kernels repo — candidate contribution #1b.
- **llama.cpp SYCL Q4 fusion + all-reduce** patches (MiniMax/Qwen, mmvq2/swiglu fusion, root-copy all-reduce)
  — relevant if we go llama.cpp-SYCL for the MoE (task #9).
- **MiniMax-M27** REAP-AutoRound MoE custom-op tuning (their 89 output / 119 total tok/s headline), model-slot
  configs, and a "LocalMaxxing" optimization leaderboard workflow.
- Cloned at `/mnt/vm_8tb/b70/b70-optimization-lab` on the box for reference.

## How to use this doc

1. Before starting a new experiment, check whether someone already has a number here to beat (or a
   dead end to skip — see the [NEG] rows).
2. When we beat or match one of these on our hardware, record OUR number in FINDINGS.md and link back
   to the row here we were chasing.
3. Add new community data points as rows in the scoreboard; promote anything we decide to actively
   chase into its own [TARGET] section like the 27B run above.
