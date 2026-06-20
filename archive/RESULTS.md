# B70 Results — live scoreboard

Single Intel Arc Pro B70 (32 GB, Battlemage), unless noted. Backend = upstream vLLM-XPU
built from source (`vllm 0.20.2rc1.dev2+gc51df4300`, image `vllm-xpu-env:tf`) unless noted.
Bench = `vllm bench serve`, random dataset, 512 in / 128 out, `--ignore-eos`. `--enforce-eager`,
no prefix cache, unless noted. Decode/TTFT are means. Newest at top.

## Qwen3.6-27B (Gated-DeltaNet) — RUNS ON A SINGLE B70 (int4, v0.23.0)  [2026-06-18]

After being blocked all session (llama.cpp SYCL has no DeltaNet; both vLLM FP8 paths had XPU kernel gaps),
**Qwen3.6-27B generates on ONE Arc Pro B70** via `Lorbus/Qwen3.6-27B-int4-AutoRound` on `vllm-xpu-env:v0230`
(vLLM 0.23.0). Kernels: `Triton/FLA GDN prefill kernel` + AutoRound int4. Output is coherent (emits
Qwen3.6 `<think>` tokens correctly).

| Metric | Value |
|---|---|
| Fits | **YES** — model 17.56 GiB + KV 7.54 GiB (75,776 tok, 18.5x conc @ 4k ctx) |
| Coherent single-stream decode | **7.89 t/s** (slow — Triton/FLA DeltaNet kernel unoptimized on XPU) |
| First-token (cold) | ~19 s (GDN kernel JIT on first request); steady after |
| Quality | coherent, thinking-mode works |

Concurrency (warm; `vllm bench serve`, GDN cold-JIT ~19 s must warm first): C4 = **21.6 t/s output /
108 t/s total**, TPOT 129 ms, TTFT ~3.7 s. Aggregate scales with concurrency but less efficiently than
dense FP8 (GDN/linear-attn kernel batches poorly). (Full 1-32 sweep tool is flaky on GDN models - cold
requests time out; warm single runs are clean.)

Significance: **the only known way to run Qwen3.6-27B on a single B70 today.** Decode is bandwidth+kernel
limited (15 GB int4 -> ~40 t/s ceiling, only ~20% achieved -> XPU DeltaNet kernels are the bottleneck,
not VRAM). Optimization target: faster XPU GDN/linear-attention kernels. 2-card BF16/FP8 will be much faster.

## Qwen3-14B (dense, GQA, native Qwen3ForCausalLM)

### W8A8 (ours) vs FP8 — PREFILL head-to-head — 2026-06-18  [INT8 WINS PREFILL ~1.6x]
Same model (Qwen3-14B), v0230, identical bench (random, varying in:out:conc). Prefill tok/s ~= in_len/TTFT.
**The native-INT8 systolic advantage is real and measured:**

| in:out:conc | Prefill tok/s ours / FP8 | speedup | TTFT ms ours / FP8 | decode t/s ours / FP8 |
|---|---|---|---|---|
| 4096:8:1  | **6353 / 3997** | **1.59x** | **645 / 1025** | 7.9 / 6.5 |
| 2048:64:8 | **14648 / 9176** | **1.60x** | 1119 / 1786 | 84.8 / 82.9 |
| 4096:8:8  | **10888 / 6802** | **1.60x** | 3010 / 4817 | 11.9 / 7.5 |
| 512:128:1 | (JIT-cold) / 3235 | - | -/158 | 13.2* / **29.5** |
| 8192:8:1  | (max-len fail) / 3684 | - | -/2224 | -/3.3 |

- **Prefill / compute-bound: INT8 W8A8 ~1.6x FP8** (FP8 = conversion-based on Xe2; INT8 = native s8s8s32).
  This is the value proposition for the kernel. TTFT ~1.6x lower too.
- **Decode / batch-1: FP8 wins** (~29 vs ~13 t/s) -- dragged by the `dynamic_per_token_int8_quant_ref`
  reference quant (eager, per-layer). FIX = fused SYCL per-token int8 quant op (make-it-fast #11).
- **[FUSED-QUANT UPDATE 2026-06-18]** Wrote a fused SYCL `dynamic_per_token_int8_quant` op (one work-group/row,
  sub-group absmax reduction; q-match 100% vs ref) -> **single-stream decode 13 -> 22.6 t/s (1.7x), now ~78%
  of FP8 (29)**; TTFT 142 ms; prefill unchanged (6325, still 1.6x FP8); C32 decode agg 465. So INT8 W8A8 now
  WINS prefill 1.6x AND nearly matches decode. Remaining decode gap = the M=1 int8 GEMM (oneDNN single-row) vs
  FP8 -- diminishing returns. Net: INT8 W8A8 is the well-rounded throughput champion on B70.
- CAVEATS: 512:128:1 W8A8 row is JIT-cold (first-req oneDNN compile, TTFT 2065ms) -- ignore. 8192 failed only
  on --max-model-len 8192. C32 rows omitted (confounded: FP8 --max-num-seqs 16 vs W8A8 default).

### W8A8 INT8 via OUR XPUInt8ScaledMMLinearKernel — 2026-06-18  [WORKS — we wrote the kernel]
We implemented the missing XPU INT8 W8A8 path: a native `int8_gemm_w8a8` oneDNN op (s8 x s8 -> s32 ->
dequant) in vllm-xpu-kernels + `XPUInt8ScaledMMLinearKernel` + `_POSSIBLE_INT8_KERNELS[XPU]` in vLLM (see
docs/kernel/01 + contrib/vllm_int8_xpu/ + scripts/44,45). The self-quant `Qwen3-14B-W8A8-INT8` that
**hard-crashed on stock vLLM** (`KeyError: PlatformEnum.XPU`) now SERVES:

| Metric | Value |
|---|---|
| Kernel selected | **`Selected XPUInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8`** (was KeyError) |
| Serves? | **YES** — `Application startup complete`, HEALTHY |
| Numerical | op verified max_abs_err 2.4e-4 vs reference (fp16 rounding) |
| Footprint | model 15.34 GiB + KV 10.85 GiB (71,040 tok, 8.67x conc @ 8k) |
| Generation | coherent ("...is Paris...") via the int8 kernel at inference |
| Status | MAKE IT WORK done; perf bench (prefill/large-batch vs FP8) + asym/AZP = make-it-fast/right TODO |

First working INT8 W8A8 on Battlemage in vLLM (novel). The chooser `.get()` hardening also fixes the
GDN-FP8 KeyError family. Upstream PR targets: vllm-xpu-kernels (the op) + vLLM (kernel class + registry).

### W4A8-INT (self-quant) — 2026-06-18  [OK kernel engaged; decode kernel UNOPTIMIZED]
Self-quantized `Qwen3-14B-W4A8-INT` (int4 sym group-128 weights + per-token dynamic int8 activations,
data-free RTN). Served on vLLM 0.23.0, `--dtype float16`. **Confirmed kernel: `Using XPUW4A8IntLinearKernel
for CompressedTensorsW4A8Int`** (oneDNN `int4_gemm_w4a8`) — the ONLY upstream path that lights the INT8 XMX
datapath. Coherent output. Model load **9.3 GiB** (smallest of all; vs FP8 15.2), KV 15.0 GiB / 98,496 tok /
**12.0x** concurrency @ 8k ctx.

| Concurrency | Aggregate out tok/s | Per-stream decode | Mean TTFT |
|---:|---:|---:|---:|
| 1  | 16.4  | **16.6** | 155 ms |
| 4  | 63.4  | 16.4 | 322 ms |
| 8  | 121.1 | 16.0 | 500 ms |
| 16 | 214.7 | 14.5 | 770 ms |
| 32 | **374.4** | 12.9 | 1067 ms |

**Head-to-head vs FP8** (both Qwen3-14B, v0230, identical bench harness, random 512/128):

| C | FP8 agg / per-stream | W4A8 agg / per-stream |
|--:|--:|--:|
| 1  | **28.7 / 29.5** | 16.4 / 16.6 |
| 4  | **104.9 / 28.3** | 63.4 / 16.4 |
| 8  | **195.0 / 27.6** | 121.1 / 16.0 |
| 16 | **329.2 / 24.4** | 214.7 / 14.5 |
| 32 | 329.8 / 23.1 | 374.4 / 12.9 |

**Verdict:** the INT8-XMX path WORKS and is the lightest-VRAM option, but **FP8 wins on speed at every
matched concurrency** — per-stream ~1.7-1.8x faster (29.5 vs 16.6 at C1; 24.4 vs 14.5 at C16), and higher
aggregate through C16. W4A8 single-stream (16.6) is only ~25% of its 9.3 GB bandwidth ceiling (~65) -> the
`int4_gemm_w4a8` decode kernel is unoptimized (same story as GDN). **CAVEAT on the C32 row:** FP8 here ran
with `--max-num-seqs 16` (script 36 default) and W4A8 with vLLM's higher default, so the C32 aggregate is
NOT comparable -- the apparent W4A8 "edge" at C32 is a max-num-seqs artifact, not a real win. **Bottom line:
use FP8 for everything; W4A8's only genuine advantage is footprint (9.3 vs 15.2 GB).** Optimization target:
faster `int4_gemm_w4a8` decode kernel.

### W8A8 INT8 (self-quant, compressed-tensors) — 2026-06-18  [HARD FAIL — no XPU kernel]
Produced `Qwen3-14B-W8A8-INT8` (16 GB) from local BF16 via llm-compressor data-free RTN (int8 per-channel
weights + per-token dynamic int8 activations). Served on vLLM 0.23.0 (XPU). vLLM picked scheme
`CompressedTensorsW8A8Int8`, then **crashed at layer-0 `create_weights`**:
`choose_scaled_mm_linear_kernel -> possible_kernels[XPU]` => **`KeyError: <PlatformEnum.XPU: 4>`**.

| Metric | Value |
|---|---|
| Serves? | **NO** — engine core init fails at model load (no bench possible) |
| Footprint | 16 GB checkpoint on disk (never reached VRAM alloc) |
| Root cause | INT8 scaled-MM kernel registry has **no XPU entry** — same chooser/KeyError as the GDN-FP8 path |
| Verdict | **W8A8 INT8 is unusable on Battlemage vLLM.** Use FP8. INT8-XMX only reachable via W4A8. |

(Empirically confirms docs/literature/05_w8a8_recipe.md, which was source-read only. Closes the
"verify kernel" TODO below.)


### FP8 PEAK throughput (max-num-seqs=64, v0.23.0 eager) — 2026-06-18
**The earlier ~324 t/s "plateau" was an artifact of the default `--max-num-seqs 16` capping concurrency.**
Raising it to 64 (+ `--max-num-batched-tokens 8192`) unlocks the real curve:

| Concurrency | Aggregate out tok/s | Per-stream decode | Mean TTFT |
|---:|---:|---:|---:|
| 8  | 202 | 30.8 | 0.9 s |
| 16 | 330 | 29.5 | 1.9 s |
| 32 | 459 | 21.0 | 2.9 s |
| 48 | 525 | 15.3 | 3.4 s |
| 64 | **556** | 12.0 | 4.1 s |

**Headline: a single B70 serves Qwen3-14B FP8 at ~558 tok/s aggregate, saturating at C64.** Confirmed
ceiling: C64=558, C96=558, C110=556 — beyond C64 throughput is flat and only TTFT grows (4.1->10.3->13.4 s),
i.e. compute-bound saturation. Throughput-vs-latency knob: **~558 t/s @ C64** (max throughput) vs
**~330 t/s @ C16** (29.5 t/s/stream, low latency). Tune `--max-num-seqs` to your workload — the default 16
caps you at ~330.

### FP8 (online, XPUFP8ScaledMMLinearKernel) — 2026-06-17
Model 15.2 GiB + KV 11.7 GiB (76,544 tok), max ctx 16k, util 0.90, FlashAttn v2.

| Concurrency | req/s | Aggregate out tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Per-stream decode (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.19 | 24.8  | 1032 | 32.5 | 30.8 |
| 4  | 0.85 | 108.9 | 402  | 33.9 | 29.5 |
| 8  | 1.41 | 181.1 | 1003 | 36.6 | 27.3 |
| 16 | 2.52 | 322.5 | 1044 | 41.7 | 24.0 |
| 32 | 2.53 | 324.1 | 6719 | 43.7 | 22.9 |

Notes: single-stream ~31 t/s = ~77% of the ~40 t/s 15 GB-bandwidth ceiling. Aggregate plateaus
~324 t/s at C16-32 (~10.5x batching). TTFT balloons at C32 (6.7 s) — enforce-eager + chunked-prefill
contention; compilation (piecewise cudagraph) is the obvious next lever.

### FP8 + inductor compilation (no enforce-eager) — 2026-06-17
Same as above but `--compilation-config '{"cudagraph_mode":"PIECEWISE",...}'`. NOTE: XPU has no
CUDA-graph capture (log: "Skipping CUDA graph capture") so only **inductor torch.compile** applies
(graphs cached to SSD; first start +~3.5 min). KV dropped to 9.86 GiB (64,640 tok) — compile buffers.

| Concurrency | req/s | Aggregate out tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Per-stream decode (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.26 | 33.0  | **176**  | 29.1 | **34.3** |
| 4  | 0.95 | 121.9 | 399  | 29.9 | 33.4 |
| 8  | 1.61 | 205.6 | 689  | 33.7 | 29.6 |
| 16 | 2.57 | 329.4 | 1045 | 40.7 | 24.6 |
| 32 | 2.57 | 329.5 | 6601 | 43.0 | 23.3 |

**Verdict: compile WINS at low concurrency** — TTFT 1032->176 ms (5.9x), decode +11% (30.8->34.3) at C1;
negligible at C16-32 (saturated). Keep compile for latency; eager fine for max-batch. (Counter to the
generic "use enforce-eager on XPU" advice — measured otherwise on this build.)

### FP8 + draft-model spec-decode (Qwen3-0.6B, num_spec=4) — 2026-06-17  [NEGATIVE RESULT]
Coherent single-stream (3 prompts, 200 tok). Same FP8+compile config + Qwen3-0.6B drafter.

| Config | Coherent decode (tok/s) | TTFT | Acceptance |
|---|---:|---:|---|
| FP8+compile (baseline) | **35.28** | ~62 ms | — |
| FP8+compile + draft (n=4) | **10.33** | ~105 ms | 45.8% (403/880), mean accept len ~1.83 |

**Spec-decode is 3.4x SLOWER on B70 — do NOT use draft-model spec-decode here (for now).**
Root cause: XPU has **no CUDA-graph capture**, so per-forward kernel-launch overhead is high; spec-decode
runs **5 forwards/step** (4 draft + 1 verify) vs 1, and ~1.8x token acceptance can't pay for 5x the launches.
Per-position accepts: pos0=164, pos1=110, pos2=78, pos3=51 (normal decay). This will only flip positive once
XPU gets graph capture or a much cheaper drafter (EAGLE-style single-pass, not yet on XPU).

### F16 / BF16 raw (no quant) — 2026-06-17
util 0.95, max-model-len 2048 (VRAM-tight: 28 GB weights leave little KV), eager. Coherent single-stream.

| Metric | Value |
|---|---|
| Coherent single-stream decode | **18.67 t/s** |
| TTFT (short prompt) | ~83 ms |
| Footprint | ~28 GB (barely fits; tiny KV; no room for compile/concurrency) |

Bandwidth ceiling 608/28 ~= 21.7 t/s -> 86% efficiency. **FP8 is ~1.9x faster (35.3 vs 18.7)** AND near-lossless,
so F16 is dominated on a single B70 (more bytes/token + barely fits). Use FP8.

#### Single-B70 Qwen3-14B summary so far (single-stream decode)
| Quant | Footprint | Decode t/s | TTFT | Notes |
|---|---|---:|---:|---|
| F16/BF16 | ~28 GB | 18.7 | ~83 ms | tight, full quality, slowest |
| FP8 (+compile) | ~15 GB | **35.3** | **~62 ms** | near-lossless, sweet spot |
| FP8 + draft spec-decode | ~16 GB | 10.3 | ~105 ms | NEGATIVE (no XPU cudagraph) |
| W4A8-INT (self-quant) | ~9.3 GB | 16.6 | ~155 ms | INT8-XMX kernel ENGAGED (XPUW4A8IntLinearKernel) but decode kernel unoptimized (~half FP8). Lightest VRAM, best high-conc aggregate. See W4A8 block above |

### vLLM v0.23.0 vs 0.20.2 — Qwen3-14B FP8 (eager) — 2026-06-18
Built v0.23.0 from source (torch 2.11.0+xpu). NOTE: **compilation BROKE on v0.23.0** (torch 2.11 + inductor;
the flagged regression) — ran eager. Same bench (random 512/128).

| C | 0.20.2 eager agg t/s | v0.23.0 eager agg t/s | 0.20.2 TTFT | **v0.23.0 TTFT** | decode (both ~) |
|---:|---:|---:|---:|---:|---:|
| 1  | 24.8  | 30.0  | 1032 ms | **147 ms** | ~30.8 |
| 8  | 181.1 | 189.6 | 1003 ms | 878 ms | ~28 |
| 32 | 324.1 | **333.7** | 6601 ms | 6501 ms | ~23.5 |

**Verdict: newest (v0.23.0) is the better backend.** ~Same throughput, but **~7x better out-of-box TTFT**
at C1 (147 vs 1032 ms — matches 0.20.2-*compiled* without needing compile), AND it's the **only** version
that loads Qwen3.6 DeltaNet on XPU. Downside: inductor compile broke (torch 2.11). Pin **v0.23.0** as primary.

<!-- new result blocks above this line, newest first within each model -->

## To fill in (night campaign)
- [ ] Qwen3-14B F16/BF16 (raw, ~28 GB — VRAM-tight) sweep
- [x] Qwen3-14B W8A8 INT8 (self-quant, INT8 XMX path) sweep + verify kernel -> **HARD FAIL** (KeyError XPU at load; no XPU INT8 kernel). See W8A8 block above + JOURNAL 06-18.
- [ ] Qwen3-14B sym_int4 (Intel native int4) sweep
- [ ] Qwen3-14B FP8 WITH compilation (piecewise cudagraph, no enforce-eager) — TTFT/throughput
- [ ] Qwen3-14B + draft-model speculative decode (Qwen3-0.6B) — single-stream speedup
- [ ] FP8 KV cache (kv-cache-dtype fp8) effect
- [ ] Newer vLLM (latest main / v0.23.0) comparison — "is newest best?"
