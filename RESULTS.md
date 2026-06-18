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

**Headline: a single B70 serves Qwen3-14B FP8 at ~556 tok/s aggregate @ C64** (still rising; ceiling ~600).
Throughput-vs-latency knob: **~556 t/s @ C64** (max throughput, high TTFT) vs **~330 t/s @ C16**
(29.5 t/s/stream, low latency). Tune `--max-num-seqs` to your workload — the default 16 leaves a lot on the table.

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
| int4 / W4A8 | ~8 GB | deferred | — | no official Qwen3-14B GPTQ-Int4; XPU int4 fragile (AWQ->CUDA torchao #269, GPTQ #39474). Revisit w/ self-quant |

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
- [ ] Qwen3-14B W8A8 INT8 (self-quant, INT8 XMX path) sweep + verify kernel
- [ ] Qwen3-14B sym_int4 (Intel native int4) sweep
- [ ] Qwen3-14B FP8 WITH compilation (piecewise cudagraph, no enforce-eager) — TTFT/throughput
- [ ] Qwen3-14B + draft-model speculative decode (Qwen3-0.6B) — single-stream speedup
- [ ] FP8 KV cache (kv-cache-dtype fp8) effect
- [ ] Newer vLLM (latest main / v0.23.0) comparison — "is newest best?"
