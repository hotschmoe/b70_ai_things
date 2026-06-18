# B70 Results — live scoreboard

Single Intel Arc Pro B70 (32 GB, Battlemage), unless noted. Backend = upstream vLLM-XPU
built from source (`vllm 0.20.2rc1.dev2+gc51df4300`, image `vllm-xpu-env:tf`) unless noted.
Bench = `vllm bench serve`, random dataset, 512 in / 128 out, `--ignore-eos`. `--enforce-eager`,
no prefix cache, unless noted. Decode/TTFT are means. Newest at top.

## Qwen3-14B (dense, GQA, native Qwen3ForCausalLM)

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

<!-- new result blocks above this line, newest first within each model -->

## To fill in (night campaign)
- [ ] Qwen3-14B F16/BF16 (raw, ~28 GB — VRAM-tight) sweep
- [ ] Qwen3-14B W8A8 INT8 (self-quant, INT8 XMX path) sweep + verify kernel
- [ ] Qwen3-14B sym_int4 (Intel native int4) sweep
- [ ] Qwen3-14B FP8 WITH compilation (piecewise cudagraph, no enforce-eager) — TTFT/throughput
- [ ] Qwen3-14B + draft-model speculative decode (Qwen3-0.6B) — single-stream speedup
- [ ] FP8 KV cache (kv-cache-dtype fp8) effect
- [ ] Newer vLLM (latest main / v0.23.0) comparison — "is newest best?"
