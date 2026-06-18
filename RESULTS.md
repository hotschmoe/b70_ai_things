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

<!-- new result blocks above this line, newest first within each model -->

## To fill in (night campaign)
- [ ] Qwen3-14B F16/BF16 (raw, ~28 GB — VRAM-tight) sweep
- [ ] Qwen3-14B W8A8 INT8 (self-quant, INT8 XMX path) sweep + verify kernel
- [ ] Qwen3-14B sym_int4 (Intel native int4) sweep
- [ ] Qwen3-14B FP8 WITH compilation (piecewise cudagraph, no enforce-eager) — TTFT/throughput
- [ ] Qwen3-14B + draft-model speculative decode (Qwen3-0.6B) — single-stream speedup
- [ ] FP8 KV cache (kv-cache-dtype fp8) effect
- [ ] Newer vLLM (latest main / v0.23.0) comparison — "is newest best?"
