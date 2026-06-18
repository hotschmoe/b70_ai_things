# Dual-B70 readiness — plug-and-play test plan for card #2

Card #2 arrives ~2026-06-19/20. We now have a **known-good stack** (`vllm-xpu-env:v0230`, vLLM 0.23.0)
that runs everything we've tried, including Qwen3.6 DeltaNet. Serve script: `scripts/43_serve_multi.sh`
(TP/PP + the #41663 Battlemage multi-GPU stability env baked in).

## Hardware reality (from docs/literature/02_multigpu.md)
- **No GPU-to-GPU P2P on Arc.** TP all-reduce round-trips through host RAM over PCIe; oneCCL is CPU-driven.
- Our slots are **PCIe 3.0 x16** (~16 GB/s). => **pipeline-parallel (PP) moves ~1000x less data than TP** —
  test both, expect PP to win on our slow link.
- Realistic expectation: **~1.0-1.3x single-stream** from 2 cards on a dense model (you buy capacity +
  concurrency, not single-stream latency). Real wins: bigger models, longer context, more concurrency.

## When card #2 is installed — run in order
0. **Verify both cards:** `ls /dev/dri` (expect card0/card1 + renderD128/renderD129);
   `docker run --rm --device /dev/dri --entrypoint sycl-ls vllm-xpu-env:v0230` (expect 2 level_zero GPUs).
   `lspci -vv` both slots for link width.
1. **The headline goal — Qwen3.6-27B FP8 at long context, TP=2** (fits 64 GB pooled WITH KV, unlike 1 card):
   `runremote 43_serve_multi.sh MODEL=/models/Qwen_Qwen3.6-27B-FP8 SERVED=qwen36-27b QUANT=none TP=2 MAXLEN=32768 UTIL=0.90`
   Then `41_gen.sh` + `35_sweep_bench.sh`. This is the "8-bit Qwen3.6 at long context" target.
2. **Scaling check — Qwen3-14B FP8 TP=2 vs 1 card** (single-card baseline: 35 t/s / 324 agg @ C32):
   `43_serve_multi.sh QUANT=fp8 TP=2` -> sweep. Measure single-stream (expect ~1.0-1.3x) + aggregate.
3. **TP vs PP:** repeat #2 with `TP=1 PP=2`. Compare (PP should fare better on PCIe3).
4. **Two independent instances (data-parallel):** one model per card (ZE_AFFINITY_MASK=0 and =1, ports
   18080/18081). Likely best total throughput, ~0 cross-card penalty. (Need a per-card-pinned serve variant.)
5. **Bigger model (capacity unlock):** Qwen3.6-27B BF16 (54 GB) TP=2, or a 70B-4bit / 35B-A3B MoE.
6. **Qwen3.6 MTP via TP2** — the user's data point says TP2+MTP is NEGATIVE; confirm/skip.

## Watch for
- vLLM #41663 dual-B70 **GP-fault / `xe ... engine reset`** — mitigations in 43_serve_multi.sh
  (CCL_ENABLE_SYCL_KERNELS=0, SYCL_UR_USE_LEVEL_ZERO_V2=0, CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0).
  If it still GP-faults, match Intel's validated BOM (the host kernel/oneCCL versions).
- TP must divide attention head count (Qwen3-14B: 40 Q / 8 KV heads -> TP=2,4,8 ok).
- Power/cooling: 2x 230 W TBP in the case.
- No XPU cudagraph -> keep `--enforce-eager` (compile broke on v0.23.0 anyway).

## Pre-staged
- Qwen3.6-27B-FP8 (29 GB) + Qwen3-14B (28 GB) + Qwen3.6-27B-int4 (15 GB) already on SSD.
- vllm-xpu-env:v0230 (DeltaNet) + :tf (0.20.2 dense) images built.
