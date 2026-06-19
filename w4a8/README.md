# W4A8-INT8 for single-card B70 -- dedicated workstream

Goal: a **W4A8-INT8** Qwen3.6-27B (and Gemma-4-31B) that fits ONE B70 and beats packed
W4A16 by lighting the int8-XMX datapath (`XPUW4A8IntLinearKernel`, oneDNN `int4_gemm_w4a8`).
14B is the test bed; nail it before touching 27B/31B.

Status as of 2026-06-20: **GPU busy (~1h) -- ALL work queued, nothing running.** Per user:
wait for the B70 to free and **use it to accelerate quantization** (don't burn the wait on slow
CPU quant). Decisions locked: recoverability method = **AutoRound** (B70-accelerated); packing
= **test first** (CPU, runs in parallel once we start). Scripts written, validate on first run.

Compute placement: AutoRound rounding -> **B70 (XPU)** via `10_quant_autoround_w4a8.sh DEVICE=xpu`
(retest the old "XPU calibration unreliable" caveat -- that was llm-compressor SmoothQuant, not
AutoRound). The trivial data-free RTN packing probe (`11_test_packed_export.sh`) has no GPU
benefit, so it can run on CPU concurrently while the B70 does AutoRound.

---

## Why W4A8 at all (the bar it must clear)

Single-card fit already works today with **packed W4A16 / int4-AutoRound** (fp16 activations,
weight-only). W4A8's ONLY reason to exist over that is int8 activations -> int8-XMX -> better
throughput under concurrency / prefill. So W4A8 is only worth shipping if it BEATS this bar:

| 14B quant | size | decode t/s | HumanEval+ plus | activations | kernel |
|-----------|------|-----------|-----------------|-------------|--------|
| fp8 (anchor) | ~15 GB | 32.1 | 0.890 | fp8 | XPUFp8 (good) |
| w8a8 | 16 GB | 23.8 | 0.860 | int8 dyn | XPUInt8 (ours) |
| **w4a16 (gptq, packed)** | **9.3 GB** | **29.1** | **0.829** | fp16 | wNa16 (good) |
| **w4a8-int (current, RTN)** | **16 GB** | **16.5** | **0.817** | int8 dyn | int4_gemm_w4a8 (unoptimized) |

Today W4A8 is **worst-of-both**: W8A8-sized, slowest decode, lowest accuracy. Packed W4A16
strictly dominates it on size, speed, AND accuracy. To flip that we need all three wins below.

Size ladder (measured, on Unraid `/mnt/vm_8tb/b70/models/`):
```
  14B  W4A8-INT (unpacked I8)   16 GB     <- no fit win vs W8A8
  14B  W4A16-gptq (packed)       9.3 GB
  14B  W8A8-gptq                 16 GB
  27B  W4A16 (packed)            25 GB    <- fits 1x today
  27B  W8A8-INT8-RTN             33 GB    <- == what 27B W4A8-unpacked would be; won't fit 1x
  27B  BF16                      72 GB
```

---

## The three wins (all required; track each)

### Win 1 -- PACKING (the single-card gate)  [status: TESTING]
Our `scripts/43_quantize_w4a8.sh` saved weights as **`int-quantized` = full int8 tensors**
(verified: `gate_proj.weight` is `dtype I8, shape [17408,5120]`, 1 byte/weight). 4-bit in
value, 8-bit on disk -> 16 GB, identical to W8A8. A 27B would be ~33 GB -> never fits 1x.

- **Target:** int4-PACKED storage (pack-quantized / int32-packed, like W4A16-gptq's 9.3 GB).
  14B -> ~9 GB, 27B -> ~16-17 GB (fits 1x with KV).
- **Open question (make-or-break):** does vLLM-XPU's `CompressedTensorsW4A8Int` +
  `XPUW4A8IntLinearKernel` LOAD packed weights, or does it require the unpacked int8 layout?
  Also: does the kernel repack int8->int4 at load (so resident VRAM < 16 GB even now)? We have
  no serve-log footprint captured -- MUST grep `model weights take X GiB` on next serve.
- **Approach:** (a) CPU export with `quantization_format="pack-quantized"` -> check size + dtype
  [`w4a8/11_test_packed_export.sh`]; (b) serve on B70, confirm kernel still selected + measure
  resident weight GiB. If packing isn't accepted -> W4A8 single-card thesis is dead, pivot to
  packed W4A16 (escalate decision).
- **Note:** AutoRound's compressed-tensors export (Win 2) may ALSO pack -> could close Win 1+2
  together. The quick RTN pack-export proof de-risks the format question first.

### Win 2 -- ACCURACY (recoverability)  [status: READY, method=AutoRound]
Current W4A8 is **data-free RTN** -> worst quality (0.817 plus). Replace with recoverability.
- **Method (chosen): AutoRound** (Intel sign-gradient int4 rounding; what the strong 27B int4
  0.927-plus used). AutoRound optimizes int4 WEIGHTS; we add **int8 dynamic activations**
  (`--act_bits 8 --act_dynamic`) and export to a compressed-tensors W4A8 that routes to
  `XPUW4A8IntLinearKernel`. Script: `w4a8/10_quant_autoround_w4a8.sh`.
- **Fallback / comparison:** SmoothQuant+GPTQ natively does the full W4A8 scheme in one recipe
  (mirror `scripts/40_quantize_w8a8.sh` DATAFREE=0, swap scheme W8A8->W4A8). Keep as the
  baseline to beat; GPTQ gave +4pts on w4a16/w8a8.
- **Target:** 14B plus-pass 0.817 -> **>= 0.84** (beat packed w4a16's 0.829), ideally approach
  w8a8's 0.860. Eval = HumanEval+ Tier-1 (164, thinking-off, greedy), same harness as the table.
- **Runtime caveat:** XPU calibration is unreliable -> AutoRound runs on **CPU**. 14B AutoRound
  with default iters is SLOW (likely many hours). Start with a low-iters smoke run to validate
  flags + export format, then a full overnight run.

### Win 3 -- KERNEL (decode speed)  [status: BACKLOG, longer effort]
`int4_gemm_w4a8` DECODE is unoptimized: 16.5 t/s vs w4a16's 29.1 at smaller weights. Per-token
dynamic int8 activation-quant overhead + an unoptimized oneDNN int4 decode path. (Prefill is
fine: 4403 t/s, int8-XMX compute-bound.)
- **Approach:** profile `int4_gemm_w4a8` decode vs `int8_gemm_w8a8` / `wNa16` in
  `contrib/vllm_int8_xpu` / `vllm-xpu-kernels`; find the bottleneck (act-quant fusion? int4
  unpack in the GEMV path?); optimize / fuse.
- **Success:** single-stream decode >= w4a16 (~29 t/s) while keeping the int8-XMX batch/prefill
  edge. Until then, W4A8's case rests on aggregate throughput under concurrency (measure vs
  fp8/w8a8 at C16-64), not single-stream latency.

---

## Execution order (ALL gated on GPU free -- nothing runs until then)
0. [trigger] B70 free -> begin. (CPU packing probe may run in parallel with the B70 quant.)
1. [B70] `10_quant_autoround_w4a8.sh DEVICE=xpu ITERS=50` smoke -> validate XPU toolchain +
   --act/--format flags + that export PACKS (size, dtype). (Wins 1+2 together if it packs.)
   [CPU, parallel] `11_test_packed_export.sh` -> independent proof pack-quantized -> ~9 GB.
2. [B70] serve the smoke output -> XPUW4A8IntLinearKernel selected? resident GiB? coherent? (Win 1b)
3. [B70] full AutoRound run (ITERS=200, NSAMPLES>=128).
4. [B70] serve + HumanEval+ Tier-1 -> compare to table; decide vs the packed-w4a16 bar (0.829).
5. Only if 14B clears the bar: replicate on Qwen3.6-27B + Gemma-4-31B (DeltaNet/MoE/vision
   calibration caveats apply).

## Experiment log
(newest at bottom; config -> command -> result -> verdict)

- 2026-06-20 -- workstream opened. Confirmed current `Qwen3-14B-W4A8-INT` is 16 GB unpacked I8
  (no fit win), decode 16.5 t/s (worst), plus 0.817 (worst). Three-win plan above. Method
  AutoRound, packing test first. (Corrected JOURNAL mislabel: w4a8 is 16 GB, NOT 9.3 GB --
  9.3 GB is W4A16-gptq.)
