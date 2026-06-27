# SGLang Qwen3.6-27B (qwen3_5 GDN) on 2x Arc B70 -- perf optimization campaign

Goal: make the *correct* SGLang serve (no GDN NaN, unlike vLLM) FAST enough to be a daily driver.
Bench = `sglang/bench2048.sh` (random IN=2048 OUT=128, ignore-eos; matches vLLM scripts/121).
Columns: decode_tps = 1000/TPOT (single-stream TG); prefill_tps (PP) = IN*1000/TTFT; TTFT ms.

## Reference points (from JOURNAL / prior vLLM work)
- vLLM int4 daily driver (single-card, GRAPH): ~30.8 tok/s decode -- but GDN-NaN-prone ("!!!!").
- The whole point of SGLang: CORRECT under mixed prefill+decode. We trade some speed for that;
  the campaign goal is to claw the speed back while staying correct.

## Results table (config -> result)

| # | config                              | conc | decode_tps | TTFT ms | prefill_tps | notes |
|---|-------------------------------------|------|-----------|---------|-------------|-------|
| 0 | bf16 TP=2 (baseline, CTX8192 MF.93) | 1    | 9.03      | 661     | 3098        | out_tok 8.51; the documented baseline |
| 0 | bf16 TP=2 (baseline)                | 4    | 8.18      | 974     | 2103        | aggregate out 24.92 tok/s (4 streams) |
| A | bf16 TP=2 +FLA_FAST +numdecode2     | 1    | 9.34      | 599     | 3419        | coherent OK; TTFT -9% prefill +10% (FLA_FAST helps) |
| A | bf16 TP=2 +FLA_FAST +numdecode2     | 4    | 5.12      | 983     | 2084        | c4 REGRESSED (numdecode2 hurts concurrency) |
| B | bf16 PP=2 (--tp 1 --pp-size 2)      | -    | BROKEN    | -       | -           | /health 200 but ALL gen requests time out -> 500 (scheduler deadlock on GDN); NO-GO |

## Next: AWQ track (the decode win). See sglang/AWQ_RECIPE.md.
- De-risk #1 (fp16-through-GDN, the AWQ act dtype): re-serve UNQUANTIZED bf16 with `--dtype float16` + gdn_nan_repro.
  Isolates the fp16 question from quant before producing any checkpoint.
- De-risk #2 (AWQ XPU speed): repack text-only W4A16 -> AWQ (CPU), serve, bench vs bf16.
- Then production: AutoRound auto_awq full-VLM (vision-retaining).

## Image capability map (VERIFIED 2026-06-27, image sglang-xpu:bmg)
Verified by listing sgl_kernel `.so` files + `torch.ops.sgl_kernel` registration (decisive: registration,
not just `dir(sgl_kernel)` Python wrappers, which exist for unregistered ops):
- WORKING XPU quant GEMM: **AWQ only** -- `awq_dequantize` is the lone registered quant op (2 .so), dequant
  4-bit -> fp16 then native `torch.matmul`. Needs `--quantization awq --dtype float16` + an AutoAWQ-format ckpt.
- DEAD on XPU (no .so / not registered): `int8_scaled_mm`, `fp8_scaled_mm`, `qserve_w4a8_*`, compressed-tensors
  W4A16/W8A8-int8/W8A8-fp8 (WNA16 -> Marlin, CUDA-gated), GPTQ/Marlin. So our W4A16/W4A8/W8A8 compressed-tensors
  AND AutoRound int4 (auto_gptq) checkpoints CANNOT serve on this XPU build.
- MXFP4 W4A16 group-gemm kernels (`GroupGemmMxfp4W4A16Xe20`) EXIST but are MoE-group-gemm only (N/A to dense 27B).
- DEAD (need new kernels/port, NOT flags): MTP/spec-decode (NEXTN draft wired but EAGLE has no intel_xpu attn +
  no xpu graph-runner key), CUDA-graph decode capture (XPUAttentionBackend implements no graph methods),
  `--enable-torch-compile` (no-op on XPU; decode is EagerRunner).

## Levers, ranked
1. **AWQ W4A16 checkpoint** (`--quantization awq --dtype float16`) -- THE decode win (4-bit weight bandwidth),
   may enable TP=1 single-card -> no all-reduce -> DP=2. Needs: produce an AutoAWQ-format ckpt (retain vision +
   GDN exclusions) + validate fp16 GDN numerics (gdn_nan_repro). MAIN THRUST.
2. Pure-flag decode levers (no checkpoint, try-now): `--enable-linear-replayssm`,
   `--num-continuous-decode-steps 2`, env `FLA_USE_FAST_OPS=1`. (Mostly aggregate/batch wins; A/B for single-stream.)
3. `--pp-size 2` vs `--tp 2` A/B (avoid per-layer all-reduce; aggregate/KV win, not single-stream latency).
4. Prefix cache recovery for agentic TTFT: `--mamba-radix-cache-strategy no_buffer --page-size 1` +
   `--attention-backend triton` (drops intel_xpu XMX attn) + `--schedule-policy lpm`. Bigger change; later.
5. Custom dense int8/int4 GEMV SYCL kernel (frontier; the user-offered lever if AWQ underperforms).
