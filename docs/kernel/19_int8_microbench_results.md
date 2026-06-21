# 19 -- INT8 vs BF16 GEMM/GEMV microbench RESULTS on the B70 (Xe2)

Date: 2026-06-21. Ran `scripts/68_int8_gemm_gemv_microbench.py` (341-row sweep) on one Arc Pro B70 via the
`:int8` image. INT8 path = the REAL custom op `torch.ops._xpu_C.int8_gemm_w8a8` (our oneDNN s8s8s32 kernel,
contrib/vllm_int8_xpu), bf16 = `torch.matmul`. Shapes = the per-model (K,N) table from
[docs/literature/08](../literature/08_int8_gemm_gemv_xe2_frontier.md). CSV: `results/microbench_gemm_gemv_*.csv`
(also `evals/results/microbench/`). This is the empirical backing for the int8-fast-path claims.

================================================================================
HEADLINE
================================================================================
- **GEMM (prefill, large M): int8 is 1.06-2.13x bf16 (median 1.68x, mean 1.59x).** Speedup GROWS with M as the
  XMX systolic array fills: 14B mlp gate_up = 1.59x @M64 -> 1.86x @M1024 -> **1.97x @M4096**. Peak measured
  **250.9 INT8 TFLOP/s** (14B mlp_down, M=2048). => the int8-XMX prefill win is real and scales with batch/context.
- **GEMV (decode, M=1): int8 is 1.12-2.12x bf16 (median 1.55x), and it is BANDWIDTH-bound + shape-dependent:**
  - LARGE-N dense shapes (the 14B/27B attn+MLP decode GEMVs): **~2x** -- 14B mlp gate_up 2.12x @427 GB/s,
    14B attn QO 2.02x @349 GB/s, 27B attn Q/O ~1.78x. int8 reads half the weight bytes of bf16 -> ~2x.
  - SMALL-N shapes (35B MoE experts N=512/2048; the KV projection N=1024): only **~1.1x** @14-70 GB/s --
    these are too small to saturate BW; they are launch/latency-bound, so halving the bytes barely helps.
  - Peak GEMV bandwidth **433 GB/s** (14B mlp_down) -- near the card's practical ceiling.

================================================================================
WHY THIS RECONCILES THE SERVED-MODEL BENCH (the decode-bytes ordering)
================================================================================
Decode is weight-bandwidth-bound; per-token time ~ (weight bytes read)/BW. So fewer weight bytes = faster decode:
    int4 weight  <  int8 weight  <  bf16 weight      (bytes read, per token)
    => decode t/s:  W4A16/W4A8  >  W8A8  >  bf16
This microbench measures int8-vs-bf16 (int8 wins ~2x at GEMV: half the bytes). The 14B served sweep measured
W8A8 (int8 wt) decode at ~HALF of W4A16/W4A8 (int4 wt) -- same mechanism, one bit-width down. Both consistent:
  - PREFILL (compute-bound): int8-XMX gives ~1.6-2x over bf16 -> W8A8 AND W4A8 cut TTFT (the 14B W4A8 -29% TTFT).
  - DECODE (BW-bound): bytes-read rules -> int4-weight (W4A8/W4A16) > int8-weight (W8A8). W4A8 is the all-rounder
    (int4 decode BW + int8 prefill compute). Exactly what the 14B ctx-2048 ladder showed.

================================================================================
GEMM int8 speedup vs M (14B mlp gate_up, K=5120 N=17408) -- the systolic-fill curve
================================================================================
  M:      64    128   256   512   1024  2048  4096
  int8x:  1.59  1.70  1.60  1.67  1.86  1.94  1.97
  TFLOP/s:45    85    153   212   226   243   245
  -> below M~512 the array is under-fed; >=1024 it saturates near ~1.9-2x and ~245 INT8 TFLOP/s.

================================================================================
GEMV int8 speedup at M=1 (decode), per shape -- where int8 helps vs not
================================================================================
  HELPS (~2x, BW-bound, large N):  14B mlp gate_up 2.12x | 14B attn QO 2.02x | 27B attn Q 1.79x | 27B attn O 1.76x
  PARTIAL (~1.5-1.7x):             14B mlp down 1.71x | ref_sq8192 1.69x | ref_ffn11008 1.59x
  BARELY (~1.1x, overhead-bound):  35B expert gate_up 1.16x | 35B expert down 1.12x | 14B attn KV 1.14x |
                                   35B attn Q/O ~1.13x | 35B dense_sq 1.14x
  => the 35B-A3B MoE expert GEMVs (small N=512/2048) get ~no int8 decode benefit -> reinforces W4A16-int4 as the
     35B decode recipe (kernel/15/18) and explains why an int8 MoE kernel would be a prefill/throughput play only.

================================================================================
NEXT OPTIMIZATION (carried from doc 08)
================================================================================
The small-N / KV-proj GEMVs (~1.1x, 14-70 GB/s) are the headroom: they are NOT hitting BW. The doc-08 P4+P5
lever (column-reorder weight layout + dp4a-style vectorized int8 GEMV) targets exactly these -- coalesce the
16-lane sub-group loads to a contiguous cache line so the small GEMVs reach BW. Worth a focused kernel pass:
re-bench `35B_expert_*` and `14B_attn_KV` after a col-major int8 GEMV variant; goal >1.5x (toward BW) on N<=2048.
