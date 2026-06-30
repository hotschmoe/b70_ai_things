# zml W8A8 int8 GEMM/GEMV sweep results (B70, oneAPI)

Characterization of the int8 (`s8 x s8 -> s32` `dotAcc`) GEMM vs bf16 on ONE Intel Arc Pro B70
(Battlemage/Xe2, INT8 XMX), via `//examples/w8a8_sweep` (median-of-iters timing, warmup discard).
Peaks: 367 INT8-TOPS, 608 GB/s. Generated 2026-06-30 on the zml-w8a8-optimize branch. Variants:
`bf16` (baseline), `i8` (s8 dot, i32 store), `i8b` (s8 dot, bf16 store), `w8a8` (full dynamic
per-token act-quant + s8 dot + dequant), `woq` (weight-only int8: dequant weight to bf16, bf16 act).

## q_proj (K=5120, N=12288) -- M sweep, median ms/call, int8 vs bf16

| M (tokens) | bf16 | i8 (vs bf16) | i8 TFLOP/s (%367) | i8 GB/s (%608) | w8a8 (vs bf16) |
|---|---|---|---|---|---|
| 1 (decode GEMV) | 0.274 ms | **1.57x** | 0.7 | 361 (59%) | 1.20x |
| 512 | 1.229 | **3.65x** | 191 (52%) | 269 (44%) | 2.16x |
| 2048 | 2.256 | 2.12x | 243 (66%) | 164 (27%) | 1.39x |
| 4096 | 3.665 | 1.58x | 222 (60%) | 123 (20%) | 1.32x |
| 8192 | 7.068 | 1.74x | **253 (69%)** | 125 (20%) | STALL* |

## sq4096 (K=N=4096) -- saturation curve

| M | i8 (vs bf16) | i8 TFLOP/s (%367) |
|---|---|---|
| 512 | 2.11x | 111 (30%) |
| 2048 | 2.62x | 196 (53%) |
| 4096 | 2.25x | 223 (61%) |
| 8192 | 1.80x | **246 (67%)** |

## Findings

1. **int8 absolute throughput saturates ~245-253 TFLOP/s = ~67-69% of the 367-TOPS peak** at large M
   (across shapes). This is the realistic int8 ceiling on this oneDNN/Xe2 path -- feeding / occupancy
   bound, NOT the systolic peak. (bf16 tops out ~140-145 TFLOP/s = ~38-40% of the same axis.)
2. **int8 vs bf16 ratio is 1.5x - 3.65x depending on M.** The big ratios at M=512-2048 partly reflect
   bf16 UNDER-performing at mid-M (52-114 TFLOP/s) while int8 already scales; both converge toward their
   ceilings at M>=8192 (ratio ~1.7-1.8x). int8 is a clear 2x+ PREFILL lever.
3. **Decode (M=1) is bandwidth-bound:** bf16 GEMV reaches 460 GB/s (76% of 608); int8 only 361 GB/s
   (59%), so int8 = 1.57x (not ~2x) -- oneDNN's int8 GEMV leaves ~30% bandwidth on the table (the
   llama.cpp #21517 weak-int8-GEMV trap). A better int8 GEMV kernel is the decode headroom.
4. **Full W8A8 (act-quant) caps well below raw i8** (1.20x @ M=1, 1.39x @ M=2048 vs i8's 1.57x/2.12x)
   -- the dynamic per-token act-quant prologue is the overhead. The act-quant DEDUP (share the quant
   across q/k/v and gate/up; landed on this branch) recovered +5.4% decode (13.0 -> 13.7 t/s).
5. **Layout: nk weight {n,k} is OPTIMAL; kn {k,n} is 1.7x SLOWER** (q_proj M=512 i8: 2.23x -> 0.92x).
   The model already stores {n,k} -- do NOT transpose.
6. **Store: i8 (i32 output) BEATS i8b (bf16 output)** -- the i32 accumulate-store is NOT a tax; the
   in-graph down-convert costs more. Keep the i32 dotAcc result.
7. **woq (weight-only int8) is a DEAD END as a perf path** (0.5-1.2x, <= bf16) -- dequant-to-bf16
   materializes the weight, no bandwidth win. (It is still used for the TP row-parallel layers, where
   it is the COHERENT choice -- see ZML_W8A8.md M5 -- not for speed.)

## Measurement caveats (why median timing matters)

- *The M=8192 w8a8 "STALL": every call hit the harness 10s cap (med 10000 ms) -- a hard stall in the
  act-quant path at the largest shape; one i8 M=8192 call also hit 10s (median correctly rejected it
  -> 4.07 ms). This is the SAME one-off-stall family as the earlier "M=2048 = 2502 ms/call" anomaly,
  which was a SINGLE ~250s GPU stall the old mean-of-100 averaged in (real median = 2.56 ms). Always
  read the MEDIAN; the mean/max columns expose the stalls. Large-M act-quant batches occasionally
  stall hard on this plugin -- worth an ONEDNN_VERBOSE diff of the M=8192 vs M=2048 primitive, later.

## Actionable summary

- Keep `nk` weight layout + `i32` dotAcc store (both already in place).
- int8 is a 2x+ PREFILL lever (saturates ~67% of peak); ~1.3-1.6x at DECODE (act-quant + GEMV-BW bound).
- Decode headroom: a better int8 GEMV kernel (close the 361->460 GB/s gap) + MTP.
- Avoid the largest-M (>=8192) act-quant path until the stall is understood.
