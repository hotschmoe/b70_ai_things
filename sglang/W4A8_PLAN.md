# W4A8 on sglang-XPU (B70) -- plan + thesis (opened 2026-06-28)

Goal (user): a W4A8 Qwen3.6-27B for sglang-XPU that HANDILY beats int4/W4A16 on
prefill (PP), TTFT, and decode (TG); retains vision; minimizes size (packed int4);
uses our own GEMV/GEMM to activate the Intel int4/int8 fastpaths. sglang is target.

## The thesis (why W4A8 can win on B70)
Two measured kernel facts from the sglang campaign (sglang/PERF.md):
- int4 `woqgemm` (auto_round_kernel): DECODE M=1 = 2.17-2.31x bf16 (realizes 4-bit
  weight bandwidth), but PREFILL M=512 = 0.56x bf16 (SLOWER -- it dequantizes int4
  -> fp16 and runs an fp16 matmul; no int4 compute fastpath).
- `torch._int_mm` (oneDNN INT8 XMX/DPAS): PREFILL = 1.24-1.61x bf16 at M>=512, but
  DECODE M=1 launch-bound (quant+_int_mm+dequant >= 3 launches/layer vs woqgemm's 1).

=> W4A8 = int4 weights (memory + decode bandwidth, fits 1 card, big KV) + int8 acts
   (prefill via INT8 XMX). HYBRID kernel: decode -> woqgemm int4; prefill -> int8 XMX.
   Expected: prefill ~1.2-1.6x bf16 = ~2.2-2.9x int4-woq's 0.56x (HANDILY beats int4
   on PP/TTFT), decode ~= int4 champion (woqgemm), memory ~= int4. Accuracy >= int4
   (int8 acts >> the implicit fp16-with-int4-weight of W4A16 only in that acts are
   quantized; need an int8-act-aware calibration if naive int8 acts hurt).

## The core kernel problem: group-128 weight scales
Both int4 checkpoints (Lorbus AutoRound, the sqgptq W4A8) use group_size=128 along K
with per-(out-channel, K-group) scales: scales shape [K/128, out].
`torch._int_mm` does ONE int32 accumulation over the FULL K then applies ONE scale --
it cannot apply a per-128-group weight scale mid-accumulation. So the int8-XMX prefill
on group-128 int4 weights needs one of:
  (A) grouped accumulation: split K into K/128 chunks, _int_mm each [M,128]x[128,N],
      scale by the group's weight scale, accumulate. K/128=136 small int8 GEMMs/layer.
      Risk: small-K (128) GEMMs may underutilize XMX + launch overhead.
  (B) per-channel re-quant: dequant int4(group128)->fp, requant int8 per-channel
      (group_size=-1), single _int_mm. Kernel-clean, but per-forward dequant+requant
      cost (O(weight)) and a small accuracy hit (per-channel < per-group).
  (C) custom fused SYCL int4w x int8a grouped GEMM (the real QServe-style kernel).
      Largest effort; triton-int8 is a DEAD END on B70 (10x slower than _int_mm).

## Decision gate (this session): sglang/w4a8_probe.py
Microbench on a REAL Lorbus layer-20 down_proj (in=17408, out=5120), card 0, image
sglang-xpu:woq. Compare at M=1 (decode) and M=2048 (prefill), with HONEST int4->int8
conversion cost included for the W4A8 prefill candidates:
  - bf16 matmul (baseline)
  - woqgemm int4 (decode champion / int4 prefill)
  - W8A8 per-channel _int_mm (the proven 1.2-1.6x prefill)
  - W4A8-A grouped-128 _int_mm loop
  - W4A8-B per-channel-requant _int_mm
GATE: a W4A8 prefill candidate must HANDILY beat bf16 (and thus 2x+ int4-woq) at
M=2048 while decode (woqgemm) stays at int4-champion level. If yes -> integrate
(w4a8_shim.py). If no -> int8-XMX prefill is not realizable from group-128 int4 on
B70 without a custom SYCL kernel; report honestly + scope (C).

## Plan after the gate
1. w4a8_shim.py: reuse woq_shim QuantLinearGPTQ int4 weight creation (decode path);
   add the winning int8 prefill branch (M>threshold). Gate B70_XPU_W4A8=1.
2. Build sglang-xpu:w4a8 (woq + shim). Serve Lorbus int4 weights AS W4A8.
3. Validate: coherence, vision, gdn_nan_repro under mixed load.
4. Accuracy: HumanEval+ vs int4 champion. If naive int8-acts hurt -> requant with
   SmoothQuant/GPTQ int8-act-aware calibration (keep vision via graft_vision.py,
   pack int4, quantize GDN -- avoid the over-broad bf16 ignore list).
5. End-to-end bench vs 23.5 t/s int4+XPUGraph champion + bf16 TP=2. Stack XPUGraph
   on the W4A8 decode path (woqgemm is graph-captured already in the int4 driver).

## GATE RESULTS (2026-06-28, card 0, sglang-xpu:woq, real Lorbus down_proj 17408x5120)
Microbenches: w4a8_probe.py / probe2 / probe3 / probe4. ms/call, warm.

Gate-1 (w4a8_probe.py):
  candidate         M=1        M=2048    speedup vs bf16
  bf16              0.328      2.889     1.00x / 1.00x
  woqgemm int4      0.145      3.762     2.26x / 0.77x   <- decode champ; prefill weak
  w8a8 _int_mm NAIVE 0.375     3.806     0.87x / 0.76x   <- un-fused epilogue kills it
  w4a8 grouped-128  27.1       89.6      0.01x / 0.03x   <- 136 per-group _int_mm in py = DEAD
  int4->int8 unpack microcost = 4.26 ms (> the whole bf16 matmul) -> per-forward materialize is DEAD

Gate-2 (torch.compile fusion): the int8-XMX prefill win IS real WITH fusion:
  w8a8 compiled: M=512 1.67x, M=2048 1.84x bf16 (M=1 only 1.07x -> int8 acts help PREFILL not decode).
  => int8-XMX prefill is reachable ONLY via oneDNN torch._int_mm + a FUSED epilogue + a MATERIALIZED int8 weight.

Gate-3/4 (auto_round_kernel API): the fused int4w x int8a kernel EXISTS but is HARD-GATED on this box:
  woqgemm(A,B,bias,n,k,gs,compute_type,weight_type,scale_type,asym): weight_type=int4, compute_type=int8
    -> M=1 runs (GEMV path, correct, relerr 0.0017) but NOT faster (the fp16-vs-others delta was cold-first noise)
    -> M=2048 FAILS: "no matrix hardware on the target device, joint_matrix is not supported"
       (the oneAPI<2026 SYCL int8-XMX gate the woqgemm banner warns about). So fused W4A8 PREFILL is blocked.
  woqgemm_s8(A,B,scaleB,bias): fused int8w x int8a (W8A8), single-launch, 1.44x bf16 @M=2048 -- but int8 WEIGHT
    (not int4) and returns C dtype=A.dtype (int8 -> overflow with int8 A; needs fp A or a fixed convention).
  igemm_s8s8s32(A,B): pure int8xint8->int32 (== torch._int_mm).

## VERDICT: W4A8 fast-prefill is KERNEL-WALLED on B70 + oneAPI<2026
There is NO cheap path to int4-stored-weight + int8-XMX-prefill on this stack:
  (a) auto_round woqgemm int8 compute -> joint_matrix unsupported (FAILS at M>1).
  (b) oneDNN torch._int_mm int8-XMX works (1.84x) but needs a MATERIALIZED int8 weight
      (per-forward int4->int8 = 4.26ms/layer = prohibitive; storing both int4+int8 ~= 40 GB > 32 GB card).
  (c) grouped-128 _int_mm accumulation in eager Python = dead (0.03x).
So on this box today, a W4A8 served from int4 weights performs == W4A16 (woqgemm fp16) on prefill; no win.

## THE UNLOCKS (ranked, next iterations)
1. **Port vLLM's `torch.ops._xpu_C.int4_gemm_w4a8` (oneDNN int4w x int8a) into sglang.** This op powered the
   vLLM-era W4A8 (14B prefill 4403 t/s) and is oneDNN-based (does NOT hit the auto_round joint_matrix gate, the
   same way torch._int_mm works here). It dequants int4->int8 INSIDE the GEMM mainloop (no materialization,
   keeps int4 memory, group-128 aware). Present in vllm-xpu-env images (vllm_xpu_kernels .so). VERIFY it is fast
   on B70 (decode + prefill) then port the .so into sglang + write w4a8_shim around it. HIGHEST leverage.
2. **oneAPI >= 2026 upgrade** in the sglang image -> unlocks woqgemm(compute_type=int8) joint_matrix -> fused
   W4A8 (decode 1 launch + prefill int8-XMX) with zero materialization. Clean but image/ABI-rebuild risk.
3. **Custom SYCL int4w x int8a GEMM** via the oneDNN DPAS path (not joint_matrix) -- re-implements (1) from
   scratch. Only if porting the existing op fails.

## Reference artifacts on disk
- Lorbus_Qwen3.6-27B-int4-AutoRound: WORKING woq base (vision, GDN quantized, 17.7GB,
  group-128, packing auto_round:auto_gptq). The W4A8 weight source.
- Qwen3.6-27B-W4A8-sqgptq-prepacked: compressed-tensors int-quantized (int4 W + int8
  dyn act). BAD artifact: 0 vision tensors, only 256 MLP linears int4 (GDN left bf16),
  25.9GB, format sglang-XPU cannot load. Do NOT use as-is.
