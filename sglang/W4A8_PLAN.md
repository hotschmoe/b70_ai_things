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

## UPDATE 2026-06-28b: int4_gemm_w4a8 VERIFIED FAST, but the drop-in port is ABI-BLOCKED
Agent verified vLLM `torch.ops._xpu_C.int4_gemm_w4a8` (oneDNN int4w x int8a) on B70 (real sqgptq layer):
  - decode M=1: 0.083 ms op-only = 3.75x bf16 (FASTER than int4-woqgemm 0.145ms); e2e 0.21ms (act-quant un-fused).
  - prefill M=2048: 1.73 ms = 1.86x bf16 AND 2.17x faster than int4-woqgemm prefill (3.76ms). relerr 2e-04.
  - It is oneDNN-based (NOT joint_matrix-gated) -> runs clean on this box. Existing sqgptq ckpt layout already
    matches the op ([N,K/8] i32 weight + [N,K/g] scale; pass weight.t()). Call:
      int4_gemm_w4a8(actInt8[M,K], actScale[M,1] fp16, actZero[M,1] i32, qweight[K/8,N] i32,
                     wscale[K/g,N], wzp=tensor([8]) i8, group_size, g_idx=None, bias=None) -> [M,N] fp16
    Activation quant is OUTSIDE the op (per-token sym int8). Output always fp16 (serve --dtype float16).
ABI GATE (w4a8_abi_test.py): the v0230 `_xpu_C.abi3.so` is built against torch 2.11; sglang has torch 2.12.
  torch.ops.load_library -> "undefined symbol: _ZNR5torch7Library4_def..." = a torch C++ ABI break (2.11 vs 2.12).
  Also DT_NEEDED libsycl.so.8/libimf/libintlc (oneAPI 2025.3 runtime) absent. So the .so CANNOT drop into sglang.
=> Need int4_gemm_w4a8 BUILT AGAINST torch 2.12 (sglang's). Net-new build step. Probe set: sglang/int4_gemm_w4a8_probe.py
   (vllm img), sglang/w4a8_abi_test.py (sglang img), extracted glue sglang/_v0230_kernels/*.py.

## THE UNLOCKS (ranked, post-ABI-gate)
GOAL: get int4_gemm_w4a8 (proven fast, oneDNN) callable in the sglang torch-2.12 image, then write
w4a8_shim (decode + prefill via the op; reuse woq_shim layer selection; fuse the act-quant to recover decode).
1. **Obtain int4_gemm_w4a8 built against torch 2.12**, by the cheapest working route:
   (a) `pip install vllm-xpu-kernels` (or equiv) INTO sglang-xpu:woq -- if it ships/builds a cp312+torch2.12 XPU
       wheel, the op registers ABI-matched. CHECK FIRST (lowest effort).
   (b) extract `_xpu_C.abi3.so` + sibling libs from a NEWER vllm-xpu image built on torch 2.12 (if one exists).
   (c) build ONLY the int4_gemm_w4a8 op from the vllm-xpu-kernels SOURCE (github) as a standalone torch-2.12
       C++/SYCL+oneDNN extension. Largest, but reliable (oneDNN int8 proven here). Source path in the .so strings:
       /workspace/vllm_xpu_kernel/csrc/xpu/onednn/...
   Then bake into sglang-xpu:w4a8; write w4a8_shim.py around torch.ops._xpu_C.int4_gemm_w4a8 + the
   dynamic_per_token_int8_quant (port from _v0230_kernels/_xpu_ops.py; torch.compile-fuse it for decode).
2. **oneAPI/compute-runtime upgrade** in sglang image -> maybe unlocks the ALREADY-PRESENT auto_round
   woqgemm(compute_type=int8) joint_matrix path (the "no matrix hardware" runtime gate). Possibly a runtime lib
   swap (no torch rebuild) -- uncertain it suffices since oneDNN int8 already works but the SYCL kernel path is gated.
3. **Custom SYCL int4w x int8a GEMM** (oneDNN DPAS) -- == rebuilding (1c) from scratch. Last resort.

## Reference artifacts on disk
- Lorbus_Qwen3.6-27B-int4-AutoRound: WORKING woq base (vision, GDN quantized, 17.7GB,
  group-128, packing auto_round:auto_gptq). The W4A8 weight source.
- Qwen3.6-27B-W4A8-sqgptq-prepacked: compressed-tensors int-quantized (int4 W + int8
  dyn act). BAD artifact: 0 vision tensors, only 256 MLP linears int4 (GDN left bf16),
  25.9GB, format sglang-XPU cannot load. Do NOT use as-is.

## UPDATE 2026-06-28c: W4A8/W4A16 hybrid SERVED on the PROVEN Lorbus multimodal path (vision kept)
The sqgptq/text-only path is ABANDONED (Qwen3_5ForCausalLM is the decoder body -> no lm_head ->
sample() crashes; no vision; MLP-only quant). New approach: serve the PROVEN Lorbus int4-AutoRound
ckpt (multimodal Qwen3_5ForConditionalGeneration, vision + full GDN+MLP int4) through sglang's
proven multimodal model path, but dispatch its int4 linears to the oneDNN int4_gemm ops instead of
auto_round woqgemm. Reuses ALL the working vision+arch+logits plumbing.

### THE GATE (sglang/w4a8_from_woq_probe.py) -- PASSED
Crux: convert auto_gptq packing (qweight [K/8,N] i32 contiguous, qzeros [K/g,N/8], scales [K/g,N])
-> the int4_gemm op's expected B (verified on COMPRESSED-TENSORS: [K/8,N] NT view, B_zp=[8] sym,
B_scale [K/g,N]). Found by reading auto_round_kernel.qlinear.unpack_to_8bit_signed + post_init:
  - ARK sym (asym=False) dequant = (nibble - 8) * scale; nibble i at bit 4i = K-index k8*8+i;
    qzeros are IGNORED for sym (the ckpt packs nibble 7 = the GPTQ -1 offset, i.e. effective zero 8).
  - This is BYTE-IDENTICAL to the op's compressed-tensors convention. Conversion is a PURE relayout:
    auto_gptq qweight is already [K/8,N] with the same nibble semantics but contiguous (stride0==N);
    the op needs stride0==1 (NT) -> B = qw.t().contiguous().t() (keep the [N,K/8] buffer alive).
    B_scale = scales.to(fp16) (already [K/g,N]); B_zp = tensor([8]) (sym).
Reference = auto_round_kernel woqgemm (QuantLinearGPTQ) built EXACTLY as woq_shim does, SAME layer.
Result (card 0, sglang-xpu:woq, real Lorbus layer-20 weights):
  layer                    M=1 w4a16 relerr   M=2048 w4a16   w4a8 (int8 act) relerr
  down_proj (MLP)          1.40e-03           5.7e-06        ~1.0e-02
  gate_proj (MLP)          9.62e-04           0.0            ~8.8e-03
  in_proj_qkv (GDN fused)  9.69e-04           0.0            ~8.6e-03
  out_proj (GDN)           1.02e-03           0.0            ~8.0e-03
=> w4a16 (decode) MATCHES woqgemm to 1e-3 on MLP AND GDN; w4a8 (prefill) ~9e-3 (int8-act-quant
   error, expected, finite). GATE PASS -> serve is numerically justified.

### Integration (left for user to commit)
- sglang/patches/woq_shim.py: added `_XpuW4A8WoqKernel` + `_load_int4_gemm_op()`; gated by
  env B70_XPU_W4A8_WOQ=1. process_weights_after_loading converts the auto_gptq buffers ONCE
  (relayout above); apply: M==1 -> int4_gemm_w4a16 (fp16 act, no quant), M>1 -> EAGER per-token
  sym int8 act-quant + int4_gemm_w4a8. Routes the SAME GPTQLinearScheme._init_kernel hook as woqgemm
  (so it only swaps the kernel; auto_round layer-selection, vision/GDN-bf16 ignore list, lm_head all
  unchanged). Act-quant is EAGER on purpose (torch.compile of it HANGS serve startup).
- sglang/serve_w4a8_woq.sh: serves Lorbus int4 via the multimodal class on card 0 (TP=1),
  --dtype bfloat16 --attention-backend triton --linear-attn-backend triton --skip-server-warmup
  --disable-radix-cache --mem-fraction-static 0.85; image sglang-xpu:mtp; mounts the updated
  woq_shim.py + the built _xpu_C.abi3.so; B70_XPU_C_SO + oneAPI LD_LIBRARY_PATH set in-container.
  GRAPH=1 stacks XPUGraph decode capture (bs=1) -- only AFTER a clean eager bench.

### Serve validation (card 0, eager, ctx=4096)
- LOADS as Qwen3_5ForConditionalGeneration (multimodal, VISION RETAINED), quant=auto-round bits=4,
  17.44 GB weights; 304 int4 linears routed to the W4A8/W4A16 hybrid (grep "w4a8-woq] layer ready").
- COHERENT: "Why is the sky blue" -> correct Rayleigh-scattering answer (not garbage). No NaN/"!!!!".
- WARM bench (sglang.bench_serving, 2048-in/128-out, discard 1st):
    c1: decode 9.88 t/s   TTFT 1105 ms   agg 8.79 t/s
    c4: decode 4.39 t/s   TTFT 1954 ms   agg 14.09 t/s
  vs int4-woqgemm EAGER champion (~9.4 t/s decode, ~1244 ms TTFT): decode ON PAR/slightly ahead
  (same int4 weights, bandwidth-bound M=1), TTFT ~11% FASTER (the int8-act w4a8 prefill helps).
  NOTE: this is EAGER. The 23.5 t/s headline is the XPUGraph-captured woqgemm driver; matching it
  needs the int4_gemm ops captured under XPUGraph (GRAPH=1) -- separate attempt.
  SOAK (2000-tok single stream, 400-tok windows): 9.98 -> 9.78 -> 9.77 -> 9.62 -> 9.44 t/s,
  OVERALL 9.72 t/s, first/last ratio 1.06x (STABLE, no degradation), coherence OK, TTFT 1085 ms.
  => eager W4A8/W4A16 hybrid = on-par decode + faster TTFT vs woqgemm eager, STABLE, vision kept.

### XPUGraph stack (GRAPH=1) -- THE WIN: BEATS the int4-woqgemm graph champion
The int4_gemm_w4a16 decode op is XPUGraph-CAPTURABLE (the bs=1 decode graph captured cleanly in
49 s; under bs=1 decode the apply() takes the M==1 w4a16 branch = a single op, no data-dependent
act-quant, so capture is clean). Served Lorbus int4 via the multimodal class + GRAPH=1 (B70_XPU_
CUDAGRAPH=1, --cuda-graph-bs-decode 1 --max-running-requests 1, ATTN=triton), card 0, vision kept.
- COHERENT under capture (Rayleigh). 304 layers routed. No NaN.
- WARM c1: decode 25.15 t/s   TTFT 1110 ms
- WARM c4: per-stream 25.52 t/s (agg 16.25; TTFT 8866 ms -- c4 SERIALIZES under maxreq=1, the
  known single-stream-graph limit; for concurrency use DP=2, not a higher maxreq, same as champion).
- SOAK (2000-tok): 26.52 -> 25.63 -> 24.79 -> 23.99 -> 23.32 t/s, OVERALL 24.80 t/s, ratio 1.14x
  (stable; the mild decline is the B70 idle/thermal downclock, same as the champion), coherence OK,
  TTFT 968 ms.
=> vs the int4-woqgemm XPUGraph champion (~23.5 t/s decode, ~1244 ms TTFT): the W4A8/W4A16 hybrid
   DECODES FASTER (25.15 warm / 24.8 soak vs 23.5, +5-7%) AND has lower TTFT (~970-1110 ms), while
   retaining vision and the full GDN+MLP int4 quant. The int4_gemm_w4a16 op (oneDNN) beats woqgemm
   on the captured decode; the int8-act int4_gemm_w4a8 op carries the (eager) prefill/TTFT win.

## VERDICT 2026-06-28c: GOAL MET. Vision-retaining W4A8/W4A16-hybrid sglang serve BEATS the int4 champion.
Daily-driver upgrade path: serve Lorbus int4 (the SAME proven ckpt, vision + GDN) via
sglang/serve_w4a8_woq.sh with GRAPH=1. Decode 24.8-25.2 t/s (> 23.5 woqgemm-graph), TTFT ~1 s,
stable + coherent, vision retained. Eager fallback (GRAPH=0): 9.7 t/s decode (== woqgemm eager) +
faster TTFT. Conversion + kernel routing numerically gated (sglang/w4a8_from_woq_probe.py).
Open follow-ups: (1) DP=2 for concurrency at 24.8 t/s/stream (mirror serve_dp2_graph.sh); (2) the
prefill int4_gemm_w4a8 is eager under GRAPH (prefill graph auto-disabled) -- a captured/compiled
act-quant could push TTFT lower; (3) HumanEval+ accuracy vs the int4 champion (numerics gated to
relerr 1e-3, so expected == int4, but confirm).

## ACCURACY GATE 2026-06-28d: PASS -- W4A8 == int4 same-stack, the int8-act prefill does NOT degrade code
HumanEval+ (164 problems, thinking-OFF, greedy, EvalPlus sandboxed grading), run through the repo harness
(evals/orchestrator/run_evals.py --tiers 1) against the LIVE sglang serves. To isolate the int8-act-prefill
effect from the vLLM-vs-sglang stack difference, BOTH the W4A8 hybrid AND the int4-woqgemm champion were
served on the SAME sglang stack (GRAPH=1, card 0) from the SAME Lorbus int4 weights and scored identically:

  config (sglang GRAPH=1, same Lorbus int4 weights)         pass@1 base   pass@1 plus   base-fails (n=164)
  W4A8 hybrid (int4_gemm: int8-act prefill, fp16-act decode)   0.921          0.896          13
  int4-graph champion (auto_round woqgemm, fp16-act)           0.921          0.896          13
  => DELTA 0.000 / 0.000. Failed sets overlap 9/13; the 4 that differ (W4A8: 91,118,140,148 vs
     int4: 67,109,126,130) are marginal problems flipped by greedy-decode non-determinism ACROSS kernels
     (different numerics -> a different token at one branch), netting to zero. The int8-act prefill
     (relerr ~9e-3 on prompt encoding only; decode is fp16-act) is statistically indistinguishable.

Note on the absolute number: 0.921 is DEPRESSED by output verbosity, not quant damage. 9 of the 13 base
failures are TRUNCATION (the model writes verbose, comment-heavy solutions that hit max_tokens=2048 with the
code fence never closed); only 4 are genuine wrong answers. The recorded vLLM int4 number (0.963/0.927,
evals/results/SUMMARY.md) is the SAME weights on a different stack whose chat template yields terser output ->
less truncation -> a higher absolute pass@1. That gap is a stack/methodology artifact, NOT the kernel: on the
APPLES-TO-APPLES same-stack comparison, W4A8 ties int4 exactly. GATE PASS -> W4A8 is a real daily driver.
Result dirs: evals/results/20260628T090922Z__qwen36-27b-w4a8woq__w4a8-graph (W4A8) and
.../20260628T095310Z__qwen36-27b-int4-graph__int4-graph-sglang (int4 baseline).

## CLEAN SAME-SESSION HEAD-TO-HEAD 2026-06-28e: W4A8 beats int4-graph on decode AND TTFT AND PP
Replaces the earlier "vs documented 23.5" comparison with a back-to-back one: BOTH serves on card 0, GRAPH=1,
SAME machine state, SAME bench (sglang/bench2048.sh, random IN=2048/OUT=128, c1, WARM = 1st run discarded,
2 recorded runs). Each serve started fresh, benched, stopped (single gpu-run --card 0 lease per serve).

  config (GRAPH=1, card 0, warm c1)   decode t/s        TTFT ms          prefill t/s (PP = IN*1000/TTFT)
  int4-graph (auto_round woqgemm)      23.50 / 23.49     1158.9 / 1159.5  1767 / 1766
  W4A8 hybrid (EAGER act-quant)        25.14 / 25.21     1053.1 / 1054.3  1945 / 1943
  W4A8 hybrid (TRITON act-quant)       25.32 / 25.35      936.4 /  934.9  2187 / 2191
  c4 (per-stream decode / agg / TTFT): int4 23.60/15.08/9538  W4A8eager 25.56/16.25/8856  W4A8triton 25.58/16.89/8459

=> The decode win is REAL on identical same-session conditions (NOT a stack artifact): W4A8 ~25.3 vs int4 ~23.5
   = +7.8% decode. The win is ALSO TTFT (-19.3%, 1159 -> 936 ms) and PP (+24%, 1766 -> 2189) -- it is NOT
   decode-only. (Decode = int4_gemm_w4a16 oneDNN, faster than woqgemm even captured; TTFT/PP = int8-act
   int4_gemm_w4a8 prefill.)

## TTFT/PP UNLOCK 2026-06-28e: prefill act-quant fused to ONE Triton launch (no startup hang)
The prefill per-token sym int8 act-quant was the eager ~8-launch chain (1.78 ms/call @M=2048 on a real
down_proj K=17408 -- x304 linears => the bulk of TTFT). torch.compile of it HANGS serve startup (inductor
async-compile-worker deadlock inside the sglang scheduler proc). FIX: a single-launch Triton kernel
(sglang/patches/w4a8_actquant_triton.py, per_token_int8: amax-reduce + quantize, 2 streaming passes over K,
1 launch). triton.jit compiles IN-PROCESS (no worker pool) -> NO hang; triton-xpu is already the attn backend.
  - Standalone gate (sglang/w4a8_triton_aq_probe.py, card 0, real down_proj): act-quant eager 1.78 ms ->
    triton 0.214 ms @M=2048 = 8.3x; full w4a8 layer 3.75 -> 1.93 ms = 1.94x. Numerics: q differs <=1 LSB on
    <1.03% of elements, scale identical; int4_gemm_w4a8 OUTPUT relerr(eager vs triton) = 3e-3 (BELOW the
    accepted ~9e-3 int8-act-vs-fp16 error). round-half-away-from-zero vs torch's round-half-even = the only diff.
  - End-to-end (table above): W4A8 TRITON vs W4A8 EAGER = TTFT -11.2% (1054 -> 936 ms), PP +12.6%
    (1944 -> 2189), decode UNCHANGED (25.18 -> 25.34; decode is the w4a16 fp16-act path, no act-quant).
  - Coherence + soak (perf_regime, GRAPH=1, triton): coherence GATE OK (Rayleigh); WARM c1 25.32 t/s TTFT 997 ms;
    SOAK 2000-tok OVERALL 24.70 t/s, first/last ratio 1.14x (stable, the usual idle-downclock), coherence OK.
  Wired in woq_shim.py _XpuW4A8WoqKernel (prefill branch), gated B70_W4A8_TRITON_AQ (default on; eager fallback
  if triton import/JIT fails or =0). serve.sh / serve_w4a8_woq.sh mount w4a8_actquant_triton.py into site-packages.

## PP AXIS (task 3) -- W4A8 prefill throughput vs int4, fully documented
PP = prompt prefill tok/s = IN(2048)*1000 / TTFT_ms (warm c1, the table above):
  int4-graph (woqgemm fp16-act prefill):        1766 tok/s
  W4A8 hybrid, EAGER int8 act-quant prefill:    1944 tok/s   (+10.1% vs int4)
  W4A8 hybrid, TRITON int8 act-quant prefill:   2189 tok/s   (+23.9% vs int4, +12.6% vs the eager W4A8)
The PP win is the int8-XMX int4_gemm_w4a8 prefill op (oneDNN, 1.9x woqgemm's int4->fp16 prefill) MINUS the
act-quant overhead; Triton recovers most of that overhead (eager act-quant was eating ~12% of TTFT).

## act-quant autotune (2026-06-28f, DECISION: SHIPPED CONFIG IS AT THE CEILING -- NO CHANGE)
Tried to shave the single-launch Triton per-token int8 act-quant (patches/w4a8_actquant_triton.py, gate
B70_W4A8_TRITON_AQ; shipped config = two-pass, BLOCK_K=2048, num_warps=8). Probe: sglang/w4a8_actquant_autotune.py
(card 0, sglang-xpu:woq, microbench, NO serve). Swept BLOCK_K {512,1024,2048,4096}, num_warps {4,8,16,32},
num_stages {1,2,3}, AND a SINGLE-PASS full-row strategy (load the whole row once = read x ONCE instead of
twice) at the real Lorbus linear K's {5120 (gate/up/qkv), 17408 (down)}, M in {1,512,2048}. Gate: q within
<=1 LSB of eager on >99% of elements + scale bit-exact (all surviving configs pass; max|dq|=1, ~0.9% mismatch).

Results (warm, ms/call, two reproducing runs):
  K= 5120 M=2048: shipped 0.0494-0.0496 ms; best 0.0490-0.0493 ms = +0.6..0.8% (NOISE). Every config (incl
    M=1/512/2048) lands ~0.049-0.054 ms -> K=5120 act-quant is LAUNCH/FIXED-OVERHEAD bound, not BW bound;
    no config helps (the compute is negligible vs the kernel-launch floor).
  K=17408 M=2048: shipped 0.2122 ms; best two-pass (BLOCK_K=4096, num_warps=16) 0.2076 ms = +2.2% (reproducible).
    SINGLE-PASS full-row (BLOCK_K=32768) = only +1.3%: the read-once BW saving is eaten by register/SLM spill
    from the 32768-wide block, so it does NOT beat the streaming two-pass. The kernel is not cleanly BW-bound.
  The +2.2% config (4096/16) is NEUTRAL-to-SLOWER at K=5120, so a SINGLE global config can't even capture the
    2.2% -- the best config valid+fastest across BOTH K just TIES the shipped (2048/8) at geomean 1.000x.
  Numerics held: int4_gemm_w4a8 output relerr(eager vs best) = 3.0e-3 (< the accepted 9e-3), finite.

VERDICT: the act-quant is ~11% of the 1.69 ms W4A8 prefill GEMM, so the max realizable act-quant win (+2.2%,
and only on the K=17408 down layers) is ~0.2% of prefill time -- below the run-to-run noise of TTFT/PP
(the shipped table shows TTFT 936.4/934.9 ms, PP 2187/2191 -- ~0.1% jitter). No config clears the >=5%
e2e bar; the shipped (two-pass BLOCK_K=2048 num_warps=8) Triton kernel is at the ceiling. NO CHANGE SHIPPED.
Committed: this note + the autotune probe (sglang/w4a8_actquant_autotune.py). w4a8_actquant_triton.py unchanged.

## W4A4 MXFP4 speed gate (2026-06-28, DECISION: DO NOT PURSUE on B70)
Cheap single-card microbench (card 0, synthetic, no serve, no accuracy concern) to answer one question
BEFORE committing to an MXFP4 requant + accuracy validation: does the built-but-unused MXFP4 w4a4 op
`torch.ops._xpu_C.fp4_gemm` (mxfp4 weight x mxfp4 act, e8m0 block-32 scales) have a FAST compute path on
B70, or is it emulated/slow like FP8?

  - Op: registered name is `fp4_gemm` (NOT `fp4_gemm_w4a4`). Schema:
    `_xpu_C::fp4_gemm(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale, ScalarType? out_dtype, Tensor? bias_) -> Tensor`
    A = act fp4 [M,K/2] (float4_e2m1fn_x2), B = weight.T [K/2,N] NT (stride[0]==1), A_scale/B_scale = e8m0
    [M,K/32]/[N,K/32] (float8_e8m0fnu), block-32. Layout mirrors tests/test_fp4_gemm_onednn.py + fp4_gemm_w4a4.h.
  - Probe: sglang/w4a4_probe.py. Image sglang-xpu:woq, _xpu_C.abi3.so via CDLL RTLD_GLOBAL, LD_LIBRARY_PATH
    prepend oneapi 2025.3/lib. MXFP4 quant via the vetted to_mxfp helper (tests/ops/mx_utils.py, mounted).
    Real down_proj shape K=17408 N=5120, warm (20 warmup + discard 1st), M=1 (decode) and M=2048 (prefill).

  RESULT (ms/call, x-vs-bf16; all paths produce FINITE output -- the op RUNS, no "joint_matrix not supported"):
    M=1    bf16 0.305 | fp4 w4a4 op 0.432 (0.71x) | int4 w4a8 0.082 (3.72x) | int4 w4a16 0.082 (3.73x)
    M=2048 bf16 2.402 | fp4 w4a4 op 11.06 (0.22x) | int4 w4a8 1.828 (1.31x) | int4 w4a16 2.710 (0.89x)
  (fp4 op+eager-mxfp4-act-quant is worse still: 1.24 ms @M=1, 21.8 ms @M=2048 -- the to_mxfp act path is
  unfused, but the OP ITSELF is already the killer.)

  VERDICT: W4A4 MXFP4 is NOT worth pursuing on B70. fp4_gemm is FUNCTIONAL but EMULATED -- there is no
  XMX/joint_matrix fast path for e2m1xe2m1 + e8m0 here. It is SLOWER THAN BF16 at both M (0.71x decode,
  0.22x prefill) and, decisively, ~6.0x SLOWER than the shipped int4_gemm_w4a8 at prefill (11.06 vs 1.83 ms)
  and ~5.3x slower than int4_gemm_w4a16 at decode (0.432 vs 0.082 ms). An MXFP4 requant + accuracy validation
  would buy a large perf REGRESSION, not a gain. DO NOT requant to MXFP4. The W4A8 hybrid (int8-act XMX
  int4_gemm_w4a8 prefill + int4_gemm_w4a16 fp16-act decode) remains the frontier on this box; this matches
  the earlier FP8-is-emulated finding (B70's only fast low-precision matmul path is INT8 XMX, used by w4a8).

## int4 lm_head (2026-06-28g, SHIPPED: +7.9% decode, accuracy HELD) -- the new decode lever
The Lorbus int4 ckpt EXCLUDES lm_head from quant (block_name_to_quantize = language_model.layers + mtp only),
so lm_head.weight stays BF16 [vocab=248320, hidden=5120] = 2.54 GB, read in FULL every decode step. At 25 t/s
(~40 ms/step) the bf16 lm_head GEMV is ~4.27 ms = ~11% of the step (bandwidth-bound: 2.54 GB / ~600 GB/s).
Lever: RTN-quantize lm_head to int4 group-32 sym ONCE at load time and route the logits GEMV through the SAME
captured decode op the body uses (int4_gemm_w4a16). lm_head is output-sensitive -> HARD-gated on HumanEval+.

MICROBENCH (sglang/lmhead_int4_probe.py, card 0, REAL lm_head weight, warm):
  - int4_gemm_w4a16 M=1: 1.29 ms (g32) / 1.11 ms (g128) vs 4.27 ms fp16 = 3.3x / 3.8x, FINITE, op relerr 3e-4.
  - naive-RTN weight quant relerr is HIGH (lm_head is outlier-heavy): g128 12.6%, g64 11.3%, g32 10.0%
    (vs the body's calibrated-AutoRound ~few %). g32 is best accuracy at ~negligible speed cost -> use g32.
    The op supports g=32 (160 groups along K=5120); int4 size 0.64 GB (g32) vs 2.54 GB bf16.

IMPLEMENTATION (sglang/patches/woq_shim.py, OPT-IN B70_W4A8_QUANT_LMHEAD=1, group B70_W4A8_LMHEAD_GROUP=32):
  monkeypatches ModelRunner.load_model (quantize lm_head AFTER weights load, chunked over N to bound the fp32
  scratch, BEFORE graph capture) + LogitsProcessor._compute_lm_head (reroute the logits matmul to
  int4_gemm_w4a16 when lm_head._b70_int4 is set). The bf16 weight is KEPT resident (revertible; get_embed_and_head
  /PP/spec paths still see lm_head.weight; net +0.64 GB) -- only the matmul is rerouted. The lm_head op is
  XPUGraph-CAPTURABLE: the logits GEMV is inside the captured decode forward (model.forward returns
  LogitsProcessorOutput) and captured cleanly at bs=1 (M=1). serve.sh: LMHEAD=1 default (LMHEAD=0 reverts).

CLEAN SAME-SESSION HEAD-TO-HEAD (card 0, GRAPH=1, warm c1, bench2048 IN2048/OUT128, 1st run discarded, 2 runs):
  config (GRAPH=1, card 0, same Lorbus int4 weights)   decode t/s     TTFT ms        prefill PP
  W4A8 hybrid, bf16 lm_head (prior shipped)            25.32 / 25.35  940.4 / 941.1  2178 / 2176
  W4A8 hybrid, int4 lm_head g32 (THIS)                 27.37 / 27.31  935.1 / 937.2  2190 / 2185
  => decode +7.9% (25.34 -> 27.34); TTFT and prefill UNCHANGED (lm_head is M=1 even in prefill -- sglang
     prunes to last-token logits -- so only DECODE benefits; the win is pure lm_head-GEMV bandwidth).

ACCURACY GATE -- PASS (HumanEval+ 164, thinking-off, greedy, sandbox evalplus-sandbox:0.3.1, same harness):
  config                            pass@1 base   pass@1 plus
  W4A8 bf16 lm_head (baseline)         0.921         0.896
  W4A8 int4 lm_head g32 (THIS)         0.933         0.896
  => base +0.012 (NOT a regression; +2 problems via greedy cross-kernel non-determinism, same effect as the
     int4-vs-W4A8 4-problem flips above), plus IDENTICAL. The 10% naive-RTN weight relerr does NOT translate
     to coding-accuracy loss: code logits have large argmax margins, so greedy is robust to the perturbation.
  Result dir: evals/results/20260628T144209Z__qwen36-27b-w4a8-graph__w4a8-lmhead-int4-g32.

VERDICT: SHIPPED. int4 lm_head is the new W4A8-graph default (LMHEAD=1): decode 27.3 t/s (+7.9%), TTFT/prefill
  and accuracy unchanged, vision retained. Decode is now ~16% over the int4-woqgemm champion (23.5 -> 27.3).
  Revert with LMHEAD=0 (bf16 lm_head). Probe: sglang/lmhead_int4_probe.py.

## serve-config knob sweep (2026-06-28h, DECISION: CEILING -- NO CHANGE; the 27.3->34 gap is GRAPHED COMPUTE, not serving overhead)
Hypothesis under test (from PERF.md "NEXT: --stream-interval N"): the shipped 27.3 t/s decode sits ~20% below the
int4-weight bandwidth floor (~34 t/s); some of that gap might be per-token serving/streaming/scheduler overhead
that is NOT inside the captured graph, recoverable by CHEAP serve-config flags (no kernel/requant changes).
Swept one flag at a time from the shipped baseline (rdy_to_serve/qwen36-27b-w4a8-graph/serve.sh: GRAPH=1,
LMHEAD=1, card 0, --attention-backend triton, --disable-overlap-schedule, --skip-server-warmup, --page-size 64,
--disable-radix-cache, --max-running-requests 1). Bench = sglang/bench2048.sh methodology (random IN=2048, c1,
WARM = 1st run discarded, 2 recorded runs back-to-back per config; each config a FRESH serve start+stop under its
own gpu-run --card 0 lease). Harness recorded out_tps (Output token throughput = total tokens / wall-clock) as the
DRIFT-IMMUNE and stream-interval-IMMUNE combined client metric, alongside TPOT-derived decode + TTFT + ITL.

  config (vs shipped baseline)            decode t/s   TTFT ms   out_tps   ITL ms   delta (decode / out_tps)   verdict
  baseline (shipped)                      27.37        935.8     20.93     36.5     --                         --
  --stream-interval 4                     27.36        938.2     20.91     143      -0.0% / -0.1%              FLAT, ITL 4x worse
  --stream-interval 8                     27.52        935.7     21.01     275      +0.5% / +0.4% (noise)      FLAT, ITL 8x worse
  --num-continuous-decode-steps 2         27.20        938.0     20.82     36.5     -0.6% / -0.5%              flat-to-NEGATIVE
  --triton-attention-num-kv-splits 16     27.00        935.3     20.71     37.0     -1.4% / -1.1%             NEGATIVE
  --triton-attention-num-kv-splits 4      25.27        935.5     19.74     39.5     -7.7% / -5.7%             NEGATIVE (default 8 best)
  env FLA_USE_FAST_OPS=1                   27.32        935.4     20.90     36.5     -0.2% / -0.1%             FLAT (no-op here)
  --chunked-prefill-size 4096             27.36        936.0     20.92     36.5      0.0% /  0.0%             FLAT (2048 prompt = 1 chunk already)
  baseline OUT=512                        27.29        967.2     21.80     36.4     --                         --
  --stream-interval 4 OUT=512             27.32        967.5     21.82     144      +0.1% / +0.1% (noise)      FLAT, ITL 4x worse

THE SMOKING GUN -- stream-interval does NOT change server-side per-token cost: TPOT is byte-for-byte identical
across stream-interval 1/4/8 (36.52 / 36.54 / 36.33 ms at OUT=128). stream-interval only batches more tokens per
SSE chunk -> ITL scales linearly (36.5 -> 143 -> 275 ms = 1x/4x/8x) with ZERO throughput or e2e-latency gain
(out_tps 20.93 -> 20.91 -> 21.01; OUT=512 e2e 4458 vs 4454 ms). The PERF.md hypothesis (that stream_interval=1
detok/HTTP overhead sits on the decode critical path and a bigger interval would recover it) is REFUTED for this
graphed W4A8 driver: with --disable-overlap-schedule + a captured bs=1 graph, the streaming/detok work is NOT on
the per-token critical path. The client-measured 27.3 t/s ALREADY EQUALS the server-side "cuda graph: True" rate
(TPOT 36.5 ms = the real per-token kernel cost: int4_gemm_w4a16 GEMVs + triton attention + GDN recurrent step +
int4 lm_head GEMV). So the 27.3->34 gap lives in the GRAPHED COMPUTE (kernel/bandwidth-realization), not in
ungraphed serving/streaming/scheduler overhead -- it is NOT reachable by serve-config flags.

Not separately benched (with rationale, to avoid wasted GPU time): --stream-interval 2 (monotonically between the
flat si1 and flat si4 on throughput, only ITL changes -> no point); --num-continuous-decode-steps 4 (ncds2 already
-0.6%, and JOURNAL 5332-5335 showed ncds4==ncds2==no change on the MTP driver); --schedule-conservativeness (a
multi-request ADMISSION-control knob; with --max-running-requests 1 there is exactly one in-flight request at c1,
so it cannot affect single-stream TTFT or decode).

VERDICT: CEILING -- NO CHANGE SHIPPED. No flag or env (nor any combination, since every individual lever is
flat-to-negative) clears the >=5% client-e2e decode bar or yields a TTFT win without a decode loss; the only
non-noise movers are NEGATIVE (kv-splits!=8, num-continuous-decode-steps>1) or pure-smoothness-LOSSES (stream-
interval, which trades 4-8x worse ITL for no throughput). The shipped defaults (stream-interval 1, kv-splits 8,
continuous-decode-steps 1, chunked-prefill default, no FLA env) are already at the serve-config optimum. Further
W4A8 decode gains require KERNEL work (faster int4 GEMV / lower-overhead attention to close the 27.3->34 floor),
not flags. serve.sh / serve_w4a8_woq.sh / serve_dp2_w4a8.sh UNCHANGED. Sweep harness: scratchpad sweep_one.sh
(start+warm-bench+stop per config); raw bench captures retained under the scratchpad.
