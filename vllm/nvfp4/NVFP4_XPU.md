# NVFP4 on B70/XPU (YOLO experiment, night of 2026-07-04)

Goal: load and run an NVFP4 (NVIDIA FP4) checkpoint on Intel Arc B70 -- a format
with zero Intel hardware or software support. Chosen model:
`nvidia/Qwen3-8B-NVFP4` (6.0GB, dense, ungated, ModelOpt-produced).

## Format (validated in 01_crack_format.py)

NVFP4 = W4A4 microscaling format:

- `weight`          uint8 [N, K/2]   -- 2x E2M1 fp4 codes per byte, LOW nibble first
- `weight_scale`    f8e4m3 [N, K/16] -- per-16-elem block scale along K
- `weight_scale_2`  f32 scalar       -- global scale, = amax/(6*448)
- `input_scale`     f32 scalar       -- activation global scale (W4A4 runtime quant)
- `k_scale/v_scale` f32 scalars      -- FP8 KV scales (all 1.0 in this ckpt; ignored)
- E2M1 value set: +/- {0, 0.5, 1, 1.5, 2, 3, 4, 6}
- dequant(w) = e2m1_lut[nibble] * e4m3(block_scale) * weight_scale_2

Crack results (layer0 q_proj): std 0.026, mean ~0, absmax 0.54, 0 NaN/Inf;
fp4 code histogram symmetric, all 16 codes used; block scales 2.25..384.
Format understanding CONFIRMED.

## Why it does not run stock on XPU

vLLM v0.24.0 has the complete ModelOpt NVFP4 config/load path, and its
EmulationNvFp4LinearKernel is device-generic (pure torch + e4m3 view; triton
paths gated behind is_cuda_alike()). The ONLY blocker:
`_POSSIBLE_NVFP4_KERNELS` (vllm/model_executor/kernels/linear/__init__.py:407)
maps CUDA and ROCM only -- no PlatformEnum.XPU entry -> "Failed to find a
kernel that can implement the NVFP4 linear layer". XPU has no
supported_quantization allowlist, so nothing else rejects modelopt.

## The shim (patches/sitecustomize.py)

Registers kernels for PlatformEnum.XPU. Two modes (NVFP4_XPU_MODE):

- `emul`:    stock EmulationNvFp4LinearKernel. Weight dequant EVERY forward +
             activation fake-quant to nvfp4. True W4A4 emulation; numerics reference.
- `dequant`: XPUDequantAtLoadNvFp4LinearKernel (ours). One-time NVFP4 -> BF16
             dequant in process_weights_after_loading, then plain F.linear.
             W4A16-style, fast; ~15GB bf16 weights on card.

Serve: `vllm/nvfp4/serve_nvfp4.sh` (single card, port 8077, enforce-eager).

## Roadmap

- [x] M0: format crack + dequant math validated on CPU (numpy, no torch)
- [x] M1: emul mode SERVES COHERENTLY on 1x B70 (true per-forward fp4 math on XPU).
      "Paris" + coherent Qwen3 reasoning chat. Loaded 5.98 GiB, KV cache fp8_e4m3
      280k tokens, 34x concurrency @ 8192. BUT: <1 tok/s decode (128 tok did not
      finish in 120s) -- the emul kernel re-dequantizes EVERY weight EVERY forward.
      Correctness reference only, unusable for serving. (needed 2 shim fixes:
      register XPU kernel + KVCacheScaleParameter shard_id tolerance.)
- [x] M1b: dequant mode SERVES COHERENTLY + FAST. "Paris" + 31.0 tok/s single-stream
      (enforce-eager, card 0). Weights 15.27 GiB bf16 (one-time NVFP4->bf16 at load),
      KV fp8_e4m3 10.79 GiB / 19x concurrency @ 8192. 31x faster than emul.
      THIS is the usable NVFP4 serve path on B70 today. numerically it is a
      W4A16 read of the W4A4 ckpt (weights exact-dequant, activations stay bf16 ->
      if anything slightly HIGHER quality than the intended W4A4).
- [x] M2: INT8-XMX packed-weight path (the kernel flex). DONE.
      Route (b): E2M1*2 is EXACT int8 ({0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}), so
      nvfp4 weight == s8 weight + per-16-K-group fp scale
      (e4m3_block_scale * weight_scale_2 / 2). 02_int8_repack.py proves it
      BIT-EXACT on real tensors. 03_xmx_microbench.py: oneDNN int8_gemm_w8a16 XMX
      is 1.16-1.66x faster than bf16 F.linear on B70 + int8 weights (half bf16).
      BLOCKER FOUND + FIXED: the kernel's is_block_quant branch hardcoded SQUARE
      {g,g} groups -> wrong numerics (rel-err 0.46-0.80) for NVFP4's K-only
      grouping. FIX (int8_gemm_w8a16.h): infer {grp_k, grp_n} from the scale
      shape [K/grp_k, N/grp_n]; NVFP4 [K/16,N] -> {16,1}. Built via
      build_nvfp4_kernel.sh to a SEPARATE nvfp4_kernel/_xpu_C.abi3.so (daily
      driver untouched; it only uses the per-channel branch). Post-fix microbench:
      rel-err 0.004-0.006 (bf16 scale rounding only) = EXACT, 1.16-1.66x.
      Rejected route (a): fp4_gemm EXISTS but is MXFP4-only (asserts e8m0 @
      group32; NVFP4 is e4m3 @ group16) -> rejects our tensors outright.
      Deferred route (c): Triton-XPU fused dequant-GEMM keeping weights 4-bit in
      VRAM -> the real M=1-decode bandwidth win; later frontier.
- [x] M3: int8xmx mode SERVES COHERENTLY on 1x B70 via our custom kernel.
      XPUInt8XmxNvFp4LinearKernel (shim): repack NVFP4->s8 + [K/16,N] bf16 group
      scale at load, oneDNN int8_gemm_w8a16 each forward. "Paris" + coherent chat,
      31.7 tok/s single-stream (== bf16 dequant speed, decode is BW-bound at 8B)
      BUT weights 9.63 GiB vs 15.27 GiB -> KV cache 18.25 GiB vs 10.79 (2x
      headroom). serve: `MODE=int8xmx ./vllm/nvfp4/serve_nvfp4.sh`.

## Bottom line

NVFP4 runs on Intel B70 three ways, all coherent: emul (reference, <1 tok/s),
dequant-at-load (31 tok/s, 15.3GB bf16), and int8xmx (31.7 tok/s, 9.6GB int8 via
our own oneDNN K-group kernel fix -- the flex). The int8xmx path is the keeper:
same speed, half the weight VRAM, and it actually uses B70's INT8 XMX units on a
format Intel has zero support for.

## 27B (MIXED_PRECISION) -- nvidia/Qwen3.6-27B-NVFP4, 2026-07-04

The real headline target: the ACTUAL NVIDIA NVFP4 build of our daily-driver model.
It is NOT uniform NVFP4 like the 8B -- it is a ModelOpt MIXED_PRECISION checkpoint
(21.9 GB on disk), same Qwen3_5 GDN-hybrid VLM arch as the daily driver:

- MLP gate/up/down     -> W4A16_NVFP4 (4-bit E2M1 weight, per-16-K block scale, bf16 acts)
- self_attn + GDN in_proj -> FP8 (per-tensor E4M3 weight + activation scale)
- norms / conv1d / embed / lm_head / vision tower / mtp -> BF16 (unquantized)
- KV cache -> FP8

vLLM v0.24.0 loads this natively via ModelOptMixedPrecisionConfig (quant_algo
MIXED_PRECISION, per-layer dispatch from the `quantized_layers` map). Confirmed on
XPU: it resolves Qwen3_5ForConditionalGeneration and dispatches FP8/NVFP4/W4A16_NVFP4
per layer. TWO XPU blockers, both fixed in patches/sitecustomize.py:
  (1)  W4A4 layers -> _POSSIBLE_NVFP4_KERNELS has no XPU entry (same as 8B; shim
       registers it).
  (1b) W4A16_NVFP4 layers -> ModelOptNvFp4W4A16LinearMethod HARDCODES a CUDA-only
       MarlinNvFp4LinearKernel (modelopt.py:1277), bypassing the registry, so it
       asserts is_supported() on XPU. Shim replaces its .kernel with an XPU
       4-bit-resident dequant kernel.
  FP8 attention layers need NO shim -- vLLM v0.24.0 already ships
  XPUFP8ScaledMMLinearKernel.

### EXACT single-card VRAM (measured from the real checkpoint headers)

  keep-4bit resident (emul / fused kernel):  21.9 GB  -> FITS one ~30GB B70 card + KV
  dequant NVFP4->int8 at load (int8xmx):     31.1 GB  -> does NOT fit one card
  full bf16 dequant:                         56.7 GB  -> no

  KEY LESSON (answers "why not just int8xmx like the 8B?"): on the 27B, int8xmx
  DOUBLES the 4-bit footprint to 31 GB and no longer fits one card. And dequant-to-
  int8 is only "a worse-packed W8A16" anyway -- the repacked int8 weight takes just
  15 distinct values ({0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}) = 4 bits of real info in an
  8-bit byte, so it costs w8a8's VRAM for int4's precision. The ONLY viable FAST
  single-card path on the 27B is to keep the weights 4-bit resident and dequant on
  the fly in-kernel (the fused E2M1 kernel; see INT4_SPOOF_EXPERIMENTS.md). int8xmx
  stays the keeper for the DENSE 8B (where 2x of 4-bit still fits); it is a deadend
  for the 27B on one card.

- [x] M4: MIXED_PRECISION 27B LOADS on XPU (mixed-precision loader + both NVFP4 XPU
      shims). emul mode (4-bit resident, ~22GB) = the fits+coherence reference.
      BENCH: <pending emul coherence + t/s>.
- [ ] M5: fused E2M1 4-bit-in-VRAM dequant kernel -> fast single-card 27B serve at
      the 22GB footprint. The real deliverable (kernel prototyped card 1).

### Native int4 DPAS on B70 -- verdict (see INT4_DPAS_RESEARCH.md)

Xe2/Battlemage DPAS silicon HAS int4/int2 matrix modes (int4 = 4x bf16 rate), but
NO software stack exposes a true int4xint4 GEMM: oneDNN s4/u4 is weight-decompression
only (decodes to int8/f16 before DPAS), SYCL joint_matrix + Triton-XPU tl.dot both
floor at int8. AND it would not help decode anyway (M=1 is weight-bandwidth bound,
30-300x below the compute roofline -- more FLOPS on an idle unit). The decode lever
is BYTES READ: keep weights 4-bit in VRAM, dequant in registers. NVFP4 also cannot
reuse oneDNN's s4 path: E2M1 is a 4-bit FLOAT LUT, not two's-complement int4, and
the E2M1*2 int trick's +-12 overflows s4's [-8,7]. So a custom E2M1 LUT kernel is
mandatory. Native int4 DPAS (route a) = confirmed deadend; fused 4-bit-in-VRAM
(route b) = the decode win; int8-XMX repack (route c) = prefill/dense-8B only.

## Status log

- 2026-07-04 04:20 model downloaded to models/files/qwen3-8b/nvfp4-modelopt/
- 2026-07-04 04:3x M0 done; shim + serve script written. GPU phase next
  (requires taking the daily driver down; user approved GPU use tonight).
- 2026-07-04 07:1x 27B session: DD down, both cards freed. Downloaded
  nvidia/Qwen3.6-27B-NVFP4 (21.9GB) to models/files/qwen3.6-27b/nvfp4-modelopt/.
  Two parallel agents: int4-DPAS research (card-free, done) + fused-kernel
  prototyping (card 1). 27B serve prep on card 0.
- 2026-07-04 07:3x M4: MIXED_PRECISION loads on XPU; W4A16 Marlin blocker fixed;
  exact sizing confirms int8xmx-on-27B is a deadend (31GB); emul serve on card 0.
