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

## Status log

- 2026-07-04 04:20 model downloaded to models/files/qwen3-8b/nvfp4-modelopt/
- 2026-07-04 04:3x M0 done; shim + serve script written. GPU phase next
  (requires taking the daily driver down; user approved GPU use tonight).
