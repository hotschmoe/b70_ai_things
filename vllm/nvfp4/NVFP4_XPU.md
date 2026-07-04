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
- [ ] M1: emul mode serves coherently on 1x B70 (true fp4 math on XPU)
- [ ] M1b: dequant mode serves coherently + bench (expect ~bf16-8B speed)
- [ ] M2: real packed-weight fast path. Candidates:
      (a) `torch.ops._xpu_C.fp4_gemm` -- EXISTS in vllm-xpu-kernels for MXFP4
          W4A4 on XPU (kernels/linear/mxfp4/xpu.py). Check whether the underlying
          kernel can take e4m3 block scales at group 16 (nvfp4) vs e8m0 at
          group 32 (mxfp4).
      (b) INT8-XMX trick: E2M1*2 is EXACT in int8 ({0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}),
          so nvfp4 = s8 weights with group-16 fp scales folded as
          (e4m3_scale * weight_scale_2 / 2) -> ride the proven oneDNN int8 woq
          path (w8a16 kernel) if it supports group scales at G=16.
      (c) Triton-XPU fused dequant-GEMM (LUT in registers), weights stay 4-bit
          in VRAM -> the bandwidth win (decode is BW-bound on B70).

## Status log

- 2026-07-04 04:20 model downloaded to models/files/qwen3-8b/nvfp4-modelopt/
- 2026-07-04 04:3x M0 done; shim + serve script written. GPU phase next
  (requires taking the daily driver down; user approved GPU use tonight).
