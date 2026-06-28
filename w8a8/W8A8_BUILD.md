# Building the fused int8 W8A8 oneDNN ops for sglang-XPU (torch 2.12) -- 2026-06-28

Adds the two ops that make W8A8 fast on B70, mirroring the W4A8 kernel build
(sglang/W4A8_BUILD.md). Both are oneDNN weights/activation matmuls -> NOT
joint_matrix-gated, run clean on Battlemage Xe2.

## The two ops

- `int8_gemm_w8a16(A_f16[M,K], B_s8[K,N] NT, B_scale[N], bias?) -> [M,N]`  -- **DECODE**.
  f16/bf16 activation x s8 weight, per-channel weight scale dequantized in the matmul
  epilogue = ONE fused launch. At M=1 decode is weight-BW-bound, so the activation stays
  f16 (no act-quant) -- the int8 analog of int4_gemm_w4a16. NEW this session
  (`csrc/xpu/onednn/int8_gemm_w8a16.h` + `f16_int8`/`bf16_int8` joint_dtypes mappers).
- `int8_gemm_w8a8(A_s8[M,K], A_scale[M,1], A_zp?, B_s8[K,N] NT, B_scale[N], azp?, bias?, out_dtype) -> [M,N]`
  -- **PREFILL**. s8 act x s8 weight, per-token x per-channel output scale fused in oneDNN
  epilogue (s8s8s32 XMX). Was ALREADY staged in the source tree (the old PP1 work) but never
  compiled into the deployed w4a8 .so; this build includes it.
- `dynamic_per_token_int8_quant(x, sym=True, bits=8) -> (q_s8, scale, zp)` -- fused per-token
  symmetric int8 act-quant (single launch; for the prefill path). Also pre-existing in tree.

## Result (VERIFIED 2026-06-28, card 0, real Qwen3.6-27B shapes, synthetic weights, warm; fp16 baseline)

```
                       DECODE M=1 (x bf16)      PREFILL M=2048 (x bf16)
  int8_gemm_w8a16      1.86-1.91x (the DECODE op, captured 1.73-1.89x)   0.98-1.06x (don't use for prefill)
  int8_gemm_w8a8       1.83-1.88x (op-only)     1.95-2.07x  (the PREFILL op)
  fp8_gemm_w8a16 (bar) 1.88-1.95x               0.98-1.00x  (no XMX -> no prefill win)
  bf16                 1.00x                    1.00x
```
=> the HYBRID (decode=int8_gemm_w8a16, prefill=int8_gemm_w8a8) HANDILY beats fp8 (decode ~tie
   ~1.9x, prefill 2.0x vs 1.0x) AND bf16 (~2x both) at the kernel level. int8-accurate (relerr
   ~9e-3 synthetic, vs fp8's e4m3). This is the task target met at the kernel level. The old
   3-kernel _int_mm chain was 0.64-0.80x decode (eager) / 0.7-0.8x prefill -- this replaces it.

## Persisted artifacts (NOT in git -- 51MB binary)
/mnt/vm_8tb/b70/w8a8_kernel/_xpu_C.abi3.so  (sha256 bc643c3f8a61..., 51095992 B)
/mnt/vm_8tb/b70/w8a8_kernel/build_xpu_c.sh
/mnt/vm_8tb/b70/vllm-xpu-kernels-w8a8/    (the EDITED source tree; writable copy I own)
In git: w8a8/int8_gemm_kernel.patch (the source diff vs upstream), w8a8/kernel_src/*.h.

## Build recipe (clean configure -- the stale-cache gotcha below)
1. Source tree = a WRITABLE copy of vllm-xpu-kernels with the patch applied:
   `rsync -a --exclude=.git /mnt/vm_8tb/b70/vllm-xpu-kernels/ /mnt/vm_8tb/b70/vllm-xpu-kernels-w8a8/`
   then `git apply` (or hand-apply) `w8a8/int8_gemm_kernel.patch` + drop in
   `w8a8/kernel_src/int8_gemm_w8a16.h` -> `csrc/xpu/onednn/`.
2. GOTCHA: the prior build's `build/` dir cmake cache is pinned to the OLD mount path (`/src`).
   Reusing it -> "CMakeCache.txt directory ... is different / source does not match" -> RC=1 at
   configure. FIX: `rm -rf build .deps vllm_xpu_kernels/_xpu_C*.so` for a CLEAN configure.
3. Build INSIDE sglang-xpu:woq (no GPU), tree mounted at /build/vllm-xpu-kernels:
   ```
   docker run --rm --name w8a8build \
     -v /mnt/vm_8tb/b70/vllm-xpu-kernels-w8a8:/build/vllm-xpu-kernels \
     -v /mnt/vm_8tb/b70/w4a8_kernel/build_xpu_c.sh:/build/build_xpu_c.sh:ro \
     sglang-xpu:woq bash -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; bash /build/build_xpu_c.sh"
   ```
   Same scope env as W4A8 (XPU_SPECIFIC_KERNELS_ENABLED=ON, all else OFF, VLLM_XPU_AOT_DEVICES=bmg).
   ~14 min clean. Produces `vllm_xpu_kernels/_xpu_C.abi3.so`. (Build artifacts come out root-owned
   since the container runs as root; the .so is 644 world-readable -> copy it out.)

## Loading (same ABI gotchas as W4A8)
- `ctypes.CDLL(so, RTLD_GLOBAL)` AFTER `import torch` (torch.ops.load_library FAILS).
- `export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH` (PREPEND, don't replace).

## Calling convention (VERIFIED card 0)
- int8_gemm_w8a16: B = weight.t() VIEW (weight stored [N,K] s8 -> B [K,N] stride[0]==1, NO contiguous);
  B_scale = per-channel [N] (f16). Per-tensor [1] or block [K/g, N] also supported. Symmetric (no zp).
- int8_gemm_w8a8: A s8 [M,K], A_scale per-token [M,1] (f16), A_zp=None (sym); B s8 [K,N] NT,
  B_scale per-channel [N] (f16); azp_adj=None, bias=None, out_dtype=torch.float16.
- Probe: w8a8/w8a8_fused_probe.py.  Next: wire into w8a8_shim.py (decode->w8a16, prefill->w8a8).
