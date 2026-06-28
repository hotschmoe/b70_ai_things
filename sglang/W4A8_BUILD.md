# Building int4_gemm_w4a8 + int4_gemm_w4a16 for sglang-XPU (torch 2.12) -- 2026-06-28

THE UNLOCK. vLLM's oneDNN ops `torch.ops._xpu_C.int4_gemm_w4a8` (int4w x int8a) and
`int4_gemm_w4a16` (int4w x fp16a) are fast on B70 but every prebuilt .so fails to load
in the sglang image (undefined symbol `torch::Library::_def` -- ABI mismatch vs sglang's
torch 2.12.0+xpu). FIX: build _xpu_C from vllm-xpu-kernels SOURCE against the sglang
image's exact torch. The sglang image has the Intel DPC++ toolchain (icpx 2025.3).

## Result (VERIFIED 2026-06-28, card 0, real Lorbus/sqgptq down_proj 17408x5120, warm)
The freshly-built _xpu_C.abi3.so (50.96 MB) LOADS in sglang-xpu:woq (torch 2.12.0+xpu),
registers both ops, runs FINITE. The HYBRID (both ops, same compressed-tensors int4 weights):
  DECODE  M=1   : int4_gemm_w4a16 (fp16 act)            = 0.079 ms = 2.13x fp16, 1.83x FASTER than woqgemm (0.145)
  PREFILL M=2048: int4_gemm_w4a8 (int8 act, compile-fused quant) = 1.96 ms = 1.13x fp16, 1.9x FASTER than woqgemm (3.76)
  (int4_gemm_w4a8 op-only = 0.079/1.69 ms; the eager per-token act-quant (1.91ms@M=2048) MUST be torch.compile-fused.)
=> int4-weight hybrid beats the int4-woqgemm champion on BOTH decode AND prefill. The W4A8 (int8-act) op
   is the PREFILL/TTFT win; the W4A16 (fp16-act) op is the DECODE win (int8 acts don't help bandwidth-bound M=1).

## Persisted artifacts (NOT in git -- 50MB binary)
/mnt/vm_8tb/b70/w4a8_kernel/_xpu_C.abi3.so   (sha256 63c8be3d26c8...)  <- bake into sglang-xpu:w4a8
/mnt/vm_8tb/b70/w4a8_kernel/build_xpu_c.sh   (the build script below)
/mnt/vm_8tb/b70/w4a8_kernel/src/             (onednn op source for reference)

## Build recipe (scoped to ONLY _xpu_C; skips the 1.68GB attn AOT)
1. clone: git clone https://github.com/vllm-project/vllm-xpu-kernels  (main; requires torch 2.12.0+xpu)
2. build INSIDE sglang-xpu:woq (no GPU needed), repo mounted at /build/vllm-xpu-kernels, run build_xpu_c.sh:
   key env (the minimal scope): BUILD_SYCL_TLA_KERNELS=OFF BASIC_KERNELS_ENABLED=OFF FA2_KERNELS_ENABLED=OFF
     MOE_KERNELS_ENABLED=OFF GDN_KERNELS_ENABLED=OFF MQA_LOGITS_KERNELS_ENABLED=OFF XPUMEM_ALLOCATOR_ENABLED=OFF
     XPU_SPECIFIC_KERNELS_ENABLED=ON  (<- the onednn int4 ops)  VLLM_XPU_AOT_DEVICES=bmg  VLLM_VERSION_OVERRIDE=0.1.3.1
   then: pip install setuptools_scm ; python setup.py build_ext --inplace   (~14 min, 699 ninja targets, oneDNN bundled static)
   produces vllm_xpu_kernels/_xpu_C.abi3.so. (cmd: docker run -d --name w4a8build -v <scratch>:/build sglang-xpu:woq bash /build/build_xpu_c.sh)

## Loading it at runtime (the ABI gotchas)
- It loads only via `import vllm_xpu_kernels._xpu_C` OR ctypes.CDLL(so, RTLD_GLOBAL) AFTER `import torch`.
  (torch.ops.load_library FAILS -- RTLD_LOCAL doesn't resolve libtorch deps.)
- LD_LIBRARY_PATH must PREPEND the oneAPI compiler libs, keeping the image default:
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  (REPLACING it clobbers the level-zero runtime -> "No XPU devices are available".)

## Calling convention (VERIFIED)
int4_gemm_w4a8(A_int8[M,K], A_scale[M,1] fp16, A_zp[M,1] int32, B[K/8,N] int32, B_scale[K/g,N], B_zp, group_size, g_idx=None, bias=None) -> fp16 [M,N]
int4_gemm_w4a16(A_fp16[M,K], B[K/8,N] int32, bias|None, B_scale[K/g,N], B_zp, group_size, g_idx=None) -> fp16 [M,N]
  - B MUST be NT format: B = weight.t() as a VIEW (stride[0]==1); weight on disk is [N,K/8] int32 -> weight.t() = [K/8,N]. NO .contiguous()!
  - B_scale = weight_scale.t() ([K/g,N]); weight_scale on disk [N,K/g].
  - B_zp: a 1-D tensor (e.g. tensor([8], int8)) selects the SYMMETRIC path (zp.dim()==1); 2-D [K/g,N/8] = asymmetric.
  - A activation quant is EXTERNAL (per-token symmetric int8); torch.compile-fuse it (eager is ~1.9ms@M=2048 = the bottleneck).
  - sqgptq-prepacked ckpt layout already matches exactly. Serve --dtype float16 (op emits fp16).
