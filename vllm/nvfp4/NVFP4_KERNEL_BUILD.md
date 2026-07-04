# nvfp4_gemm_w4a16 -- custom oneDNN NVFP4 weight-decompression op (build recipe)

`torch.ops._xpu_C.nvfp4_gemm_w4a16` is a bit-exact NVFP4 (E2M1) weight-only matmul: the
weights stay 4-bit (f4_e2m1) resident in VRAM and are decompressed inside the oneDNN JIT
gemm. It is the FAST single-card path for the 27B NVFP4 serve (the int8 repack does not fit
one card at 31 GB; this keeps ~22-24 GB). Microbench: 2.85x bf16 F.linear at decode,
bit-exact vs the E2M1 reference. Forward signature:

    y = torch.ops._xpu_C.nvfp4_gemm_w4a16(
            A_bf16[M,K], B_f4e2m1_packed[K/2,N] uint8, bias?, B_scale_bf16[K/16,N], group_size=16)

Built from the `vllm-xpu-kernels-v0240` source tree (has GDN + int4/fp4 gemm). The op is a
small port of `int4_gemm_w4a16` (weight dtype f4_e2m1 instead of s4/u4, group-16 K-scale, NO
zero-point since E2M1 is a symmetric float grid). Kernel header source-of-truth:
`kernels/nvfp4_gemm_w4a16.h`.

## Source edits (vs stock vllm-xpu-kernels-v0240)

1. `csrc/xpu/onednn/nvfp4_gemm_w4a16.h` -- NEW (== `kernels/nvfp4_gemm_w4a16.h`). Mirrors
   `dnnl_matmul_w4a16_int4` with jd=bf16_f4e2m1, weight [K/2,N] (ldb = byte-stride * 2),
   set_scales(DNNL_ARG_WEIGHTS, mask (1<<0)+(1<<1), {group_size,1}, bf16), NO set_zero_points,
   fpmath bf16, and the zero-point-free create_and_cache overload (single group_size arg).

2. `csrc/xpu/onednn/onednn_ext.h` -- add the joint dtype:
   - enum `joint_dtypes_t { ... bf16_f4e2m1, ... }`
   - `onednn_types_mapper<bf16_f4e2m1>::get()` -> make_tuple(bf16, f4_e2m1, bf16)
   - runtime dispatch: `case joint_dtypes_t::bf16_f4e2m1: return
     matmul_primitive_create_and_cache<joint_dtypes_t::bf16_f4e2m1, F>(...)` (mirror the
     bf16_int4 case). oneDNN already exposes `memory::data_type::f4_e2m1` (dnnl 3.x, =14).

3. `csrc/xpu/ops.h` -- declare:
   `torch::Tensor nvfp4_gemm_w4a16(const at::Tensor& A, const at::Tensor& B,
   const std::optional<at::Tensor>& bias, const at::Tensor& B_scale, int64_t group_size);`

4. `csrc/xpu/onednn/onednn_matmul.cpp` -- entrypoint: checks B is 2D NT packed (K contiguous),
   allocs the bf16 result via check_and_create_output_tensor, calls
   `oneDNN::dnnl_matmul_w4a16_nvfp4(result, A, B, bias, B_scale, group_size)`.

5. `csrc/xpu/torch_bindings.cpp` -- register:
   `xpu_ops.def("nvfp4_gemm_w4a16(Tensor A, Tensor B, Tensor? bias, Tensor B_scale, int group_size) -> Tensor");`
   `xpu_ops.impl("nvfp4_gemm_w4a16", torch::kXPU, &nvfp4_gemm_w4a16);`

## Build (two variants)

The op is precision-agnostic to GDN, but the 27B is a GDN-hybrid VLM, so the SERVE .so must
include the GDN attention kernels; the standalone op microbench does not need them.

- Op validation / microbench (fast, ~8 min): GDN OFF.
    docker run ... vllm-xpu-env:v0240 -c 'source setvars; GDN_KERNELS_ENABLED=OFF \
      python setup.py build_ext --inplace'  ->  _xpu_C.abi3.so (~51 MB, nvfp4 op, no GDN)
    output copied to /mnt/vm_8tb/b70/nvfp4_fused_kernel/_xpu_C.abi3.so

- Serve (~20 min): GDN ON (default) -> a .so with BOTH gdn_attention_core AND nvfp4_gemm_w4a16
    ... GDN_KERNELS_ENABLED=ON ...  ->  _xpu_C.abi3.so (~61 MB)
    output to /mnt/vm_8tb/b70/nvfp4_fused_kernel_gdn/_xpu_C.abi3.so
    (+ the GDN sidecar libgdn_attn_kernels_xe_2.so from w8a8_kernel_v0240)

## Serve

    MODE=fused ./vllm/nvfp4/serve_nvfp4_27b.sh   # FUSED_SO -> nvfp4_fused_kernel_gdn .so

The shim (patches/sitecustomize.py) MODE=fused wires `_XPUW4A16NvFp4Kernel.apply_weights` to
`torch.ops._xpu_C.nvfp4_gemm_w4a16`; process_weights_after_loading repacks the resident weight
to [K/2,N] and folds the E4M3 block scale x fp32 global scale into a [K/16,N] bf16 tensor ONCE
at load, so the weights stay 4-bit resident.
