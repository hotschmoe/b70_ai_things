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

## Native E4M3 block-scale small-M path (2026-07-22)

The checkpoint already stores one E4M3 block scale per 16 K values plus a scalar
FP32 global scale. Expanding and folding those scales to BF16 doubles scale traffic
in the bandwidth-bound decode GEMM. The new op keeps the checkpoint representation:

    y = torch.ops._xpu_C.nvfp4_gemm_w4a16_f8scale(
            A_bf16[M,K], B_f4e2m1_packed[K/2,N] uint8, bias?,
            B_scale_e4m3[K/16,N], B_global_scale_fp32[1], group_size=16)

oneDNN receives the E4M3 tensor as the grouped weight scale and the global FP32
scalar as a source scale. The latter is algebraically equivalent to folding the
global value into every weight block scale, but avoids both BF16 scale rounding and
the extra scale bytes. A separate primitive-cache key prevents aliasing the folded
BF16 and native-E4M3 attribute layouts.

The shim dispatches the native-scale op only when flattened M is at or below
`B70_NVFP4_F8_SCALE_M_MAX`; the measured shelf value is 8. Larger matrices retain
the existing folded-BF16 XMX path because it is faster for prefill. Both NT scale
layouts are prepared once at load, then the unused row-major staging tensors are
deleted.

Real Qwen3.6-27B layer-0 gate projection, N=17408 and K=5120, B70 card 0, median
of five A-B-A rounds:

| M | folded BF16 scale | native E4M3 scale | speedup |
|---:|---:|---:|---:|
| 1 | 0.1156 ms | 0.0948 ms | 1.220x |
| 2 | 0.1055 ms | 0.0962 ms | 1.096x |
| 4 | 0.1064 ms | 0.0972 ms | 1.094x |
| 6 | 0.1070 ms | 0.0974 ms | 1.098x |
| 8 | 0.1075 ms | 0.0982 ms | 1.095x |
| 16 | 0.1289 ms | 0.1306 ms | 0.987x |
| 512 | 0.8723 ms | 0.9198 ms | 0.948x |

The real-model relative L2 delta versus the folded-BF16-scale op is about 0.0034;
this is the expected removal of BF16 scale-folding roundoff, not an alternate FP4
decode. The packed E2M1 weights are unchanged.

Reproducible sources:

- `kernels/nvfp4_gemm_w4a16.h`: both oneDNN implementations.
- `kernels/nvfp4_f8scale_integration.patch`: op declaration, binding, and entry point.
- `vllm/nvfp4/bench_f8scale.py`: real-checkpoint A/B microbenchmark.
- `vllm/nvfp4/build_nvfp4_f8scale_gdn.sh`: fresh-tree GDN-enabled serve build.

Build without modifying the production artifact:

    bash vllm/nvfp4/build_nvfp4_f8scale_gdn.sh

Output is `/mnt/vm_8tb/b70/nvfp4_f8scale_kernel_gdn/`, including the ABI-matched
GDN sidecar. The production shelf selects this directory explicitly; the older
`nvfp4_fused_kernel_gdn` artifact remains available for rollback.
