#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

// NVFP4 weight-only matmul: weights stay 4-bit (f4_e2m1) resident in VRAM and are
// decompressed in the oneDNN JIT gemm. Mirrors dnnl_matmul_w4a16_int4 but:
//   - weight dtype is f4_e2m1 (E2M1 float 4-bit), NOT two's-complement int4;
//   - packed 2 nibbles/byte, so the leading-dim byte-stride is *2 (int4 packs
//     8 per int32 -> *8);
//   - per-16-K-element group scale (NVFP4 block scale, folded with the fp32
//     global scale into a single bf16 tensor at load), mask (1<<0)+(1<<1),
//     group {group_size, 1};
//   - NO zero-point: E2M1 is a symmetric float grid, so the asymmetric int4
//     zero-point machinery is dropped entirely.
static inline void dnnl_matmul_w4a16_nvfp4(
    torch::Tensor& result,      // dst, [b, m, n] bf16
    const torch::Tensor& mat1,  // src, [b, m, k] bf16
    const torch::Tensor& mat2,  // f4_e2m1 weight packed 2/byte, [k/2, n]
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& scale,  // [k/group_size, n] bf16 (block*global folded)
    int64_t group_size) {
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();  // presume channel last format
  const int k = *(src_sz.end() - 1);

  // get joint dtypes: bf16 activations, f4_e2m1 weights
  joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  if (in_dtype == at::ScalarType::BFloat16) {
    jd = joint_dtypes_t::bf16_f4e2m1;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for nvfp4-w4a16 matmul: ", in_dtype);
  }

  // get bias type
  bias_type_t b_type = get_bias_type(bias, m, n);

  // get lda ldb and ldc
  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false, "Unsupported input dimension for nvfp4-w4a16 matmul: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  // weight packs 2 f4 nibbles per byte -> leading dim in elements = byte-stride * 2
  int64_t ldb = mat2.strides()[mat2.dim() - 1] * 2;
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    pattr.set_scales(
        DNNL_ARG_WEIGHTS,
        /* mask */ (1 << 0) + (1 << 1),
        {group_size, 1},
        get_onednn_dtype(scale));
    // NO zero-point: E2M1 is symmetric.
    pattr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
  };

  // get device, engine, stream
  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd,
      trans_type_t::nt,
      b_type,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      dev_id,
      f_attr,
      group_size);

  int arg_off = 0;
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      scale.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(scale), engine, scale.data_ptr());
      });

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);
  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (get_shape(b_type) != bias_shape_t::none) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }

  int scratchpad_size = matmul_ext.get_scratchpad_size();
  torch::Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto& strm = GpuStreamManager::Instance().get_stream();
  matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}

// Scale-preserving NVFP4 variant. The checkpoint stores one e4m3 block scale
// per 16 K elements plus one fp32 global weight scale. Keeping those as two
// oneDNN scale attributes avoids expanding every e4m3 scale to bf16 at model
// load and halves scale traffic in the bandwidth-bound small-M path.
static inline void dnnl_matmul_w4a16_nvfp4_f8scale(
    torch::Tensor& result,
    const torch::Tensor& mat1,
    const torch::Tensor& mat2,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& block_scale,
    const torch::Tensor& global_scale,
    int64_t group_size) {
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();
  const int k = *(src_sz.end() - 1);

  TORCH_CHECK(
      mat1.scalar_type() == at::ScalarType::BFloat16,
      "nvfp4 f8-scale matmul expects bf16 activations");
  TORCH_CHECK(
      block_scale.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "nvfp4 f8-scale matmul expects e4m3 block scales");
  TORCH_CHECK(
      global_scale.scalar_type() == at::ScalarType::Float &&
          global_scale.numel() == 1,
      "nvfp4 f8-scale matmul expects one fp32 global scale");

  joint_dtypes_t jd = joint_dtypes_t::bf16_f4e2m1;
  bias_type_t b_type = get_bias_type(bias, m, n);

  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false,
        "Unsupported input dimension for nvfp4 f8-scale matmul: ",
        mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] * 2;
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    pattr.set_scales(
        DNNL_ARG_WEIGHTS,
        /* mask */ (1 << 0) + (1 << 1),
        {group_size, 1},
        get_onednn_dtype(block_scale));
    // Multiplying the source by the scalar global weight scale is
    // algebraically identical to folding it into every weight block scale.
    pattr.set_scales(
        DNNL_ARG_SRC, /* mask */ 0, {}, get_onednn_dtype(global_scale));
    pattr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
  };

  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  // Keep this primitive distinct from the folded-bf16-scale variant in the
  // thread-local cache: the attribute count and scale dtypes differ.
  int64_t cache_key = (1LL << 30) | group_size;
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd,
      trans_type_t::nt,
      b_type,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      dev_id,
      f_attr,
      cache_key);

  int arg_off = 0;
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      block_scale.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(block_scale), engine, block_scale.data_ptr());
      });
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
      global_scale.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(global_scale), engine, global_scale.data_ptr());
      });

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);
  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (get_shape(b_type) != bias_shape_t::none) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }

  int scratchpad_size = matmul_ext.get_scratchpad_size();
  torch::Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto& strm = GpuStreamManager::Instance().get_stream();
  matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}
}  // namespace oneDNN
