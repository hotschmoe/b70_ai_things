#pragma once

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>

#include <dnnl.hpp>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

// W8A16 weights-decompression matmul: f16/bf16 activation x s8 (int8) weight ->
// f16/bf16. The DECODE counterpart of int8_gemm_w8a8 (s8 x s8). At M=1 decode is
// weight-bandwidth-bound, so int8-quantizing the activation buys nothing and only
// adds an act-quant launch; instead keep the activation in f16/bf16 and let oneDNN
// dequantize the s8 weight with a per-channel (or per-block / per-tensor) scale in
// the matmul epilogue -- ONE fused launch, mirroring fp8_gemm_w8a16 but s8 weight.
// Symmetric weights only (no weight zero point).
static inline void dnnl_matmul_w8a16_int8(
    torch::Tensor& result,      // dst, [b, m, n]
    const torch::Tensor& mat1,  // src, [b, m, k] f16/bf16
    const torch::Tensor& mat2,  // quantized weight, [k, n] s8 (NT)
    bool is_nt,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& m2_sc,
    const int64_t group_size = 0) {
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();  // presume channel last format
  const int k = *(src_sz.end() - 1);

  // block quant param: m2_sc is 2D with more than 1 element for block quant
  // Weight scale layout: [k/group_size, n/group_size]
  bool is_block_quant = (m2_sc.dim() == 2) && (m2_sc.numel() > 1);
  int64_t blk_group_size = -1;
  if (is_block_quant) {
    blk_group_size = k / m2_sc.size(0);
  }

  // get joint dtypes
  joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  auto wei_dtype = mat2.scalar_type();
  TORCH_CHECK(
      wei_dtype == at::ScalarType::Char,
      "int8 w8a16 matmul expects s8 (torch.int8) weight, got ",
      wei_dtype);
  if (in_dtype == at::ScalarType::Half) {
    jd = joint_dtypes_t::f16_int8;
  } else if (in_dtype == at::ScalarType::BFloat16) {
    jd = joint_dtypes_t::bf16_int8;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported activation dtype for int8 w8a16 matmul: ", in_dtype);
  }

  // get bias type
  bias_type_t b_type = get_bias_type(bias, m, n);

  trans_type_t tt = trans_type_t::nn;
  if (is_nt) {
    // transpose mat2
    tt = trans_type_t::nt;
  }

  // get lda ldb and ldc
  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false, "Unsupported input dimension for int8 w8a16 matmul: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
                    ? mat2.strides()[mat2.dim() - 2]
                    : mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    if (is_block_quant) {
      // Infer BOTH group sizes from the weight-scale shape [K/grp_k, N/grp_n]
      // instead of assuming a square {g,g} block. NVFP4 stores a per-16-K-group,
      // per-output-channel scale -> shape [K/16, N] -> {grp_k=16, grp_n=1}. The
      // old hardcoded square gave wrong numerics for any K-only grouping.
      const int64_t grp_k = k / m2_sc.size(0);
      const int64_t grp_n = n / m2_sc.size(1);
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 0) + (1 << 1),
          {grp_k, grp_n},
          get_onednn_dtype(m2_sc));
      /* per block quant (K-group x N-group, NVFP4 = {16,1}) */
    } else if (m2_sc.numel() == 1) {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ 0,
          {},
          get_onednn_dtype(m2_sc));
      /* per tensor quant */
    } else {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 1),
          {},
          get_onednn_dtype(m2_sc));
      /* per channel quant */
    }
    pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    if (in_dtype == at::ScalarType::BFloat16) {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
    }
  };

  int arg_off = 0;

  // ************************************************************
  // get device, engine, stream
  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  int m2_sc_group_size = m2_sc.numel();
  int sc_group_size = (group_size << 8) | m2_sc_group_size;
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd, tt, b_type, m, n, k, lda, ldb, ldc, dev_id, f_attr, sc_group_size);

  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      m2_sc.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(m2_sc), engine, m2_sc.data_ptr());
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
  auto qint8_matmul_event =
      matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}
}  // namespace oneDNN
