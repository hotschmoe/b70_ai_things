#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

// BLOCK-SCALED INT8 matmul for NVFP4-as-int8 prefill.
//
// This is the INT8-XMX prefill path for NVFP4: the E2M1 weight is losslessly
// repacked to s8 codes ({0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}) with a per-16-K-group
// scale g = block_scale * global_scale / 2 (proven bit-exact, 02_int8_repack.py).
// Activations are per-token dynamic-int8 quantized. So:
//   src  s8 [m,k], per-token src scale [m,1]  (symmetric, no zp)
//   wei  s8 [k,n] NT, per-16-K-GROUP + per-N-channel weight scale [k/grp,n]
//        (mask (1<<0)+(1<<1), {grp_k, grp_n}) -- BLOCK-SCALED, the crux
//   out  bf16/f16
//
// oneDNN accumulates s8*s8 in s32 and dequantizes PER GROUP inside the reduction
// (a grouped weight scale is applied before the K-sum), then applies the per-token
// src scale -- i.e. genuine block-scaled int8. Symmetric-only (E2M1 grid is
// symmetric; per-token int8 act quant is symmetric), so NO zero points. Uses the
// s8_s8 joint dtype so it rides B70 INT8 XMX (s8s8s32), not a bf16 upcast.
static inline void dnnl_matmul_nvfp4_w4a8(
    torch::Tensor& result,       // dst, [b, m, n] bf16/f16
    const torch::Tensor& mat1,   // quantized src, [b, m, k] s8
    const torch::Tensor& m1_sc,  // src scale, [m, 1] per-token (or [1] per-tensor)
    const torch::Tensor& mat2,   // s8 weight, [k, n] (NT)
    const torch::Tensor& m2_sc,  // weight scale [k/grp, n] (block) or [n] (per-chan) or [1]
    int64_t group_size) {
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();
  const int k = *(src_sz.end() - 1);

  auto in_dtype = mat1.scalar_type();
  auto wei_dtype = mat2.scalar_type();
  auto out_dtype = result.scalar_type();
  TORCH_CHECK(
      in_dtype == at::ScalarType::Char && wei_dtype == at::ScalarType::Char,
      "nvfp4_w4a8 (block-scaled int8) expects s8 src and s8 weight, got src=",
      in_dtype, " weight=", wei_dtype);
  joint_dtypes_t jd = out_dtype == at::ScalarType::BFloat16
                          ? joint_dtypes_t::s8_s8_bf16
                          : joint_dtypes_t::s8_s8_f16;

  bias_type_t b_type = get_bias_type(std::nullopt, m, n);

  // weight stored transposed [k,n] with k-major stride -> NT
  bool is_nt = mat2.strides()[mat2.dim() - 2] == 1;
  trans_type_t tt = is_nt ? trans_type_t::nt : trans_type_t::nn;

  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(false, "Unsupported input dim for nvfp4_w4a8: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
                    ? mat2.strides()[mat2.dim() - 2]
                    : mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[leading_dim];

  // block-scaled weight iff m2_sc is 2D with >1 element (NVFP4 [k/16, n])
  bool is_block_quant = (m2_sc.dim() == 2) && (m2_sc.numel() > 1);

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // src (activation) scale: per-token or per-tensor. Symmetric -> no zp.
    if (m1_sc.numel() == 1) {
      pattr.set_scales(DNNL_ARG_SRC, /*mask*/ 0, {}, get_onednn_dtype(m1_sc));
    } else {
      pattr.set_scales(
          DNNL_ARG_SRC, /*mask*/ (1 << 0) + (1 << 1), {1, k},
          get_onednn_dtype(m1_sc));  // per token
    }

    // weight scale: block (K-group x N-group), per-channel, or per-tensor.
    if (is_block_quant) {
      const int64_t grp_k = k / m2_sc.size(0);
      const int64_t grp_n = n / m2_sc.size(1);
      pattr.set_scales(
          DNNL_ARG_WEIGHTS, /*mask*/ (1 << 0) + (1 << 1), {grp_k, grp_n},
          get_onednn_dtype(m2_sc));  // NVFP4 = {16, 1}
    } else if (m2_sc.numel() == 1) {
      pattr.set_scales(DNNL_ARG_WEIGHTS, /*mask*/ 0, {}, get_onednn_dtype(m2_sc));
    } else {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS, /*mask*/ (1 << 1), {}, get_onednn_dtype(m2_sc));  // per channel
    }
    // NO fpmath_mode override: keep the integer s8s8s32 accumulate (INT8 XMX).
  };

  int arg_off = 0;
  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  int m1_sc_group_size = m1_sc.numel();
  int m2_sc_group_size = m2_sc.numel();
  // Encode both scale numels + a block flag into the cache key so a block-scaled
  // primitive can never alias a per-channel one (different attr topology).
  int64_t sc_group_size = (static_cast<int64_t>(is_block_quant ? 1 : 0) << 40) |
                          (static_cast<int64_t>(m1_sc_group_size) << 20) |
                          static_cast<int64_t>(m2_sc_group_size);
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd, tt, b_type, m, n, k, lda, ldb, ldc, dev_id, f_attr, sc_group_size);

  matmul_ext.set_attribute(
      arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m2_sc.data_ptr(),
      [&]() {
        return make_onednn_memory(get_onednn_md(m2_sc), engine, m2_sc.data_ptr());
      });
  matmul_ext.set_attribute(
      arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc.data_ptr(), [&]() {
        return make_onednn_memory(get_onednn_md(m1_sc), engine, m1_sc.data_ptr());
      });

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);
  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  int scratchpad_size = matmul_ext.get_scratchpad_size();
  torch::Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto& strm = GpuStreamManager::Instance().get_stream();
  matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}
}  // namespace oneDNN
