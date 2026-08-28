#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "int8_gemm_w8a8.h"

namespace oneDNN {

// BF16/FP16 activation times symmetric INT8 weight with per-tensor or
// per-output-channel scales. This decode path deliberately avoids activation
// quantization while preserving the current INT8 scratchpad and queue handoff.
static inline void dnnl_matmul_w8a16_int8(
    torch::Tensor& result,
    const torch::Tensor& mat1,
    const torch::Tensor& mat2,
    const torch::Tensor& m2_sc,
    bool is_nt,
    const std::optional<torch::Tensor>& bias) {
  const auto src_sz = mat1.sizes();
  const auto output_sz = result.sizes();
  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = output_sz.back();
  const int k = src_sz.back();

  TORCH_CHECK(
      mat1.scalar_type() == at::ScalarType::Half ||
          mat1.scalar_type() == at::ScalarType::BFloat16,
      "input must be float16 or bfloat16 for int8 W8A16 matmul");
  TORCH_CHECK(
      mat2.scalar_type() == at::ScalarType::Char,
      "weight must be int8 for int8 W8A16 matmul");
  TORCH_CHECK(
      m2_sc.numel() == 1 || m2_sc.numel() == n,
      "weight scales must be per-tensor or per-output-channel for int8 "
      "W8A16 matmul");

  const joint_dtypes_t jd =
      mat1.scalar_type() == at::ScalarType::BFloat16
      ? joint_dtypes_t::bf16_int8
      : joint_dtypes_t::f16_int8;
  const bias_type_t b_type = get_bias_type(bias, m, n);
  const trans_type_t tt = is_nt ? trans_type_t::nt : trans_type_t::nn;

  const auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false,
        "Unsupported input dimension for int8 W8A16 matmul: ",
        mat1.dim());
  }

  const int64_t lda = mat1_strides[leading_dim];
  const int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
      ? mat2.strides()[mat2.dim() - 2]
      : mat2.strides()[mat2.dim() - 1];
  const int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    if (m2_sc.numel() == 1) {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS, /* mask */ 0, {}, get_onednn_dtype(m2_sc));
    } else {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 1),
          {},
          get_onednn_dtype(m2_sc));
    }
  };

  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  const at::Device current_device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(current_device);
  const int scale_group = m2_sc.numel() == 1 ? 1 : 2;
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd,
      tt,
      b_type,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      dev_id,
      f_attr,
      scale_group);

  int arg_off = 0;
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

  const int scratchpad_size = matmul_ext.get_scratchpad_size();
  torch::Tensor& scratchpad_tensor =
      get_int8_gemm_scratchpad_cache(dev_id, scratchpad_size, mat1.options());
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto& stream = GpuStreamManager::Instance().get_stream();
  std::vector<sycl::event> deps;
  if (const char* dependency = std::getenv(
          "VLLM_XPU_ONEDNN_INT8_INPUT_DEPENDENCY");
      dependency != nullptr && dependency[0] == '1' &&
      dependency[1] == '\0') {
    auto& queue = c10::xpu::getCurrentXPUStream().queue();
    deps.emplace_back(queue.ext_oneapi_submit_barrier());
    TORCH_WARN_ONCE("VLLM_XPU_ONEDNN_INT8_INPUT_DEPENDENCY reached");
  }
  auto done = matmul_ext.execute(
      stream, engine, std::move(arg_handles), arg_off, std::move(deps));
  if (const char* barrier = std::getenv(
          "VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER");
      barrier != nullptr && barrier[0] == '1' && barrier[1] == '\0') {
    auto& queue = c10::xpu::getCurrentXPUStream().queue();
    queue.ext_oneapi_submit_barrier({done});
    TORCH_WARN_ONCE("VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER reached");
  }
}

}  // namespace oneDNN
