#pragma once

#include <array>
#include <cstdlib>

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

struct int8_gemm_scratchpad_ring_t {
  std::vector<torch::Tensor> tensors;
  int next = 0;
};

static inline int get_int8_gemm_scratchpad_ring_size() {
  static const int ring_size = []() {
    const char* env = std::getenv("VLLM_XPU_INT8_GEMM_SCRATCHPAD_RING_SIZE");
    if (env == nullptr || env[0] == '\0') {
      return 1;
    }
    int parsed = std::atoi(env);
    if (parsed < 1) {
      parsed = 1;
    }
    if (parsed > 16) {
      parsed = 16;
    }
    return parsed;
  }();
  return ring_size;
}

static inline torch::Tensor& get_int8_gemm_scratchpad_cache(
    const int device_id,
    const int scratchpad_size,
    const torch::TensorOptions& options) {
  static thread_local std::array<
      ska::flat_hash_map<int, int8_gemm_scratchpad_ring_t>,
      16>
      scratchpads;

  auto& device_scratchpads = scratchpads[device_id];
  auto& entry = device_scratchpads[scratchpad_size];
  const int ring_size = get_int8_gemm_scratchpad_ring_size();
  if (static_cast<int>(entry.tensors.size()) != ring_size) {
    entry.tensors.clear();
    entry.tensors.reserve(ring_size);
    for (int i = 0; i < ring_size; ++i) {
      entry.tensors.push_back(
          at::empty({scratchpad_size}, options.dtype(at::kByte), c10::nullopt));
    }
    entry.next = 0;
  } else {
    for (int i = 0; i < ring_size; ++i) {
      if (!entry.tensors[i].defined()) {
        entry.tensors[i] = at::empty(
            {scratchpad_size}, options.dtype(at::kByte), c10::nullopt);
      }
    }
  }

  torch::Tensor& tensor = entry.tensors[entry.next % ring_size];
  entry.next = (entry.next + 1) % ring_size;
  return tensor;
}

// per-token dynamic-int8 activations x per-channel int8 weights -> f16/bf16
//
// Symmetric-only (no src/weight zero points). oneDNN accumulates in s32
// internally and applies the src (per-token) and weight (per-channel) output
// scales to produce an f16/bf16 result. This mirrors the fp8 w8a8 scale layout
// but with s8 src + s8 weights (s8s8s32 native on Battlemage XMX).
static inline void dnnl_matmul_w8a8_int8(
    torch::Tensor& result,      // dst, [b, m, n]
    const torch::Tensor& mat1,  // quantized src, [b, m, k] s8
    const torch::Tensor& mat2,  // quantized weight, [k, n] s8 (NT)
    bool is_nt,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& m1_sc,  // src scale, [m, 1] (per-token) or [1]
    const torch::Tensor& m2_sc) {  // weight scale, [1, n] (per-channel) or [1]
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();  // presume channel last format
  const int k = *(src_sz.end() - 1);

  // get joint dtypes
  joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  auto wei_dtype = mat2.scalar_type();
  auto out_dtype = result.scalar_type();

  TORCH_CHECK(
      in_dtype == at::ScalarType::Char && wei_dtype == at::ScalarType::Char,
      "int8 w8a8 matmul expects s8 (torch.int8) src and weight, got src=",
      in_dtype,
      " weight=",
      wei_dtype);

  jd = out_dtype == at::ScalarType::BFloat16 ? joint_dtypes_t::s8_s8_bf16
                                             : joint_dtypes_t::s8_s8_f16;

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
        false, "Unsupported input dimension for int8 matmul: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
                    ? mat2.strides()[mat2.dim() - 2]
                    : mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // src (activation) scale: per-token or per-tensor. Symmetric -> no zp.
    if (m1_sc.numel() == 1) {
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */ 0,
          {},
          get_onednn_dtype(m1_sc));
      /* per tensor quant */
    } else {
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */ (1 << 0) + (1 << 1),
          {1, k},
          get_onednn_dtype(m1_sc));
      /* per token quant */
    }

    // weight scale: per-channel or per-tensor. Symmetric -> no zp.
    if (m2_sc.numel() == 1) {
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
  };

  int arg_off = 0;

  // ************************************************************
  // get device, engine, stream
  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  int m1_sc_group_size = m1_sc.numel();
  int m2_sc_group_size = m2_sc.numel();
  int sc_group_size = (m1_sc_group_size << 8) | m2_sc_group_size;
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
  matmul_ext.set_attribute(
      arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc.data_ptr(), [&]() {
        return make_onednn_memory(
            get_onednn_md(m1_sc), engine, m1_sc.data_ptr());
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
  torch::Tensor& scratchpad_tensor =
      get_int8_gemm_scratchpad_cache(dev_id, scratchpad_size, mat1.options());
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto& strm = GpuStreamManager::Instance().get_stream();
  if (const char* dependency =
          std::getenv("VLLM_XPU_ONEDNN_INT8_INPUT_DEPENDENCY");
      dependency != nullptr && dependency[0] == '1' &&
      dependency[1] == '\0') {
    // This oneDNN stream wraps PyTorch's current in-order XPU queue. Put an
    // explicit queue tail between activation quantization and primitive
    // submission without relying on the newer execute(deps) overload.
    auto& queue = c10::xpu::getCurrentXPUStream().queue();
    queue.ext_oneapi_submit_barrier();
    TORCH_WARN_ONCE("VLLM_XPU_ONEDNN_INT8_INPUT_DEPENDENCY reached");
  }
  auto done = matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
  // D9/K1: publish oneDNN completion onto the PyTorch XPU stream so a TP
  // collective cannot consume logits on oneCCL's stream first. Async device
  // dependency, not a host wait. Default off.
  if (const char* barrier =
          std::getenv("VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER");
      barrier != nullptr && barrier[0] == '1' && barrier[1] == '\0') {
    auto& queue = c10::xpu::getCurrentXPUStream().queue();
    queue.ext_oneapi_submit_barrier({done});
    TORCH_WARN_ONCE("VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER reached");
  }
}
}  // namespace oneDNN
