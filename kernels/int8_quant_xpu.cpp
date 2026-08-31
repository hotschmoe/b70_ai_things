// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Dynamic symmetric per-token INT8 activation quantization for XPU.
// Scales are float32 to match compressed-tensors W8A8 and oneDNN.

#include <cstdlib>
#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "utils.h"

namespace {

template <typename scalar_t>
float to_float(scalar_t value) {
  return static_cast<float>(value);
}

int get_int8_quant_max_local() {
  static const int max_local = []() {
    const char* env = std::getenv("B70_XPU_INT8_QUANT_MAX_LOCAL");
    if (env == nullptr || env[0] == '\0') {
      return 256;
    }
    const int parsed = std::atoi(env);
    switch (parsed) {
      case 32:
      case 64:
      case 128:
      case 256:
      case 512:
        return parsed;
      default:
        TORCH_WARN_ONCE(
            "B70_XPU_INT8_QUANT_MAX_LOCAL must be 32, 64, 128, 256, or "
            "512; using 256");
        return 256;
    }
  }();
  return max_local;
}

int choose_int8_quant_local(int64_t cols) {
  int local = 32;
  const int max_local = get_int8_quant_max_local();
  while (local < max_local && static_cast<int64_t>(local) * 32 < cols) {
    local *= 2;
  }
  return local;
}

template <typename scalar_t>
void launch_per_token_quant_int8(
    const scalar_t* x,
    int8_t* q,
    float* scales,
    int64_t rows,
    int64_t cols) {
  const int block_size = choose_int8_quant_local(cols);
  auto& queue = vllm::xpu::vllmGetQueue();
  sycl::range<1> local(block_size);
  sycl::range<1> global(rows * block_size);

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_max(local, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const int64_t row = item.get_group(0);
          const int local_id = item.get_local_id(0);
          const int local_range = item.get_local_range(0);

          float thread_max = 0.0f;
          const int64_t row_offset = row * cols;
          for (int64_t col = local_id; col < cols; col += local_range) {
            const float value = to_float(x[row_offset + col]);
            thread_max = sycl::fmax(thread_max, sycl::fabs(value));
          }

          local_max[local_id] = thread_max;
          item.barrier(sycl::access::fence_space::local_space);

          for (int stride = local_range / 2; stride > 0; stride >>= 1) {
            if (local_id < stride) {
              local_max[local_id] =
                  sycl::fmax(local_max[local_id], local_max[local_id + stride]);
            }
            item.barrier(sycl::access::fence_space::local_space);
          }

          const float absmax = sycl::fmax(local_max[0], 1.0e-10f);
          const float scale = absmax / 127.0f;
          if (local_id == 0) {
            scales[row] = scale;
          }

          const float inv_scale = 127.0f / absmax;
          for (int64_t col = local_id; col < cols; col += local_range) {
            float value = to_float(x[row_offset + col]) * inv_scale;
            // Match torch.round and compressed-tensors: nearest integer with
            // halfway cases rounded to even. sycl::round instead rounds ties
            // away from zero and changes exact activation bytes.
            value = sycl::rint(value);
            value = sycl::fmin(127.0f, sycl::fmax(-127.0f, value));
            q[row_offset + col] = static_cast<int8_t>(value);
          }
        });
  });
}

std::vector<int64_t> make_scale_shape(const torch::Tensor& x) {
  std::vector<int64_t> scale_shape;
  scale_shape.reserve(x.dim());
  for (int64_t i = 0; i < x.dim() - 1; ++i) {
    scale_shape.push_back(x.size(i));
  }
  scale_shape.push_back(1);
  return scale_shape;
}

bool shape_matches(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& expected) {
  if (tensor.dim() != static_cast<int64_t>(expected.size())) {
    return false;
  }
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (tensor.size(i) != expected[i]) {
      return false;
    }
  }
  return true;
}

void check_quant_input(const torch::Tensor& x, const char* op_name) {
  TORCH_CHECK(x.is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
          x.scalar_type() == torch::kBFloat16,
      op_name,
      " only supports fp32, fp16, and bf16 inputs, got ",
      x.scalar_type());
}

void check_out_tensor(
    const torch::Tensor& x,
    const torch::Tensor& tensor,
    const char* tensor_name,
    c10::ScalarType expected_dtype,
    const std::vector<int64_t>& expected_shape,
    const char* op_name) {
  TORCH_CHECK(tensor.is_xpu(), tensor_name, " must be an XPU tensor");
  TORCH_CHECK(
      tensor.device() == x.device(),
      tensor_name,
      " must be on the same XPU device as x");
  TORCH_CHECK(
      tensor.scalar_type() == expected_dtype,
      tensor_name,
      " has wrong dtype for ",
      op_name);
  TORCH_CHECK(
      tensor.is_contiguous(), tensor_name, " must be contiguous for ", op_name);
  TORCH_CHECK(
      shape_matches(tensor, expected_shape),
      tensor_name,
      " has wrong shape for ",
      op_name);
}

void check_quant_out_tensors(
    const torch::Tensor& x,
    const torch::Tensor& q,
    const torch::Tensor& scales,
    const char* op_name) {
  TORCH_CHECK(
      x.is_contiguous(),
      op_name,
      " requires contiguous x to avoid hidden out-variant allocation");
  check_out_tensor(x, q, "q", torch::kInt8, x.sizes().vec(), op_name);
  check_out_tensor(
      x,
      scales,
      "scales",
      torch::kFloat32,
      make_scale_shape(x),
      op_name);
}

void launch_checked(
    const torch::Tensor& x,
    torch::Tensor& q,
    torch::Tensor& scales,
    int64_t rows,
    int64_t cols) {
  if (rows == 0 || cols == 0) {
    return;
  }
  if (x.scalar_type() == torch::kFloat32) {
    launch_per_token_quant_int8(
        x.data_ptr<float>(), q.data_ptr<int8_t>(), scales.data_ptr<float>(), rows, cols);
  } else if (x.scalar_type() == torch::kFloat16) {
    launch_per_token_quant_int8(
        reinterpret_cast<const sycl::half*>(x.data_ptr()),
        q.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        rows,
        cols);
  } else {
    launch_per_token_quant_int8(
        reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(x.data_ptr()),
        q.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        rows,
        cols);
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> per_token_quant_int8_xpu(
    const torch::Tensor& x) {
  check_quant_input(x, "per_token_quant_int8_xpu");
  auto x_contig = x.contiguous();
  const int64_t cols = x_contig.size(-1);
  const int64_t rows = x_contig.numel() / cols;
  auto q = torch::empty_like(x_contig, x_contig.options().dtype(torch::kInt8));
  auto scales = torch::empty(
      make_scale_shape(x_contig), x_contig.options().dtype(torch::kFloat32));
  launch_checked(x_contig, q, scales, rows, cols);
  return {q, scales};
}

std::tuple<torch::Tensor, torch::Tensor> per_token_quant_int8_xpu_out(
    const torch::Tensor& x,
    torch::Tensor& q,
    torch::Tensor& scales) {
  check_quant_input(x, "per_token_quant_int8_xpu_out");
  check_quant_out_tensors(x, q, scales, "per_token_quant_int8_xpu_out");
  const int64_t cols = x.size(-1);
  const int64_t rows = x.numel() / cols;
  launch_checked(x, q, scales, rows, cols);
  return {q, scales};
}
