// SPDX-License-Identifier: Apache-2.0
//
// Fused per-row top-1 reduction for a vocabulary-parallel logits shard.
//
// The vLLM MTP local-argmax path needs both the maximum value and its local
// index. On XPU, composing aten::argmax with an indexed gather is correct but
// slower than the full-vocabulary collective it is intended to replace. This
// kernel reads the shard once, performs one work-group reduction per row, and
// returns the value in fp32 plus the index in int64.

#include <limits>

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "dispatch_utils.h"
#include "utils.h"

namespace vllm {

template <typename scalar_t>
class shard_top1_kernel {
 public:
  shard_top1_kernel(
      const scalar_t* __restrict__ logits_,
      float* __restrict__ values_,
      int64_t* __restrict__ indices_,
      int64_t rows_,
      int64_t width_)
      : logits(logits_),
        values(values_),
        indices(indices_),
        rows(rows_),
        width(width_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (const sycl::nd_item<1>& item) const {
    const int64_t row = item.get_group(0);
    if (row >= rows) {
      return;
    }

    const int64_t lane = item.get_local_id(0);
    const int64_t lanes = item.get_local_range(0);
    const scalar_t* __restrict__ row_logits = logits + row * width;

    float lane_value = -std::numeric_limits<float>::infinity();
    int64_t lane_index = width;
    for (int64_t col = lane; col < width; col += lanes) {
      const float value = vllm::xpu::to_float(row_logits[col]);
      if (value > lane_value || (value == lane_value && col < lane_index)) {
        lane_value = value;
        lane_index = col;
      }
    }

    const auto group = item.get_group();
    const float row_value =
        sycl::reduce_over_group(group, lane_value, sycl::maximum<float>());
    const int64_t row_index = sycl::reduce_over_group(
        group,
        lane_value == row_value ? lane_index : width,
        sycl::minimum<int64_t>());

    if (lane == 0) {
      values[row] = row_value;
      indices[row] = row_index;
    }
  }

 private:
  const scalar_t* __restrict__ logits;
  float* __restrict__ values;
  int64_t* __restrict__ indices;
  const int64_t rows;
  const int64_t width;
};

}  // namespace vllm

std::tuple<torch::Tensor, torch::Tensor> xpu_shard_top1(
    const torch::Tensor& logits) {
  TORCH_CHECK(logits.is_xpu(), "xpu_shard_top1: logits must be on XPU");
  TORCH_CHECK(logits.dim() >= 1, "xpu_shard_top1: logits must have rank >= 1");
  TORCH_CHECK(
      logits.scalar_type() == at::ScalarType::Float ||
          logits.scalar_type() == at::ScalarType::Half ||
          logits.scalar_type() == at::ScalarType::BFloat16,
      "xpu_shard_top1: logits must be fp32, fp16, or bf16");

  const int64_t width = logits.size(-1);
  TORCH_CHECK(width > 0, "xpu_shard_top1: last dimension must be non-empty");

  const at::DeviceGuard device_guard(logits.device());
  const torch::Tensor input = logits.contiguous();
  const int64_t rows = input.numel() / width;
  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end() - 1);

  torch::Tensor values =
      at::empty({rows}, input.options().dtype(at::ScalarType::Float));
  torch::Tensor indices =
      at::empty({rows}, input.options().dtype(at::ScalarType::Long));

  constexpr int64_t kWorkGroupSize = 256;
  const sycl::range<1> local(kWorkGroupSize);
  const sycl::range<1> global(rows * kWorkGroupSize);
  auto& queue = vllm::xpu::vllmGetQueue(input.device().index());

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "xpu_shard_top1", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        const auto* input_ptr =
            reinterpret_cast<const sycl_t*>(input.data_ptr<scalar_t>());
        auto* values_ptr = values.data_ptr<float>();
        auto* indices_ptr = indices.data_ptr<int64_t>();

        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(global, local),
              vllm::shard_top1_kernel<sycl_t>(
                  input_ptr, values_ptr, indices_ptr, rows, width));
        });
      });

  return std::make_tuple(
      values.reshape(output_shape), indices.reshape(output_shape));
}
