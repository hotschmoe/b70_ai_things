// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared per-token dynamic symmetric int8 activation-quant SYCL kernel.
//
// Used by BOTH:
//   - the standalone `dynamic_per_token_int8_quant` op
//     (csrc/xpu/sycl/dynamic_per_token_int8_quant.cpp), and
//   - the FUSED `int8_gemm_w8a8_fusedq` op (csrc/xpu/onednn/onednn_matmul.cpp),
//     which quantizes an f16/bf16 activation inline and then runs the oneDNN
//     s8s8 matmul on the SAME in-order XPU stream -- one opaque graph node, no
//     inductor-fusion loss (docs/kernel/23_b70_gemv_gemm_roofline.md, B1).
//
// For an input activation row x[m, :] (f16 / bf16 / f32) this computes
//     absmax   = max_k |x[m, k]|
//     scale[m] = max(absmax / 127, 1e-5)
//     q[m, k]  = clamp(round(x[m, k] / scale[m]), -127, 127)   (int8)
// and returns q [M, K] int8 + scale [M, 1] (input dtype). zero_point is 0.
//
// PARALLELISM FIX (B1): the reduction was always a group reduce
// (sycl::reduce_over_group over the whole work-group), but the launch used a
// 32-wide work-group (one sub-group per row). At M=1 decode a single 32-lane
// work-group cannot hide global-memory latency -> ~101us on K=17408 (the
// capture-persistent hotspot). Here the launcher sizes the work-group to
// multiple sub-groups (up to 512 lanes/row) so far more loads are in flight per
// row; the group reduce still yields an identical absmax to every lane. Purely a
// launch-geometry change -- the numerics are bit-identical to the 32-lane path.

#pragma once

#include <sycl/sycl.hpp>
#include <torch/torch.h>

#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t>
class dynamic_per_token_int8_quant_kernel {
 public:
  dynamic_per_token_int8_quant_kernel(
      const scalar_t* __restrict__ input_,
      int8_t* __restrict__ q_out_,
      scalar_t* __restrict__ scale_out_,
      const int64_t num_rows_,
      const int64_t k_,
      const int64_t row_stride_)
      : input(input_),
        q_out(q_out_),
        scale_out(scale_out_),
        num_rows(num_rows_),
        k(k_),
        row_stride(row_stride_) {}

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (const sycl::nd_item<1>& item) const {
    const int64_t row = item.get_group(0);
    if (row >= num_rows) return;

    const int lid = item.get_local_id(0);
    const int lrange = item.get_local_range(0);
    auto group = item.get_group();

    const scalar_t* __restrict__ row_in = input + row * row_stride;

    // Pass 1: thread-local absmax in fp32, then a WORK-GROUP reduction. The
    // work-group may span several sub-groups (see the launcher); the SYCL group
    // algorithm reduces across all of them (SLM + barrier) and broadcasts the
    // result back to every lane -- so all lanes derive the same scale below.
    float thread_absmax = 0.0f;
    for (int64_t i = lid; i < k; i += lrange) {
      float v = vllm::xpu::to_float(row_in[i]);
      thread_absmax = sycl::fmax(thread_absmax, sycl::fabs(v));
    }
    float absmax =
        sycl::reduce_over_group(group, thread_absmax, sycl::maximum<float>());

    // scale = max(absmax / 127, 1e-5), matching the reference clamp(min=1e-5).
    float scale = absmax / 127.0f;
    scale = sycl::fmax(scale, 1e-5f);
    const float inv_scale = 1.0f / scale;

    if (lid == 0) {
      // Store scale in the input dtype (reference casts scale to input dtype).
      scalar_t s_out;
      vllm::xpu::from_float(s_out, scale);
      scale_out[row] = s_out;
    }

    // Pass 2: quantize. Round-to-nearest-even via sycl::rint, then clamp to
    // [-127, 127] (symmetric int8; -128 is intentionally excluded to match the
    // reference qmax=127 symmetric scale and keep dequant symmetric).
    int8_t* __restrict__ row_out = q_out + row * k;
    for (int64_t i = lid; i < k; i += lrange) {
      float v = vllm::xpu::to_float(row_in[i]);
      float q = sycl::rint(v * inv_scale);
      q = sycl::fmin(sycl::fmax(q, -127.0f), 127.0f);
      row_out[i] = static_cast<int8_t>(q);
    }
  }

 private:
  const scalar_t* __restrict__ input;
  int8_t* __restrict__ q_out;
  scalar_t* __restrict__ scale_out;
  const int64_t num_rows;
  const int64_t k;
  const int64_t row_stride;
};

// Pick a work-group size (multiple of the 32-lane sub-group) that gives enough
// in-flight loads per row to hide memory latency, without over-allocating for
// small K. Target ~32 elements/lane, capped at 512 lanes (16 sub-groups).
static inline int64_t choose_int8_quant_local(int64_t k) {
  constexpr int64_t kSub = 32;
  constexpr int64_t kMaxLocal = 512;
  int64_t want = ((k / kSub) + (kSub - 1)) / kSub * kSub;  // round(k/32) up to *32
  if (want < kSub) want = kSub;
  if (want > kMaxLocal) want = kMaxLocal;
  return want;
}

// Launch the per-token int8 quant over a contiguous [M, K] activation `x2d`,
// writing q [M, K] int8 and scale [M, 1] (x2d dtype). Submitted on `queue`
// (the current in-order XPU stream). Caller owns the output tensors + any
// reshape back to original leading dims.
static inline void launch_dynamic_per_token_int8_quant(
    sycl::queue& queue,
    const torch::Tensor& x2d,
    torch::Tensor& q,
    torch::Tensor& scale) {
  const int64_t m = x2d.size(0);
  const int64_t k = x2d.size(1);
  if (m == 0 || k == 0) return;
  const int64_t row_stride = x2d.stride(0);
  const int64_t local = choose_int8_quant_local(k);

  sycl::range<1> global(static_cast<size_t>(m) * static_cast<size_t>(local));
  sycl::range<1> local_r(static_cast<size_t>(local));

  VLLM_DISPATCH_FLOATING_TYPES(
      x2d.scalar_type(), "dynamic_per_token_int8_quant", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        auto in_ptr = reinterpret_cast<const sycl_t*>(x2d.data_ptr<scalar_t>());
        auto q_ptr = q.data_ptr<int8_t>();
        auto sc_ptr = reinterpret_cast<sycl_t*>(scale.data_ptr<scalar_t>());

        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(global, local_r),
              vllm::dynamic_per_token_int8_quant_kernel<sycl_t>(
                  in_ptr, q_ptr, sc_ptr, m, k, row_stride));
        });
      });
}

}  // namespace vllm
