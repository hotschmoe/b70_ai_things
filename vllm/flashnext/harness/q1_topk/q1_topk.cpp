// Q1 isolated top-k experiment. Original algorithm, one reduction change.
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/library.h>
#include <torch/torch.h>
#include <bit>
#include <cstdint>
namespace b70_q1_topk_r1 {
constexpr int kTopkThreads = 256;
template <bool UseGroup> class TopkKernel;
void record_all(std::initializer_list<torch::Tensor> ts) {
  for (const auto& t : ts) if (t.defined() && t.numel() && t.is_xpu())
    c10::xpu::XPUCachingAllocator::recordStream(t.storage().data_ptr(), c10::xpu::getCurrentXPUStream());
}
void check_step(const torch::Tensor& t, int n) {
 TORCH_CHECK(t.is_xpu() && t.is_contiguous() && t.scalar_type() == torch::kInt32 && t.dim() == 2 && t.size(0) == n && t.size(1) == 4, "step");
}
uint32_t order_f32(float score) {
 uint32_t bits = std::bit_cast<uint32_t>(score);
 uint32_t mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
 return bits ^ mask;
}
uint64_t pack_key(float score, uint32_t index) {
 uint32_t inv = 0xFFFFFFFFu ^ index;
 return (static_cast<uint64_t>(order_f32(score)) << 32) | static_cast<uint64_t>(inv);
}
template <bool UseGroup>
torch::Tensor topk_impl(const torch::Tensor& scores, const torch::Tensor& ids, const torch::Tensor& step) {
  TORCH_CHECK(scores.is_xpu() && scores.is_contiguous() && scores.scalar_type() == torch::kFloat && scores.dim() == 2, "scores");
  TORCH_CHECK(ids.is_xpu() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32 && ids.dim() == 2, "ids");
  const int n_q = static_cast<int>(scores.size(0));
  TORCH_CHECK(ids.size(0) == n_q, "ids batch");
  check_step(step, n_q);
  const int scores_n = static_cast<int>(scores.size(1));
  const int ids_n = static_cast<int>(ids.size(1));
  const at::DeviceGuard guard(scores.device());
  record_all({scores, ids, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const float* scores_p = scores.data_ptr<float>();
  int32_t* ids_p = ids.data_ptr<int32_t>();
  const int32_t* step_p = step.data_ptr<int32_t>();
  auto nd = sycl::nd_range<1>(
      static_cast<size_t>(n_q) * kTopkThreads, static_cast<size_t>(kTopkThreads));
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<long long, 1> red(sycl::range<1>(kTopkThreads), h);
    h.parallel_for<TopkKernel<UseGroup>>(nd, [=](sycl::nd_item<1> it) {
      const int qi = static_cast<int>(it.get_group(0));
      const int t = static_cast<int>(it.get_local_id(0));
      const int32_t* st = step_p + static_cast<long long>(qi) * 4;
      int n = st[1];
      int width = st[3];
      if (n > scores_n) {
        n = scores_n;
      }
      if (n < 0) {
        n = 0;
      }
      if (width > n) {
        width = n;
      }
      if (width > ids_n) {
        width = ids_n;
      }
      if (width <= 0 || n <= 0) {
        return;
      }
      const float* row = scores_p + static_cast<long long>(qi) * scores_n;
      int32_t* out = ids_p + static_cast<long long>(qi) * ids_n;
      uint64_t prefix = 0;
      uint64_t decided = 0;
      long long need = width;
      for (int bit = 63; bit >= 0; --bit) {
        const uint64_t bitv = 1ull << bit;
        long long local = 0;
        for (int i = t; i < n; i += kTopkThreads) {
          const uint64_t key = pack_key(row[i], static_cast<uint32_t>(i));
          if ((key & decided) == prefix && (key & bitv) != 0) {
            local = local + 1;
          }
        }
        long long cnt;
        if constexpr (UseGroup) {
          cnt = sycl::reduce_over_group(it.get_group(), local, sycl::plus<long long>());
        } else {
          red[t] = local;
          it.barrier(sycl::access::fence_space::local_space);
          for (int s = kTopkThreads / 2; s > 0; s >>= 1) {
            if (t < s) red[t] = red[t] + red[t + s];
            it.barrier(sycl::access::fence_space::local_space);
          }
          cnt = red[0];
          it.barrier(sycl::access::fence_space::local_space);
        }
        decided = decided | bitv;
        if (cnt >= need) {
          prefix = prefix | bitv;
        } else {
          need = need - cnt;
        }
      }
      if (t == 0) {
        int w = 0;
        for (int i = 0; i < n; ++i) {
          const uint64_t key = pack_key(row[i], static_cast<uint32_t>(i));
          if (key >= prefix && w < width) {
            out[w] = i;
            w = w + 1;
          }
        }
      }
    });
  });
  return ids;
}


} // namespace b70_q1_topk_r1
TORCH_LIBRARY(b70_q1_topk_r1, m) {
 m.def("tree(Tensor scores, Tensor ids, Tensor step) -> Tensor");
 m.def("group(Tensor scores, Tensor ids, Tensor step) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_q1_topk_r1, XPU, m) {
 m.impl("tree", &b70_q1_topk_r1::topk_impl<false>);
 m.impl("group", &b70_q1_topk_r1::topk_impl<true>);
}
