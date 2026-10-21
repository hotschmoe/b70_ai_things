// Isolated exact key pack and threshold compaction. No native QSA edits.
#include "key.h"
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/library.h>
#include <torch/torch.h>
namespace b70_q1_packcompact_r1 {
class PackKernel;
class CompactKernel;
void record_all(std::initializer_list<torch::Tensor> tensors) {
  for (const auto& t : tensors) if (t.numel())
    c10::xpu::XPUCachingAllocator::recordStream(t.storage().data_ptr(), c10::xpu::getCurrentXPUStream());
}
void check(const torch::Tensor& scores, const torch::Tensor& step) {
  TORCH_CHECK(scores.is_xpu() && scores.is_contiguous() && scores.scalar_type()==torch::kFloat && scores.dim()==2, "scores");
  TORCH_CHECK(scores.size(1)>0 && scores.size(1)<2147483647ll, "score width");
  TORCH_CHECK(step.device()==scores.device() && step.is_contiguous() && step.scalar_type()==torch::kInt32 && step.dim()==2 && step.size(0)==scores.size(0) && step.size(1)==4, "step");
}
torch::Tensor pack(const torch::Tensor& scores, const torch::Tensor& keys, const torch::Tensor& step) {
  check(scores,step);
  TORCH_CHECK(keys.device()==scores.device() && keys.is_contiguous() && keys.scalar_type()==torch::kInt64 && keys.sizes()==scores.sizes(), "keys");
  const at::DeviceGuard guard(scores.device());
  record_all({scores,keys,step});
  const int cols=static_cast<int>(scores.size(1));
  const int64_t count=scores.numel();
  const float* sp=scores.data_ptr<float>();
  int64_t* kp=keys.data_ptr<int64_t>();
  const int32_t* st=step.data_ptr<int32_t>();
  auto& queue=c10::xpu::getCurrentXPUStream().queue();
  auto nd=sycl::nd_range<1>(static_cast<size_t>((count+255)/256*256),256);
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<PackKernel>(nd,[=](sycl::nd_item<1> it) {
      const int64_t offset=it.get_global_id(0);
      if(offset>=count)return;
      const int qi=offset/cols;
      const int i=offset-static_cast<int64_t>(qi)*cols;
      const int n=st[static_cast<int64_t>(qi)*4+1];
      kp[offset]=i<n ? mapped_key(sp[offset],static_cast<uint32_t>(i)) : kInvalid;
    });
  });
  return keys;
}
torch::Tensor compact(const torch::Tensor& scores, const torch::Tensor& selected_values,
                      const torch::Tensor& ids, const torch::Tensor& step) {
  check(scores,step);
  const int batch=static_cast<int>(scores.size(0));
  TORCH_CHECK(ids.device()==scores.device() && ids.is_contiguous() && ids.scalar_type()==torch::kInt32 && ids.dim()==2 && ids.size(0)==batch,"ids");
  TORCH_CHECK(selected_values.device()==scores.device() && selected_values.is_contiguous() && selected_values.scalar_type()==torch::kInt64 && selected_values.dim()==2 && selected_values.size(0)==batch,"selected values");
  const at::DeviceGuard guard(scores.device());
  record_all({scores,selected_values,ids,step});
  const int cols=static_cast<int>(scores.size(1));
  const int cap=static_cast<int>(ids.size(1));
  const int selected_k=static_cast<int>(selected_values.size(1));
  const float* sp=scores.data_ptr<float>();
  const int64_t* vp=selected_values.data_ptr<int64_t>();
  int32_t* ip=ids.data_ptr<int32_t>();
  const int32_t* st=step.data_ptr<int32_t>();
  auto& queue=c10::xpu::getCurrentXPUStream().queue();
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<CompactKernel>(sycl::range<1>(batch),[=](sycl::id<1> item) {
      const int qi=item[0];
      int n=st[static_cast<int64_t>(qi)*4+1];
      int width=st[static_cast<int64_t>(qi)*4+3];
      n=sycl::min(sycl::max(n,0),cols);
      width=sycl::min(sycl::min(sycl::min(width,n),cap),selected_k);
      if(width<=0 || n<=0)return;
      const int64_t threshold=vp[static_cast<int64_t>(qi)*selected_k+width-1];
      const float* row=sp+static_cast<int64_t>(qi)*cols;
      int32_t* out=ip+static_cast<int64_t>(qi)*cap;
      int w=0;
      for(int i=0;i<n;++i) {
        if(mapped_key(row[i],static_cast<uint32_t>(i))>=threshold && w<width)
          out[w++]=i;
      }
    });
  });
  return ids;
}
}
TORCH_LIBRARY(b70_q1_packcompact_r1,m) {
  m.def("pack(Tensor scores, Tensor keys, Tensor step) -> Tensor");
  m.def("compact(Tensor scores, Tensor selected_values, Tensor ids, Tensor step) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_q1_packcompact_r1,XPU,m) {
  m.impl("pack",&b70_q1_packcompact_r1::pack);
  m.impl("compact",&b70_q1_packcompact_r1::compact);
}
