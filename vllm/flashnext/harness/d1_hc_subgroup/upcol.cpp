// Scratch-only D1 C1 HC up+collapse candidate. No serving integration.
#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <torch/library.h>
namespace b70_d1_hc_sg_r1 {
inline float bf16f(uint16_t v) { return sycl::bit_cast<float>((uint32_t)v << 16); }
class UpCollapseM1;
// Original LPR8 chunk/FMA/XOR order for each row is retained. One subgroup
// owns h and streams0/1 in acc0, streams2/3 in acc2. No cross-SG SLM/barrier.
void launch(sycl::queue& q, const uint16_t* x, const int8_t* w,
            const sycl::half* s, const uint16_t* flat, uint16_t* out,
            int H, int K) {
    const int chunks = K / 16;
    const int groups = (H + 3) / 4;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<UpCollapseM1>(sycl::nd_range<1>((size_t)groups * 64, 64),
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            const int lane = (int)sg.get_local_linear_id();
            const int hh = (int)it.get_group(0) * 4 + ((int)it.get_local_id(0) >> 4);
            const int sub = lane % 8;
            const int st0 = lane / 8;
            float acc0 = 0.f, acc2 = 0.f;
            if (hh < H) {
                for (int c = sub; c < chunks; c += 8) {
                    const uint16_t* xr = x + (size_t)c * 16;
                    const sycl::uint4 a = *reinterpret_cast<const sycl::uint4*>(xr);
                    const sycl::uint4 b = *reinterpret_cast<const sycl::uint4*>(xr + 8);
                    const uint32_t xs[8] = {a.x(),a.y(),a.z(),a.w(),b.x(),b.y(),b.z(),b.w()};
                    // Separate acc rows, same inner16 FMA order and scale update.
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int row = (st0 + 2 * half) * H + hh;
                        const int8_t* wr = w + (size_t)row * K;
                        const sycl::int4 wv = *reinterpret_cast<const sycl::int4*>(wr + (size_t)c * 16);
                        const float sc = (float)s[(size_t)row * (K / 32) + (c >> 1)];
                        float wf[16];
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int word = j==0 ? wv.x() : j==1 ? wv.y() : j==2 ? wv.z() : wv.w();
#pragma unroll
                            for (int b = 0; b < 4; ++b)
                                wf[j*4+b] = (float)(int8_t)((word >> (8*b)) & 0xff);
                        }
                        float d = 0.f;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            d = sycl::fma(wf[2*j], bf16f((uint16_t)(xs[j] & 0xffff)), d);
                            d = sycl::fma(wf[2*j+1], bf16f((uint16_t)(xs[j] >> 16)), d);
                        }
                        if (half==0) acc0 = sycl::fma(d,sc,acc0);
                        else acc2 = sycl::fma(d,sc,acc2);
                    }
                }
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                acc0 += sycl::permute_group_by_xor(sg,acc0,o);
                acc2 += sycl::permute_group_by_xor(sg,acc2,o);
            }
            // Broadcast the SAME sub==0 reduction lanes used by launch_short<8>.
            const float b0 = (float)sycl::ext::oneapi::bfloat16(acc0);
            const float b2 = (float)sycl::ext::oneapi::bfloat16(acc2);
            const float gates[4] = {sycl::select_from_group(sg,b0,0),
                                    sycl::select_from_group(sg,b0,8),
                                    sycl::select_from_group(sg,b2,0),
                                    sycl::select_from_group(sg,b2,8)};
            if (lane==0 && hh<H) {
                float sum = 0.f;
                for (int st = 0; st < 4; ++st) {
                    const float g = gates[st];
                    const float gs = (float)sycl::ext::oneapi::bfloat16(1.f/(1.f+sycl::exp(-g)));
                    const float xv = bf16f(flat[(size_t)st*H+hh]);
                    sum += (float)sycl::ext::oneapi::bfloat16(xv*gs);
                }
                out[hh] = sycl::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(sum/4));
            }
        });
    });
}
at::Tensor up_collapse_m1(const at::Tensor& act, const at::Tensor& w,
                         const at::Tensor& s, const at::Tensor& flat) {
    TORCH_CHECK(act.is_xpu() && act.scalar_type()==at::kBFloat16 &&
                act.is_contiguous() && act.dim()==2 && act.size(0)==1 && act.size(1)==320,"M1 K320 only");
    const auto dev=act.device();
    TORCH_CHECK(w.device()==dev && w.scalar_type()==at::kChar && w.is_contiguous() &&
                w.dim()==2 && w.size(0)==10240 && w.size(1)==320,"w10240x320");
    TORCH_CHECK(s.device()==dev && s.scalar_type()==at::kHalf && s.is_contiguous() &&
                s.dim()==2 && s.size(0)==10240 && s.size(1)==10,"s10240x10");
    TORCH_CHECK(flat.device()==dev && flat.scalar_type()==at::kBFloat16 && flat.is_contiguous() &&
                flat.dim()==2 && flat.size(0)==1 && flat.size(1)==10240,"flat1x10240");
    const at::DeviceGuard guard(dev);
    const auto stream=c10::xpu::getCurrentXPUStream();
    for (const auto& t : {act,w,s,flat})
        c10::xpu::XPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
    auto out=at::empty({1,2560},act.options());
    launch(stream.queue(),static_cast<const uint16_t*>(act.data_ptr()),w.data_ptr<int8_t>(),
           reinterpret_cast<const sycl::half*>(s.data_ptr()),
           static_cast<const uint16_t*>(flat.data_ptr()),static_cast<uint16_t*>(out.data_ptr()),2560,320);
    return out;
}
}
TORCH_LIBRARY(b70_d1_hc_sg_r1,m) {
    m.def("up_collapse_m1(Tensor act, Tensor w, Tensor s, Tensor flat) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_d1_hc_sg_r1,XPU,m) {
    m.impl("up_collapse_m1",&b70_d1_hc_sg_r1::up_collapse_m1);
}
