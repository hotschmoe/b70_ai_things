// N1: batch-invariant GDN b|a projection. Derived from O4 bf_gemv.
#include <sycl/sycl.hpp>
#include <vector>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

namespace b70_n1_impl {

constexpr int kMaxM = 8;

inline float bf16f(uint16_t v) { return sycl::bit_cast<float>((uint32_t)v << 16); }

template <int KS>
class n1_bf_gemm_k;

// bf16 weights, same schedule: 8 weights (16 bytes) per lane step.
template <int KS>
void launch_bf(sycl::queue& q, const uint16_t* x, const uint16_t* w, uint16_t* y, int M, int N, int K) {
    constexpr int NSG = KS > 4 ? KS : 4;
    constexpr int R = NSG / KS;
    const int chunks = K / 8;
    const int per = (chunks + KS - 1) / KS;
    const int row_groups = (N + R - 1) / R;
    const int ngroups = row_groups * ((M + kMaxM - 1) / kMaxM);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(NSG * kMaxM, h);
        h.parallel_for<n1_bf_gemm_k<KS>>(
            sycl::nd_range<1>((size_t)ngroups * NSG * 16, NSG * 16),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_linear_id();
                const int sgid = (int)it.get_local_id(0) >> 4;
                const int tile = (int)it.get_group(0) / row_groups;
                const int m0 = tile * kMaxM;
                const int rows = sycl::min(kMaxM, M - m0);
                const int row = ((int)it.get_group(0) % row_groups) * R + sgid / KS;
                const int part = sgid % KS;
                float acc[kMaxM];
#pragma unroll
                for (int m = 0; m < kMaxM; ++m) acc[m] = 0.f;
                if (row < N) {
                    const int c0 = part * per;
                    const int c1 = sycl::min(c0 + per, chunks);
                    const uint16_t* wr = w + (size_t)row * K;
                    for (int c = c0 + lane; c < c1; c += 16) {
                        const sycl::uint4 wv = *reinterpret_cast<const sycl::uint4*>(wr + (size_t)c * 8);
                        const uint32_t ws[4] = {wv.x(), wv.y(), wv.z(), wv.w()};
#pragma unroll
                        for (int m = 0; m < kMaxM; ++m) {
                            if (m < rows) {
                                const sycl::uint4 a = *reinterpret_cast<const sycl::uint4*>(x + (size_t)(m0 + m) * K + (size_t)c * 8);
                                const uint32_t xs[4] = {a.x(), a.y(), a.z(), a.w()};
                                float d = acc[m];
#pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    d = sycl::fma(bf16f((uint16_t)(ws[j] & 0xffff)), bf16f((uint16_t)(xs[j] & 0xffff)), d);
                                    d = sycl::fma(bf16f((uint16_t)(ws[j] >> 16)), bf16f((uint16_t)(xs[j] >> 16)), d);
                                }
                                acc[m] = d;
                            }
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < kMaxM; ++m) {
                    if (m < rows) acc[m] = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
                }
                if (lane == 0) {
                    for (int m = 0; m < rows; ++m) slm[sgid * kMaxM + m] = acc[m];
                }
                sycl::group_barrier(it.get_group());
                if (part == 0 && lane < rows && row < N) {
                    float t = 0.f;
                    for (int p = 0; p < KS; ++p) t += slm[(sgid + p) * kMaxM + lane];
                    const sycl::ext::oneapi::bfloat16 v(t);
                    y[(size_t)(m0 + lane) * N + row] = sycl::bit_cast<uint16_t>(v);
                }
            });
    });
}

// Same eight-row lane arithmetic and KS selection as O4 bf_gemv, with
// independent token tiles added to the work-group grid. No cross-tile sum.
at::Tensor bf_gemm(at::Tensor x, at::Tensor w) {
    TORCH_CHECK(x.is_xpu() && w.is_xpu() && x.device() == w.device(), "same-device XPU tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x bf16 [M, K]");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.dim() == 2 && w.is_contiguous(), "w bf16 [N, K]");
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    TORCH_CHECK(w.size(1) == K && K % 8 == 0 && K > 0 && N > 0, "shapes");
    auto y = at::empty({M, N}, x.options());
    if (M == 0) return y;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    const auto* xp = static_cast<const uint16_t*>(x.data_ptr());
    const auto* wp = static_cast<const uint16_t*>(w.data_ptr());
    auto* yp = static_cast<uint16_t*>(y.data_ptr());
    // Match gguf/linear.py _dense for the concatenated GDN b|a weight.
    if (N >= 256) launch_bf<2>(q, xp, wp, yp, M, N, K);
    else launch_bf<8>(q, xp, wp, yp, M, N, K);
    return y;
}
}  // namespace b70_n1_impl

TORCH_LIBRARY(b70_n1, m) { m.def("bf_gemm(Tensor x, Tensor w) -> Tensor"); }
TORCH_LIBRARY_IMPL(b70_n1, XPU, m) { m.impl("bf_gemm", b70_n1_impl::bf_gemm); }
