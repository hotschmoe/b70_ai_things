// O4: int8 weight x bf16 activation GEMV for small M (decode).
// w: int8 [N, K] row-major, s: fp16 [N, K/32] (one scale per 32 weights,
// Q8_0 granularity), x: bf16 [M, K], y: bf16 [M, N] (fp32 accumulate,
// round-to-nearest-even). Used for the Flash-Next hyper-connection
// down/up projections (B70_DECODE_HC_Q8=1).
//
// A work-group of 4 sub-groups (64 lanes) owns R rows. KS sub-groups
// split K for one row (KS * R == 4). Each lane reads 16 int8 weights per
// step, so a sub-group reads 256 consecutive bytes.
#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

namespace {

constexpr int kMaxM = 8;

inline float bf16f(uint16_t v) { return sycl::bit_cast<float>((uint32_t)v << 16); }

template <int KS>
class w8_gemv_k;
template <int KS>
class bf_gemv_k;

// bf16 weights, same schedule: 8 weights (16 bytes) per lane step.
template <int KS>
void launch_bf(sycl::queue& q, const uint16_t* x, const uint16_t* w, uint16_t* y, int M, int N, int K) {
    constexpr int NSG = KS > 4 ? KS : 4;
    constexpr int R = NSG / KS;
    const int chunks = K / 8;
    const int per = (chunks + KS - 1) / KS;
    const int ngroups = (N + R - 1) / R;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(NSG * kMaxM, h);
        h.parallel_for<bf_gemv_k<KS>>(
            sycl::nd_range<1>((size_t)ngroups * NSG * 16, NSG * 16),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_linear_id();
                const int sgid = (int)it.get_local_id(0) >> 4;
                const int row = (int)it.get_group(0) * R + sgid / KS;
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
                            if (m < M) {
                                const sycl::uint4 a = *reinterpret_cast<const sycl::uint4*>(x + (size_t)m * K + (size_t)c * 8);
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
                    if (m < M) acc[m] = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
                }
                if (lane == 0) {
                    for (int m = 0; m < M; ++m) slm[sgid * kMaxM + m] = acc[m];
                }
                sycl::group_barrier(it.get_group());
                if (part == 0 && lane < M && row < N) {
                    float t = 0.f;
                    for (int p = 0; p < KS; ++p) t += slm[(sgid + p) * kMaxM + lane];
                    const sycl::ext::oneapi::bfloat16 v(t);
                    y[(size_t)lane * N + row] = sycl::bit_cast<uint16_t>(v);
                }
            });
    });
}

template <int KS>
void launch(sycl::queue& q, const uint16_t* x, const int8_t* w, const sycl::half* s, uint16_t* y, int M, int N,
            int K) {
    constexpr int R = 4 / KS;
    const int chunks = K / 16;                // 16 weights per lane step
    const int per = (chunks + KS - 1) / KS;   // chunks per sub-group
    const int ngroups = (N + R - 1) / R;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(4 * kMaxM, h);
        h.parallel_for<w8_gemv_k<KS>>(
            sycl::nd_range<1>((size_t)ngroups * 64, 64),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_linear_id();
                const int sgid = (int)it.get_local_id(0) >> 4;
                const int row = (int)it.get_group(0) * R + sgid / KS;
                const int part = sgid % KS;
                float acc[kMaxM];
#pragma unroll
                for (int m = 0; m < kMaxM; ++m) acc[m] = 0.f;
                if (row < N) {
                    const int c0 = part * per;
                    const int c1 = sycl::min(c0 + per, chunks);
                    const int8_t* wr = w + (size_t)row * K;
                    const sycl::half* sr = s + (size_t)row * (K / 32);
                    for (int c = c0 + lane; c < c1; c += 16) {
                        const sycl::int4 wv = *reinterpret_cast<const sycl::int4*>(wr + (size_t)c * 16);
                        const float sc = (float)sr[c >> 1];
                        float wf[16];
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int word = j == 0 ? wv.x() : j == 1 ? wv.y() : j == 2 ? wv.z() : wv.w();
#pragma unroll
                            for (int b = 0; b < 4; ++b) wf[j * 4 + b] = (float)(int8_t)((word >> (8 * b)) & 0xff);
                        }
#pragma unroll
                        for (int m = 0; m < kMaxM; ++m) {
                            if (m < M) {
                                const uint16_t* xr = x + (size_t)m * K + (size_t)c * 16;
                                const sycl::uint4 a = *reinterpret_cast<const sycl::uint4*>(xr);
                                const sycl::uint4 b = *reinterpret_cast<const sycl::uint4*>(xr + 8);
                                const uint32_t xs[8] = {a.x(), a.y(), a.z(), a.w(), b.x(), b.y(), b.z(), b.w()};
                                float d = 0.f;
#pragma unroll
                                for (int j = 0; j < 8; ++j) {
                                    d = sycl::fma(wf[2 * j], bf16f((uint16_t)(xs[j] & 0xffff)), d);
                                    d = sycl::fma(wf[2 * j + 1], bf16f((uint16_t)(xs[j] >> 16)), d);
                                }
                                acc[m] = sycl::fma(d, sc, acc[m]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < kMaxM; ++m) {
                    if (m < M) acc[m] = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
                }
                if constexpr (KS == 1) {
                    if (lane == 0 && row < N) {
                        for (int m = 0; m < M; ++m) {
                            const sycl::ext::oneapi::bfloat16 v(acc[m]);
                            y[(size_t)m * N + row] = sycl::bit_cast<uint16_t>(v);
                        }
                    }
                } else {
                    if (lane == 0) {
                        for (int m = 0; m < M; ++m) slm[sgid * kMaxM + m] = acc[m];
                    }
                    sycl::group_barrier(it.get_group());
                    if (part == 0 && lane < M && row < N) {
                        float t = 0.f;
                        for (int p = 0; p < KS; ++p) t += slm[(sgid + p) * kMaxM + lane];
                        const sycl::ext::oneapi::bfloat16 v(t);
                        y[(size_t)lane * N + row] = sycl::bit_cast<uint16_t>(v);
                    }
                }
            });
    });
}

at::Tensor w8_gemv(at::Tensor x, at::Tensor w, at::Tensor s, int64_t ksplit) {
    TORCH_CHECK(x.is_xpu() && w.is_xpu() && s.is_xpu(), "xpu tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x bf16 [M, K]");
    TORCH_CHECK(w.scalar_type() == at::kChar && w.dim() == 2 && w.is_contiguous(), "w int8 [N, K]");
    TORCH_CHECK(s.scalar_type() == at::kHalf && s.dim() == 2 && s.is_contiguous(), "s fp16 [N, K/32]");
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    TORCH_CHECK(w.size(1) == K && K % 32 == 0 && s.size(0) == N && s.size(1) == K / 32, "shapes");
    TORCH_CHECK(M >= 1 && M <= kMaxM, "M must be 1..8");
    auto y = at::empty({M, N}, x.options());
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    const auto* xp = static_cast<const uint16_t*>(x.data_ptr());
    auto* yp = static_cast<uint16_t*>(y.data_ptr());
    const auto* sp = reinterpret_cast<const sycl::half*>(s.data_ptr());
    if (ksplit == 4) launch<4>(q, xp, w.data_ptr<int8_t>(), sp, yp, M, N, K);
    else if (ksplit == 2) launch<2>(q, xp, w.data_ptr<int8_t>(), sp, yp, M, N, K);
    else launch<1>(q, xp, w.data_ptr<int8_t>(), sp, yp, M, N, K);
    return y;
}

at::Tensor bf_gemv(at::Tensor x, at::Tensor w, int64_t ksplit) {
    TORCH_CHECK(x.is_xpu() && w.is_xpu(), "xpu tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x bf16 [M, K]");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.dim() == 2 && w.is_contiguous(), "w bf16 [N, K]");
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    TORCH_CHECK(w.size(1) == K && K % 8 == 0 && M >= 1 && M <= kMaxM, "shapes");
    auto y = at::empty({M, N}, x.options());
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    const auto* xp = static_cast<const uint16_t*>(x.data_ptr());
    const auto* wp = static_cast<const uint16_t*>(w.data_ptr());
    auto* yp = static_cast<uint16_t*>(y.data_ptr());
    switch (ksplit) {
    case 16: launch_bf<16>(q, xp, wp, yp, M, N, K); break;
    case 8: launch_bf<8>(q, xp, wp, yp, M, N, K); break;
    case 4: launch_bf<4>(q, xp, wp, yp, M, N, K); break;
    case 2: launch_bf<2>(q, xp, wp, yp, M, N, K); break;
    default: launch_bf<1>(q, xp, wp, yp, M, N, K); break;
    }
    return y;
}

// O4: a uint8 [rows, cols] XPU tensor whose storage is sycl::malloc_host
// (pinned host USM). Device kernels read it over PCIe (PLE table).
at::Tensor host_table(int64_t rows, int64_t cols) {
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    const size_t n = (size_t)rows * (size_t)cols;
    void* p = sycl::malloc_host(n, q);
    TORCH_CHECK(p != nullptr, "malloc_host failed for ", n, " bytes");
    // from_blob rejects a host-USM pointer on an XPU device, so build the
    // storage from a DataPtr directly.
    struct Owner {
        void* p;
        sycl::context ctx;
    };
    auto* own = new Owner{p, q.get_context()};
    const at::Device dev(at::kXPU, c10::xpu::current_device());
    at::DataPtr dp(p, own, [](void* c) {
        auto* o = static_cast<Owner*>(c);
        sycl::free(o->p, o->ctx);
        delete o;
    }, dev);
    c10::Storage st(c10::Storage::use_byte_size_t(), n, std::move(dp), nullptr, false);
    auto t = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(dev));
    t.set_(st, 0, {rows, cols}, {cols, 1});
    return t;
}

// CPU alias of a host-USM tensor (same bytes; lifetime is the caller's).
at::Tensor host_cpu_view(at::Tensor t) {
    return at::from_blob(t.data_ptr(), t.sizes(), t.strides(), at::TensorOptions().dtype(t.scalar_type()).device(at::kCPU));
}

}  // namespace

TORCH_LIBRARY(b70_o4, m) {
    m.def("host_table(int rows, int cols) -> Tensor");
    m.def("host_cpu_view(Tensor t) -> Tensor");
    m.def("w8_gemv(Tensor x, Tensor w, Tensor s, int ksplit) -> Tensor");
    m.def("bf_gemv(Tensor x, Tensor w, int ksplit) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_o4, CompositeExplicitAutograd, m) {
    m.impl("host_table", host_table);
}
TORCH_LIBRARY_IMPL(b70_o4, XPU, m) {
    m.impl("host_cpu_view", host_cpu_view);
    m.impl("w8_gemv", w8_gemv);
    m.impl("bf_gemv", bf_gemv);
}
