// C10: bandwidth-oriented GGUF dequant plus the index kernels the bf16
// grouped GEMM path needs. One work-item owns one 32-wide group. Neighboring
// lanes in a subgroup write neighboring 64-byte output chunks (one cache
// line each) through two 32-byte vector stores. Decode is b70_decode32, so
// the bf16 bits match dequant_bf16 / the CPU reference.
#include "dequant_fast.hpp"
#include "decode.hpp"

#include <sycl/sycl.hpp>

#include <c10/xpu/XPUStream.h>

#include <stdexcept>

// Helpers stay in this TU. Kernel name types are at global scope so the
// exported ops (matched to dequant_fast.hpp) can name them.
static sycl::queue& q_of() { return c10::xpu::getCurrentXPUStream().queue(); }

static void check_xpu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_xpu(), name, " must be an XPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

template <bool FP16>
inline void b70_store32(uint16_t* dst, const float* v) {
    alignas(32) uint16_t tmp[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        if constexpr (FP16) tmp[j] = sycl::bit_cast<uint16_t>(sycl::half(v[j]));
        else tmp[j] = b70_fp32_to_bf16(v[j]);
    }
    auto* d = reinterpret_cast<sycl::vec<uint16_t, 16>*>(dst);
    auto* s = reinterpret_cast<const sycl::vec<uint16_t, 16>*>(tmp);
    d[0] = s[0];
    d[1] = s[1];
}

template <int TYPE, bool FP16>
class b70_c10_dequant;

template <int TYPE, bool FP16>
void launch_dequant(sycl::queue& q, const uint8_t* src, uint16_t* dst, int64_t rows, int64_t cols, int row_bytes) {
    const int gpr = (int)(cols / 32);
    const int64_t total = rows * (int64_t)gpr;
    constexpr int WG = 256;
    const int64_t cap = (int64_t)WG * 4096;
    int64_t nthreads = total < cap ? total : cap;
    if (nthreads < 1) nthreads = 1;
    const size_t grid = (size_t)(((nthreads + WG - 1) / WG) * WG);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_c10_dequant<TYPE, FP16>>(
            sycl::nd_range<1>(grid, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int64_t stride = (int64_t)it.get_global_range(0);
                for (int64_t g = (int64_t)it.get_global_id(0); g < total; g += stride) {
                    const int64_t row = g / gpr;
                    const int gi = (int)(g - row * gpr);
                    float v[32];
                    b70_decode32<TYPE>(src + row * (int64_t)row_bytes, gi, v);
                    b70_store32<FP16>(dst + (row * cols + (int64_t)gi * 32), v);
                }
            });
    });
}

template <typename Fn>
void dispatch_type(int type, Fn fn) {
    switch (type) {
    case B70_Q4_K: fn.template operator()<B70_Q4_K>(); break;
    case B70_Q5_K: fn.template operator()<B70_Q5_K>(); break;
    case B70_Q6_K: fn.template operator()<B70_Q6_K>(); break;
    case B70_IQ2_XXS: fn.template operator()<B70_IQ2_XXS>(); break;
    case B70_IQ2_XS: fn.template operator()<B70_IQ2_XS>(); break;
    case B70_IQ2_S: fn.template operator()<B70_IQ2_S>(); break;
    case B70_IQ3_XXS: fn.template operator()<B70_IQ3_XXS>(); break;
    case B70_IQ3_S: fn.template operator()<B70_IQ3_S>(); break;
    case B70_IQ4_NL: fn.template operator()<B70_IQ4_NL>(); break;
    case B70_IQ4_XS: fn.template operator()<B70_IQ4_XS>(); break;
    case B70_Q2_K: fn.template operator()<B70_Q2_K>(); break;
    case B70_Q3_K: fn.template operator()<B70_Q3_K>(); break;
    case B70_Q2_0: fn.template operator()<B70_Q2_0>(); break;
    default: throw std::runtime_error("b70_gguf: prefill_dequant unsupported type");
    }
}

static int expert_of(const int32_t* starts, int E, int p) {
    int lo = 0, hi = E;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (starts[mid] <= p) lo = mid;
        else hi = mid;
    }
    return lo;
}

static bool two_byte(const at::Tensor& t) {
    const auto d = t.scalar_type();
    return d == at::kBFloat16 || d == at::kHalf;
}

class b70_c10_hist;
class b70_c10_scan;
class b70_c10_place;
class b70_c10_gather;
class b70_c10_scatter;
class b70_c10_pad;
class b70_c10_unpad;

at::Tensor b70_prefill_dequant_op(at::Tensor q, int64_t ggml_type, int64_t rows, int64_t cols, at::Tensor out) {
    check_xpu(q, "q");
    check_xpu(out, "out");
    TORCH_CHECK(q.scalar_type() == at::kByte, "q must be uint8");
    TORCH_CHECK(two_byte(out), "out must be bf16 or fp16");
    TORCH_CHECK(rows > 0 && cols > 0 && cols % 32 == 0, "prefill_dequant shape");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == rows && out.size(1) == cols, "out shape");
    const int rb = b70_row_bytes((int)ggml_type, (int)cols);
    TORCH_CHECK(rb > 0, "bad type or cols");
    TORCH_CHECK(q.numel() >= rows * (int64_t)rb, "q is shorter than rows*row_bytes");
    const bool fp16 = out.scalar_type() == at::kHalf;
    auto& queue = q_of();
    const uint8_t* src = q.data_ptr<uint8_t>();
    uint16_t* dst = static_cast<uint16_t*>(out.data_ptr());
    dispatch_type((int)ggml_type, [&]<int TYPE>() {
        if (fp16) launch_dequant<TYPE, true>(queue, src, dst, rows, cols, rb);
        else launch_dequant<TYPE, false>(queue, src, dst, rows, cols, rb);
    });
    return out;
}

at::Tensor b70_prefill_group_ids_op(at::Tensor ids, int64_t E, at::Tensor order, at::Tensor starts, at::Tensor counts,
                                    at::Tensor cursor) {
    check_xpu(ids, "ids");
    check_xpu(order, "order");
    check_xpu(starts, "starts");
    check_xpu(counts, "counts");
    check_xpu(cursor, "cursor");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 1, "ids int32 [M]");
    TORCH_CHECK(E > 0 && E <= 4096, "E");
    const int M = (int)ids.size(0);
    TORCH_CHECK(M > 0 && order.numel() >= M, "order");
    TORCH_CHECK(starts.numel() >= E + 1 && counts.numel() >= E && cursor.numel() >= E, "starts/counts/cursor");
    TORCH_CHECK(starts.scalar_type() == at::kInt && counts.scalar_type() == at::kInt && order.scalar_type() == at::kInt &&
                    cursor.scalar_type() == at::kInt,
                "index scratch must be int32");
    auto& q = q_of();
    const int32_t* idp = ids.data_ptr<int32_t>();
    int32_t* orderp = order.data_ptr<int32_t>();
    int32_t* startsp = starts.data_ptr<int32_t>();
    int32_t* countsp = counts.data_ptr<int32_t>();
    int32_t* cursorp = cursor.data_ptr<int32_t>();
    const int Ee = (int)E;
    q.memset(countsp, 0, sizeof(int32_t) * (size_t)Ee);
    constexpr int WG = 256;
    int groups = M < WG ? 1 : (M + WG - 1) / WG;
    if (groups > 1024) groups = 1024;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> lc(Ee, h);
        h.parallel_for<b70_c10_hist>(sycl::nd_range<1>((size_t)groups * WG, WG), [=](sycl::nd_item<1> it) {
            const int tid = (int)it.get_local_id(0);
            int* bin = lc.get_multi_ptr<sycl::access::decorated::no>().get();
            for (int e = tid; e < Ee; e += WG) bin[e] = 0;
            it.barrier(sycl::access::fence_space::local_space);
            const int gs = (int)it.get_global_range(0);
            for (int i = (int)it.get_global_id(0); i < M; i += gs) {
                const int e = idp[i];
                if ((unsigned)e < (unsigned)Ee) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        ar(bin[e]);
                    ar.fetch_add(1);
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int e = tid; e < Ee; e += WG) {
                const int c = bin[e];
                if (c) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        ag(countsp[e]);
                    ag.fetch_add(c);
                }
            }
        });
    });
    q.single_task<b70_c10_scan>([=]() {
        int sum = 0;
        startsp[0] = 0;
        for (int e = 0; e < Ee; ++e) {
            cursorp[e] = sum;
            sum += countsp[e];
            startsp[e + 1] = sum;
        }
    });
    const size_t place_n = (size_t)((M + 255) & ~255);
    q.parallel_for<b70_c10_place>(sycl::range<1>(place_n), [=](sycl::item<1> it) {
        const int i = (int)it.get_id(0);
        if (i >= M) return;
        const int e = idp[i];
        if ((unsigned)e >= (unsigned)Ee) return;
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ag(cursorp[e]);
        const int pos = ag.fetch_add(1);
        orderp[pos] = i;
    });
    return counts;
}

at::Tensor b70_prefill_row_gather_op(at::Tensor src, at::Tensor dst, at::Tensor index, int64_t mv) {
    check_xpu(src, "src");
    check_xpu(dst, "dst");
    check_xpu(index, "index");
    TORCH_CHECK(two_byte(src) && src.scalar_type() == dst.scalar_type(), "gather dtype");
    TORCH_CHECK(src.dim() == 2 && dst.dim() == 2 && src.size(1) == dst.size(1), "gather shape");
    const int K = (int)src.size(1);
    const int M = (int)src.size(0);
    TORCH_CHECK(K % 8 == 0 && K > 0, "K%8");
    TORCH_CHECK(mv >= 0 && mv <= dst.size(0) && mv <= index.numel(), "mv");
    if (mv == 0) return dst;
    auto& q = q_of();
    const uint16_t* sp = static_cast<const uint16_t*>(src.data_ptr());
    uint16_t* dp = static_cast<uint16_t*>(dst.data_ptr());
    const int32_t* ip = index.data_ptr<int32_t>();
    constexpr int WG = 128;
    const int nwg = (int)mv < 4096 ? (int)mv : 4096;
    const int vecs = K / 8;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_c10_gather>(sycl::nd_range<1>((size_t)nwg * WG, WG), [=](sycl::nd_item<1> it) {
            const int tid = (int)it.get_local_id(0);
            const int wg = (int)it.get_group(0);
            for (int p = wg; p < (int)mv; p += nwg) {
                const int src_row = ip[p];
                if ((unsigned)src_row >= (unsigned)M) continue;
                const auto* in = reinterpret_cast<const sycl::vec<uint16_t, 8>*>(sp + (size_t)src_row * K);
                auto* out = reinterpret_cast<sycl::vec<uint16_t, 8>*>(dp + (size_t)p * K);
                for (int v = tid; v < vecs; v += WG) out[v] = in[v];
            }
        });
    });
    return dst;
}

at::Tensor b70_prefill_row_scatter_op(at::Tensor src, at::Tensor dst, at::Tensor index, int64_t mv) {
    check_xpu(src, "src");
    check_xpu(dst, "dst");
    check_xpu(index, "index");
    TORCH_CHECK(two_byte(src) && src.scalar_type() == dst.scalar_type(), "scatter dtype");
    TORCH_CHECK(src.dim() == 2 && dst.dim() == 2 && src.size(1) == dst.size(1), "scatter shape");
    const int K = (int)src.size(1);
    const int M = (int)dst.size(0);
    TORCH_CHECK(K % 8 == 0 && mv >= 0 && mv <= src.size(0) && mv <= index.numel(), "scatter mv");
    if (mv == 0) return dst;
    auto& q = q_of();
    const uint16_t* sp = static_cast<const uint16_t*>(src.data_ptr());
    uint16_t* dp = static_cast<uint16_t*>(dst.data_ptr());
    const int32_t* ip = index.data_ptr<int32_t>();
    constexpr int WG = 128;
    const int nwg = (int)mv < 4096 ? (int)mv : 4096;
    const int vecs = K / 8;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_c10_scatter>(sycl::nd_range<1>((size_t)nwg * WG, WG), [=](sycl::nd_item<1> it) {
            const int tid = (int)it.get_local_id(0);
            const int wg = (int)it.get_group(0);
            for (int p = wg; p < (int)mv; p += nwg) {
                const int dst_row = ip[p];
                if ((unsigned)dst_row >= (unsigned)M) continue;
                const auto* in = reinterpret_cast<const sycl::vec<uint16_t, 8>*>(sp + (size_t)p * K);
                auto* out = reinterpret_cast<sycl::vec<uint16_t, 8>*>(dp + (size_t)dst_row * K);
                for (int v = tid; v < vecs; v += WG) out[v] = in[v];
            }
        });
    });
    return dst;
}

at::Tensor b70_prefill_pad_grouped_op(at::Tensor xflat, at::Tensor xpad, at::Tensor order, at::Tensor starts, int64_t mv) {
    check_xpu(xflat, "xflat");
    check_xpu(xpad, "xpad");
    check_xpu(order, "order");
    check_xpu(starts, "starts");
    TORCH_CHECK(two_byte(xflat) && xflat.scalar_type() == xpad.scalar_type(), "pad dtype");
    TORCH_CHECK(xpad.dim() == 3 && xflat.dim() == 2 && xpad.size(2) == xflat.size(1), "pad shape");
    const int K = (int)xflat.size(1);
    const int M = (int)xflat.size(0);
    const int E = (int)xpad.size(0);
    const int mmax = (int)xpad.size(1);
    TORCH_CHECK(K % 8 == 0 && mv >= 0 && mv <= order.numel(), "pad mv");
    if (mv == 0) return xpad;
    auto& q = q_of();
    const uint16_t* sp = static_cast<const uint16_t*>(xflat.data_ptr());
    uint16_t* dp = static_cast<uint16_t*>(xpad.data_ptr());
    const int32_t* op = order.data_ptr<int32_t>();
    const int32_t* stp = starts.data_ptr<int32_t>();
    constexpr int WG = 128;
    const int nwg = (int)mv < 4096 ? (int)mv : 4096;
    const int vecs = K / 8;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_c10_pad>(sycl::nd_range<1>((size_t)nwg * WG, WG), [=](sycl::nd_item<1> it) {
            const int tid = (int)it.get_local_id(0);
            const int wg = (int)it.get_group(0);
            for (int p = wg; p < (int)mv; p += nwg) {
                const int e = expert_of(stp, E, p);
                const int local = p - stp[e];
                const int src_row = op[p];
                if ((unsigned)local >= (unsigned)mmax || (unsigned)src_row >= (unsigned)M) continue;
                const auto* in = reinterpret_cast<const sycl::vec<uint16_t, 8>*>(sp + (size_t)src_row * K);
                auto* out = reinterpret_cast<sycl::vec<uint16_t, 8>*>(dp + ((size_t)e * mmax + local) * K);
                for (int v = tid; v < vecs; v += WG) out[v] = in[v];
            }
        });
    });
    return xpad;
}

at::Tensor b70_prefill_unpad_scatter_op(at::Tensor ypad, at::Tensor slots, at::Tensor order, at::Tensor starts, int64_t mv) {
    check_xpu(ypad, "ypad");
    check_xpu(slots, "slots");
    check_xpu(order, "order");
    check_xpu(starts, "starts");
    TORCH_CHECK(two_byte(ypad) && ypad.scalar_type() == slots.scalar_type(), "unpad dtype");
    TORCH_CHECK(ypad.dim() == 3 && slots.dim() == 2 && ypad.size(2) == slots.size(1), "unpad shape");
    const int K = (int)slots.size(1);
    const int M = (int)slots.size(0);
    const int E = (int)ypad.size(0);
    const int mmax = (int)ypad.size(1);
    TORCH_CHECK(K % 8 == 0 && mv >= 0 && mv <= order.numel(), "unpad mv");
    if (mv == 0) return slots;
    auto& q = q_of();
    const uint16_t* sp = static_cast<const uint16_t*>(ypad.data_ptr());
    uint16_t* dp = static_cast<uint16_t*>(slots.data_ptr());
    const int32_t* op = order.data_ptr<int32_t>();
    const int32_t* stp = starts.data_ptr<int32_t>();
    constexpr int WG = 128;
    const int nwg = (int)mv < 4096 ? (int)mv : 4096;
    const int vecs = K / 8;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_c10_unpad>(sycl::nd_range<1>((size_t)nwg * WG, WG), [=](sycl::nd_item<1> it) {
            const int tid = (int)it.get_local_id(0);
            const int wg = (int)it.get_group(0);
            for (int p = wg; p < (int)mv; p += nwg) {
                const int e = expert_of(stp, E, p);
                const int local = p - stp[e];
                const int dst_row = op[p];
                if ((unsigned)local >= (unsigned)mmax || (unsigned)dst_row >= (unsigned)M) continue;
                const auto* in = reinterpret_cast<const sycl::vec<uint16_t, 8>*>(sp + ((size_t)e * mmax + local) * K);
                auto* out = reinterpret_cast<sycl::vec<uint16_t, 8>*>(dp + (size_t)dst_row * K);
                for (int v = tid; v < vecs; v += WG) out[v] = in[v];
            }
        });
    });
    return slots;
}
