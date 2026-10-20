// SYCL kernels for GGUF decode on Intel XPU.
// Q4_K / Q5_K / Q6_K / IQ4_XS GEMV uses the KREG schedule in kreg.hpp
// (Strata native_mmvq.dp.cpp:2236). x is quantized to Q8_1 once per call.
// The other types stage one row in SLM
// and reduce a scalar dequant across the sub-group.
// Grouped prefill is the night joint_matrix path (night_grouped_expert.inc:230):
// each lane decodes 32 weights of one output column into SLM, then DPAS
// 8x16x16 fp16->fp32. MT=16 covers 128 activation rows per weight read
// (one pass for the M=128 expert bench). GRF 256, as Strata does for MT>4.
#include "kernels.h"
#include "decode.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdio>
#include <stdexcept>

namespace {

constexpr int kSg = 16;
constexpr int kNsg = 4;
constexpr int kWg = kSg * kNsg;       // 64, grouped tile
constexpr int kMaxRow = 8192;         // bytes staged per output row
constexpr int kLdb = 40;              // 32 decoded fp16 values + pad, night LDB
constexpr int kMt = 16;               // 128 activation rows per XMX pass
constexpr int kGemvSlmCap = 32768;    // bytes of SLM for one GEMV work-group

sycl::queue& q_of(void* queue) { return *static_cast<sycl::queue*>(queue); }

template <int TYPE>
class b70_dequant_k;

template <int TYPE>
void launch_dequant(sycl::queue& q, const uint8_t* blocks, int64_t rows, int64_t cols, int row_bytes, uint16_t* out) {
    const int groups = (int)(cols / 32);
    const int64_t total = rows * groups;
    constexpr int WG = 256;
    const size_t grid = (size_t)((total + WG - 1) / WG) * WG;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_dequant_k<TYPE>>(sycl::nd_range<1>(grid, WG), [=](sycl::nd_item<1> it) {
            const int64_t g = (int64_t)it.get_global_id(0);
            if (g >= total) return;
            const int64_t r = g / groups;
            const int gi = (int)(g - r * groups);
            float v[32];
            b70_decode32<TYPE>(blocks + r * row_bytes, gi, v);
            uint16_t* o = out + r * cols + (int64_t)gi * 32;
            for (int j = 0; j < 32; ++j) o[j] = b70_fp32_to_bf16(v[j]);
        });
    });
}

template <int TYPE>
class b70_gemv_k;

template <int TYPE>
void launch_gemv(sycl::queue& q, const uint16_t* x, const uint8_t* w, const int32_t* expert_ids, float* y,
                 int M, int T, int topk, int E, int N, int K, int row_bytes) {
    const bool moe = expert_ids != nullptr;
    // More sub-groups per work-group while the staged rows fit in SLM.
    // A fixed 32 KB reservation (4 * 8192) left most of that idle and
    // capped how many work-groups a subslice could keep in flight.
    int nsg = kNsg;
    while (nsg < 32 && (nsg * 2) * row_bytes <= kGemvSlmCap) nsg *= 2;
    const int wg_size = nsg * kSg;
    const int tiles = (N + nsg - 1) / nsg;
    const int groups = moe ? (T * topk * tiles) : tiles;
    const size_t global = (size_t)groups * (size_t)wg_size;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> slm(nsg * row_bytes, h);
        h.parallel_for<b70_gemv_k<TYPE>>(sycl::nd_range<1>(global, wg_size), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
            auto sg = it.get_sub_group();
            const int lane = (int)sg.get_local_linear_id();
            const int sgid = (int)it.get_local_id(0) / kSg;
            const int wg = (int)it.get_group(0);
            int row = 0;
            int token = 0;
            const uint8_t* src = w;
            bool live = false;
            if (!moe) {
                row = wg * nsg + sgid;
                live = row < N;
                if (live) src = w + (size_t)row * row_bytes;
            } else {
                const int which = wg / tiles;
                const int tile = wg - which * tiles;
                token = which / topk;
                const int slot = which - token * topk;
                row = tile * nsg + sgid;
                const int expert = expert_ids[(size_t)token * topk + slot];
                live = row < N && expert >= 0 && expert < E;
                if (live) src = w + ((size_t)expert * N + row) * row_bytes;
            }
            uint8_t* dst = slm.get_multi_ptr<sycl::access::decorated::no>().get() + sgid * row_bytes;
            if (live) {
                const bool align16 = ((uintptr_t)src & 15u) == 0;
                if (align16) {
                    const int n16 = row_bytes & ~15;
                    for (int off = lane * 16; off < n16; off += kSg * 16) {
                        const uint32_t* in = reinterpret_cast<const uint32_t*>(src + off);
                        uint32_t* o = reinterpret_cast<uint32_t*>(dst + off);
                        o[0] = in[0]; o[1] = in[1]; o[2] = in[2]; o[3] = in[3];
                    }
                    for (int off = n16 + lane; off < row_bytes; off += kSg) dst[off] = src[off];
                } else {
                    const int n2 = row_bytes & ~1;
                    for (int off = lane * 2; off < n2; off += kSg * 2) {
                        *reinterpret_cast<uint16_t*>(dst + off) = *reinterpret_cast<const uint16_t*>(src + off);
                    }
                }
            }
            sycl::group_barrier(sg);
            float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const int groups32 = K / 32;
            const int mlim = moe ? 1 : M;
            if (live) {
                for (int g = lane; g < groups32; g += kSg) {
                    float v[32];
                    b70_decode32<TYPE>(dst, g, v);
                    const int k0 = g * 32;
                    for (int m = 0; m < mlim; ++m) {
                        const uint16_t* xm = x + (size_t)(moe ? token : m) * K + k0;
                        float s = 0.f;
#pragma unroll
                        for (int j = 0; j < 32; ++j) s = sycl::fma(b70_bf16_to_fp32(xm[j]), v[j], s);
                        acc[m] += s;
                    }
                }
            }
            for (int m = 0; m < 8; ++m) {
                const float part = (m < mlim) ? acc[m] : 0.f;
                const float red = sycl::reduce_over_group(sg, part, sycl::plus<float>());
                if (lane == 0 && live && m < mlim) {
                    if (!moe) y[(size_t)m * N + row] = red;
                    else {
                        const int which = wg / tiles;
                        y[(size_t)which * N + row] = red;
                    }
                }
            }
        });
    });
}

template <int TYPE>
class b70_grouped_k;

template <int TYPE>
void launch_grouped(sycl::queue& q, const uint16_t* x, const uint8_t* w, const int32_t* entries, float* y,
                    int n_ent, int E, int M, int N, int K, int row_bytes) {
    constexpr int TILE = kMt * 8;
    const int ntiles = N / (kNsg * kSg);
    namespace jm = sycl::ext::oneapi::experimental::matrix;
    namespace jmi = sycl::ext::intel::experimental::matrix;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> slm(kNsg * kSg * kLdb, h);
        h.parallel_for<b70_grouped_k<TYPE>>(
            sycl::nd_range<2>(sycl::range<2>((size_t)n_ent, (size_t)ntiles * kWg), sycl::range<2>(1, kWg)),
            props,
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSg)]] {
            auto sg = it.get_sub_group();
            const int lane = (int)sg.get_local_linear_id();
            const int sgid = (int)it.get_local_id(1) / kSg;
            const int ent = (int)it.get_group(0);
            const int n0 = (int)it.get_group(1) * (kNsg * kSg) + sgid * kSg;
            const int vr = n0 + lane;
            const int expert = entries[ent * 3 + 0];
            const int row0 = entries[ent * 3 + 1];
            const int rows = entries[ent * 3 + 2];
            const bool ok = expert >= 0 && expert < E && row0 >= 0 && rows > 0 && (row0 + rows) <= M;
            if (!ok) return;
            const uint8_t* rowp = w + ((size_t)expert * (size_t)N + (size_t)vr) * (size_t)row_bytes;
            auto sp = slm.template get_multi_ptr<sycl::access::decorated::no>() + sgid * kSg * kLdb;
            const sycl::half* X = reinterpret_cast<const sycl::half*>(x);
            using Acc = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, kSg>;
            Acc acc[kMt];
            for (int rbase = 0; rbase < rows; rbase += TILE) {
                const int rows_here = (rows - rbase) < TILE ? (rows - rbase) : TILE;
#pragma unroll
                for (int i = 0; i < kMt; ++i) jm::joint_matrix_fill(sg, acc[i], 0.f);
                auto xp = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(
                    const_cast<sycl::half*>(X + (size_t)(row0 + rbase) * K));
                for (int k0 = 0; k0 < K; k0 += 32) {
                    float v[32];
                    if (ok) b70_decode32<TYPE>(rowp, k0 / 32, v);
                    else
                        for (int j = 0; j < 32; ++j) v[j] = 0.f;
                    sycl::half* my = sp.get() + lane * kLdb;
                    for (int j = 0; j < 32; ++j) my[j] = sycl::half(v[j]);
                    sycl::group_barrier(sg);
                    for (int kk = 0; kk < 32; kk += 16) {
                        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, kSg, jm::layout::col_major> b;
                        jm::joint_matrix_load(sg, b, sp + kk, kLdb);
#pragma unroll
                        for (int i = 0; i < kMt; ++i) {
                            if (i * 8 < rows_here) {
                                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> a;
                                jmi::joint_matrix_load_checked(sg, a, xp, (size_t)K, (size_t)rows_here, (size_t)K,
                                                               (size_t)(i * 8), (size_t)(k0 + kk));
                                jm::joint_matrix_mad(sg, acc[i], a, b, acc[i]);
                            }
                        }
                    }
                    sycl::group_barrier(sg);
                }
                auto yp = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                   sycl::access::decorated::no>(y + (size_t)(row0 + rbase) * N);
#pragma unroll
                for (int i = 0; i < kMt; ++i) {
                    if (i * 8 < rows_here) {
                        jmi::joint_matrix_store_checked(sg, acc[i], yp, (size_t)N, jm::layout::row_major,
                                                        (size_t)rows_here, (size_t)N, (size_t)(i * 8), (size_t)n0);
                    }
                }
            }
        });
    });
}

#define B70_DISPATCH(TYPE)                                                                                             \
    case TYPE: fn.template operator()<TYPE>(); break;

template <typename Fn>
void dispatch(int type, Fn fn) {
    switch (type) {
        B70_DISPATCH(B70_Q4_K)
        B70_DISPATCH(B70_Q5_K)
        B70_DISPATCH(B70_Q6_K)
        B70_DISPATCH(B70_IQ2_XXS)
        B70_DISPATCH(B70_IQ2_XS)
        B70_DISPATCH(B70_IQ2_S)
        B70_DISPATCH(B70_IQ3_XXS)
        B70_DISPATCH(B70_IQ3_S)
        B70_DISPATCH(B70_IQ4_NL)
        B70_DISPATCH(B70_IQ4_XS)
        B70_DISPATCH(B70_Q2_0)
    default: throw std::runtime_error("b70_gguf: unsupported ggml type");
    }
}

#include "kreg.hpp"

}  // namespace

void b70_dequant_bf16(void* queue, int type, const uint8_t* q, int64_t rows, int64_t cols, uint16_t* out) {
    const int rb = b70_row_bytes(type, (int)cols);
    if (rb <= 0 || rows <= 0) throw std::runtime_error("b70_gguf: bad dequant shape");
    auto& sy = q_of(queue);
    dispatch(type, [&]<int TYPE>() { launch_dequant<TYPE>(sy, q, rows, cols, rb, out); });
}

void b70_gemv(void* queue, int type, const uint16_t* x, const uint8_t* w, float* y, int M, int N, int K) {
    const int rb = b70_row_bytes(type, K);
    if (rb <= 0 || rb > kMaxRow || M <= 0 || N <= 0) throw std::runtime_error("b70_gguf: bad gemv shape");
    auto& sy = q_of(queue);
    const bool kreg = (type == B70_Q4_K || type == B70_Q5_K || type == B70_Q6_K || type == B70_IQ4_XS) && (K % 256) == 0;
    if (kreg) {
        for (int m0 = 0; m0 < M; m0 += 8) {
            const int mm = (M - m0) < 8 ? (M - m0) : 8;
            dispatch(type, [&]<int TYPE>() {
                b70_kreg_m<TYPE>(sy, x + (size_t)m0 * K, w, y + (size_t)m0 * N, mm, N, K);
            });
        }
        return;
    }
    for (int m0 = 0; m0 < M; m0 += 8) {
        const int mm = (M - m0) < 8 ? (M - m0) : 8;
        dispatch(type, [&]<int TYPE>() {
            launch_gemv<TYPE>(sy, x + (size_t)m0 * K, w, nullptr, y + (size_t)m0 * N, mm, 0, 0, 0, N, K, rb);
        });
    }
}

void b70_moe_gemv(void* queue, int type, const uint16_t* x, const uint8_t* w, const int32_t* expert_ids, float* y,
                  int T, int topk, int E, int N, int K) {
    const int rb = b70_row_bytes(type, K);
    if (rb <= 0 || rb > kMaxRow || T <= 0 || topk <= 0 || E <= 0 || N <= 0)
        throw std::runtime_error("b70_gguf: bad moe gemv shape");
    auto& sy = q_of(queue);
    dispatch(type, [&]<int TYPE>() {
        launch_gemv<TYPE>(sy, x, w, expert_ids, y, 1, T, topk, E, N, K, rb);
    });
}

int b70_grouped_gemm(void* queue, int type, const uint16_t* x, const uint8_t* w, const int32_t* entries, float* y,
                     int n_ent, int E, int M, int N, int K) {
    const int rb = b70_row_bytes(type, K);
    if (rb <= 0 || n_ent <= 0 || E <= 0 || M <= 0 || N <= 0) return 1;
    if (N % (kNsg * kSg) != 0 || K % 32 != 0) return 2;
    auto& sy = q_of(queue);
    dispatch(type, [&]<int TYPE>() { launch_grouped<TYPE>(sy, x, w, entries, y, n_ent, E, M, N, K, rb); });
    return 0;
}
