// C2 prefill GEMM.
// One launch covers every expert of a projection. Tokens are sorted on device,
// activations gathered into expert-major order, then a joint_matrix kernel
// (Strata night_grouped_expert.inc:230, with the next-group preload at :275).
// Each work-group owns as many output columns as SLM allows so the activation
// tile is staged once and reused across those columns.
// mode 0: fp16 DPAS 8x16x16 (the Strata tile). mode 1: bf16 DPAS 8x16x16.
// mode 2: int8 DPAS 8x16x32 with per-32-block scales (evaluated, extra rounding).
#include "grouped_api.hpp"
#include "grouped_raw.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <c10/xpu/XPUStream.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace b70pf {

namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace jmi = sycl::ext::intel::experimental::matrix;

constexpr int kSg = 16;
constexpr int kLdb = 40;
constexpr int kLda = 32;

struct Split {
    double sort = 0, gather = 0, gemm = 0, gemm2 = 0, swiglu = 0, reduce = 0, total = 0;
    int cols = 0, mt = 0, mode = 0;
};
Split g_split;

int split_on() {
    const char* e = std::getenv("B70_PREFILL_SPLIT");
    return (e && e[0] == '1') ? 1 : 0;
}

struct Layout {
    int cols = 64;
    size_t total = 0, a_off = 0, sc_off = 0, i32_off = 0;
};

Layout make_layout(int cols, int mt, int mode) {
    Layout L;
    L.cols = cols;
    const int nsg = cols / kSg;
    if (mode == 2) {
        const size_t b = (size_t)nsg * kSg * kLdb;
        L.a_off = (b + 63) & ~size_t{63};
        const size_t a = (size_t)mt * 8 * kLda;
        L.sc_off = (L.a_off + a + 63) & ~size_t{63};
        const size_t sc = (size_t)mt * 8 * sizeof(float);
        L.i32_off = (L.sc_off + sc + 63) & ~size_t{63};
        const size_t i32 = (size_t)nsg * 8 * kSg * sizeof(int);
        L.total = L.i32_off + i32;
    } else {
        const size_t b = (size_t)nsg * kSg * kLdb * 2;
        L.a_off = (b + 63) & ~size_t{63};
        const size_t a = (size_t)mt * 8 * kLda * 2;
        L.total = L.a_off + a;
    }
    return L;
}

Layout choose_layout(sycl::queue& q, int N, int mt, int mode) {
    auto dev = q.get_device();
    const size_t local = dev.get_info<sycl::info::device::local_mem_size>();
    const size_t maxwg = dev.get_info<sycl::info::device::max_work_group_size>();
    size_t maxsg = 32;
    try {
        maxsg = dev.get_info<sycl::info::device::max_num_sub_groups>();
    } catch (...) {
        maxsg = 32;
    }
    if (maxsg < 4) maxsg = 32;
    static int printed = 0;
    if (!printed) {
        std::fprintf(stderr, "prefill device local_mem=%zu max_wg=%zu max_sg=%zu\n", local, maxwg, maxsg);
        printed = 1;
    }
    // 16KB for compiler spill. 128B so the kernel can align the base.
    const size_t reserve = 16384 + 128;
    const size_t budget = local > reserve ? local - reserve : local / 2;
    const int cands[] = {1024, 640, 512, 384, 256, 128, 64};
    Layout best = make_layout(64, mt, mode);
    for (int c : cands) {
        if (c % kSg != 0 || N % c != 0) continue;
        if ((size_t)c > maxwg) continue;
        if ((size_t)(c / kSg) > maxsg) continue;
        Layout L = make_layout(c, mt, mode);
        if (L.total <= budget) return L;
    }
    if (best.total > budget) {
        std::fprintf(stderr, "prefill SLM %zu exceeds budget %zu (local %zu) at cols=64\n", best.total, budget, local);
    }
    return best;
}

int pick_mt(int M, int E) {
    const int avg = (M + E - 1) / E;
    return avg <= 24 ? 4 : 16;
}

struct GemmArgs {
    const uint16_t* X;
    const uint8_t* gate;
    const uint8_t* up;
    const int32_t* starts;
    float* Y;
    int E, N, K, rb, n_wrow;
    int gu;
    Layout lay;
};

template <int TYPE>
inline void prefetch_decode(const uint8_t* base, int wrow, int K, int k0, B70Raw& cur, float* v) {
    const int64_t flat = (int64_t)wrow * K + k0;
    const int ib = (int)((flat & 255) >> 5);
    if (k0 + 32 < K) {
        const int64_t fn = flat + 32;
        B70Raw nxt = b70_load_raw<TYPE>(base, fn >> 8, (int)((fn & 255) >> 5));
        b70_decode_raw<TYPE>(cur, ib, v);
        cur = nxt;
    } else {
        b70_decode_raw<TYPE>(cur, ib, v);
    }
}

template <typename T>
auto as_local(uint8_t* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
        reinterpret_cast<T*>(p));
}

template <int TYPE, int MT, int MODE>
class b70_pf_gemm;

template <int TYPE, int MT, int MODE>
void launch_gemm(sycl::queue& q, const GemmArgs& a) {
    const Layout& L = a.lay;
    const int cols = L.cols;
    const int nsg = cols / kSg;
    const int nwg_n = a.N / cols;
    const int groups = a.E * nwg_n;
    const size_t wg = (size_t)cols;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<(MT > 4 ? 256 : 128)>};
    const int gu = a.gu;
    const int E = a.E, N = a.N, K = a.K, rb = a.rb, n_wrow = a.n_wrow;
    const size_t a_off = L.a_off, sc_off = L.sc_off, i32_off = L.i32_off;
    const uint16_t* X = a.X;
    const uint8_t* gate = a.gate;
    const uint8_t* up = a.up;
    const int32_t* starts = a.starts;
    float* Y = a.Y;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> slm(L.total + 128, h);
        h.parallel_for<b70_pf_gemm<TYPE, MT, MODE>>(
            sycl::nd_range<1>((size_t)groups * wg, wg), props,
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
                auto sg = it.get_sub_group();
                const int tid = (int)it.get_local_id(0);
                const int lane = (int)sg.get_local_linear_id();
                const int sgid = tid / kSg;
                const int group = (int)it.get_group(0);
                const int expert = group / nwg_n;
                const int n0 = (group - expert * nwg_n) * cols;
                const int row0 = starts[expert];
                const int rows = starts[expert + 1] - row0;
                if (rows <= 0) return;
                const int col = n0 + sgid * kSg + lane;
                const uint8_t* gate_e = gate + ((size_t)expert * (size_t)n_wrow * (size_t)rb);
                const uint8_t* up_e = gu ? up + ((size_t)expert * (size_t)n_wrow * (size_t)rb) : gate_e;
                const uint8_t* base = (gu && (col & 1)) ? up_e : gate_e;
                const int wrow = gu ? (col >> 1) : col;
                uint8_t* raw = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                const uintptr_t mis = reinterpret_cast<uintptr_t>(raw) & 63u;
                uint8_t* sm = raw + (mis ? (64u - mis) : 0u);
                const int64_t flat0 = (int64_t)wrow * K;
                constexpr int TILE = MT * 8;
                for (int rbase = 0; rbase < rows; rbase += TILE) {
                    const int rows_here = (rows - rbase) < TILE ? (rows - rbase) : TILE;
                    const int rows_pad = (rows_here + 7) & ~7;
                    if constexpr (MODE == 2) {
                        int8_t* As = reinterpret_cast<int8_t*>(sm + a_off);
                        float* sc = reinterpret_cast<float*>(sm + sc_off);
                        const int an = MT * 8 * kLda;
                        for (int i = tid; i < an; i += cols) As[i] = 0;
                        for (int i = tid; i < MT * 8; i += cols) sc[i] = 0.f;
                    } else {
                        uint16_t* As = reinterpret_cast<uint16_t*>(sm + a_off);
                        const int an = MT * 8 * kLda;
                        for (int i = tid; i < an; i += cols) As[i] = 0;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    B70Raw cur = b70_load_raw<TYPE>(base, flat0 >> 8, (int)((flat0 & 255) >> 5));
                    if constexpr (MODE == 2) {
                        float facc[MT][8];
#pragma unroll
                        for (int i = 0; i < MT; ++i)
                            for (int r = 0; r < 8; ++r) facc[i][r] = 0.f;
                        for (int k0 = 0; k0 < K; k0 += 32) {
                            float v[32];
                            prefetch_decode<TYPE>(base, wrow, K, k0, cur, v);
                            float amax = 0.f;
                            for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(v[j]));
                            const float sb = amax > 0.f ? amax / 127.f : 0.f;
                            const float inv = amax > 0.f ? 127.f / amax : 0.f;
                            int8_t* my = reinterpret_cast<int8_t*>(sm) + (size_t)(sgid * kSg + lane) * kLdb;
                            for (int j = 0; j < 32; ++j) {
                                float q = sycl::rint(sycl::fmin(127.f, sycl::fmax(-127.f, v[j] * inv)));
                                my[j] = (int8_t)q;
                            }
                            int8_t* As = reinterpret_cast<int8_t*>(sm + a_off);
                            float* sc = reinterpret_cast<float*>(sm + sc_off);
                            for (int r = tid; r < rows_here; r += cols) {
                                const uint16_t* src = X + ((size_t)(row0 + rbase + r) * K + k0);
                                float mx = 0.f;
                                float tmp[32];
                                for (int c = 0; c < 32; ++c) {
                                    tmp[c] = (float)sycl::bit_cast<sycl::half>(src[c]);
                                    mx = sycl::fmax(mx, sycl::fabs(tmp[c]));
                                }
                                sc[r] = mx > 0.f ? mx / 127.f : 0.f;
                                const float ainv = mx > 0.f ? 127.f / mx : 0.f;
                                for (int c = 0; c < 32; ++c) {
                                    float q = sycl::rint(sycl::fmin(127.f, sycl::fmax(-127.f, tmp[c] * ainv)));
                                    As[r * kLda + c] = (int8_t)q;
                                }
                            }
                            it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                            for (int i = 0; i < MT; ++i) {
                                if (i * 8 >= rows_pad) break;
                                jm::joint_matrix<sycl::sub_group, int8_t, jm::use::a, 8, 32, jm::layout::row_major> am;
                                jm::joint_matrix<sycl::sub_group, int8_t, jm::use::b, 32, kSg, jm::layout::col_major> bm;
                                jm::joint_matrix<sycl::sub_group, int, jm::use::accumulator, 8, kSg> cm;
                                jm::joint_matrix_fill(sg, cm, 0);
                                auto bp = as_local<int8_t>(sm + (size_t)sgid * kSg * kLdb);
                                auto ap = as_local<int8_t>(sm + a_off + (size_t)(i * 8) * kLda);
                                jm::joint_matrix_load(sg, bm, bp, kLdb);
                                jm::joint_matrix_load(sg, am, ap, kLda);
                                jm::joint_matrix_mad(sg, cm, am, bm, cm);
                                auto ip = as_local<int>(sm + i32_off + (size_t)sgid * 8 * kSg * sizeof(int));
                                jm::joint_matrix_store(sg, cm, ip, kSg, jm::layout::row_major);
                                sycl::group_barrier(sg);
                                int* tile = reinterpret_cast<int*>(sm + i32_off) + sgid * 8 * kSg;
                                for (int r = 0; r < 8; ++r) {
                                    const int gr = i * 8 + r;
                                    if (gr < rows_here) {
                                        facc[i][r] = sycl::fma((float)tile[r * kSg + lane], sb * sc[gr], facc[i][r]);
                                    }
                                }
                                sycl::group_barrier(sg);
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        for (int i = 0; i < MT; ++i) {
                            for (int r = 0; r < 8; ++r) {
                                const int gr = i * 8 + r;
                                if (gr < rows_here) Y[((size_t)(row0 + rbase + gr) * N) + col] = facc[i][r];
                            }
                        }
                    } else if constexpr (MODE == 0) {
                        using Acc = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, kSg>;
                        Acc acc[MT];
#pragma unroll
                        for (int i = 0; i < MT; ++i) jm::joint_matrix_fill(sg, acc[i], 0.f);
                        for (int k0 = 0; k0 < K; k0 += 32) {
                            float v[32];
                            prefetch_decode<TYPE>(base, wrow, K, k0, cur, v);
                            sycl::half* my = reinterpret_cast<sycl::half*>(sm) + (size_t)(sgid * kSg + lane) * kLdb;
                            for (int j = 0; j < 32; ++j) my[j] = sycl::half(v[j]);
                            auto* As = reinterpret_cast<sycl::vec<uint16_t, 8>*>(sm + a_off);
                            for (int r = tid; r < rows_here; r += cols) {
                                const auto* src = reinterpret_cast<const sycl::vec<uint16_t, 8>*>(
                                    X + ((size_t)(row0 + rbase + r) * K + k0));
                                As[(r * kLda) / 8 + 0] = src[0];
                                As[(r * kLda) / 8 + 1] = src[1];
                                As[(r * kLda) / 8 + 2] = src[2];
                                As[(r * kLda) / 8 + 3] = src[3];
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            for (int kk = 0; kk < 32; kk += 16) {
                                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, kSg, jm::layout::col_major> bm;
                                auto bp = as_local<sycl::half>(sm + (size_t)sgid * kSg * kLdb * sizeof(sycl::half) + kk * sizeof(sycl::half));
                                jm::joint_matrix_load(sg, bm, bp, kLdb);
#pragma unroll
                                for (int i = 0; i < MT; ++i) {
                                    if (i * 8 < rows_pad) {
                                        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> am;
                                        auto ap = as_local<sycl::half>(sm + a_off + ((size_t)(i * 8) * kLda + kk) * sizeof(sycl::half));
                                        jm::joint_matrix_load(sg, am, ap, kLda);
                                        jm::joint_matrix_mad(sg, acc[i], am, bm, acc[i]);
                                    }
                                }
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        auto yp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                            Y + (size_t)(row0 + rbase) * N);
#pragma unroll
                        for (int i = 0; i < MT; ++i) {
                            if (i * 8 < rows_here) {
                                jmi::joint_matrix_store_checked(sg, acc[i], yp, (size_t)N, jm::layout::row_major, (size_t)rows_here,
                                                                (size_t)N, (size_t)(i * 8), (size_t)(n0 + sgid * kSg));
                            }
                        }
                    } else {
                        using Acc = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, kSg>;
                        Acc acc[MT];
#pragma unroll
                        for (int i = 0; i < MT; ++i) jm::joint_matrix_fill(sg, acc[i], 0.f);
                        for (int k0 = 0; k0 < K; k0 += 32) {
                            float v[32];
                            prefetch_decode<TYPE>(base, wrow, K, k0, cur, v);
                            // unsigned short is matrix_type::uint16, which B70 rejects.
                            // bfloat16 is matrix_type::bf16 and has the same 16-bit layout.
                            using bf16 = sycl::ext::oneapi::bfloat16;
                            uint16_t* my = reinterpret_cast<uint16_t*>(sm) + (size_t)(sgid * kSg + lane) * kLdb;
                            for (int j = 0; j < 32; ++j) my[j] = b70_fp32_to_bf16(v[j]);
                            auto* As = reinterpret_cast<sycl::vec<uint16_t, 8>*>(sm + a_off);
                            for (int r = tid; r < rows_here; r += cols) {
                                const auto* src = reinterpret_cast<const sycl::vec<uint16_t, 8>*>(
                                    X + ((size_t)(row0 + rbase + r) * K + k0));
                                As[(r * kLda) / 8 + 0] = src[0];
                                As[(r * kLda) / 8 + 1] = src[1];
                                As[(r * kLda) / 8 + 2] = src[2];
                                As[(r * kLda) / 8 + 3] = src[3];
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            for (int kk = 0; kk < 32; kk += 16) {
                                jm::joint_matrix<sycl::sub_group, bf16, jm::use::b, 16, kSg, jm::layout::col_major> bm;
                                auto bp = as_local<bf16>(sm + (size_t)sgid * kSg * kLdb * 2 + (size_t)kk * 2);
                                jm::joint_matrix_load(sg, bm, bp, kLdb);
#pragma unroll
                                for (int i = 0; i < MT; ++i) {
                                    if (i * 8 < rows_pad) {
                                        jm::joint_matrix<sycl::sub_group, bf16, jm::use::a, 8, 16, jm::layout::row_major> am;
                                        auto ap = as_local<bf16>(sm + a_off + ((size_t)(i * 8) * kLda + kk) * 2);
                                        jm::joint_matrix_load(sg, am, ap, kLda);
                                        jm::joint_matrix_mad(sg, acc[i], am, bm, acc[i]);
                                    }
                                }
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        auto yp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                            Y + (size_t)(row0 + rbase) * N);
#pragma unroll
                        for (int i = 0; i < MT; ++i) {
                            if (i * 8 < rows_here) {
                                jmi::joint_matrix_store_checked(sg, acc[i], yp, (size_t)N, jm::layout::row_major, (size_t)rows_here,
                                                                (size_t)N, (size_t)(i * 8), (size_t)(n0 + sgid * kSg));
                            }
                        }
                    }
                }
            });
    });
}

// fp16 (mode 0) is built for every Flash-Next type at MT=4 and MT=16.
// bf16 and int8 are built only at MT=16, and only for the types the bench
// measures. Callers force mt=16 when mode!=0 so the SLM layout matches.
// B70_PREFILL_SMOKE builds IQ3_XXS and Q2_0, fp16 only, so a compile error
// in the kernel body shows up before the full instantiation set.
template <int MODE, int MT>
void dispatch_fp16_all(sycl::queue& q, int type, const GemmArgs& a) {
    switch (type) {
    case B70_Q4_K: launch_gemm<B70_Q4_K, MT, MODE>(q, a); break;
    case B70_Q5_K: launch_gemm<B70_Q5_K, MT, MODE>(q, a); break;
    case B70_Q6_K: launch_gemm<B70_Q6_K, MT, MODE>(q, a); break;
    case B70_IQ2_XXS: launch_gemm<B70_IQ2_XXS, MT, MODE>(q, a); break;
    case B70_IQ2_XS: launch_gemm<B70_IQ2_XS, MT, MODE>(q, a); break;
    case B70_IQ2_S: launch_gemm<B70_IQ2_S, MT, MODE>(q, a); break;
    case B70_IQ3_XXS: launch_gemm<B70_IQ3_XXS, MT, MODE>(q, a); break;
    case B70_IQ3_S: launch_gemm<B70_IQ3_S, MT, MODE>(q, a); break;
    case B70_IQ4_NL: launch_gemm<B70_IQ4_NL, MT, MODE>(q, a); break;
    case B70_IQ4_XS: launch_gemm<B70_IQ4_XS, MT, MODE>(q, a); break;
    case B70_Q2_0: launch_gemm<B70_Q2_0, MT, MODE>(q, a); break;
    default: throw std::runtime_error("b70 prefill: unsupported ggml type");
    }
}

template <int MODE>
void dispatch_eval16(sycl::queue& q, int type, const GemmArgs& a) {
    switch (type) {
    case B70_Q4_K: launch_gemm<B70_Q4_K, 16, MODE>(q, a); break;
    case B70_Q5_K: launch_gemm<B70_Q5_K, 16, MODE>(q, a); break;
    case B70_Q6_K: launch_gemm<B70_Q6_K, 16, MODE>(q, a); break;
    case B70_IQ3_XXS: launch_gemm<B70_IQ3_XXS, 16, MODE>(q, a); break;
    case B70_IQ3_S: launch_gemm<B70_IQ3_S, 16, MODE>(q, a); break;
    case B70_IQ4_NL: launch_gemm<B70_IQ4_NL, 16, MODE>(q, a); break;
    case B70_IQ4_XS: launch_gemm<B70_IQ4_XS, 16, MODE>(q, a); break;
    case B70_Q2_0: launch_gemm<B70_Q2_0, 16, MODE>(q, a); break;
    default: throw std::runtime_error("b70 prefill: bf16/int8 is built for Q4_K Q5_K Q6_K IQ3_XXS IQ3_S IQ4_NL IQ4_XS Q2_0 only");
    }
}

void run_gemm(sycl::queue& q, int type, int mode, int mt, const GemmArgs& a) {
#ifdef B70_PREFILL_SMOKE
    if (mode != 0 || (type != B70_IQ3_XXS && type != B70_Q2_0)) {
        throw std::runtime_error("b70 prefill smoke build: fp16 IQ3_XXS and Q2_0 only");
    }
    if (type == B70_IQ3_XXS) {
        if (mt == 4) launch_gemm<B70_IQ3_XXS, 4, 0>(q, a);
        else launch_gemm<B70_IQ3_XXS, 16, 0>(q, a);
    } else if (mt == 4) {
        launch_gemm<B70_Q2_0, 4, 0>(q, a);
    } else {
        launch_gemm<B70_Q2_0, 16, 0>(q, a);
    }
#else
    if (mode == 1 || mode == 2) {
        g_split.mt = 16;
        if (mode == 1) dispatch_eval16<1>(q, type, a);
        else dispatch_eval16<2>(q, type, a);
    } else if (mt == 4) {
        dispatch_fp16_all<0, 4>(q, type, a);
    } else {
        dispatch_fp16_all<0, 16>(q, type, a);
    }
#endif
}

class b70_pf_hist;
class b70_pf_scan;
class b70_pf_scatter;
class b70_pf_gather;
class b70_pf_swiglu;
class b70_pf_reduce;
class b70_pf_dense_starts;

void sort_ids(sycl::queue& q, const int32_t* ids, int32_t* counts, int32_t* starts, int32_t* cursor, int32_t* order, int M,
              int E) {
    q.memset(counts, 0, sizeof(int32_t) * (size_t)E);
    const int wg = 256;
    const int groups = M < wg ? 1 : (M + wg - 1) / wg;
    const int gcap = groups > 1024 ? 1024 : groups;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> lc(E, h);
        h.parallel_for<b70_pf_hist>(sycl::nd_range<1>((size_t)gcap * wg, wg), [=](sycl::nd_item<1> it) {
            const int tid = (int)it.get_local_id(0);
            int* bin = lc.get_multi_ptr<sycl::access::decorated::no>().get();
            for (int e = tid; e < E; e += wg) bin[e] = 0;
            it.barrier(sycl::access::fence_space::local_space);
            const int gs = (int)it.get_global_range(0);
            for (int i = (int)it.get_global_id(0); i < M; i += gs) {
                const int e = ids[i];
                if ((unsigned)e < (unsigned)E) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        ar(bin[e]);
                    ar.fetch_add(1);
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int e = tid; e < E; e += wg) {
                const int c = bin[e];
                if (c) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        ag(counts[e]);
                    ag.fetch_add(c);
                }
            }
        });
    });
    q.single_task<b70_pf_scan>([=]() {
        int sum = 0;
        starts[0] = 0;
        for (int e = 0; e < E; ++e) {
            cursor[e] = sum;
            sum += counts[e];
            starts[e + 1] = sum;
        }
    });
    q.parallel_for<b70_pf_scatter>(sycl::range<1>((size_t)((M + 255) & ~255)), [=](sycl::item<1> it) {
        const int i = (int)it.get_id(0);
        if (i >= M) return;
        const int e = ids[i];
        if ((unsigned)e >= (unsigned)E) return;
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ag(cursor[e]);
        const int pos = ag.fetch_add(1);
        order[pos] = i;
    });
}

void gather_rows(sycl::queue& q, const uint16_t* src, uint16_t* dst, const int32_t* order, const int32_t* starts, int K,
                 int E, int id_div, int M) {
    const int wg = 128;
    const int groups = M > 0 ? M : 1;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_pf_gather>(sycl::nd_range<1>((size_t)groups * wg, wg), [=](sycl::nd_item<1> it) {
            const int pos = (int)it.get_group(0);
            const int mvalid = starts[E];
            if (pos >= mvalid) return;
            const int tid = (int)it.get_local_id(0);
            const int vecs = K / 8;
            int src_row = order[pos];
            if (id_div > 1) src_row /= id_div;
            const auto* in = reinterpret_cast<const sycl::vec<uint16_t, 8>*>(src + (size_t)src_row * K);
            auto* out = reinterpret_cast<sycl::vec<uint16_t, 8>*>(dst + (size_t)pos * K);
            for (int v = tid; v < vecs; v += wg) out[v] = in[v];
        });
    });
}

void scatter_rows2(sycl::queue& q, const float* src, float* dst, const int32_t* order, const int32_t* starts, int N, int E,
                   int groups_hint) {
    const int wg = 256;
    const int groups = groups_hint > 0 ? groups_hint : 1;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<class b70_pf_yscatter2>(sycl::nd_range<1>((size_t)groups * wg, wg), [=](sycl::nd_item<1> it) {
            const int pos = (int)it.get_group(0);
            const int mvalid = starts[E];
            if (pos >= mvalid) return;
            const int dst_row = order[pos];
            const int tid = (int)it.get_local_id(0);
            const float* in = src + (size_t)pos * N;
            float* out = dst + (size_t)dst_row * N;
            for (int c = tid; c < N; c += wg) out[c] = in[c];
        });
    });
}

void swiglu_kernel(sycl::queue& q, const float* gu, uint16_t* h, const int32_t* starts, int E, int M, int F, int mode) {
    const int wg = 128;
    q.submit([&](sycl::handler& hnd) {
        hnd.parallel_for<b70_pf_swiglu>(sycl::nd_range<1>((size_t)M * wg, wg), [=](sycl::nd_item<1> it) {
            const int row = (int)it.get_group(0);
            if (row >= starts[E]) return;
            const int tid = (int)it.get_local_id(0);
            for (int f = tid; f < F; f += wg) {
                const float g = gu[(size_t)row * (2 * F) + (size_t)(2 * f)];
                const float u = gu[(size_t)row * (2 * F) + (size_t)(2 * f + 1)];
                const float sig = 1.f / (1.f + sycl::exp(-g));
                const float y = g * sig * u;
                if (mode == 1) h[(size_t)row * F + f] = b70_fp32_to_bf16(y);
                else h[(size_t)row * F + f] = sycl::bit_cast<uint16_t>(sycl::half(y));
            }
        });
    });
}

void reduce_kernel(sycl::queue& q, const float* slot, float* out, const float* rw, int T, int topk, int H) {
    const int wg = 256;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_pf_reduce>(sycl::nd_range<1>((size_t)T * wg, wg), [=](sycl::nd_item<1> it) {
            const int t = (int)it.get_group(0);
            const int tid = (int)it.get_local_id(0);
            for (int c = tid; c < H; c += wg) {
                float s = 0.f;
                for (int k = 0; k < topk; ++k) {
                    const float w = rw ? rw[(size_t)t * topk + k] : 1.f;
                    s = sycl::fma(slot[((size_t)t * topk + k) * H + c], w, s);
                }
                out[(size_t)t * H + c] = s;
            }
        });
    });
}

using Clock = std::chrono::steady_clock;
void tick_begin(sycl::queue& q, bool on, Clock::time_point& t0) {
    if (!on) return;
    q.wait();
    t0 = Clock::now();
}
void tick_end(sycl::queue& q, bool on, Clock::time_point t0, double& dst) {
    if (!on) return;
    q.wait();
    dst += std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

at::Tensor cast_x(const at::Tensor& x, int mode) {
    if (mode == 1) return x.scalar_type() == at::kBFloat16 ? x : x.to(at::kBFloat16);
    return x.scalar_type() == at::kHalf ? x : x.to(at::kHalf);
}

void check_xpu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_xpu(), name, " must be an XPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void fill_dense_starts(sycl::queue& q, int32_t* starts, int M) {
    q.single_task<b70_pf_dense_starts>([=]() {
        starts[0] = 0;
        starts[1] = M;
    });
}

}  // namespace b70pf

using namespace b70pf;

at::Tensor b70_prefill_grouped_op(at::Tensor x, at::Tensor q, at::Tensor expert_ids, int64_t ggml_type, int64_t N,
                                  int64_t K, int64_t mode) {
    check_xpu(x, "x");
    check_xpu(q, "q");
    check_xpu(expert_ids, "expert_ids");
    TORCH_CHECK(mode >= 0 && mode <= 2, "mode must be 0 fp16, 1 bf16, or 2 int8");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K, "x must be [M, K]");
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt && expert_ids.dim() == 1, "expert_ids must be int32 [M]");
    TORCH_CHECK(expert_ids.size(0) == x.size(0), "expert_ids length must match M");
    TORCH_CHECK(q.scalar_type() == at::kByte, "q must be uint8");
    const int rb = b70_row_bytes((int)ggml_type, (int)K);
    TORCH_CHECK(rb > 0, "bad type or K is not a multiple of the block");
    TORCH_CHECK(N % 64 == 0 && K % 32 == 0, "need N%64==0 and K%32==0");
    TORCH_CHECK(q.numel() % (N * rb) == 0, "q numel is not E*N*row_bytes");
    const int E = (int)(q.numel() / (N * rb));
    const int M = (int)x.size(0);
    TORCH_CHECK(E > 0 && E <= 4096 && M > 0, "bad E or M");
    at::Tensor xf = cast_x(x, (int)mode);
    auto y = at::zeros({M, N}, xf.options().dtype(at::kFloat));
    auto counts = at::empty({E}, xf.options().dtype(at::kInt));
    auto starts = at::empty({E + 1}, xf.options().dtype(at::kInt));
    auto cursor = at::empty({E}, xf.options().dtype(at::kInt));
    auto order = at::empty({M}, xf.options().dtype(at::kInt));
    auto xpack = at::empty({M, K}, xf.options());
    auto ypack = at::empty({M, N}, y.options());
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    const bool on = split_on();
    g_split = Split{};
    g_split.mode = (int)mode;
    auto t0 = Clock::now();
    Clock::time_point ts;
    const int mt = mode != 0 ? 16 : pick_mt(M, E);
    GemmArgs args;
    args.X = static_cast<const uint16_t*>(xpack.data_ptr());
    args.gate = q.data_ptr<uint8_t>();
    args.up = nullptr;
    args.starts = starts.data_ptr<int32_t>();
    args.Y = ypack.data_ptr<float>();
    args.E = E;
    args.N = (int)N;
    args.K = (int)K;
    args.rb = rb;
    args.n_wrow = (int)N;
    args.gu = 0;
    args.lay = choose_layout(queue, (int)N, mt, (int)mode);
    g_split.cols = args.lay.cols;
    g_split.mt = mt;
    tick_begin(queue, on, ts);
    sort_ids(queue, expert_ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
             cursor.data_ptr<int32_t>(), order.data_ptr<int32_t>(), M, E);
    tick_end(queue, on, ts, g_split.sort);
    tick_begin(queue, on, ts);
    gather_rows(queue, static_cast<const uint16_t*>(xf.data_ptr()), static_cast<uint16_t*>(xpack.data_ptr()),
                order.data_ptr<int32_t>(), starts.data_ptr<int32_t>(), (int)K, E, 1, M);
    tick_end(queue, on, ts, g_split.gather);
    tick_begin(queue, on, ts);
    run_gemm(queue, (int)ggml_type, (int)mode, mt, args);
    tick_end(queue, on, ts, g_split.gemm);
    tick_begin(queue, on, ts);
    scatter_rows2(queue, ypack.data_ptr<float>(), y.data_ptr<float>(), order.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
                  (int)N, E, M);
    tick_end(queue, on, ts, g_split.reduce);
    if (on) {
        g_split.total = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        std::fprintf(stderr, "PREFILL_SPLIT grouped sort=%.1f gather=%.1f gemm=%.1f scatter=%.1f total=%.1f cols=%d mt=%d mode=%d\n",
                     g_split.sort, g_split.gather, g_split.gemm, g_split.reduce, g_split.total, g_split.cols, g_split.mt,
                     g_split.mode);
    }
    return y;
}

at::Tensor b70_prefill_dense_op(at::Tensor x, at::Tensor q, int64_t ggml_type, int64_t N, int64_t K, int64_t mode) {
    check_xpu(x, "x");
    check_xpu(q, "q");
    TORCH_CHECK(mode >= 0 && mode <= 2, "mode must be 0 fp16, 1 bf16, or 2 int8");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K, "x must be [M, K]");
    TORCH_CHECK(q.scalar_type() == at::kByte, "q must be uint8");
    const int rb = b70_row_bytes((int)ggml_type, (int)K);
    TORCH_CHECK(rb > 0, "bad type or K");
    TORCH_CHECK(N % 64 == 0 && K % 32 == 0, "need N%64==0 and K%32==0");
    TORCH_CHECK(q.numel() >= N * rb, "q is shorter than N*row_bytes");
    const int M = (int)x.size(0);
    TORCH_CHECK(M > 0, "M");
    at::Tensor xf = cast_x(x, (int)mode);
    auto y = at::empty({M, N}, xf.options().dtype(at::kFloat));
    auto starts = at::empty({2}, xf.options().dtype(at::kInt));
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    fill_dense_starts(queue, starts.data_ptr<int32_t>(), M);
    const bool on = split_on();
    g_split = Split{};
    g_split.mode = (int)mode;
    const int mt = mode != 0 ? 16 : pick_mt(M, 1);
    GemmArgs args;
    args.X = static_cast<const uint16_t*>(xf.data_ptr());
    args.gate = q.data_ptr<uint8_t>();
    args.up = nullptr;
    args.starts = starts.data_ptr<int32_t>();
    args.Y = y.data_ptr<float>();
    args.E = 1;
    args.N = (int)N;
    args.K = (int)K;
    args.rb = rb;
    args.n_wrow = (int)N;
    args.gu = 0;
    args.lay = choose_layout(queue, (int)N, mt, (int)mode);
    g_split.cols = args.lay.cols;
    g_split.mt = mt;
    auto t0 = Clock::now();
    Clock::time_point ts;
    tick_begin(queue, on, ts);
    run_gemm(queue, (int)ggml_type, (int)mode, mt, args);
    tick_end(queue, on, ts, g_split.gemm);
    if (on) {
        g_split.total = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        std::fprintf(stderr, "PREFILL_SPLIT dense gemm=%.1f total=%.1f cols=%d mt=%d mode=%d M=%d N=%lld K=%lld\n",
                     g_split.gemm, g_split.total, g_split.cols, g_split.mt, g_split.mode, M, (long long)N, (long long)K);
    }
    return y;
}

at::Tensor b70_prefill_moe_layer_op(at::Tensor x, at::Tensor ids, at::Tensor q_gate, at::Tensor q_up, at::Tensor q_down,
                                    at::Tensor router, int64_t type_gu, int64_t type_down, int64_t H, int64_t F,
                                    int64_t mode) {
    check_xpu(x, "x");
    check_xpu(ids, "ids");
    check_xpu(q_gate, "q_gate");
    check_xpu(q_up, "q_up");
    check_xpu(q_down, "q_down");
    TORCH_CHECK(mode >= 0 && mode <= 2, "mode");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == H, "x must be [T, H]");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 2 && ids.size(0) == x.size(0), "ids [T, topk] int32");
    const int T = (int)x.size(0);
    const int topk = (int)ids.size(1);
    const int M = T * topk;
    const int rb_gu = b70_row_bytes((int)type_gu, (int)H);
    const int rb_dn = b70_row_bytes((int)type_down, (int)F);
    TORCH_CHECK(rb_gu > 0 && rb_dn > 0, "bad expert type or shape");
    TORCH_CHECK(H % 64 == 0 && F % 32 == 0 && (2 * F) % 64 == 0, "H%64, F%32, 2F%64");
    TORCH_CHECK(q_gate.numel() % (F * rb_gu) == 0, "gate pack");
    const int E = (int)(q_gate.numel() / (F * rb_gu));
    TORCH_CHECK(q_up.numel() == q_gate.numel(), "up pack must match gate");
    TORCH_CHECK(q_down.numel() == (int64_t)E * H * rb_dn, "down pack");
    TORCH_CHECK(E > 0 && E <= 4096 && M > 0, "bad E or M");
    const float* rw = nullptr;
    at::Tensor rw_hold;
    if (router.defined() && router.numel() > 0) {
        check_xpu(router, "router");
        TORCH_CHECK(router.scalar_type() == at::kFloat && router.numel() == M, "router must be fp32 [T, topk]");
        rw = router.data_ptr<float>();
        rw_hold = router;
    }
    at::Tensor xf = cast_x(x, (int)mode);
    auto counts = at::empty({E}, xf.options().dtype(at::kInt));
    auto starts = at::empty({E + 1}, xf.options().dtype(at::kInt));
    auto cursor = at::empty({E}, xf.options().dtype(at::kInt));
    auto order = at::empty({M}, xf.options().dtype(at::kInt));
    auto xpack = at::empty({M, H}, xf.options());
    auto gu = at::empty({M, 2 * F}, xf.options().dtype(at::kFloat));
    auto h = at::empty({M, F}, xf.options());
    auto ydown = at::empty({M, H}, xf.options().dtype(at::kFloat));
    auto slot = at::zeros({M, H}, xf.options().dtype(at::kFloat));
    auto y = at::empty({T, H}, xf.options().dtype(at::kFloat));
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    const bool on = split_on();
    g_split = Split{};
    g_split.mode = (int)mode;
    const int mt = mode != 0 ? 16 : pick_mt(M, E);
    auto t0 = Clock::now();
    Clock::time_point ts;
    tick_begin(queue, on, ts);
    sort_ids(queue, ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
             cursor.data_ptr<int32_t>(), order.data_ptr<int32_t>(), M, E);
    tick_end(queue, on, ts, g_split.sort);
    tick_begin(queue, on, ts);
    gather_rows(queue, static_cast<const uint16_t*>(xf.data_ptr()), static_cast<uint16_t*>(xpack.data_ptr()),
                order.data_ptr<int32_t>(), starts.data_ptr<int32_t>(), (int)H, E, topk, M);
    tick_end(queue, on, ts, g_split.gather);
    GemmArgs gu_args;
    gu_args.X = static_cast<const uint16_t*>(xpack.data_ptr());
    gu_args.gate = q_gate.data_ptr<uint8_t>();
    gu_args.up = q_up.data_ptr<uint8_t>();
    gu_args.starts = starts.data_ptr<int32_t>();
    gu_args.Y = gu.data_ptr<float>();
    gu_args.E = E;
    gu_args.N = (int)(2 * F);
    gu_args.K = (int)H;
    gu_args.rb = rb_gu;
    gu_args.n_wrow = (int)F;
    gu_args.gu = 1;
    gu_args.lay = choose_layout(queue, (int)(2 * F), mt, (int)mode);
    g_split.cols = gu_args.lay.cols;
    g_split.mt = mt;
    tick_begin(queue, on, ts);
    run_gemm(queue, (int)type_gu, (int)mode, mt, gu_args);
    tick_end(queue, on, ts, g_split.gemm);
    tick_begin(queue, on, ts);
    swiglu_kernel(queue, gu.data_ptr<float>(), static_cast<uint16_t*>(h.data_ptr()), starts.data_ptr<int32_t>(), E, M,
                  (int)F, (int)mode);
    tick_end(queue, on, ts, g_split.swiglu);
    GemmArgs dn;
    dn.X = static_cast<const uint16_t*>(h.data_ptr());
    dn.gate = q_down.data_ptr<uint8_t>();
    dn.up = nullptr;
    dn.starts = starts.data_ptr<int32_t>();
    dn.Y = ydown.data_ptr<float>();
    dn.E = E;
    dn.N = (int)H;
    dn.K = (int)F;
    dn.rb = rb_dn;
    dn.n_wrow = (int)H;
    dn.gu = 0;
    dn.lay = choose_layout(queue, (int)H, mt, (int)mode);
    tick_begin(queue, on, ts);
    run_gemm(queue, (int)type_down, (int)mode, mt, dn);
    tick_end(queue, on, ts, g_split.gemm2);
    tick_begin(queue, on, ts);
    scatter_rows2(queue, ydown.data_ptr<float>(), slot.data_ptr<float>(), order.data_ptr<int32_t>(),
                  starts.data_ptr<int32_t>(), (int)H, E, M);
    reduce_kernel(queue, slot.data_ptr<float>(), y.data_ptr<float>(), rw, T, topk, (int)H);
    tick_end(queue, on, ts, g_split.reduce);
    if (on) {
        g_split.total = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        std::fprintf(stderr,
                     "PREFILL_SPLIT layer sort=%.1f gather=%.1f gu=%.1f swiglu=%.1f down=%.1f reduce=%.1f total=%.1f cols=%d mt=%d mode=%d T=%d\n",
                     g_split.sort, g_split.gather, g_split.gemm, g_split.swiglu, g_split.gemm2, g_split.reduce, g_split.total,
                     g_split.cols, g_split.mt, g_split.mode, T);
    }
    return y;
}

at::Tensor b70_prefill_split_us_op() {
    auto t = at::empty({8}, at::TensorOptions().dtype(at::kFloat));
    float* p = t.data_ptr<float>();
    p[0] = (float)g_split.sort;
    p[1] = (float)g_split.gather;
    p[2] = (float)g_split.gemm;
    p[3] = (float)g_split.gemm2;
    p[4] = (float)g_split.swiglu;
    p[5] = (float)g_split.reduce;
    p[6] = (float)g_split.total;
    p[7] = (float)(g_split.cols * 1000 + g_split.mt);
    return t;
}
