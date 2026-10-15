// S75 private canonical-Q8 IQ3_S C6 DPAS oracle; ControlSplit is S65 original.
// Main source and global Split21/expert dispatch remain untouched.
#pragma once
#include "strata/sycl_math.hpp"
#include "strata/kernels/dp4a.hpp"
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include <type_traits>
#include <cstdint>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <stdexcept>
namespace s75 {
#define ggml_cuda_dp4a(a, b, c) strata::dp4a((a), (b), (c))
__dpct_inline__ int get_int_b2(const void *x, const int &i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__dpct_inline__ int get_int_b4(const void *x, const int &i32) {
    return ((const int *)x)[i32];
}
struct ControlSplit {   // IQ3_S
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const sycl::int2 qs_packed = sycl::int2(get_int_b2(bq3->qs, iqs + 0),
                                                get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const sycl::int2 grid_pos =
                sycl::int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                           iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x03) << 7) |
                    ((signs_packed_8[l0 / 2] & 0x0C) << 21),
                0x00000000, std::not_equal_to<>());
            const int signs1 = dpct::vectorized_binary<sycl::uchar4>(
                ((signs_packed_8[l0 / 2] & 0x30) << 3) |
                    ((signs_packed_8[l0 / 2] & 0xC0) << 17),
                0x00000000, std::not_equal_to<>());
            r.g[l0 + 0] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.x() ^ signs0, signs0, std::minus<>());
            r.g[l0 + 1] = dpct::vectorized_binary<sycl::uchar4>(
                grid_pos.y() ^ signs1, signs1, std::minus<>());
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = sycl::vec<sycl::half, 1>(bq3->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi *= r.ls;
        const float d = r.dw * bq8_1[iqs / 2].ds[0];
        return d * sumi;
    }
};
__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}
template<int TY> struct Fmt;
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2; };
template<int TY> inline constexpr bool kStageIqGrid = (TY == 16 || TY == 17 || TY == 22 || TY == 29);
template <int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false, bool FAST = false>
__dpct_inline__ void
row_dot_multi(const uint8_t *row, const block_q8_1 *x, const int (&off)[NC],
              int n, int nb, int lane, float (&s)[NC],
              const uint32_t *__restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = ControlSplit;
    static_assert(!FAST);
    static_assert(TY == 21 && NC == 6 && EXACT_N && !STAGE_GRID);
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        /*
        DPCT1084: The function call "Split::load" has multiple migration
        results in different template instantiations that could not be unified.
        You may need to adjust the code.
        */
        const typename S::W w =
            S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (EXACT_N || c < n) s[c] = warp_sum(s[c]);
}
template <int TY, bool STAGE_GRID = kStageIqGrid<TY>>
__dpct_inline__ const uint32_t *
stage_iq_grid(uint32_t *s_buf, int tid, int nthreads) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if constexpr (!STAGE_GRID) {
        return nullptr;
    } else if constexpr (TY == 16) {
        sycl::uint2 *dst = (sycl::uint2 *)s_buf;
        const sycl::uint2 *src = (const sycl::uint2 *)iq2xxs_grid;
#pragma unroll
        for (int i = tid; i < 256; i += nthreads) dst[i] = src[i];
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        return s_buf;
    } else if constexpr (TY == 17) {
        sycl::uint2 *dst = (sycl::uint2 *)s_buf;
        const sycl::uint2 *src = (const sycl::uint2 *)iq2xs_grid;
#pragma unroll
        for (int i = tid; i < 512; i += nthreads) dst[i] = src[i];
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        return s_buf;
    } else if constexpr (TY == 22) {
        sycl::uint2 *dst = (sycl::uint2 *)s_buf;
        const sycl::uint2 *src = (const sycl::uint2 *)iq2s_grid;
#pragma unroll
        for (int i = tid; i < 1024; i += nthreads) dst[i] = src[i];
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        return s_buf;
    } else if constexpr (TY == 29) {
        sycl::uint2 *dst = (sycl::uint2 *)s_buf;
        const sycl::uint2 *src = (const sycl::uint2 *)iq1s_grid_gpu;
#pragma unroll
        for (int i = tid; i < 1024; i += nthreads) dst[i] = src[i];
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        return s_buf;
    } else {
        return nullptr;
    }
}

template<int TY, bool STAGE_GRID = kStageIqGrid<TY>>
struct IqGridWords {
    static constexpr int value = !STAGE_GRID ? 4 : (TY == 22 || TY == 29) ? 2048 : (TY == 17) ? 1024 : (TY == 16) ? 512 : 4;
};
template <int TY, int NC, bool STAGE_GRID = kStageIqGrid<TY>,
          bool EXACT_N = false, bool FAST = false>
__dpct_inline__ void
mmvq_multi_kernel(const uint8_t *__restrict__ w, size_t row_bytes,
                  const block_q8_1 *__restrict__ x, float *__restrict__ y,
                  int n_in, int n_out, int ncols) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_grid_buf = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    uint32_t[IqGridWords<TY, STAGE_GRID>::value]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const uint32_t *s_grid = stage_iq_grid<TY, STAGE_GRID>(
        s_grid_buf, item_ct1.get_local_id(1) * 32 + item_ct1.get_local_id(2),
        128);
    const int row = item_ct1.get_group(2) * 4 + item_ct1.get_local_id(1);
    if (row >= n_out) return;
    const int lane = item_ct1.get_local_id(2);
    const int nb = n_in / Fmt<TY>::qk, xb = n_in / 32;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    if constexpr (EXACT_N) {
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = c * xb;
        float s[NC];
        row_dot_multi<TY, NC, true, STAGE_GRID, FAST>(wr, x, off, NC, nb, lane, s, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (lane == c) y[(size_t) c * n_out + row] = s[c];
    } else {
        for (int c0 = 0; c0 < ncols; c0 += NC) {
            const int n = sycl::min(NC, ncols - c0);
            int off[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c)
                off[c] = (c0 + sycl::min(c, n - 1)) * xb;
            float s[NC];
            row_dot_multi<TY, NC, false, STAGE_GRID, FAST>(wr, x, off, n, nb, lane, s, s_grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n && lane == c) y[(size_t) (c0 + c) * n_out + row] = s[c];
        }
    }
}

// C6 DPAS: one SG16/workgroup, 16 output columns, padded token tile8.
// Matrix logical coordinates are (token c, output-within-tile r).
class DpasC6;
inline void launch_dpas(const void* W, const void* X, float* Y,
                        int K, int M, sycl::queue& q) {
    if (!((K == 2560 && M == 6144) || (K == 6144 && M == 2560)))
        throw std::invalid_argument("S75 fixed two-shape C6 geometry required");
    constexpr int CT = 8, C = 6, RT = 16, KT = 32;
    const int xb = K / KT;
    const size_t rb = size_t(K / 256) * 110;
    const auto* weights = static_cast<const uint8_t*>(W);
    const auto* x = static_cast<const block_q8_1*>(X);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<std::int8_t, 1> a(CT * KT, h), b(RT * KT, h);
        sycl::local_accessor<float, 1> dw(RT, h), partial(C * RT * 32, h);
        sycl::local_accessor<int, 1> ls(RT, h);
        sycl::local_accessor<std::int32_t, 1> tile(CT * RT, h);
        sycl::local_accessor<sycl::half, 1> dx(C, h);
        h.parallel_for<DpasC6>(
            sycl::nd_range<1>(sycl::range<1>(size_t(M)), sycl::range<1>(RT)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                namespace jm = sycl::ext::oneapi::experimental::matrix;
                namespace im = sycl::ext::intel::experimental::matrix;
                const auto sg = it.get_sub_group();
                const int lid = int(it.get_local_linear_id());
                const int r0 = int(it.get_group_linear_id()) * RT;
                // Each shared ledger cell has exactly one initializer.
                for (int i = lid; i < C * RT * 32; i += RT) partial[i] = 0.0f;
                it.barrier(sycl::access::fence_space::local_space);
                // Uniform p loop: every subgroup member executes every matrix
                // load/fill/mad/apply and every workgroup barrier.
                for (int p = 0; p < xb; ++p) {
                    const auto decoded = ControlSplit::load<false>(
                        weights + size_t(r0 + lid) * rb, p / 8, 2 * (p % 8));
                    dw[lid] = decoded.dw;
                    ls[lid] = decoded.ls;
                    // B is column-major KT x RT: contiguous K for each output.
                    // Only unpack existing signed bytes; IQ math is ControlSplit.
                    for (int k = 0; k < KT; ++k) {
                        const unsigned u = (unsigned(decoded.g[k / 4]) >> (8 * (k % 4))) & 255u;
                        b[lid * KT + k] = std::int8_t(u < 128u ? int(u) : int(u) - 256);
                    }
                    for (int i = lid; i < CT * KT; i += RT) {
                        const int c = i / KT, k = i % KT;
                        a[i] = c < C ? x[c * xb + p].qs[k] : std::int8_t(0);
                    }
                    if (lid < C) dx[lid] = x[lid * xb + p].ds[0];
                    // Publish A/B/scales before collective local matrix loads.
                    it.barrier(sycl::access::fence_space::local_space);
                    jm::joint_matrix<sycl::sub_group, std::int8_t, jm::use::a,
                                     CT, KT, jm::layout::row_major> ma;
                    jm::joint_matrix<sycl::sub_group, std::int8_t, jm::use::b,
                                     KT, RT, jm::layout::col_major> mb;
                    jm::joint_matrix<sycl::sub_group, std::int32_t,
                                     jm::use::accumulator, CT, RT> mc;
                    jm::joint_matrix_load(sg, ma,
                        a.template get_multi_ptr<sycl::access::decorated::no>(), KT);
                    jm::joint_matrix_load(sg, mb,
                        b.template get_multi_ptr<sycl::access::decorated::no>(), KT);
                    jm::joint_matrix_fill(sg, mc, 0);
                    jm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    // Pure coordinate transform on fragment elements only. No
                    // shared ledger writes or inferred per-WI matrix ownership.
                    im::joint_matrix_apply(sg, mc,
                        [&](std::int32_t& value, size_t, size_t r) {
                            value *= ls[r];
                        });
                    jm::joint_matrix_store(sg, mc,
                        tile.template get_multi_ptr<sycl::access::decorated::no>(),
                        RT, jm::layout::row_major);
                    // Publish the scaled INT32 tile before explicit WI owners
                    // read it. Each WI owns exactly one output-column ledger.
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int c = 0; c < C; ++c) {
                        const int sumi = tile[c * RT + lid];
                        const float d = dw[lid] * dx[c];
                        const float v = d * sumi;
                        partial[(c * RT + lid) * 32 + (p % 32)] += v;
                    }
                    // Finish all tile/scales reads before next iteration
                    // overwrites A/B/scales. Also orders ledger p and p+32.
                    it.barrier(sycl::access::fence_space::local_space);
                }
                // Last iteration's bottom barrier publishes the complete ledger.
                // WI lid owns output column r0+lid, and writes only six tokens.
                for (int c = 0; c < C; ++c) {
                    const int base = (c * RT + lid) * 32;
                    float tree[16];
                    // Native mmvq stores lane=c. This permutation preserves its
                    // operand order, not just lane0's unordered reduction tree.
                    for (int i = 0; i < 16; ++i)
                        tree[i] = partial[base + (i ^ c)] + partial[base + ((i + 16) ^ c)];
                    for (int offset = 8; offset > 0; offset >>= 1)
                        for (int i = 0; i < offset; ++i) tree[i] += tree[i + offset];
                    Y[size_t(c) * M + r0 + lid] = tree[0];
                }
            });
    });
}
template<bool FAST> class PrivateC6;
template<bool FAST> void launch(const void* W, const void* X, float* Y, int K, int M, sycl::queue& q) {
    if constexpr (FAST) {
        launch_dpas(W, X, Y, K, M, q);
    } else {
    const dpct::dim3 grid((unsigned)((M + 3) / 4)), block(32, 4);
    const size_t rb = size_t(K / 256) * 110;
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<PrivateC6<FAST>>(
            sycl::nd_range<3>(grid * block, block), exp_props,
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                mmvq_multi_kernel<21, 6, false, true, FAST>(
                    static_cast<const uint8_t*>(W), rb,
                    static_cast<const block_q8_1*>(X), Y, K, M, 6);
            });
    });
    }
}
#undef ggml_cuda_dp4a
} // namespace s75
