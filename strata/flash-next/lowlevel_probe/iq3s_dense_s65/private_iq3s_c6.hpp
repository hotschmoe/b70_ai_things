// S65 private fixed IQ3_S C6 oracle; source pin and transformations in extraction.json.
// Main source and global Split21/expert dispatch remain untouched.
#pragma once
#include "strata/sycl_math.hpp"
#include "strata/kernels/dp4a.hpp"
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include <type_traits>
namespace s65 {
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
#ifndef STRATA_SWAR
#define STRATA_SWAR 1   // 1: the SWAR forms; 0: dpct's per-byte loops (A/B 2026-09-30: both correct)
#endif
#if STRATA_SWAR
__dpct_inline__ int swar_ne4(unsigned x) {
    x = (x | (x >> 4)) & 0x0F0F0F0Fu;
    x = (x | (x >> 2)) & 0x03030303u;
    x = (x | (x >> 1)) & 0x01010101u;
    return (int) (x * 0xFFu);
}
__dpct_inline__ int swar_sub4(unsigned a, unsigned s) {
    return (int) (((a & 0x7F7F7F7Fu) + (s & 0x01010101u)) ^ (a & 0x80808080u));
}
#else
__dpct_inline__ int swar_ne4(unsigned x) { return dpct::vectorized_binary<sycl::uchar4>(x, 0, std::not_equal_to<>()); }
__dpct_inline__ int swar_sub4(unsigned a, unsigned s) { return dpct::vectorized_binary<sycl::uchar4>(a, s, std::minus<>()); }
#endif
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
struct CandidateSplit : ControlSplit {
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
            const int signs0 = swar_ne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                    ((signs_packed_8[l0 / 2] & 0x0C) << 21));
            const int signs1 = swar_ne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                    ((signs_packed_8[l0 / 2] & 0xC0) << 17));
            r.g[l0 + 0] = swar_sub4(grid_pos.x() ^ signs0, signs0);
            r.g[l0 + 1] = swar_sub4(grid_pos.y() ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = sycl::vec<sycl::half, 1>(bq3->d)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        return r;
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
    using S = std::conditional_t<FAST, CandidateSplit, ControlSplit>;
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
template<bool FAST> class PrivateC6;
template<bool FAST> void launch(const void* W, const void* X, float* Y, int K, int M, sycl::queue& q) {
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
#undef ggml_cuda_dp4a
} // namespace s65
