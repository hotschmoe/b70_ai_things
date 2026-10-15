// S98 PRIVATE IQ2_XS routed GU original/multi oracle, SOURCE ONLY.
// Exact extracted source and transformations are pinned in extraction.json.
#pragma once
#include "strata/sycl_math.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dp4a.hpp"
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include <type_traits>
namespace s98 {
using strata::kernels::NativeExpertLayout;
static_assert(sizeof(block_iq2_xs) == 74 && sizeof(block_q8_1) == 36);
constexpr int kExpertLanes = 8;
#define ggml_cuda_dp4a(a, b, c) strata::dp4a((a), (b), (c))
#ifndef STRATA_GRID_SLM
#define STRATA_GRID_SLM 0
#endif
__dpct_inline__ int get_int_b2(const void *x, const int &i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__dpct_inline__ int get_int_b4(const void *x, const int &i32) {
    return ((const int *)x)[i32];
}
__dpct_inline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = sycl::popcount(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
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
__dpct_inline__ float vec_dot_iq2_xs_q8_1(const void *__restrict__ vbq,
                                          const block_q8_1 *__restrict__ bq8_1,
                                          const int &kbx, const int &iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    // SYCL port: the four 16-bit codes by shifts. ggml reads them through a uint16_t* to an int2, which is type
    // punning the SYCL device compiler does not honour (non-char aliasing): IQ2_XS products came out garbage
    // (iq_multi_parity, Swift 1.5 IQ2_XS) while the dequantizer, reading the codes directly, was exact.
    const uint32_t q2_lo = (uint32_t) get_int_b2(bq2->qs, iqs + 0), q2_hi = (uint32_t) get_int_b2(bq2->qs, iqs + 1);
    const uint16_t q2[4] = {(uint16_t) q2_lo, (uint16_t) (q2_lo >> 16), (uint16_t) q2_hi, (uint16_t) (q2_hi >> 16)};
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = swar_ne4(signs & 0x08040201);
        const int grid_l = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = swar_ne4(signs & 0x80402010);
        const int grid_h = swar_sub4(grid_pos.y() ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = sycl::vec<sycl::half, 1>(bq2->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    bq8_1[iqs / 2].ds[0];
    return d * sumi;
}
template<int TY> struct Fmt;
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template <int LANES>
__dpct_inline__ float lanes_sum(float v) {
#pragma unroll
    for (int o = LANES / 2; o > 0; o >>= 1)
        v += dpct::experimental::permute_sub_group_by_xor(0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
    return v;
}
template <int TY, int LANES>
__dpct_inline__ float row_dot_lanes(const uint8_t *row, const block_q8_1 *x, int nb, int sub) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = sub; k < nb * F::ipb; k += LANES) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return lanes_sum<LANES>(s);
}
struct MultiW { int w[8]; int a = 0, b = 0; float d = 0.f; };
template<int TY> struct Multi;
template<> struct Multi<17> {
    static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* = nullptr) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    // SYCL port: the four 16-bit codes by shifts. ggml reads them through a uint16_t* to an int2, which is type
    // punning the SYCL device compiler does not honour (non-char aliasing): IQ2_XS products came out garbage
    // (iq_multi_parity, Swift 1.5 IQ2_XS) while the dequantizer, reading the codes directly, was exact.
    const uint32_t q2_lo = (uint32_t) get_int_b2(bq2->qs, iqs + 0), q2_hi = (uint32_t) get_int_b2(bq2->qs, iqs + 1);
    const uint16_t q2[4] = {(uint16_t) q2_lo, (uint16_t) (q2_lo >> 16), (uint16_t) q2_hi, (uint16_t) (q2_hi >> 16)};
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const sycl::uint2 grid_pos =
            ((const sycl::uint2 *)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = swar_ne4(signs & 0x08040201);
        const int grid_l = swar_sub4(grid_pos.x() ^ signs0, signs0);
        const int signs1 = swar_ne4(signs & 0x80402010);
        const int grid_h = swar_sub4(grid_pos.y() ^ signs1, signs1);
        m.w[l0 + 0] = grid_l;
        m.w[l0 + 1] = grid_h;
    }
        m.a = ls0;
        m.b = ls1;
        m.d = sycl::vec<sycl::half, 1>(bq2->d)
                  .convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        const int sumi = (s0 * m.a + s1 * m.b + (s0 + s1) / 2) / 4;
        const float d = m.d * ds;
        return d * sumi;
    }
};
template <int TY, int LANES, int E>
__dpct_inline__ void row_dot_multi_iq(const uint8_t* row, const block_q8_1* const* xs, int nb, int sub, float* out,
                                      const void* grid = nullptr) {
    using F = Fmt<TY>;
    using M = Multi<TY>;
    float s[E];
#pragma unroll
    for (int e = 0; e < E; ++e) s[e] = 0.f;
    for (int k = sub; k < nb * F::ipb; k += LANES) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        MultiW m;
        M::prep(row, kbx, iqs, m, grid);
#pragma unroll
        for (int e = 0; e < E; ++e) {
            int u[8];
            float ds;
            M::acts(xs[e] + kbx * (F::qk / 32), iqs, u, ds);
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int j = 0; j < M::NW / 2; ++j) s0 = ggml_cuda_dp4a(m.w[j], u[j], s0);
#pragma unroll
            for (int j = M::NW / 2; j < M::NW; ++j) s1 = ggml_cuda_dp4a(m.w[j], u[j], s1);
            s[e] += M::finish(s0, s1, m, ds);
        }
    }
#pragma unroll
    for (int e = 0; e < E; ++e) out[e] = lanes_sum<LANES>(s[e]);
}
template<int TY, int LANES, int E>
__dpct_inline__ void row_dot_multi(const uint8_t* row, const block_q8_1* const* xs,
                                  int nb, int sub, float* out, const void* grid = nullptr) {
    static_assert(TY == 17 && LANES == 8 && (E == 2 || E == 4));
    row_dot_multi_iq<TY, LANES, E>(row, xs, nb, sub, out, grid);
}
template <int TY, int LANES, bool USE_MULTI = false>
__dpct_inline__ void row_entries(const uint8_t* wr, const block_q8_1* x, int x_stride, const int32_t* ent_idx,
                                 int e0, int e1, int nb, int sub, float* dst, size_t dst_stride,
                                 const void* grid = nullptr) {
    int e = e0;
    if constexpr (USE_MULTI) {
        for (; e + 4 <= e1; e += 4) {
            const block_q8_1* xs[4] = {x + (size_t) ent_idx[e] * x_stride, x + (size_t) ent_idx[e + 1] * x_stride,
                                       x + (size_t) ent_idx[e + 2] * x_stride, x + (size_t) ent_idx[e + 3] * x_stride};
            float o[4];
            row_dot_multi<TY, LANES, 4>(wr, xs, nb, sub, o, grid);
            if (sub == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) dst[(size_t) (e + i) * dst_stride] = o[i];
        }
        if (e + 2 <= e1) {
            const block_q8_1* xs[2] = {x + (size_t) ent_idx[e] * x_stride, x + (size_t) ent_idx[e + 1] * x_stride};
            float o[2];
            row_dot_multi<TY, LANES, 2>(wr, xs, nb, sub, o, grid);
            if (sub == 0) { dst[(size_t) e * dst_stride] = o[0]; dst[(size_t) (e + 1) * dst_stride] = o[1]; }
            e += 2;
        }
    }
    for (; e < e1; ++e) {
        const float v = row_dot_lanes<TY, LANES>(wr, x + (size_t) ent_idx[e] * x_stride, nb, sub);
        if (sub == 0) dst[(size_t) e * dst_stride] = v;
    }
}
template <int TG, int LN = kExpertLanes, bool USE_MULTI = false>
__dpct_inline__ void private_gu_kernel(
    const unsigned long long *__restrict__ grp_ptr,
    const int32_t *__restrict__ grp_start, const int32_t *__restrict__ n_groups,
    const int32_t *__restrict__ ent_tok, const block_q8_1 *__restrict__ xq,
    NativeExpertLayout L, float *__restrict__ gate, float *__restrict__ up) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    // #363: group g is computed by block row g % gridDim.y (a launch sized for fewer rows than possible groups strides);
    // a block row with no group returns before the barrier below (uniform over the work-group)
    const int ng = *n_groups;
    if ((int) item_ct1.get_group(1) >= ng) return;
    const void* grid = nullptr;
#if STRATA_GRID_SLM
    if constexpr (TG == 18 || TG == 21 || TG == 22) {
        using GT = std::conditional_t<TG == 22, uint64_t, uint32_t>;
        constexpr int GN = TG == 22 ? 1024 : TG == 21 ? 512 : 256;
        auto& sg = *sycl::ext::oneapi::group_local_memory_for_overwrite<GT[GN]>(item_ct1.get_group());
        for (int i = (int) item_ct1.get_local_id(2); i < GN; i += (int) item_ct1.get_local_range(2)) {
            if constexpr (TG == 22) sg[i] = iq2s_grid[i];
            else if constexpr (TG == 21) sg[i] = iq3s_grid[i];
            else sg[i] = iq3xxs_grid[i];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        grid = &sg[0];
    }
#endif
    const int rib = item_ct1.get_local_id(2) / LN,
              sub = item_ct1.get_local_id(2) % LN;
    const int row = item_ct1.get_group(2) * (256 / LN) + rib; // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    for (int g = (int) item_ct1.get_group(1); g < ng; g += (int) item_ct1.get_group_range(1)) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        row_entries<TG, LN, USE_MULTI>(wr, xq, xb, ent_tok, e0, e1, nb, sub, (is_up ? up : gate) + r, (size_t) L.n_ff, grid);
    }
}
template<bool USE_MULTI> class PrivateGu17;
template<bool USE_MULTI>
inline void launch(const unsigned long long* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_tok,
                   const block_q8_1* xq, NativeExpertLayout L, float* gate,
                   float* up, unsigned grid_groups, void* stream) {
    constexpr int ROWS = 256 / 8;
    const unsigned gx = (unsigned)((2 * L.n_ff + ROWS - 1) / ROWS);
    auto* q = strata::q_of(stream);
    auto props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    q->parallel_for<PrivateGu17<USE_MULTI>>(
        sycl::nd_range<3>(sycl::range<3>(1, grid_groups, (size_t)gx * 256),
                          sycl::range<3>(1, 1, 256)), props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            private_gu_kernel<17, 8, USE_MULTI>(grp_ptr, grp_start, n_groups,
                                               ent_tok, xq, L, gate, up);
        });
}
inline void launch_original(const unsigned long long* p, const int32_t* s,
                            const int32_t* n, const int32_t* t,
                            const block_q8_1* x, NativeExpertLayout L,
                            float* g, float* u, unsigned grids, void* stream) {
    launch<false>(p, s, n, t, x, L, g, u, grids, stream);
}
inline void launch_multi(const unsigned long long* p, const int32_t* s,
                         const int32_t* n, const int32_t* t,
                         const block_q8_1* x, NativeExpertLayout L,
                         float* g, float* u, unsigned grids, void* stream) {
    launch<true>(p, s, n, t, x, L, g, u, grids, stream);
}
#undef ggml_cuda_dp4a
} // namespace s98
