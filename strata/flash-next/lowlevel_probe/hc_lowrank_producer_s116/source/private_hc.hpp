// S116 SOURCE ONLY. Four exact selected control bodies and two narrow candidates.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <utility>
#include <cstdint>
#include "strata/kernels/fused_gr.hpp"
namespace s116_private {
using strata::kernels::FusedGrArgs;
using strata::kernels::kFusedGrMaxT;
constexpr int N=2560, HC=4, D=N*HC, LR=320, THREADS=256, WARPS=8;
constexpr int DOWN_SPLIT=4, UPM_COLS=16;
__dpct_inline__ float s116_warp_sum(float v) {
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
__dpct_inline__ float s116_sigmoid(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

// 8 bf16 packed in a uint4 against 8 floats.
__dpct_inline__ float s116_dot8(const sycl::uint4 w, const float *x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x(), w.y(), w.z(), w.w()};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = sycl::fma(sycl::bit_cast<float>(v[j] << 16), (float)(x[2 * j]),
                        acc);
        acc = sycl::fma(sycl::bit_cast<float>(v[j] & 0xffff0000u),
                        (float)(x[2 * j + 1]), acc);
    }
    return acc;
}
struct HCMultiS116 {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
    float* part;   ///< [DOWN_SPLIT][kFusedGrMaxT][LR] partial row sums of the down kernel
    int nsplit = DOWN_SPLIT;   ///< SYCL port: splits the up kernel sums (1 after the sliced down kernel's reduce)
    float* part2 = nullptr;    ///< SYCL port: [GRS_SLICES][kFusedGrMaxT][LR + HC] slice partials of the sliced kernel
};

// SYCL port (Arc): a RUNTIME index into the by-value kernel argument (m.a[k]) makes IGC copy the whole HCMultiS116
// (~1.2 KB) into every work-item's private scratch - unitrace: ~40 KB private per SIMD32 thread on the A770 for
// gr_up_multi / gr_down_reduce / gr_norm_split. S116_GR_PICK selects ONE field with compile-time indices only (a chain
// of selects over uniform kernel arguments), so nothing is copied. Addressing only: values and orders unchanged.
template <class F, std::size_t... J>
__dpct_inline__ auto s116_pick_impl(const HCMultiS116& m, int k, F f, std::index_sequence<J...>) {
    auto v = f(m.a[0]);
    ((v = (k == (int) J) ? f(m.a[J]) : v), ...);
    return v;
}
#define S116_GR_PICK(m, k, field) \
    s116_pick_impl((m), (k), [](const FusedGrArgs& x_) { return x_.field; }, std::make_index_sequence<kFusedGrMaxT>{})

__dpct_inline__ void s116_norm_split(HCMultiS116 m) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto& part = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WARPS]>(item.get_group());
    auto& s_rs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(item.get_group());
    const int tok = (int) item.get_group(2) / HC, c = (int) item.get_group(2) % HC;
    const float* const aR = S116_GR_PICK(m, tok, R);
    const bool aapply = S116_GR_PICK(m, tok, apply);
    const float* const abo = S116_GR_PICK(m, tok, bo_prev);
    const float* const awn = S116_GR_PICK(m, tok, w_norm);
    const float aeps = S116_GR_PICK(m, tok, eps);
    float* const ars = S116_GR_PICK(m, tok, rs);
    float* xn = m.xn + (size_t) tok * D;
    const int t = (int) item.get_local_id(2), lane = t & 31, warp = t >> 5;
    const float gw = aapply ? 2.0f * s116_sigmoid(S116_GR_PICK(m, tok, inj_prev)[c] / (float) HC) : 0.0f;
    constexpr int MAXK = (N + THREADS * 4 - 1) / (THREADS * 4) + 1;   // 3: float4s of one stream per thread
    sycl::float4 keep[MAXK];
    int kept = 0;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;
        const int d = i - c * N;
        sycl::float4 r = *reinterpret_cast<const sycl::float4*>(aR + i);
        if (aapply) {
            const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(abo + d);
            r.x() = sycl::fma((float) (b.x()), gw, r.x());
            r.y() = sycl::fma((float) (b.y()), gw, r.y());
            r.z() = sycl::fma((float) (b.z()), gw, r.z());
            r.w() = sycl::fma((float) (b.w()), gw, r.w());
        }
        const sycl::float4 g = *reinterpret_cast<const sycl::float4*>(awn + i);
        ss += r.x() * r.x() + r.y() * r.y() + r.z() * r.z() + r.w() * r.w();
        if (kept < MAXK) keep[kept++] = sycl::float4(r.x() * g.x(), r.y() * g.y(), r.z() * g.z(), r.w() * g.w());
    }
    const float v = s116_warp_sum(ss);
    if (lane == 0) part[warp] = v;
    item.barrier(sycl::access::fence_space::local_space);
    if (t == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum += part[w];
        s_rs = sycl::rsqrt(sum / (float) N + aeps);
        ars[c] = s_rs;
    }
    item.barrier(sycl::access::fence_space::local_space);
    const float rs = s_rs;
    int k = 0;
    for (int i = t * 4; i < D && k < kept; i += THREADS * 4) {
        if (i / N != c) continue;
        const sycl::float4 x = keep[k++];
        *reinterpret_cast<sycl::float4*>(xn + i) = sycl::float4(x.x() * rs, x.y() * rs, x.z() * rs, x.w() * rs);
    }
}
constexpr int GRS_COLS = 128, GRS_SLICES = D / GRS_COLS, GRS_ROWS = LR + HC;   // 80 slices, 324 rows
constexpr int GRS_RB = (GRS_ROWS + WARPS * 8 - 1) / (WARPS * 8);                // 6 blocks of 64 rows
// MORE (gr_more_groups): one work-group per (slice, row block) - 480 groups instead of 80; every row's sum is the same
template <bool MORE = false>
__dpct_inline__ void s116_down_sliced(HCMultiS116 m, float* tile) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = (int) item.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int g = (int) item.get_group(2);
    const int T = m.T, sl = MORE ? g / GRS_RB : g, c0 = sl * GRS_COLS;
    const int r_first = MORE ? (g % GRS_RB) * WARPS * 8 : 0, r_end = MORE ? r_first + WARPS * 8 : GRS_ROWS;
    for (int i = t; i < T * GRS_COLS; i += THREADS) {
        const int k = i / GRS_COLS, c = i - k * GRS_COLS;
        tile[i] = m.xn[(size_t) k * D + c0 + c];
    }
    item.barrier(sycl::access::fence_space::local_space);
    const int rsub = lane >> 2, q = lane & 3;
    const bool inj = m.a[0].w_inject != nullptr;
    for (int r0 = r_first; r0 < r_end; r0 += WARPS * 8) {
        const int r = r0 + warp * 8 + rsub;
        const bool ok = r < LR || (r < GRS_ROWS && inj);
        const uint16_t* wrow = r < LR ? m.a[0].w_down + (size_t) r * D : m.a[0].w_inject + (size_t) (ok ? r - LR : 0) * D;
        sycl::uint4 wv[4];
#pragma unroll
        for (int cc = 0; cc < 4; ++cc)
            wv[cc] = ok ? reinterpret_cast<const sycl::uint4*>(wrow + c0)[q + 4 * cc] : sycl::uint4(0, 0, 0, 0);
        float acc[kFusedGrMaxT];
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            acc[k] = 0.0f;
            if (k < T) {
#pragma unroll
                for (int cc = 0; cc < 4; ++cc) acc[k] += s116_dot8(wv[cc], tile + k * GRS_COLS + (q + 4 * cc) * 8);
            }
        }
        auto sg = item.get_sub_group();
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float v = acc[k];
            v += sycl::permute_group_by_xor(sg, v, 1);
            v += sycl::permute_group_by_xor(sg, v, 2);
            if (q == 0 && ok) m.part2[((size_t) sl * kFusedGrMaxT + k) * GRS_ROWS + r] = v;
        }
    }
}
template <bool MORE = false>
__dpct_inline__ void s116_control_reduce(HCMultiS116 m) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = (int) item.get_global_id(2);
    if (i >= m.T * GRS_ROWS) return;
    const int k = i / GRS_ROWS, r = i - k * GRS_ROWS;
    if (r >= LR && m.a[0].w_inject == nullptr) return;
    float x = 0.0f;
    if constexpr (MORE) {
        // 8 loads in flight per step (the one-at-a-time loop was load-latency bound on the A770), then added one by
        // one: x += slice 0, 1, 2, ... 79, the same sequence of float adds as the loop below (-fp-model=precise)
        const float* src = m.part2 + (size_t) k * GRS_ROWS + r;
        constexpr size_t SL_STRIDE = (size_t) kFusedGrMaxT * GRS_ROWS;
        for (int sl = 0; sl < GRS_SLICES; sl += 8) {
            float pv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) pv[j] = src[(size_t) (sl + j) * SL_STRIDE];
#pragma unroll
            for (int j = 0; j < 8; ++j) x += pv[j];
        }
    } else {
        for (int sl = 0; sl < GRS_SLICES; ++sl) x += m.part2[((size_t) sl * kFusedGrMaxT + k) * GRS_ROWS + r];
    }
    if (r < LR) m.part[(size_t) k * LR + r] = x;   // split 0
    else S116_GR_PICK(m, k, inject_out)[r - LR] = x;
}
template <int COLS = UPM_COLS, bool ILP = false, bool LO_ONCE = false>   // columns per block: 16 (the original), 8 or 4 - each row's dot, butterfly and the per-column
__dpct_inline__ void s116_control_up(HCMultiS116 m) {   // stream sum unchanged, so the same bits; more blocks, fewer serial rows per warp
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[kFusedGrMaxT][LR]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kFusedGrMaxT][HC][COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = item_ct1.get_group(2) * COLS;
#pragma unroll
    for (int i = t; i < T * LR; i += THREADS) {   // the down kernel's partial sums, in split order, then silu
        const int k = i / LR, r = i % LR;
        float x = 0.0f;
#pragma unroll
        for (int sp = 0; sp < DOWN_SPLIT; ++sp) if (sp < m.nsplit) x += m.part[((size_t) sp * kFusedGrMaxT + k) * LR + r];
        x /= (float) HC;
        const float v = x / (1.0f + sycl::native::exp(-x));
        lo[k][r] = v;
        if (!LO_ONCE || item_ct1.get_group(2) == 0) S116_GR_PICK(m, k, lo)[r] = v;   // the same values in every block
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // this lane's token (lane < T), picked once - not per row
    const int pk = lane < T ? lane : 0;
    const float* const pR = S116_GR_PICK(m, pk, R);
    const float* const pWn = S116_GR_PICK(m, pk, w_norm);
    const float* const pRs = S116_GR_PICK(m, pk, rs);
    const bool pApply = S116_GR_PICK(m, pk, apply);
    const float* const pBo = S116_GR_PICK(m, pk, bo_prev);
    const float* const pIp = S116_GR_PICK(m, pk, inj_prev);
    float* const pRout = S116_GR_PICK(m, pk, R_out);
    for (int r = warp; r < HC * COLS; r += WARPS) {
        const int c = r / COLS, dd = r - c * COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(m.a[0].w_up + (size_t)i * LR);
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wa = *(w4 + lane);
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wb =
            lane < LR / 8 - 32 ? *(w4 + 32 + lane) : sycl::uint4(0, 0, 0, 0);
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            rv = pR[i];
            wn = pWn[i];
            rsc = pRs[c];
            apply = pApply;
            if (apply) { bo = pBo[d0 + dd]; ip = pIp[c]; }
        }
        float mine = 0.0f;
        if constexpr (ILP) {   // STRATA_GR_UP_ILP: the T dots first, then their butterflies interleaved step by step -
            float accs[kFusedGrMaxT];   // each value sees s116_warp_sum's shuffles in s116_warp_sum's order: the same bits
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k) {
                float acc = 0.0f;
                if (k < T) {
                    acc = s116_dot8(wa, lo[k] + lane * 8);
                    if (lane < LR / 8 - 32) acc += s116_dot8(wb, lo[k] + (32 + lane) * 8);
                }
                accs[k] = acc;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                for (int k = 0; k < kFusedGrMaxT; ++k)
                    if (k < T) accs[k] += dpct::experimental::permute_sub_group_by_xor(
                                   0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), accs[k], o);
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k) if (k < T && lane == k) mine = accs[k];
        } else {
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = s116_dot8(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += s116_dot8(wb, lo[k] + (32 + lane) * 8);
            acc = s116_warp_sum(acc);
            if (lane == k) mine = acc;
        }
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * s116_sigmoid(ip / (float)HC), rv);
                pRout[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * s116_sigmoid(mine);
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * COLS; i += THREADS) {
        const int k = i / COLS, col = i - k * COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        S116_GR_PICK(m, k, mixed)[d0 + col] = s / (float) HC;
    }
}

template <bool MORE = false>
__dpct_inline__ void s116_candidate_reduce(HCMultiS116 m) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = (int) item.get_global_id(2);
    if (i >= m.T * GRS_ROWS) return;
    const int k = i / GRS_ROWS, r = i - k * GRS_ROWS;
    if (r >= LR && m.a[0].w_inject == nullptr) return;
    float x = 0.0f;
    if constexpr (MORE) {
        // 8 loads in flight per step (the one-at-a-time loop was load-latency bound on the A770), then added one by
        // one: x += slice 0, 1, 2, ... 79, the same sequence of float adds as the loop below (-fp-model=precise)
        const float* src = m.part2 + (size_t) k * GRS_ROWS + r;
        constexpr size_t SL_STRIDE = (size_t) kFusedGrMaxT * GRS_ROWS;
        for (int sl = 0; sl < GRS_SLICES; sl += 8) {
            float pv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) pv[j] = src[(size_t) (sl + j) * SL_STRIDE];
#pragma unroll
            for (int j = 0; j < 8; ++j) x += pv[j];
        }
    } else {
        for (int sl = 0; sl < GRS_SLICES; ++sl) x += m.part2[((size_t) sl * kFusedGrMaxT + k) * GRS_ROWS + r];
    }
    if (r < LR) {
        m.part[(size_t) k * LR + r] = x;   // preserve raw split 0 exactly
        // Keep the UP expression's NEW +0 and source division. No volatile.
        // A compiler may forward this FP32 load; later ISA must prove that the
        // epilogue sees the exact 32-bit operand of the preserved raw store.
        float y = 0.0f;
        y += m.part[(size_t) k * LR + r];
        y /= (float) HC;
        const float v = y / (1.0f + sycl::native::exp(-y));
        S116_GR_PICK(m, k, lo)[r] = v;
    } else S116_GR_PICK(m, k, inject_out)[r - LR] = x;
}
template <int COLS = UPM_COLS, bool ILP = false, bool LO_ONCE = false>   // columns per block: 16 (the original), 8 or 4 - each row's dot, butterfly and the per-column
__dpct_inline__ void s116_candidate_up(HCMultiS116 m) {   // stream sum unchanged, so the same bits; more blocks, fewer serial rows per warp
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &lo = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[kFusedGrMaxT][LR]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &g = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kFusedGrMaxT][HC][COLS]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const int d0 = item_ct1.get_group(2) * COLS;
#pragma unroll
    for (int i = t; i < T * LR; i += THREADS) {   // the down kernel's partial sums, in split order, then silu
        const int k = i / LR, r = i % LR;
        lo[k][r] = S116_GR_PICK(m, k, lo)[r];   // completed producer result, same SLM/barrier
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // this lane's token (lane < T), picked once - not per row
    const int pk = lane < T ? lane : 0;
    const float* const pR = S116_GR_PICK(m, pk, R);
    const float* const pWn = S116_GR_PICK(m, pk, w_norm);
    const float* const pRs = S116_GR_PICK(m, pk, rs);
    const bool pApply = S116_GR_PICK(m, pk, apply);
    const float* const pBo = S116_GR_PICK(m, pk, bo_prev);
    const float* const pIp = S116_GR_PICK(m, pk, inj_prev);
    float* const pRout = S116_GR_PICK(m, pk, R_out);
    for (int r = warp; r < HC * COLS; r += WARPS) {
        const int c = r / COLS, dd = r - c * COLS, i = c * N + d0 + dd;
        const sycl::uint4 *w4 =
            reinterpret_cast<const sycl::uint4 *>(m.a[0].w_up + (size_t)i * LR);
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wa = *(w4 + lane);
        /*
        DPCT1098: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        const sycl::uint4 wb =
            lane < LR / 8 - 32 ? *(w4 + 32 + lane) : sycl::uint4(0, 0, 0, 0);
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            rv = pR[i];
            wn = pWn[i];
            rsc = pRs[c];
            apply = pApply;
            if (apply) { bo = pBo[d0 + dd]; ip = pIp[c]; }
        }
        float mine = 0.0f;
        if constexpr (ILP) {   // STRATA_GR_UP_ILP: the T dots first, then their butterflies interleaved step by step -
            float accs[kFusedGrMaxT];   // each value sees s116_warp_sum's shuffles in s116_warp_sum's order: the same bits
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k) {
                float acc = 0.0f;
                if (k < T) {
                    acc = s116_dot8(wa, lo[k] + lane * 8);
                    if (lane < LR / 8 - 32) acc += s116_dot8(wb, lo[k] + (32 + lane) * 8);
                }
                accs[k] = acc;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                for (int k = 0; k < kFusedGrMaxT; ++k)
                    if (k < T) accs[k] += dpct::experimental::permute_sub_group_by_xor(
                                   0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), accs[k], o);
#pragma unroll
            for (int k = 0; k < kFusedGrMaxT; ++k) if (k < T && lane == k) mine = accs[k];
        } else {
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
            if (k >= T) break;
            float acc = s116_dot8(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += s116_dot8(wb, lo[k] + (32 + lane) * 8);
            acc = s116_warp_sum(acc);
            if (lane == k) mine = acc;
        }
        }
        if (lane < T) {
            if (apply) {
                rv = sycl::fma(bo, 2.0f * s116_sigmoid(ip / (float)HC), rv);
                pRout[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * s116_sigmoid(mine);
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (int i = t; i < T * COLS; i += THREADS) {
        const int k = i / COLS, col = i - k * COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        S116_GR_PICK(m, k, mixed)[d0 + col] = s / (float) HC;
    }
}


// Exactly six private, disjoint kernel types. No other HC variant materialized.
class HCNormS116; class HCDownS116; class HCControlReduceS116;
class HCControlUpS116; class HCCandidateReduceS116; class HCCandidateUpS116;
inline void reduce_only(sycl::queue& q,HCMultiS116 m,bool candidate);
inline void prefix(sycl::queue& q,HCMultiS116 m,bool candidate) {
    q.parallel_for<dpct_kernel_name<HCNormS116>>(
        sycl::nd_range<3>(sycl::range(1,1,m.T*HC)*sycl::range(1,1,THREADS),sycl::range(1,1,THREADS)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {s116_norm_split(m);});
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float,1> tile(sycl::range<1>(kFusedGrMaxT*GRS_COLS),cgh);
        cgh.parallel_for<dpct_kernel_name<HCDownS116>>(
            sycl::nd_range<3>(sycl::range(1,1,GRS_SLICES)*sycl::range(1,1,THREADS),sycl::range(1,1,THREADS)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {s116_down_sliced<false>(m,tile.get_multi_ptr<sycl::access::decorated::no>().get());});
    });
    reduce_only(q,m,candidate);
}
inline void reduce_only(sycl::queue& q,HCMultiS116 m,bool candidate) {
    const unsigned nr=unsigned((m.T*GRS_ROWS+127)/128);
    const sycl::nd_range<3> red(sycl::range(1,1,nr)*sycl::range(1,1,128),sycl::range(1,1,128));
    if(candidate)q.parallel_for<dpct_kernel_name<HCCandidateReduceS116>>(red,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {s116_candidate_reduce<false>(m);});
    else q.parallel_for<dpct_kernel_name<HCControlReduceS116>>(red,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {s116_control_reduce<false>(m);});
}
inline void up(sycl::queue& q,HCMultiS116 m,bool candidate) {
    const sycl::nd_range<3> nr(sycl::range(1,1,N/16)*sycl::range(1,1,THREADS),sycl::range(1,1,THREADS));
    auto props=sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
    if(candidate)q.parallel_for<dpct_kernel_name<HCCandidateUpS116>>(nr,props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {s116_candidate_up<16,false,true>(m);});
    else q.parallel_for<dpct_kernel_name<HCControlUpS116>>(nr,props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {s116_control_up<16,false,true>(m);});
}
inline void whole(sycl::queue& q,HCMultiS116 m,bool candidate){prefix(q,m,candidate);up(q,m,candidate);}
#undef S116_GR_PICK
} // namespace s116_private
