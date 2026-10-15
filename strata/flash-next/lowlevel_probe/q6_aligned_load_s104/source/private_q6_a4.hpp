#pragma once
// SOURCE ONLY S104: canonical Q6 C6/A2/RPW1, weight load reconstruction only.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/dp4a.hpp"
#include "strata/sycl_math.hpp"
#include <cstdint>
#include <cstddef>
#include <stdexcept>
namespace s104 {
static_assert(sizeof(int)==4);
constexpr int QK=256,Q8K=32,WARP=32,WARPS=4;
struct Q6KBlock {uint8_t ql[128];uint8_t qh[64];int8_t scales[16];sycl::half d;};
struct Q81Block {sycl::half2 ds;int8_t qs[32];};
static_assert(sizeof(Q6KBlock)==210 && alignof(Q6KBlock)==2 && offsetof(Q6KBlock,d)==208);
static_assert(sizeof(Q81Block)==36 && offsetof(Q81Block,qs)==4);
__dpct_inline__ sycl::int4 load16_a2(const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const sycl::int4* q = reinterpret_cast<const sycl::int4*>(a & ~uintptr_t(15));
    const int k = int(a >> 2) & 3;          // dword offset within the first aligned chunk
    const bool half = (a & 2) != 0;          // and a 2-byte shift (the address is always 2-byte aligned)
    // the second chunk only when the address is unaligned: then it holds needed bytes, so (16-byte chunks never
    // straddle a page) it cannot fault even at the very end of an allocation
    const sycl::int4 lo = q[0], hi = (a & 15) ? q[1] : lo;
    const uint32_t d[8] = {(uint32_t) lo.x(), (uint32_t) lo.y(), (uint32_t) lo.z(), (uint32_t) lo.w(),
                           (uint32_t) hi.x(), (uint32_t) hi.y(), (uint32_t) hi.z(), (uint32_t) hi.w()};
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 5; ++i)              // w = d[k .. k+4] by selects (no indexed register access)
        w[i] = k == 0 ? d[i] : k == 1 ? d[i + 1] : k == 2 ? d[i + 2] : d[(i + 3) & 7];
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = half ? (w[i] >> 16) | (w[i + 1] << 16) : w[i];
    return sycl::int4((int) o[0], (int) o[1], (int) o[2], (int) o[3]);
}
// A4 scalar reads are 4-byte aligned, never misaligned int4 dereferences.
// Host launch guard requires base%4==0 and the two complete Q6 shapes.
__dpct_inline__ sycl::int4 load16_a4(const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const int* q = reinterpret_cast<const int*>(a & ~uintptr_t(3));
    const bool half = (a & 2) != 0;
    const uint32_t d[5] = {uint32_t(q[0]), uint32_t(q[1]), uint32_t(q[2]), uint32_t(q[3]), uint32_t(q[4])};
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = half ? (d[i] >> 16) | (d[i + 1] << 16) : d[i];
    return sycl::int4((int) o[0], (int) o[1], (int) o[2], (int) o[3]);
}
template<bool CAND>
void kernel(const Q6KBlock* __restrict__ w, const Q81Block* __restrict__ x,
                                 float* __restrict__ y, int n_in, int n_out) {
    constexpr int NCOLS=6,RPW=1,SG=32,BS=sizeof(Q6KBlock); constexpr bool A2=true;
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    constexpr int NWG = WARPS * WARP / SG;              // sub-groups per work-group
    const int row0 = (int(item.get_group(2)) * NWG + warp) * RPW;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;            // position group within the block, block within the iteration
    const int vh_shift = 2 * ((g & 3) >> 1);
    const int qh4_idx = 2 * (g >> 2) + (g & 1);
    const int scale_offset = 8 * (g >> 2) + (g & 3);
    const int bq8_offset = 4 * (g >> 2) + ((g & 3) >> 1);
    const int u_int4 = g & 1;                           // ints 4(g%2)..+3 of the q8 block = one int4
    float acc[RPW][NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += SG / 8) {
        const int kby = kbx * (QK / Q8K);
        sycl::int4 ql4[RPW], qh4[RPW]; float dsc0[RPW], dsc1[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const Q6KBlock* b = reinterpret_cast<const Q6KBlock*>(reinterpret_cast<const uint8_t*>(w) +
                (std::size_t(row0 + (row0 + r < n_out ? r : 0)) * blocks_per_row + kbx) * BS);
            if constexpr (A2) {
                if constexpr(CAND) {
                    ql4[r] = load16_a4(b->ql + 16 * g);
                    qh4[r] = load16_a4(b->qh + 16 * qh4_idx);
                } else {
                ql4[r] = load16_a2(b->ql + 16 * g);
                qh4[r] = load16_a2(b->qh + 16 * qh4_idx);
                }
            } else {
                ql4[r] = reinterpret_cast<const sycl::int4*>(b->ql)[g];
                qh4[r] = reinterpret_cast<const sycl::int4*>(b->qh)[qh4_idx];
            }
            const float d = b->d;
            dsc0[r] = d * (float) b->scales[scale_offset]; dsc1[r] = d * (float) b->scales[scale_offset + 4];
        }
        // the activations once per column, shared by the RPW rows
        sycl::int4 u0[NCOLS], u1[NCOLS];
        float ds0[NCOLS], ds1[NCOLS];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            const Q81Block* xb = x + std::size_t(j) * x_stride + kby + bq8_offset;
            if constexpr (A2) {   // Q8_1 qs sits at offset 4 of a 36-byte block: 4-byte aligned, so four int loads
                const int* p0 = reinterpret_cast<const int*>(xb[0].qs) + 4 * u_int4;
                const int* p1 = reinterpret_cast<const int*>(xb[2].qs) + 4 * u_int4;
                u0[j] = sycl::int4(p0[0], p0[1], p0[2], p0[3]);
                u1[j] = sycl::int4(p1[0], p1[1], p1[2], p1[3]);
            } else {
                u0[j] = reinterpret_cast<const sycl::int4*>(xb[0].qs)[u_int4];
                u1[j] = reinterpret_cast<const sycl::int4*>(xb[2].qs)[u_int4];
            }
            ds0[j] = xb[0].ds[0]; ds1[j] = xb[2].ds[0];
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int vl[4] = {ql4[r].x(), ql4[r].y(), ql4[r].z(), ql4[r].w()};
            const int vh[4] = {qh4[r].x() >> vh_shift, qh4[r].y() >> vh_shift, qh4[r].z() >> vh_shift, qh4[r].w() >> vh_shift};
            int vi0[4], vi1[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                vi0[p] = dpct::vectorized_binary<sycl::char4>((vl[p] & 0x0f0f0f0f) | ((vh[p] << 4) & 0x30303030), 0x20202020, dpct::sub_sat());
                vi1[p] = dpct::vectorized_binary<sycl::char4>(((vl[p] >> 4) & 0x0f0f0f0f) | (((vh[p] >> 4) << 4) & 0x30303030), 0x20202020, dpct::sub_sat());
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                int s0 = 0, s1 = 0;
                s0 = strata::dp4a(vi0[0], u0[j].x(), s0); s0 = strata::dp4a(vi0[1], u0[j].y(), s0);
                s0 = strata::dp4a(vi0[2], u0[j].z(), s0); s0 = strata::dp4a(vi0[3], u0[j].w(), s0);
                s1 = strata::dp4a(vi1[0], u1[j].x(), s1); s1 = strata::dp4a(vi1[1], u1[j].y(), s1);
                s1 = strata::dp4a(vi1[2], u1[j].z(), s1); s1 = strata::dp4a(vi1[3], u1[j].w(), s1);
                acc[r][j] += ds0[j] * dsc0[r] * (float) s0 + ds1[j] * dsc1[r] * (float) s1;
            }
        }
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = acc[r][j];
#pragma unroll
            for (int o = SG / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
            if (lane == 0 && row0 + r < n_out) y[std::size_t(j) * n_out + row0 + r] = v;
        }
}

template<bool CAND>class PrivateC6A4;
template<bool CAND>void launch(const void* w,const void* x,float* y,int N,int M,sycl::queue& q){
 if(!((N==6144 && M==2560)||(N==2560 && M==10240)) || (reinterpret_cast<uintptr_t>(w)&3u))
     throw std::invalid_argument("S104 needs two pinned C6 shapes and base%4==0; no fallback/retry");
 const unsigned groups=unsigned((size_t(M)+WARPS-1)/WARPS);
 q.parallel_for<PrivateC6A4<CAND>>(sycl::nd_range<3>(sycl::range<3>(1,1,groups)*sycl::range<3>(1,WARPS,WARP),sycl::range<3>(1,WARPS,WARP)),
 [=](sycl::nd_item<3>)[[sycl::reqd_sub_group_size(32)]]{kernel<CAND>(static_cast<const Q6KBlock*>(w),static_cast<const Q81Block*>(x),y,N,M);});
}
} // namespace s104
