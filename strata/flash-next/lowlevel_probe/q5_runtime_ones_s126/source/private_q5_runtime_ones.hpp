#pragma once
// S126 private Q5 C2 only: original exact source, candidate runtime ones operand.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_math.hpp"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace s126 {
constexpr int QK=256,Q8K=32,WARP=32,WARPS=4;
struct Q5KBlock {sycl::half2 dm;uint8_t scales[12];uint8_t qh[32];uint8_t qs[128];};
struct Q4KBlock {sycl::half2 dm;uint8_t scales[12];uint8_t qs[128];};
struct Q81Block {sycl::half2 ds;int8_t qs[32];};
static_assert(sizeof(Q5KBlock)==176 && offsetof(Q5KBlock,qs)==48);
static_assert(sizeof(Q81Block)==36 && offsetof(Q81Block,qs)==4);
__dpct_inline__ sycl::int4 ld_q8_16(const Q81Block* b, int half) {   // Q8_1 qs is 4-byte aligned: four int loads
    const int* q = reinterpret_cast<const int*>(b->qs) + 4 * half;
    return sycl::int4(q[0], q[1], q[2], q[3]);
}
__dpct_inline__ int dp4a4(const sycl::int4 a, const sycl::int4 b, int acc) {
    acc = strata::dp4a(a.x(), b.x(), acc); acc = strata::dp4a(a.y(), b.y(), acc);
    acc = strata::dp4a(a.z(), b.z(), acc); return strata::dp4a(a.w(), b.w(), acc);
}
__dpct_inline__ void k4_scale_min(const uint8_t* sc, int j, float& s, float& m) {   // ggml get_scale_min_k4
    if (j < 4) { s = (float) (sc[j] & 63); m = (float) (sc[j + 4] & 63); }
    else { s = (float) ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)); m = (float) ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)); }
}
struct WideQ4K {
    using Block = Q4KBlock;
    struct W { sycl::int4 lo, hi; float dl, dh, ml, mh; int sb, half; };
    static W load(const Block* b, int g) {
        W r; r.sb = 2 * (g >> 1); r.half = g & 1;
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];          // 16-byte aligned (144 = 9 x 16)
        r.lo = sycl::int4(q.x() & 0x0f0f0f0f, q.y() & 0x0f0f0f0f, q.z() & 0x0f0f0f0f, q.w() & 0x0f0f0f0f);
        r.hi = sycl::int4((q.x() >> 4) & 0x0f0f0f0f, (q.y() >> 4) & 0x0f0f0f0f, (q.z() >> 4) & 0x0f0f0f0f, (q.w() >> 4) & 0x0f0f0f0f);
        const sycl::float2 dm = b->dm.convert<float, sycl::rounding_mode::automatic>();
        float s0, m0, s1, m1; k4_scale_min(b->scales, r.sb, s0, m0); k4_scale_min(b->scales, r.sb + 1, s1, m1);
        r.dl = dm.x() * s0; r.ml = dm.y() * m0; r.dh = dm.x() * s1; r.mh = dm.y() * m1;
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) {   // xb = this column's Q8_1 blocks of the 256-block
        const Q81Block* a = xb + r.sb; const Q81Block* c = xb + r.sb + 1;
        const sycl::int4 ua = ld_q8_16(a, r.half), uc = ld_q8_16(c, r.half);
        const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
        const float da = (float) a->ds[0], dc = (float) c->ds[0];
        return da * (r.dl * (float) dp4a4(r.lo, ua, 0) - r.ml * (float) dp4a4(ones, ua, 0)) +
               dc * (r.dh * (float) dp4a4(r.hi, uc, 0) - r.mh * (float) dp4a4(ones, uc, 0));
    }
    // apply() in two parts for several rows of one sub-group: the activations depend on the lane alone (sb, half from g)
    struct A { sycl::int4 ua, uc; float da, dc; };
    static A act(const Q81Block* xb, int g) {
        const int sb = 2 * (g >> 1), half = g & 1;
        const Q81Block* a = xb + sb; const Q81Block* c = xb + sb + 1;
        A v; v.ua = ld_q8_16(a, half); v.uc = ld_q8_16(c, half); v.da = (float) a->ds[0]; v.dc = (float) c->ds[0];
        return v;
    }
    static float dot(const W& r, const A& v) {   // apply's expression, the same operations in the same order
        const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
        return v.da * (r.dl * (float) dp4a4(r.lo, v.ua, 0) - r.ml * (float) dp4a4(ones, v.ua, 0)) +
               v.dc * (r.dh * (float) dp4a4(r.hi, v.uc, 0) - r.mh * (float) dp4a4(ones, v.uc, 0));
    }
};
struct WideQ5K {
    using Block = Q5KBlock;
    using W = WideQ4K::W;
    static W load(const Block* b, int g) {
        W r; r.sb = 2 * (g >> 1); r.half = g & 1;
        const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b->qs)[g];          // qs at 48, 176 = 11 x 16
        const sycl::int4 h = reinterpret_cast<const sycl::int4*>(b->qh)[r.half];     // qh bytes of positions 16h..+15
        const int sl = r.sb, sh = r.sb + 1;
        auto hb = [](int v, int bit) { return ((v >> bit) & 0x01010101) << 4; };
        r.lo = sycl::int4((q.x() & 0x0f0f0f0f) | hb(h.x(), sl), (q.y() & 0x0f0f0f0f) | hb(h.y(), sl),
                          (q.z() & 0x0f0f0f0f) | hb(h.z(), sl), (q.w() & 0x0f0f0f0f) | hb(h.w(), sl));
        r.hi = sycl::int4(((q.x() >> 4) & 0x0f0f0f0f) | hb(h.x(), sh), ((q.y() >> 4) & 0x0f0f0f0f) | hb(h.y(), sh),
                          ((q.z() >> 4) & 0x0f0f0f0f) | hb(h.z(), sh), ((q.w() >> 4) & 0x0f0f0f0f) | hb(h.w(), sh));
        const sycl::float2 dm = b->dm.convert<float, sycl::rounding_mode::automatic>();
        float s0, m0, s1, m1; k4_scale_min(b->scales, r.sb, s0, m0); k4_scale_min(b->scales, r.sb + 1, s1, m1);
        r.dl = dm.x() * s0; r.ml = dm.y() * m0; r.dh = dm.x() * s1; r.mh = dm.y() * m1;
        return r;
    }
    static float apply(const W& r, const Q81Block* xb) { return WideQ4K::apply(r, xb); }
    using A = WideQ4K::A;
    static A act(const Q81Block* xb, int g) { return WideQ4K::act(xb, g); }
    static float dot(const W& r, const A& v) { return WideQ4K::dot(r, v); }
};
inline float apply_runtime(const WideQ5K::W& r, const Q81Block* xb, int ones_word) {   // xb = this column's Q8_1 blocks of the 256-block
        const Q81Block* a = xb + r.sb; const Q81Block* c = xb + r.sb + 1;
        const sycl::int4 ua = ld_q8_16(a, r.half), uc = ld_q8_16(c, r.half);
        const sycl::int4 ones(ones_word, ones_word, ones_word, ones_word);
        const float da = (float) a->ds[0], dc = (float) c->ds[0];
        return da * (r.dl * (float) dp4a4(r.lo, ua, 0) - r.ml * (float) dp4a4(ones, ua, 0)) +
               dc * (r.dh * (float) dp4a4(r.hi, uc, 0) - r.mh * (float) dp4a4(ones, uc, 0));
    }
template <typename F, int NCOLS>
void original_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                             float* __restrict__ y, int n_in, int n_out) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;
    const typename F::Block* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += 4) {
        const typename F::W wv = F::load(wr + kbx, g);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] += F::apply(wv, x + std::size_t(j) * x_stride + kbx * (QK / Q8K));
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}
template <typename F, int NCOLS>
void candidate_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                             float* __restrict__ y, int n_in, int n_out, int ones_word) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = int(item.get_local_id(2)), warp = int(item.get_local_id(1));
    const int row = int(item.get_group(2)) * WARPS + warp;
    if (row >= n_out) return;
    const int blocks_per_row = n_in / QK, x_stride = n_in / Q8K;
    const int g = lane & 7, sub = lane >> 3;
    const typename F::Block* wr = w + std::size_t(row) * blocks_per_row;
    float acc[NCOLS] = {};
    for (int kbx = sub; kbx < blocks_per_row; kbx += 4) {
        const typename F::W wv = F::load(wr + kbx, g);
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[j] += apply_runtime(wv, x + std::size_t(j) * x_stride + kbx * (QK / Q8K), ones_word);
    }
    auto sg = item.get_sub_group();
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}

class OriginalQ5C2;
class RuntimeOnesQ5C2;
class Raw16Sums;
inline void launch_original(const void* weights,const void* xq,float* y,int n_in,int n_out,sycl::queue& q) {
    const auto* w=static_cast<const Q5KBlock*>(weights);
    const auto* x=static_cast<const Q81Block*>(xq);
    const unsigned blocks=unsigned((std::size_t(n_out)+WARPS-1)/WARPS);
    q.parallel_for<OriginalQ5C2>(
        sycl::nd_range<3>(sycl::range(1,1,blocks)*sycl::range(1,WARPS,WARP),sycl::range(1,WARPS,WARP)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {original_kernel<WideQ5K,2>(w,x,y,n_in,n_out);});
}
// ones_word is parsed by the executable; no device-side equality guard or literal.
inline void launch_candidate(const void* weights,const void* xq,float* y,int n_in,int n_out,int ones_word,sycl::queue& q) {
    const auto* w=static_cast<const Q5KBlock*>(weights);
    const auto* x=static_cast<const Q81Block*>(xq);
    const unsigned blocks=unsigned((std::size_t(n_out)+WARPS-1)/WARPS);
    q.parallel_for<RuntimeOnesQ5C2>(
        sycl::nd_range<3>(sycl::range(1,1,blocks)*sycl::range(1,WARPS,WARP),sycl::range(1,WARPS,WARP)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {candidate_kernel<WideQ5K,2>(w,x,y,n_in,n_out,ones_word);});
}
// Validation only, never a projection producer/sidecar or captured timed node.
inline void raw16_sums(const void* xq,int* out,int n_q8_blocks,int ones_word,sycl::queue& q) {
    const auto* x=static_cast<const Q81Block*>(xq);
    q.parallel_for<Raw16Sums>(sycl::range<1>(size_t(n_q8_blocks)*2),[=](sycl::id<1> id) {
        const int k=int(id[0]);const sycl::int4 ones(ones_word,ones_word,ones_word,ones_word);
        out[k]=dp4a4(ones,ld_q8_16(x+k/2,k%2),0);
    });
}
} // namespace s126
