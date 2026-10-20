// KREG GEMV for Q4_K, Q5_K, Q6_K, IQ4_XS.
// Schedule from Strata night_kreg_kernel (native_mmvq.dp.cpp:2236):
// the bf16 activation is quantized once per gemv() into Q8_1 blocks
// (amax/127 per 32 values, same rint as the old in-kernel quant), K is
// split across the work-group (8 lanes per 256-block), each work-group
// streams R output rows, a sub-group butterfly reduces the partials, and
// SLM sums the sub-groups. Dots are dp4a on the nibble or codebook lanes.
// The half-block min term is d * sum(qs of that half), recomputed in the
// GEMV, matching night_act_k.
#pragma once

template <int TYPE, int NCOLS, int R>
class b70_kreg_k;

class b70_q8_quant_k;

// 48 bytes so qs sits at offset 16. Every block stays 16-byte aligned when
// the base is, and each 16-byte half is an aligned int4.
struct B70Q8 {
    float d;
    float pad[3];
    int8_t qs[32];
};
static_assert(sizeof(B70Q8) == 48, "q8 block");
static_assert(__builtin_offsetof(B70Q8, qs) == 16, "qs half is 16-byte aligned");

constexpr int b70_kreg_p(int p0) { return (p0 + 15) / 16 * 16; }

inline int b70_dp4a(int a, int b, int c) {
    const sycl::vec<int, 4> va =
        sycl::vec<int, 1>(a).template as<sycl::vec<int8_t, 4>>().template convert<int>();
    const sycl::vec<int, 4> vb =
        sycl::vec<int, 1>(b).template as<sycl::vec<int8_t, 4>>().template convert<int>();
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}

inline int b70_dp4a4(const sycl::int4& a, const sycl::int4& b, int c) {
    c = b70_dp4a(a.x(), b.x(), c);
    c = b70_dp4a(a.y(), b.y(), c);
    c = b70_dp4a(a.z(), b.z(), c);
    return b70_dp4a(a.w(), b.w(), c);
}

inline int b70_pack4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (int)((uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) | ((uint32_t)(uint8_t)c << 16) |
                 ((uint32_t)(uint8_t)d << 24));
}

inline sycl::int4 b70_pack16(const int8_t* q) {
    return sycl::int4(b70_pack4(q[0], q[1], q[2], q[3]), b70_pack4(q[4], q[5], q[6], q[7]),
                      b70_pack4(q[8], q[9], q[10], q[11]), b70_pack4(q[12], q[13], q[14], q[15]));
}

// Two aligned 16-byte loads plus a shift. A Q6_K row is 2-byte aligned;
// a raw misaligned vector load splits on this GPU.
inline sycl::int4 b70_ld16_a2(const void* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const sycl::int4* q = reinterpret_cast<const sycl::int4*>(a & ~uintptr_t(15));
    const int k = int(a >> 2) & 3;
    const bool half = (a & 2) != 0;
    const sycl::int4 lo = q[0];
    const sycl::int4 hi = (a & 15) ? q[1] : lo;
    const uint32_t d[8] = {(uint32_t)lo.x(), (uint32_t)lo.y(), (uint32_t)lo.z(), (uint32_t)lo.w(),
                           (uint32_t)hi.x(), (uint32_t)hi.y(), (uint32_t)hi.z(), (uint32_t)hi.w()};
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) w[i] = k == 0 ? d[i] : k == 1 ? d[i + 1] : k == 2 ? d[i + 2] : d[(i + 3) & 7];
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = half ? (w[i] >> 16) | (w[i + 1] << 16) : w[i];
    return sycl::int4((int)o[0], (int)o[1], (int)o[2], (int)o[3]);
}

inline sycl::int4 b70_ld_int4(const void* p) {
    return *reinterpret_cast<const sycl::int4*>(p);
}

struct B70Act {
    sycl::int4 u0, u1;
    float d0, d1, s0, s1;
};

inline B70Act b70_quant16_pair(const sycl::sub_group& sg, const uint16_t* p, bool live) {
    float xv[16];
    float m = 0.f;
    if (live) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            xv[i] = b70_bf16_to_fp32(p[i]);
            m = sycl::fmax(m, sycl::fabs(xv[i]));
        }
    }
    const float other = sycl::permute_group_by_xor(sg, m, 1);
    m = sycl::fmax(m, other);
    const float d = m * (1.f / 127.f);
    const float id = m > 0.f ? 127.f / m : 0.f;
    int8_t q[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        int r = 0;
        if (live) {
            r = (int)sycl::rint(xv[i] * id);
            if (r > 127) r = 127;
            if (r < -127) r = -127;
        }
        q[i] = (int8_t)r;
    }
    const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
    B70Act a{};
    a.u0 = b70_pack16(q);
    a.d0 = live ? d : 0.f;
    a.s0 = a.d0 * (float)b70_dp4a4(ones, a.u0, 0);
    return a;
}

inline B70Act b70_quant32_local(const uint16_t* p, bool live) {
    float xv[32];
    float m = 0.f;
    if (live) {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            xv[i] = b70_bf16_to_fp32(p[i]);
            m = sycl::fmax(m, sycl::fabs(xv[i]));
        }
    }
    const float d = m * (1.f / 127.f);
    const float id = m > 0.f ? 127.f / m : 0.f;
    int8_t q[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        int r = 0;
        if (live) {
            r = (int)sycl::rint(xv[i] * id);
            if (r > 127) r = 127;
            if (r < -127) r = -127;
        }
        q[i] = (int8_t)r;
    }
    B70Act a{};
    a.u0 = b70_pack16(q);
    a.u1 = b70_pack16(q + 16);
    a.d0 = live ? d : 0.f;
    a.d1 = a.d0;
    return a;
}

struct B70WK {
    sycl::int4 lo, hi;
    float dl, dh, ml, mh;
};

inline void b70_scales_k(sycl::int4 hdr, int g, float& dl, float& dh, float& ml, float& mh) {
    const int sb = 2 * (g >> 1);
    const int sh = 8 * (sb & 3);
    const uint32_t A = ((uint32_t)hdr.y() >> sh) & 0xFFFFu;
    const uint32_t B = ((uint32_t)hdr.z() >> sh) & 0xFFFFu;
    const uint32_t C = ((uint32_t)hdr.w() >> sh) & 0xFFFFu;
    const uint32_t ls0 = A & 63u, ls1 = (A >> 8) & 63u, lm0 = B & 63u, lm1 = (B >> 8) & 63u;
    const uint32_t hs0 = (C & 0xFu) | (((A >> 6) & 3u) << 4);
    const uint32_t hs1 = ((C >> 8) & 0xFu) | (((A >> 14) & 3u) << 4);
    const uint32_t hm0 = ((C >> 4) & 0xFu) | (((B >> 6) & 3u) << 4);
    const uint32_t hm1 = ((C >> 12) & 0xFu) | (((B >> 14) & 3u) << 4);
    const bool lo4 = sb < 4;
    const uint32_t s0 = lo4 ? ls0 : hs0, s1 = lo4 ? ls1 : hs1, m0 = lo4 ? lm0 : hm0, m1 = lo4 ? lm1 : hm1;
    const uint32_t hx = (uint32_t)hdr.x();
    const float d = b70_fp16_to_fp32((uint16_t)(hx & 0xFFFFu));
    const float dmin = b70_fp16_to_fp32((uint16_t)(hx >> 16));
    dl = d * (float)s0;
    dh = d * (float)s1;
    ml = dmin * (float)m0;
    mh = dmin * (float)m1;
}

inline int b70_hb(uint32_t v, int bit) { return (int)(((v >> bit) & 0x01010101u) << 4); }

inline B70WK b70_load_q4(const uint8_t* b, int g) {
    B70WK r{};
    const sycl::int4 hdr = b70_ld_int4(b);
    const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b + 16)[g];
    const uint32_t qx = (uint32_t)q.x(), qy = (uint32_t)q.y(), qz = (uint32_t)q.z(), qw = (uint32_t)q.w();
    r.lo = sycl::int4((int)(qx & 0x0f0f0f0fu), (int)(qy & 0x0f0f0f0fu), (int)(qz & 0x0f0f0f0fu), (int)(qw & 0x0f0f0f0fu));
    r.hi = sycl::int4((int)((qx >> 4) & 0x0f0f0f0fu), (int)((qy >> 4) & 0x0f0f0f0fu), (int)((qz >> 4) & 0x0f0f0f0fu),
                      (int)((qw >> 4) & 0x0f0f0f0fu));
    b70_scales_k(hdr, g, r.dl, r.dh, r.ml, r.mh);
    return r;
}

inline int b70_q5_mix(uint32_t qv, uint32_t hv, int bit, bool high) {
    const uint32_t nib = high ? (qv >> 4) & 0x0f0f0f0fu : qv & 0x0f0f0f0fu;
    return (int)(nib | (uint32_t)b70_hb(hv, bit));
}

inline B70WK b70_load_q5(const uint8_t* b, int g) {
    B70WK r{};
    const int sb = 2 * (g >> 1);
    const sycl::int4 hdr = b70_ld_int4(b);
    const sycl::int4 q = reinterpret_cast<const sycl::int4*>(b + 48)[g];
    const sycl::int4 h = b70_ld_int4(b + 16 + 16 * (g & 1));
    const uint32_t qx = (uint32_t)q.x(), qy = (uint32_t)q.y(), qz = (uint32_t)q.z(), qw = (uint32_t)q.w();
    const uint32_t hx = (uint32_t)h.x(), hy = (uint32_t)h.y(), hz = (uint32_t)h.z(), hw = (uint32_t)h.w();
    r.lo = sycl::int4(b70_q5_mix(qx, hx, sb, false), b70_q5_mix(qy, hy, sb, false), b70_q5_mix(qz, hz, sb, false),
                      b70_q5_mix(qw, hw, sb, false));
    r.hi = sycl::int4(b70_q5_mix(qx, hx, sb + 1, true), b70_q5_mix(qy, hy, sb + 1, true),
                      b70_q5_mix(qz, hz, sb + 1, true), b70_q5_mix(qw, hw, sb + 1, true));
    b70_scales_k(hdr, g, r.dl, r.dh, r.ml, r.mh);
    return r;
}

inline float b70_dot_k(const B70WK& r, const B70Act& a) {
    return r.dl * (a.d0 * (float)b70_dp4a4(r.lo, a.u0, 0)) - r.ml * a.s0 +
           r.dh * (a.d1 * (float)b70_dp4a4(r.hi, a.u1, 0)) - r.mh * a.s1;
}

struct B70W6 {
    int vi0[4], vi1[4];
    float dsc0, dsc1;
};

inline int b70_q6_center(uint32_t vl, uint32_t vh) {
    const uint32_t t = (vl & 0x0f0f0f0fu) | ((vh << 4) & 0x30303030u);
    return (int)((t + 0x60606060u) ^ 0x80808080u);
}

inline B70W6 b70_load_q6(const uint8_t* b, int g) {
    B70W6 r{};
    const int vh_shift = 2 * ((g & 3) >> 1);
    const int qh4_idx = 2 * (g >> 2) + (g & 1);
    const int scale_offset = 8 * (g >> 2) + (g & 3);
    const sycl::int4 ql4 = b70_ld16_a2(b + 16 * g);
    const sycl::int4 qh4 = b70_ld16_a2(b + 128 + 16 * qh4_idx);
    const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b + 208));
    r.dsc0 = d * (float)(int8_t)b[192 + scale_offset];
    r.dsc1 = d * (float)(int8_t)b[192 + scale_offset + 4];
    const uint32_t vl[4] = {(uint32_t)ql4.x(), (uint32_t)ql4.y(), (uint32_t)ql4.z(), (uint32_t)ql4.w()};
    const uint32_t vh[4] = {(uint32_t)qh4.x() >> vh_shift, (uint32_t)qh4.y() >> vh_shift, (uint32_t)qh4.z() >> vh_shift,
                            (uint32_t)qh4.w() >> vh_shift};
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        r.vi0[p] = b70_q6_center(vl[p], vh[p]);
        r.vi1[p] = b70_q6_center(vl[p] >> 4, vh[p] >> 4);
    }
    return r;
}

inline float b70_dot_q6(const B70W6& r, const B70Act& a) {
    int s0 = 0, s1 = 0;
    s0 = b70_dp4a(r.vi0[0], a.u0.x(), s0);
    s0 = b70_dp4a(r.vi0[1], a.u0.y(), s0);
    s0 = b70_dp4a(r.vi0[2], a.u0.z(), s0);
    s0 = b70_dp4a(r.vi0[3], a.u0.w(), s0);
    s1 = b70_dp4a(r.vi1[0], a.u1.x(), s1);
    s1 = b70_dp4a(r.vi1[1], a.u1.y(), s1);
    s1 = b70_dp4a(r.vi1[2], a.u1.z(), s1);
    s1 = b70_dp4a(r.vi1[3], a.u1.w(), s1);
    return a.d0 * r.dsc0 * (float)s0 + a.d1 * r.dsc1 * (float)s1;
}

inline uint32_t b70_iq4_pair(int i) {
    return (uint32_t)(uint8_t)kIq4nl[i & 15] | ((uint32_t)(uint8_t)kIq4nl[i >> 4] << 16);
}

struct B70W4X {
    sycl::int4 lo, hi;
    float d;
};

inline B70W4X b70_load_iq4xs(const uint8_t* b, int g, const uint32_t* tl) {
    B70W4X r{};
    const uint32_t h0 = (uint32_t)b70_ld16(b) | ((uint32_t)b70_ld16(b + 2) << 16);
    const uint32_t sl = (uint32_t)b70_ld16(b + 4) | ((uint32_t)b70_ld16(b + 6) << 16);
    const int ls = (int)(((sl >> (8 * (g >> 1) + 4 * (g & 1))) & 0x0fu) | (((h0 >> (16 + 2 * g)) & 0x03u) << 4));
    r.d = b70_fp16_to_fp32((uint16_t)(h0 & 0xFFFFu)) * (float)(ls - 32);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(b + 8 + 16 * g);
    int lo[4], hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t v = q[i];
        const uint32_t P = tl[v & 0xFFu] | (tl[(v >> 8) & 0xFFu] << 8);
        const uint32_t Q = tl[(v >> 16) & 0xFFu] | (tl[v >> 24] << 8);
        lo[i] = (int)((P & 0xFFFFu) | (Q << 16));
        hi[i] = (int)((P >> 16) | (Q & 0xFFFF0000u));
    }
    r.lo = sycl::int4(lo[0], lo[1], lo[2], lo[3]);
    r.hi = sycl::int4(hi[0], hi[1], hi[2], hi[3]);
    return r;
}

inline float b70_dot_iq4xs(const B70W4X& r, const B70Act& a) {
    return r.d * a.d0 * (float)b70_dp4a4(r.hi, a.u1, b70_dp4a4(r.lo, a.u0, 0));
}

template <int O, int N, int P>
inline void b70_kreg_tsum(float (&v)[P], const sycl::sub_group& sg, int lane) {
    if constexpr (O > 0) {
        constexpr int H = N / 2;
        const bool up = (lane & O) != 0;
#pragma unroll
        for (int i = 0; i < H; ++i) {
            const float lo = v[i];
            const float hi = v[i + H];
            const float send = up ? lo : hi;
            const float keep = up ? hi : lo;
            v[i] = keep + sycl::permute_group_by_xor(sg, send, O);
        }
        b70_kreg_tsum<O / 2, H, P>(v, sg, lane);
    }
}

// One device buffer, grown only after a queue wait. Steady-state gemv calls
// do not wait and do not reallocate. Single stream: the bench and the
// decode path submit on the current XPU stream.
inline B70Q8* b70_q8_scratch(sycl::queue& q, size_t nblk) {
    static B70Q8* ptr = nullptr;
    static size_t cap = 0;
    if (nblk <= cap) return ptr;
    q.wait();
    if (ptr) sycl::free(ptr, q);
    cap = nblk < 8192 ? 8192 : nblk;
    // size must be a multiple of the alignment. 48-byte blocks are already a
    // multiple of 16, which is what the int4 qs load needs.
    void* raw = sycl::aligned_alloc_device(16, cap * sizeof(B70Q8), q);
    if (!raw) throw std::runtime_error("b70_gguf: q8 scratch alloc failed");
    ptr = static_cast<B70Q8*>(raw);
    return ptr;
}

// One work-item per 32-value block. amax is the max abs of the whole block,
// matching the xor-1 pair in b70_quant16_pair and the local amax in
// b70_quant32_local. Submitted on q and left in flight.
inline sycl::event b70_q8_quant(sycl::queue& q, const uint16_t* x, B70Q8* y, int M, int K) {
    const int nb = K / 32;
    const int n = M * nb;
    constexpr int L = 64;
    const size_t global = (size_t)((n + L - 1) / L) * (size_t)L;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_q8_quant_k>(sycl::nd_range<1>(global, (size_t)L), [=](sycl::nd_item<1> it) {
            const int i = (int)it.get_global_id(0);
            if (i >= n) return;
            const uint16_t* p = x + (size_t)i * 32;
            float xv[32];
            float m = 0.f;
#pragma unroll
            for (int t = 0; t < 32; ++t) {
                xv[t] = b70_bf16_to_fp32(p[t]);
                m = sycl::fmax(m, sycl::fabs(xv[t]));
            }
            const float id = m > 0.f ? 127.f / m : 0.f;
            B70Q8 b{};
            b.d = m * (1.f / 127.f);
#pragma unroll
            for (int t = 0; t < 32; ++t) {
                int r = (int)sycl::rint(xv[t] * id);
                if (r > 127) r = 127;
                if (r < -127) r = -127;
                b.qs[t] = (int8_t)r;
            }
            y[i] = b;
        });
    });
}

template <int TYPE, int NCOLS, int R, int GRF>
void b70_kreg_launch(sycl::queue& q, const B70Q8* xq, const uint8_t* w, float* y, int N, int K, sycl::event dep) {
    constexpr int P0 = R * NCOLS;
    constexpr int P = (P0 + 15) / 16 * 16;
    constexpr int PL = P / 16;
    constexpr int BB = TYPE == B70_Q4_K ? 144 : TYPE == B70_Q5_K ? 176 : TYPE == B70_Q6_K ? 210 : 136;
    constexpr bool kTable = TYPE == B70_IQ4_XS;
    const int bpr = K / 256;
    const int nb = K / 32;
    const int need = bpr * 8;
    const int L = (need + 15) / 16 * 16;
    const int nsg = L / 16;
    const int tiles = (N + R - 1) / R;
    const int slm_n = nsg * P + (kTable ? 256 : 0);
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<GRF>};
    q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        sycl::local_accessor<float, 1> slm(slm_n, h);
        h.parallel_for<b70_kreg_k<TYPE, NCOLS, R>>(
            sycl::nd_range<1>((size_t)tiles * (size_t)L, (size_t)L), props,
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int lid = (int)it.get_local_id(0);
                const int lane = (int)sg.get_local_linear_id();
                const int sgi = lid / 16;
                const bool live = (lid >> 3) < bpr;
                const int kb = live ? (lid >> 3) : 0;
                const int g = lid & 7;
                const sycl::int4 ones(0x01010101, 0x01010101, 0x01010101, 0x01010101);
                B70Act a[NCOLS];
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) {
                    const B70Q8* xj = xq + (size_t)j * (size_t)nb;
                    if (!live) {
                        a[j] = B70Act{};
                    } else if constexpr (kTable) {
                        const B70Q8* blk = xj + (kb * 8 + g);
                        a[j].u0 = b70_ld_int4(blk->qs);
                        a[j].u1 = b70_ld_int4(blk->qs + 16);
                        a[j].d0 = blk->d;
                        a[j].d1 = blk->d;
                        a[j].s0 = 0.f;
                        a[j].s1 = 0.f;
                    } else {
                        int b0, b1, half;
                        if constexpr (TYPE == B70_Q6_K) {
                            const int bq8 = 4 * (g >> 2) + ((g & 3) >> 1);
                            half = g & 1;
                            b0 = kb * 8 + bq8;
                            b1 = kb * 8 + bq8 + 2;
                        } else {
                            const int sb = 2 * (g >> 1);
                            half = g & 1;
                            b0 = kb * 8 + sb;
                            b1 = kb * 8 + sb + 1;
                        }
                        const B70Q8* p0 = xj + b0;
                        const B70Q8* p1 = xj + b1;
                        a[j].u0 = b70_ld_int4(p0->qs + half * 16);
                        a[j].d0 = p0->d;
                        a[j].s0 = a[j].d0 * (float)b70_dp4a4(ones, a[j].u0, 0);
                        a[j].u1 = b70_ld_int4(p1->qs + half * 16);
                        a[j].d1 = p1->d;
                        a[j].s1 = a[j].d1 * (float)b70_dp4a4(ones, a[j].u1, 0);
                    }
                }
                const uint32_t* tbl = nullptr;
                float* sm = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                if constexpr (kTable) {
                    uint32_t* tw = reinterpret_cast<uint32_t*>(sm + nsg * P);
                    for (int i = lid; i < 256; i += L) tw[i] = b70_iq4_pair(i);
                    it.barrier(sycl::access::fence_space::local_space);
                    tbl = tw;
                }
                const int row0 = (int)it.get_group(0) * R;
                const int rlast = N - 1 - row0;
                float v[P];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int rr = r < rlast ? r : rlast;
                    const uint8_t* b = w + ((size_t)(row0 + rr) * (size_t)bpr + (size_t)kb) * (size_t)BB;
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j) {
                        float part = 0.f;
                        if (live) {
                            if constexpr (TYPE == B70_Q4_K) part = b70_dot_k(b70_load_q4(b, g), a[j]);
                            else if constexpr (TYPE == B70_Q5_K) part = b70_dot_k(b70_load_q5(b, g), a[j]);
                            else if constexpr (TYPE == B70_Q6_K) part = b70_dot_q6(b70_load_q6(b, g), a[j]);
                            else part = b70_dot_iq4xs(b70_load_iq4xs(b, g, tbl), a[j]);
                        }
                        v[r * NCOLS + j] = part;
                    }
                }
#pragma unroll
                for (int k = P0; k < P; ++k) v[k] = 0.f;
                b70_kreg_tsum<8, P, P>(v, sg, lane);
#pragma unroll
                for (int i = 0; i < PL; ++i) sm[sgi * P + lane * PL + i] = v[i];
                it.barrier(sycl::access::fence_space::local_space);
                for (int k = lid; k < P0; k += L) {
                    float s = 0.f;
                    for (int qq = 0; qq < nsg; ++qq) s += sm[qq * P + k];
                    const int r = k / NCOLS;
                    const int j = k - r * NCOLS;
                    if (row0 + r < N) y[(size_t)j * (size_t)N + (size_t)(row0 + r)] = s;
                }
            });
    });
}

template <int TYPE>
void b70_kreg_m(sycl::queue& q, const uint16_t* x, const uint8_t* w, float* y, int M, int N, int K) {
    // Quantize x once. The GEMV is queued behind it on the same in-order
    // stream. No wait between the two submits: the caller times gemv() as
    // one interval.
    const int nb = K / 32;
    B70Q8* xq = b70_q8_scratch(q, (size_t)M * (size_t)nb);
    const sycl::event ev = b70_q8_quant(q, x, xq, M, K);
    // GRF 256 only when the tile holds more than 16 partials. The head
    // (M=1, R=4) stays at GRF 128, matching Strata's night_dispatch.
    if constexpr (TYPE == B70_Q6_K) {
        switch (M) {
            case 1: b70_kreg_launch<TYPE, 1, 4, 128>(q, xq, w, y, N, K, ev); break;
            case 2: b70_kreg_launch<TYPE, 2, 4, 128>(q, xq, w, y, N, K, ev); break;
            case 3: b70_kreg_launch<TYPE, 3, 4, 128>(q, xq, w, y, N, K, ev); break;
            case 4: b70_kreg_launch<TYPE, 4, 2, 128>(q, xq, w, y, N, K, ev); break;
            case 5: b70_kreg_launch<TYPE, 5, 2, 128>(q, xq, w, y, N, K, ev); break;
            case 6: b70_kreg_launch<TYPE, 6, 2, 128>(q, xq, w, y, N, K, ev); break;
            case 7: b70_kreg_launch<TYPE, 7, 2, 128>(q, xq, w, y, N, K, ev); break;
            default: b70_kreg_launch<TYPE, 8, 2, 128>(q, xq, w, y, N, K, ev); break;
        }
    } else if constexpr (TYPE == B70_IQ4_XS) {
        switch (M) {
            case 1: b70_kreg_launch<TYPE, 1, 8, 128>(q, xq, w, y, N, K, ev); break;
            case 2: b70_kreg_launch<TYPE, 2, 4, 128>(q, xq, w, y, N, K, ev); break;
            case 3: b70_kreg_launch<TYPE, 3, 4, 128>(q, xq, w, y, N, K, ev); break;
            case 4: b70_kreg_launch<TYPE, 4, 4, 128>(q, xq, w, y, N, K, ev); break;
            case 5: b70_kreg_launch<TYPE, 5, 4, 256>(q, xq, w, y, N, K, ev); break;
            case 6: b70_kreg_launch<TYPE, 6, 4, 256>(q, xq, w, y, N, K, ev); break;
            case 7: b70_kreg_launch<TYPE, 7, 4, 256>(q, xq, w, y, N, K, ev); break;
            default: b70_kreg_launch<TYPE, 8, 4, 256>(q, xq, w, y, N, K, ev); break;
        }
    } else {
        switch (M) {
            case 1: b70_kreg_launch<TYPE, 1, 4, 128>(q, xq, w, y, N, K, ev); break;
            case 2: b70_kreg_launch<TYPE, 2, 8, 128>(q, xq, w, y, N, K, ev); break;
            case 3: b70_kreg_launch<TYPE, 3, 4, 128>(q, xq, w, y, N, K, ev); break;
            case 4: b70_kreg_launch<TYPE, 4, 4, 128>(q, xq, w, y, N, K, ev); break;
            case 5: b70_kreg_launch<TYPE, 5, 8, 256>(q, xq, w, y, N, K, ev); break;
            case 6: b70_kreg_launch<TYPE, 6, 8, 256>(q, xq, w, y, N, K, ev); break;
            case 7: b70_kreg_launch<TYPE, 7, 8, 256>(q, xq, w, y, N, K, ev); break;
            default: b70_kreg_launch<TYPE, 8, 8, 256>(q, xq, w, y, N, K, ev); break;
        }
    }
}
