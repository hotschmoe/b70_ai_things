// One 32-weight group dotted with one Q8_1 block (B70Q8).
// Prep matches Strata NQ<> (iq_kernels.dp.cpp:2454) with the ALU sign path
// that the night latch keeps (NQ<118/121/122>, STRATA_NIGHT_IQ_ALU default,
// iq_kernels.dp.cpp:5015): swar_ne4 masks, then sgn = (g ^ m) + (m & 1)
// instead of swar_sub4. IQ4_NL uses night::iq4nl_mux4 (:2437). Q2_0 uses
// night::q2_spread4 (:2431). K-quants, IQ4_XS, Q2_K and Q3_K still decode
// to float and FMA against qs * d.
#pragma once

inline int b70_swar_ne4(uint32_t x) {
    x = (x | (x >> 4)) & 0x0F0F0F0Fu;
    x = (x | (x >> 2)) & 0x03030303u;
    x = (x | (x >> 1)) & 0x01010101u;
    return (int)(x * 0xFFu);
}

// Exact for IQ grid bytes in 1..62, which is every codebook entry.
inline int b70_sgn(uint32_t g, uint32_t m) { return (int)((g ^ m) + (m & 0x01010101u)); }

// Unaligned loads. Byte-wise b70_ld16 expands to a lot of ALU; memcpy is the
// form IGC turns into one unaligned load (night Q2_0 note, iq_kernels.dp.cpp:2416).
inline uint32_t b70_loadu32(const uint8_t* p) {
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

inline uint64_t b70_loadu64(const uint8_t* p) {
    uint64_t v;
    __builtin_memcpy(&v, p, 8);
    return v;
}

// O3 (reports/2O3_moe_decode.md): weight blocks of every DP4A type start 2-byte aligned (all block sizes are
// even, rows and tensors are at least 4-aligned). Strata load8_a2 / load4_a2
// (iq_kernels.dp.cpp:371): aligned dword loads plus a funnel shift, instead of
// one d64/d32 LSC load at a 2-aligned address (2.8x slower gate/up on B70).
// Callers must pass even weight pointers (kernels.cpp b70_check_a2). The fp16 scale is one aligned
// 16-bit load and a hardware convert (Strata night::h2f) instead of two byte
// loads and the software b70_fp16_to_fp32 (a denormal loop in the ISA).
#ifndef B70_LOAD_A2
#define B70_LOAD_A2 1
#endif
#ifndef B70_HW_HALF
#define B70_HW_HALF 1
#endif
inline uint64_t b70_ld8_a2(const uint8_t* p) {
#if B70_LOAD_A2
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0], w1 = q[1];
    if (!(a & 2)) return (uint64_t)w0 | ((uint64_t)w1 << 32);
    const uint32_t w2 = q[2];
    return (uint64_t)((w0 >> 16) | (w1 << 16)) | ((uint64_t)((w1 >> 16) | (w2 << 16)) << 32);
#else
    return b70_loadu64(p);
#endif
}
inline uint32_t b70_ld4_a2(const uint8_t* p) {
#if B70_LOAD_A2
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* q = reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
    const uint32_t w0 = q[0];
    if (!(a & 2)) return w0;
    return (w0 >> 16) | (q[1] << 16);
#else
    return b70_loadu32(p);
#endif
}
inline float b70_h2f(const uint8_t* p) {
#if B70_HW_HALF
    const uint16_t h = *reinterpret_cast<const uint16_t*>(p);
    return (float)sycl::bit_cast<sycl::half>(h);
#else
    return b70_fp16_to_fp32((uint16_t)b70_ld16(p));
#endif
}

// v is the low 8 bits of a shifted sign field. Bit 7 is replaced by parity,
// matching ggml unpack_ksigns (iq_kernels.dp.cpp:60).
inline uint32_t b70_unpack_ksigns(uint32_t v) {
    v &= 0xffu;
    const uint32_t p = (uint32_t)sycl::popcount(v) & 1u;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

inline void b70_q8_i8(const B70Q8& a, int u[8]) {
    const int* p = reinterpret_cast<const int*>(a.qs);
#pragma unroll
    for (int i = 0; i < 8; ++i) u[i] = p[i];
}

inline void b70_grid_signed(uint32_t lo, uint32_t hi, uint32_t signs, int& g0, int& g1) {
    const uint32_t m0 = (uint32_t)b70_swar_ne4(signs & 0x08040201u);
    const uint32_t m1 = (uint32_t)b70_swar_ne4(signs & 0x80402010u);
    g0 = b70_sgn(lo, m0);
    g1 = b70_sgn(hi, m1);
}

// IQ2_S / IQ3_S sign byte: bits 0..7 select weights 0..7. Not ksign7.
// Same masks as NQ<122> / NQ<121> (iq_kernels.dp.cpp:2651, :2677).
inline void b70_byte_signed(uint32_t sb, uint32_t lo, uint32_t hi, int& g0, int& g1) {
    const uint32_t m0 = (uint32_t)b70_swar_ne4(((sb & 0x03u) << 7) | ((sb & 0x0Cu) << 21));
    const uint32_t m1 = (uint32_t)b70_swar_ne4(((sb & 0x30u) << 3) | ((sb & 0xC0u) << 17));
    g0 = b70_sgn(lo, m0);
    g1 = b70_sgn(hi, m1);
}

// IQ4_NL: four nibbles in the low 4 bits of each byte -> four codebook bytes.
// night::iq4nl_mux4, iq_kernels.dp.cpp:2437.
inline uint32_t b70_iq4nl_mux4(uint32_t q) {
    const uint32_t b0 = (q & 0x01010101u) * 0xFFu;
    const uint32_t b1 = ((q >> 1) & 0x01010101u) * 0xFFu;
    const uint32_t b2 = ((q >> 2) & 0x01010101u) * 0xFFu;
    const uint32_t b3 = ((q >> 3) & 0x01010101u) * 0xFFu;
    auto sel = [](uint32_t x, uint32_t y, uint32_t m) { return (x & ~m) | (y & m); };
    uint32_t r1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t c0 = (uint32_t)(uint8_t)kIq4nl[2 * i] * 0x01010101u;
        const uint32_t c1 = (uint32_t)(uint8_t)kIq4nl[2 * i + 1] * 0x01010101u;
        r1[i] = sel(c0, c1, b0);
    }
    uint32_t r2[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) r2[i] = sel(r1[2 * i], r1[2 * i + 1], b1);
    return sel(sel(r2[0], r2[1], b2), sel(r2[2], r2[3], b2), b3);
}

// Q2_0: one byte of four 2-bit fields -> four signed bytes (q - 1).
// night::q2_spread4, iq_kernels.dp.cpp:2431.
inline uint32_t b70_q2_spread4(uint32_t b) {
    uint32_t t = (b | (b << 12)) & 0x000F000Fu;
    t = (t | (t << 6)) & 0x03030303u;
    return (t + 0x7F7F7F7Fu) ^ 0x80808080u;
}

inline uint32_t b70_nibble_word(const uint8_t* qs, int base, int hi) {
    uint32_t w = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t byte = hi ? (qs[base + i] >> 4) : (qs[base + i] & 15u);
        w |= byte << (8 * i);
    }
    return w;
}

// Eight packed signed words for one 32-weight group, plus the integer
// finish (same truncating divisions as Multi<>::finish). fin:
// 0 plain sum, 1 IQ2_XXS, 2 IQ2_XS/IQ2_S split halves, 3 IQ3_XXS, 4 IQ3_S.
struct B70Part {
    int w[8];
    int a;
    int b;
    float d;
    int fin;
    // K-quant / IQ4_XS integer form (O3 stage 2): per-16 scales, per-16
    // mins, and the min multiplier. Unused by the IQ types.
    int sc0;
    int sc1;
    int mn0;
    int mn1;
    float dm;
};

// O3 stage 2: the float decode32 types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
// IQ4_XS) as signed-byte words + integer scales, dotted with DP4A like
// ggml vec_dot_*_q8_1. -DB70_KQ_DP4A=0 keeps the old decode32 + fma path.
#ifndef B70_KQ_DP4A
#define B70_KQ_DP4A 1
#endif
template <int TYPE>
inline constexpr bool b70_kq_type() {
    return B70_KQ_DP4A && (TYPE == B70_Q2_K || TYPE == B70_Q3_K || TYPE == B70_Q4_K || TYPE == B70_Q5_K ||
                           TYPE == B70_Q6_K || TYPE == B70_IQ4_XS);
}

template <int TYPE>
inline constexpr bool b70_dp4a_type() {
    return TYPE == B70_Q8_0 || TYPE == B70_IQ2_XXS || TYPE == B70_IQ2_XS || TYPE == B70_IQ2_S || TYPE == B70_IQ3_XXS ||
           TYPE == B70_IQ3_S || TYPE == B70_IQ4_NL || TYPE == B70_Q2_0 || b70_kq_type<TYPE>();
}

template <int TYPE>
inline void b70_prep(const uint8_t* row, int gi, B70Part& m) {
    if constexpr (TYPE == B70_Q8_0) {
        // qs is at offset 2, so the 32 int8 are not 4-aligned. Pack bytes.
        const uint8_t* b = row + (size_t)gi * 34;
        const uint8_t* qs = b + 2;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t w = (uint32_t)qs[4 * j] | ((uint32_t)qs[4 * j + 1] << 8) | ((uint32_t)qs[4 * j + 2] << 16) |
                         ((uint32_t)qs[4 * j + 3] << 24);
            m.w[j] = (int)w;
        }
        m.d = b70_h2f(b);
        m.fin = 0;
    } else if constexpr (TYPE == B70_IQ2_XXS) {
        const int ib = gi & 7;
        const uint8_t* b = row + (size_t)(gi >> 3) * 66;
        const uint64_t qq = b70_ld8_a2(b + 2 + 8 * ib);
        const uint32_t q2 = (uint32_t)qq;
        const uint32_t aux = (uint32_t)(qq >> 32);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t g = iq2xxs_grid[(q2 >> (8 * k)) & 0xffu];
            int g0, g1;
            b70_grid_signed((uint32_t)g, (uint32_t)(g >> 32), b70_unpack_ksigns(aux >> (7 * k)), g0, g1);
            m.w[2 * k] = g0;
            m.w[2 * k + 1] = g1;
        }
        m.a = (int)((aux >> 27) | 1u);
        m.d = b70_h2f(b);
        m.fin = 1;
    } else if constexpr (TYPE == B70_IQ2_XS) {
        const int ib = gi & 7;
        const uint8_t* b = row + (size_t)(gi >> 3) * 74;
        const uint64_t qq = b70_ld8_a2(b + 2 + 8 * ib);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q = (uint32_t)((qq >> (16 * k)) & 0xffffu);
            const uint64_t g = iq2xs_grid[q & 511u];
            int g0, g1;
            b70_grid_signed((uint32_t)g, (uint32_t)(g >> 32), b70_unpack_ksigns(q >> 9), g0, g1);
            m.w[2 * k] = g0;
            m.w[2 * k + 1] = g1;
        }
        m.a = b[66 + ib] & 0x0f;
        m.b = b[66 + ib] >> 4;
        m.d = b70_h2f(b);
        m.fin = 2;
    } else if constexpr (TYPE == B70_IQ2_S) {
        const int ib = gi & 7;
        const uint8_t* b = row + (size_t)(gi >> 3) * 82;
        const uint32_t qh = b[66 + ib];
        const uint32_t qs4 = b70_ld4_a2(b + 2 + 4 * ib);
        const uint32_t sb4 = b70_ld4_a2(b + 34 + 4 * ib);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t qs = (qs4 >> (8 * k)) & 0xffu;
            const uint32_t sb = (sb4 >> (8 * k)) & 0xffu;
            const uint32_t idx = qs | ((qh << (8 - 2 * k)) & 0x300u);
            const uint64_t g = iq2s_grid[idx];
            int g0, g1;
            b70_byte_signed(sb, (uint32_t)g, (uint32_t)(g >> 32), g0, g1);
            m.w[2 * k] = g0;
            m.w[2 * k + 1] = g1;
        }
        m.a = b[74 + ib] & 0x0f;
        m.b = b[74 + ib] >> 4;
        m.d = b70_h2f(b);
        m.fin = 2;
    } else if constexpr (TYPE == B70_IQ3_XXS) {
        const int ib = gi & 7;
        const uint8_t* b = row + (size_t)(gi >> 3) * 98;
        const uint32_t aux = b70_ld4_a2(b + 66 + 4 * ib);
        const uint64_t qq = b70_ld8_a2(b + 2 + 8 * ib);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t qw = (uint32_t)((qq >> (16 * k)) & 0xffffu);
            int g0, g1;
            b70_grid_signed(iq3xxs_grid[qw & 0xffu], iq3xxs_grid[qw >> 8], b70_unpack_ksigns(aux >> (7 * k)), g0, g1);
            m.w[2 * k] = g0;
            m.w[2 * k + 1] = g1;
        }
        m.a = (int)(aux >> 28);
        m.d = b70_h2f(b);
        m.fin = 3;
    } else if constexpr (TYPE == B70_IQ3_S) {
        const int ib = gi & 7;
        const uint8_t* b = row + (size_t)(gi >> 3) * 110;
        const uint32_t qh = b[66 + ib];
        const uint64_t qq = b70_ld8_a2(b + 2 + 8 * ib);
        const uint32_t sb4 = b70_ld4_a2(b + 74 + 4 * ib);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t qw = (uint32_t)((qq >> (16 * k)) & 0xffffu);
            const uint32_t sb = (sb4 >> (8 * k)) & 0xffu;
            const uint32_t i0 = (qw & 0xffu) | ((qh << (8 - 2 * k)) & 0x100u);
            const uint32_t i1 = (qw >> 8) | ((qh << (7 - 2 * k)) & 0x100u);
            int g0, g1;
            b70_byte_signed(sb, iq3s_grid[i0], iq3s_grid[i1], g0, g1);
            m.w[2 * k] = g0;
            m.w[2 * k + 1] = g1;
        }
        m.a = 1 + 2 * (int)((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 0x0fu);
        m.d = b70_h2f(b);
        m.fin = 4;
    } else if constexpr (TYPE == B70_IQ4_NL) {
        const uint8_t* b = row + (size_t)gi * 18;
        const uint64_t q0 = b70_ld8_a2(b + 2);
        const uint64_t q1 = b70_ld8_a2(b + 10);
        const uint32_t qw[4] = {(uint32_t)q0, (uint32_t)(q0 >> 32), (uint32_t)q1, (uint32_t)(q1 >> 32)};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            m.w[i] = (int)b70_iq4nl_mux4(qw[i] & 0x0f0f0f0fu);
            m.w[4 + i] = (int)b70_iq4nl_mux4((qw[i] >> 4) & 0x0f0f0f0fu);
        }
        m.d = b70_h2f(b);
        m.fin = 0;
    } else if constexpr (TYPE == B70_Q2_0) {
        const uint8_t* b = row + (size_t)(gi >> 1) * 18;
        const uint64_t qq = b70_ld8_a2(b + 2 + (gi & 1) * 8);
#pragma unroll
        for (int i = 0; i < 8; ++i) m.w[i] = (int)b70_q2_spread4((uint32_t)((qq >> (8 * i)) & 0xffu));
        m.d = b70_h2f(b);
        m.fin = 0;
    } else if constexpr (TYPE == B70_IQ4_XS) {
        // 136-byte block: d, scales_h (u16), scales_l[4], qs[128]. Part ib = qs[16 ib..16 ib+15].
        const int ib = gi & 7;
        const uint8_t* b = row + (size_t)(gi >> 3) * 136;
        const uint32_t hdr = b70_ld4_a2(b);           // d | scales_h
        const uint32_t sl = b70_ld4_a2(b + 4);        // scales_l
        const int ls = (int)((sl >> (4 * ib)) & 0xfu) | (int)(((hdr >> (16 + 2 * ib)) & 3u) << 4);
        const uint8_t* qs = b + 8 + 16 * ib;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t q = b70_ld4_a2(qs + 4 * i);
            m.w[i] = (int)b70_iq4nl_mux4(q & 0x0f0f0f0fu);
            m.w[4 + i] = (int)b70_iq4nl_mux4((q >> 4) & 0x0f0f0f0fu);
        }
        m.d = b70_h2f(b) * (float)(ls - 32);
        m.fin = 0;
    } else if constexpr (TYPE == B70_Q2_K) {
        // 84-byte block: scales[16], qs[64], d, dmin. Part gi8 = half*4 + j, shift 2j.
        const int gi8 = gi & 7;
        const int half = gi8 >> 2;
        const int j = gi8 & 3;
        const uint8_t* b = row + (size_t)(gi >> 3) * 84;
        const uint8_t* qs = b + 16 + half * 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) m.w[i] = (int)((b70_ld4_a2(qs + 4 * i) >> (2 * j)) & 0x03030303u);
        const uint32_t sc = b70_ld4_a2(b + half * 8 + 2 * j) & 0xffffu;
        m.sc0 = (int)(sc & 0xfu);
        m.mn0 = (int)((sc >> 4) & 0xfu);
        m.sc1 = (int)((sc >> 8) & 0xfu);
        m.mn1 = (int)((sc >> 12) & 0xfu);
        const uint32_t dd = b70_ld4_a2(b + 80);
        m.d = (float)sycl::bit_cast<sycl::half>((uint16_t)(dd & 0xffffu));
        m.dm = (float)sycl::bit_cast<sycl::half>((uint16_t)(dd >> 16));
        m.fin = 5;
    } else if constexpr (TYPE == B70_Q3_K) {
        // 110-byte block: hmask[32], qs[64], scales[12], d. Byte = q - 4 + 4*hbit in -4..3.
        const int gi8 = gi & 7;
        const int half = gi8 >> 2;
        const int j = gi8 & 3;
        const uint8_t* b = row + (size_t)(gi >> 3) * 110;
        const uint8_t* qs = b + 32 + half * 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = (b70_ld4_a2(qs + 4 * i) >> (2 * j)) & 0x03030303u;
            const uint32_t h = ((b70_ld4_a2(b + 4 * i) >> (half * 4 + j)) & 0x01010101u) << 2;
            m.w[i] = (int)(((q | h) + 0x7C7C7C7Cu) ^ 0x80808080u);
        }
        // b70_q3_scale with aligned word loads (scales at b+96, 2-aligned).
        const uint32_t a0 = b70_ld4_a2(b + 96);
        const uint32_t a1 = b70_ld4_a2(b + 100);
        const uint32_t tmp = b70_ld4_a2(b + 104);
        const uint32_t o[4] = {(a0 & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4),
                               (a1 & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4),
                               ((a0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4),
                               ((a1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4)};
        const int is0 = half * 8 + 2 * j;
        const int is1 = is0 + 1;
        m.sc0 = (int)(int8_t)((o[is0 >> 2] >> (8 * (is0 & 3))) & 0xffu) - 32;
        m.sc1 = (int)(int8_t)((o[is1 >> 2] >> (8 * (is1 & 3))) & 0xffu) - 32;
        m.d = (float)sycl::bit_cast<sycl::half>((uint16_t)(b70_ld4_a2(b + 108) & 0xffffu));
        m.fin = 6;
    } else if constexpr (TYPE == B70_Q4_K || TYPE == B70_Q5_K) {
        // 144 / 176-byte block: d, dmin, scales[12], (qh[32]), qs[128]. Part gi8: nibble half of qs[32 j64..].
        constexpr int BB = TYPE == B70_Q4_K ? 144 : 176;
        constexpr int QO = TYPE == B70_Q4_K ? 16 : 48;
        const int gi8 = gi & 7;
        const int j64 = gi8 >> 1;
        const int hi = gi8 & 1;
        const uint8_t* b = row + (size_t)(gi >> 3) * BB;
        const uint8_t* q = b + QO + 32 * j64;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t w = (b70_ld4_a2(q + 4 * i) >> (4 * hi)) & 0x0f0f0f0fu;
            if constexpr (TYPE == B70_Q5_K) w |= ((b70_ld4_a2(b + 16 + 4 * i) >> gi8) & 0x01010101u) << 4;
            m.w[i] = (int)w;
        }
        // get_scale_min_k4 on scales[12] at b+4.
        const uint32_t s0 = b70_ld4_a2(b + 4);
        const uint32_t s1 = b70_ld4_a2(b + 8);
        const uint32_t s2 = b70_ld4_a2(b + 12);
        auto byte = [](uint32_t w, int k) { return (int)((w >> (8 * k)) & 0xffu); };
        int sc, mn;
        if (gi8 < 4) {
            sc = byte(s0, gi8) & 63;
            mn = byte(s1, gi8) & 63;
        } else {
            sc = (byte(s2, gi8 - 4) & 0xF) | ((byte(s0, gi8 - 4) >> 6) << 4);
            mn = (byte(s2, gi8 - 4) >> 4) | ((byte(s1, gi8 - 4) >> 6) << 4);
        }
        m.sc0 = sc;
        m.sc1 = sc;
        m.mn0 = mn;
        m.mn1 = mn;
        const uint32_t dd = b70_ld4_a2(b);
        m.d = (float)sycl::bit_cast<sycl::half>((uint16_t)(dd & 0xffffu));
        m.dm = (float)sycl::bit_cast<sycl::half>((uint16_t)(dd >> 16));
        m.fin = 5;
    } else if constexpr (TYPE == B70_Q6_K) {
        // 210-byte block: ql[128], qh[64], scales[16] (int8), d. Byte = q6 - 32 in -32..31.
        const int gi8 = gi & 7;
        const int n = gi8 >> 2;
        const int qu = gi8 & 3;
        const uint8_t* b = row + (size_t)(gi >> 3) * 210;
        const uint8_t* ql = b + 64 * n + ((qu & 1) ? 32 : 0);
        const uint8_t* qh = b + 128 + 32 * n;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t lo = (b70_ld4_a2(ql + 4 * i) >> ((qu >> 1) * 4)) & 0x0f0f0f0fu;
            const uint32_t h2 = ((b70_ld4_a2(qh + 4 * i) >> (2 * qu)) & 0x03030303u) << 4;
            m.w[i] = (int)(((lo | h2) + 0x60606060u) ^ 0x80808080u);
        }
        const uint32_t sc = b70_ld4_a2(b + 192 + 8 * n + 2 * qu) & 0xffffu;
        m.sc0 = (int)(int8_t)(sc & 0xffu);
        m.sc1 = (int)(int8_t)(sc >> 8);
        m.d = (float)sycl::bit_cast<sycl::half>((uint16_t)(b70_ld4_a2(b + 208) & 0xffffu));
        m.fin = 6;
    }
}

template <int TYPE>
inline float b70_apply_u(const B70Part& m, const int u[8], float ad) {
    if constexpr (TYPE == B70_Q2_K || TYPE == B70_Q3_K || TYPE == B70_Q4_K || TYPE == B70_Q5_K ||
                  TYPE == B70_Q6_K) {
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) s0 = b70_dp4a(m.w[j], u[j], s0);
#pragma unroll
        for (int j = 4; j < 8; ++j) s1 = b70_dp4a(m.w[j], u[j], s1);
        float r = m.d * (float)(s0 * m.sc0 + s1 * m.sc1);
        if constexpr (TYPE == B70_Q2_K || TYPE == B70_Q4_K || TYPE == B70_Q5_K) {
            int a0 = 0, a1 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) a0 = b70_dp4a(0x01010101, u[j], a0);
#pragma unroll
            for (int j = 4; j < 8; ++j) a1 = b70_dp4a(0x01010101, u[j], a1);
            r -= m.dm * (float)(a0 * m.mn0 + a1 * m.mn1);
        }
        return r * ad;
    }
    int sumi;
    if constexpr (TYPE == B70_IQ2_XS || TYPE == B70_IQ2_S) {
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) s0 = b70_dp4a(m.w[j], u[j], s0);
#pragma unroll
        for (int j = 4; j < 8; ++j) s1 = b70_dp4a(m.w[j], u[j], s1);
        sumi = (s0 * m.a + s1 * m.b + (s0 + s1) / 2) / 4;
    } else {
        int sum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sum = b70_dp4a(m.w[j], u[j], sum);
        if constexpr (TYPE == B70_IQ2_XXS) sumi = sum * m.a / 8;
        else if constexpr (TYPE == B70_IQ3_XXS) sumi = (m.a * sum + sum / 2) / 2;
        else if constexpr (TYPE == B70_IQ3_S) sumi = sum * m.a;
        else sumi = sum;
    }
    return m.d * ad * (float)sumi;
}

template <int TYPE>
inline float b70_dot32_f(const uint8_t* row, int gi, const B70Q8& a) {
    float v[32];
    b70_decode32<TYPE>(row, gi, v);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) s = sycl::fma(v[j], (float)a.qs[j], s);
    return s * a.d;
}

template <int TYPE>
inline float b70_dot32(const uint8_t* row, int gi, const B70Q8& a) {
    if constexpr (b70_dp4a_type<TYPE>()) {
        B70Part m;
        b70_prep<TYPE>(row, gi, m);
        int u[8];
        b70_q8_i8(a, u);
        return b70_apply_u<TYPE>(m, u, a.d);
    } else {
        return b70_dot32_f<TYPE>(row, gi, a);
    }
}

// Loop stays inside the case so each arm is a straight DP4A (or float) walk.
inline float b70_acc_k(int type, const uint8_t* row, int ng, int lane, const B70Q8* xr) {
    float acc = 0.f;
    switch (type) {
    case B70_IQ2_XXS:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_IQ2_XXS>(row, g, xr[g]);
        break;
    case B70_IQ2_XS:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_IQ2_XS>(row, g, xr[g]);
        break;
    case B70_IQ2_S:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_IQ2_S>(row, g, xr[g]);
        break;
    case B70_IQ3_XXS:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_IQ3_XXS>(row, g, xr[g]);
        break;
    case B70_IQ3_S:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_IQ3_S>(row, g, xr[g]);
        break;
    case B70_IQ4_NL:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_IQ4_NL>(row, g, xr[g]);
        break;
    case B70_Q2_0:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_Q2_0>(row, g, xr[g]);
        break;
    case B70_Q8_0:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_Q8_0>(row, g, xr[g]);
        break;
    case B70_Q4_K:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_Q4_K>(row, g, xr[g]);
        break;
    case B70_Q5_K:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_Q5_K>(row, g, xr[g]);
        break;
    case B70_Q6_K:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_Q6_K>(row, g, xr[g]);
        break;
    case B70_IQ4_XS:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_IQ4_XS>(row, g, xr[g]);
        break;
    case B70_Q2_K:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_Q2_K>(row, g, xr[g]);
        break;
    case B70_Q3_K:
        for (int g = lane; g < ng; g += 16) acc += b70_dot32<B70_Q3_K>(row, g, xr[g]);
        break;
    default: break;
    }
    return acc;
}
