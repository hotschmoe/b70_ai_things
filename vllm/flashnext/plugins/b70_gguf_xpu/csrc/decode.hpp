// GGUF block decode for the Flash-Next IQ3_XXS mix.
// Arithmetic follows llama.cpp ggml dequantize_row_* (MIT) and Strata's
// night load_raw/decode (sycl/src/kernels/cuda/night_grouped_expert.inc):
// same integers, fp16 scale, sign byte = s7 | parity(s7)<<7.
// Host and device: no SYCL types, little-endian loads.
#pragma once
#include <bit>
#include <cstdint>
#include "tables.inc"

enum B70GgufType {
    B70_Q4_K = 12,
    B70_Q5_K = 13,
    B70_Q6_K = 14,
    B70_IQ2_XXS = 16,
    B70_IQ2_XS = 17,
    B70_IQ3_XXS = 18,
    B70_IQ4_NL = 20,
    B70_IQ3_S = 21,
    B70_IQ2_S = 22,
    B70_IQ4_XS = 23,
    B70_Q2_K = 10,
    B70_Q3_K = 11,
    B70_Q2_0 = 42
};

inline bool b70_type_info(int type, int& block_elems, int& block_bytes) {
    switch (type) {
    case B70_Q4_K: block_elems = 256; block_bytes = 144; return true;
    case B70_Q5_K: block_elems = 256; block_bytes = 176; return true;
    case B70_Q6_K: block_elems = 256; block_bytes = 210; return true;
    case B70_IQ2_XXS: block_elems = 256; block_bytes = 66; return true;
    case B70_IQ2_XS: block_elems = 256; block_bytes = 74; return true;
    case B70_IQ3_XXS: block_elems = 256; block_bytes = 98; return true;
    case B70_IQ4_NL: block_elems = 32; block_bytes = 18; return true;
    case B70_IQ3_S: block_elems = 256; block_bytes = 110; return true;
    case B70_IQ2_S: block_elems = 256; block_bytes = 82; return true;
    case B70_IQ4_XS: block_elems = 256; block_bytes = 136; return true;
    case B70_Q2_K: block_elems = 256; block_bytes = 84; return true;
    case B70_Q3_K: block_elems = 256; block_bytes = 110; return true;
    case B70_Q2_0: block_elems = 64; block_bytes = 18; return true;
    default: return false;
    }
}

inline int b70_row_bytes(int type, int k) {
    int be = 0, bb = 0;
    if (!b70_type_info(type, be, bb) || k <= 0 || k % be != 0) return 0;
    return (k / be) * bb;
}

inline uint32_t b70_ld16(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}

inline uint32_t b70_ld32(const uint8_t* p) {
    return b70_ld16(p) | (b70_ld16(p + 2) << 16);
}

// Q3_K 6-bit scales are packed into 12 bytes. ggml dequantize_row_q3_K
// (and strata dequant.hpp dequantize_q3_K) expands them with kmask1/kmask2
// before the -32 bias. `is` is 0..15.
inline int b70_q3_scale(const uint8_t* scales12, int is) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t a0 = b70_ld32(scales12);
    const uint32_t a1 = b70_ld32(scales12 + 4);
    const uint32_t tmp = b70_ld32(scales12 + 8);
    const uint32_t o0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
    const uint32_t o1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const uint32_t o2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    const uint32_t o3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    const uint32_t o[4] = {o0, o1, o2, o3};
    const uint32_t byte = (o[is >> 2] >> (8 * (is & 3))) & 0xffu;
    return (int)(int8_t)byte - 32;
}

inline uint32_t b70_ksign7(uint32_t s7) {
    uint32_t x = s7;
    uint32_t c = 0;
    while (x) { c += x & 1u; x >>= 1; }
    return s7 | ((c & 1u) << 7);
}

inline float b70_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h & 0x7c00u) >> 10;
    uint32_t mant = h & 0x03ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03ffu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    return std::bit_cast<float>(f);
}

inline uint16_t b70_fp32_to_bf16(float v) {
    uint32_t u = std::bit_cast<uint32_t>(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 64u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

inline float b70_bf16_to_fp32(uint16_t b) {
    return std::bit_cast<float>((uint32_t)b << 16);
}

inline float b70_grid_byte_u64(uint64_t g, int j) {
    return (float)((g >> (8 * j)) & 0xffu);
}
inline float b70_grid_byte_u32(uint32_t g, int j) {
    return (float)((g >> (8 * j)) & 0xffu);
}

// kvalues_iq4nl from ggml (non-linear 4-bit codebook).
inline constexpr int8_t kIq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
inline int b70_iq4nl(int nib) { return (int)kIq4nl[nib & 15]; }

// Q4_K / Q5_K 6-bit scale and min, ggml get_scale_min_k4.
inline void b70_scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Decode 32 weights starting at group gi (element gi*32) of one row.
template <int TYPE>
inline void b70_decode32(const uint8_t* row, int gi, float* v) {
    if constexpr (TYPE == B70_Q2_0) {
        const uint8_t* b = row + (gi / 2) * 18;
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b));
        const int e0 = (gi % 2) * 32;
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            v[j] = (float)(q - 1) * d;
        }
    } else if constexpr (TYPE == B70_IQ4_NL) {
        const uint8_t* b = row + gi * 18;
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b));
        for (int j = 0; j < 16; ++j) {
            v[j] = d * (float)b70_iq4nl(b[2 + j] & 0xf);
            v[j + 16] = d * (float)b70_iq4nl(b[2 + j] >> 4);
        }
    } else if constexpr (TYPE == B70_Q4_K) {
        const uint8_t* b = row + (gi / 8) * 144;
        const int gi8 = gi % 8;
        const int j64 = gi8 / 2;
        const int hi = gi8 % 2;
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b));
        const float dmin = b70_fp16_to_fp32((uint16_t)b70_ld16(b + 2));
        int sc, mn;
        b70_scale_min_k4(gi8, b + 4, sc, mn);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)mn;
        const uint8_t* q = b + 16 + 32 * j64;
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (q[l] >> 4) : (q[l] & 0xF);
            v[l] = d1 * (float)nib - m1;
        }
    } else if constexpr (TYPE == B70_Q5_K) {
        const uint8_t* b = row + (gi / 8) * 176;
        const int gi8 = gi % 8;
        const int j64 = gi8 / 2;
        const int hi = gi8 % 2;
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b));
        const float dmin = b70_fp16_to_fp32((uint16_t)b70_ld16(b + 2));
        int sc, mn;
        b70_scale_min_k4(gi8, b + 4, sc, mn);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)mn;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t bit = (uint8_t)(1u << gi8);
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            v[l] = d1 * (float)(nib + ((qh[l] & bit) ? 16 : 0)) - m1;
        }
    } else if constexpr (TYPE == B70_Q6_K) {
        const uint8_t* b = row + (gi / 8) * 210;
        const int gi8 = gi % 8;
        const int n = gi8 / 4;
        const int qu = gi8 % 4;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const uint8_t* scb = b + 192 + 8 * n;
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b + 208));
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            const int sc = (int)(int8_t)scb[is + 2 * qu];
            v[l] = d * (float)sc * (float)(q - 32);
        }
    } else if constexpr (TYPE == B70_IQ4_XS) {
        const uint8_t* b = row + (gi / 8) * 136;
        const int ib = gi % 8;
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b));
        const uint32_t scales_h = b70_ld16(b + 2);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) |
                        (int)(((scales_h >> (2 * ib)) & 3u) << 4);
        const float dl = d * (float)(ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            v[j] = dl * (float)b70_iq4nl(qs[j] & 0xf);
            v[j + 16] = dl * (float)b70_iq4nl(qs[j] >> 4);
        }
    } else if constexpr (TYPE == B70_IQ2_XXS) {
        const uint8_t* b = row + (gi / 8) * 66;
        const int ib = gi % 8;
        const uint32_t q0 = b70_ld16(b + 2 + 8 * ib);
        const uint32_t q1 = b70_ld16(b + 4 + 8 * ib);
        const uint32_t aux32 = b70_ld16(b + 6 + 8 * ib) | (b70_ld16(b + 8 + 8 * ib) << 16);
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b)) * (0.5f + (float)(aux32 >> 28)) * 0.25f;
        const uint32_t idx[4] = {q0 & 0xffu, q0 >> 8, q1 & 0xffu, q1 >> 8};
        for (int il = 0; il < 4; ++il) {
            const uint64_t g = iq2xxs_grid[idx[il]];
            const uint32_t signs = b70_ksign7((aux32 >> (7 * il)) & 127u);
            for (int j = 0; j < 8; ++j)
                v[8 * il + j] = d * b70_grid_byte_u64(g, j) * ((signs >> j) & 1 ? -1.f : 1.f);
        }
    } else if constexpr (TYPE == B70_IQ2_XS) {
        const uint8_t* b = row + (gi / 8) * 74;
        const int ib = gi % 8;
        const float db = b70_fp16_to_fp32((uint16_t)b70_ld16(b));
        const uint32_t scale = b[66 + ib];
        for (int il = 0; il < 4; ++il) {
            const uint32_t q = b70_ld16(b + 2 + 8 * ib + 2 * il);
            const uint64_t g = iq2xs_grid[q & 511u];
            const float d = db * (0.5f + (float)((scale >> (4 * (il / 2))) & 0xfu)) * 0.25f;
            const uint32_t signs = b70_ksign7(q >> 9);
            for (int j = 0; j < 8; ++j)
                v[8 * il + j] = d * b70_grid_byte_u64(g, j) * ((signs >> j) & 1 ? -1.f : 1.f);
        }
    } else if constexpr (TYPE == B70_IQ2_S) {
        const uint8_t* b = row + (gi / 8) * 82;
        const int ib = gi % 8;
        const float db = b70_fp16_to_fp32((uint16_t)b70_ld16(b));
        const uint32_t qh = b[66 + ib];
        const uint32_t scale = b[74 + ib];
        for (int il = 0; il < 4; ++il) {
            const uint32_t qs = b[2 + 4 * ib + il];
            const uint32_t signs = b[2 + 32 + 4 * ib + il];
            const uint32_t idx = qs | (((qh << (8 - 2 * il)) & 0x300u));
            const uint64_t g = iq2s_grid[idx];
            const float d = db * (0.5f + (float)((scale >> (4 * (il / 2))) & 0xfu)) * 0.25f;
            for (int j = 0; j < 8; ++j)
                v[8 * il + j] = d * b70_grid_byte_u64(g, j) * ((signs >> j) & 1 ? -1.f : 1.f);
        }
    } else if constexpr (TYPE == B70_IQ3_XXS) {
        const uint8_t* b = row + (gi / 8) * 98;
        const int ib = gi % 8;
        const uint32_t aux32 = b70_ld16(b + 66 + 4 * ib) | (b70_ld16(b + 68 + 4 * ib) << 16);
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b)) * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        for (int il = 0; il < 4; ++il) {
            const uint32_t qw = b70_ld16(b + 2 + 8 * ib + 2 * il);
            const uint32_t g1 = iq3xxs_grid[qw & 0xffu];
            const uint32_t g2 = iq3xxs_grid[qw >> 8];
            const uint32_t signs = b70_ksign7((aux32 >> (7 * il)) & 127u);
            for (int j = 0; j < 4; ++j) {
                v[8 * il + j] = d * b70_grid_byte_u32(g1, j) * ((signs >> j) & 1 ? -1.f : 1.f);
                v[8 * il + 4 + j] = d * b70_grid_byte_u32(g2, j) * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
            }
        }
    } else if constexpr (TYPE == B70_IQ3_S) {
        const uint8_t* b = row + (gi / 8) * 110;
        const int ib = gi % 8;
        const uint32_t scb = b[106 + ib / 2];
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b)) *
                        (float)(1 + 2 * (int)((scb >> (4 * (ib % 2))) & 0xfu));
        const uint32_t qh = b[66 + ib];
        for (int il = 0; il < 4; ++il) {
            const uint32_t qw = b70_ld16(b + 2 + 8 * ib + 2 * il);
            const uint32_t signs = b[74 + 4 * ib + il];
            const uint32_t i1 = (qw & 0xffu) | ((qh << (8 - 2 * il)) & 256u);
            const uint32_t i2 = (qw >> 8) | ((qh << (7 - 2 * il)) & 256u);
            const uint32_t g1 = iq3s_grid[i1];
            const uint32_t g2 = iq3s_grid[i2];
            for (int j = 0; j < 4; ++j) {
                v[8 * il + j] = d * b70_grid_byte_u32(g1, j) * ((signs >> j) & 1 ? -1.f : 1.f);
                v[8 * il + 4 + j] = d * b70_grid_byte_u32(g2, j) * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
            }
        }
    } else if constexpr (TYPE == B70_Q2_K) {
        // block_q2_K: scales[16], qs[64], fp16 d, fp16 dmin. 84 bytes.
        // dequantize_row_q2_K: 16 groups of 16, two 128-halves, shift 0/2/4/6.
        const uint8_t* b = row + (gi / 8) * 84;
        const int gi8 = gi % 8;
        const int half = gi8 / 4;
        const int j = gi8 % 4;
        const int shift = j * 2;
        const uint8_t* qs = b + 16 + half * 32;
        const uint8_t sc0 = b[half * 8 + 2 * j];
        const uint8_t sc1 = b[half * 8 + 2 * j + 1];
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b + 80));
        const float mn = b70_fp16_to_fp32((uint16_t)b70_ld16(b + 82));
        const float dl0 = d * (float)(sc0 & 0xF);
        const float ml0 = mn * (float)(sc0 >> 4);
        const float dl1 = d * (float)(sc1 & 0xF);
        const float ml1 = mn * (float)(sc1 >> 4);
        for (int l = 0; l < 16; ++l) {
            v[l] = dl0 * (float)((qs[l] >> shift) & 3) - ml0;
            v[l + 16] = dl1 * (float)((qs[l + 16] >> shift) & 3) - ml1;
        }
    } else if constexpr (TYPE == B70_Q3_K) {
        // block_q3_K: hmask[32], qs[64], scales[12], fp16 d. 110 bytes. No dmin.
        // The high-bit mask is NOT reset between the two 128-halves.
        const uint8_t* b = row + (gi / 8) * 110;
        const int gi8 = gi % 8;
        const int half = gi8 / 4;
        const int j = gi8 % 4;
        const int shift = j * 2;
        const int mbit = 1 << (half * 4 + j);
        const uint8_t* hm = b;
        const uint8_t* qs = b + 32 + half * 32;
        const float d = b70_fp16_to_fp32((uint16_t)b70_ld16(b + 108));
        const float dl0 = d * (float)b70_q3_scale(b + 96, half * 8 + 2 * j);
        const float dl1 = d * (float)b70_q3_scale(b + 96, half * 8 + 2 * j + 1);
        for (int l = 0; l < 16; ++l) {
            const int q0 = (qs[l] >> shift) & 3;
            const int q1 = (qs[l + 16] >> shift) & 3;
            v[l] = dl0 * (float)(q0 - ((hm[l] & mbit) ? 0 : 4));
            v[l + 16] = dl1 * (float)(q1 - ((hm[l + 16] & mbit) ? 0 : 4));
        }
    }
}

inline void b70_decode32_switch(int type, const uint8_t* row, int gi, float* v) {
    switch (type) {
    case B70_Q2_0: b70_decode32<B70_Q2_0>(row, gi, v); break;
    case B70_IQ4_NL: b70_decode32<B70_IQ4_NL>(row, gi, v); break;
    case B70_Q4_K: b70_decode32<B70_Q4_K>(row, gi, v); break;
    case B70_Q5_K: b70_decode32<B70_Q5_K>(row, gi, v); break;
    case B70_Q6_K: b70_decode32<B70_Q6_K>(row, gi, v); break;
    case B70_IQ4_XS: b70_decode32<B70_IQ4_XS>(row, gi, v); break;
    case B70_IQ2_XXS: b70_decode32<B70_IQ2_XXS>(row, gi, v); break;
    case B70_IQ2_XS: b70_decode32<B70_IQ2_XS>(row, gi, v); break;
    case B70_IQ2_S: b70_decode32<B70_IQ2_S>(row, gi, v); break;
    case B70_IQ3_XXS: b70_decode32<B70_IQ3_XXS>(row, gi, v); break;
    case B70_IQ3_S: b70_decode32<B70_IQ3_S>(row, gi, v); break;
    case B70_Q2_K: b70_decode32<B70_Q2_K>(row, gi, v); break;
    case B70_Q3_K: b70_decode32<B70_Q3_K>(row, gi, v); break;
    default: break;
    }
}

// Full-row fp32 dequant. rows * cols elements, row-major, one quant row = K = cols.
inline void b70_dequant_f32_rows(int type, const uint8_t* q, int64_t rows, int64_t cols, float* out) {
    const int rb = b70_row_bytes(type, (int)cols);
    const int groups = (int)(cols / 32);
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row = q + r * rb;
        float* y = out + r * cols;
        for (int g = 0; g < groups; ++g) b70_decode32_switch(type, row, g, y + g * 32);
    }
}
