// Prefetch split of one 32-value weight group.
// Port of Strata sycl/src/kernels/cuda/night_grouped_expert.inc:76-228
// (load_raw then decode, so the next group's loads have no data dependency
// on this group's arithmetic). K-quant types are the same split: the bytes
// b70_decode32 reads are pulled into Raw first.
// Host and device. No SYCL types.
#pragma once
#include "decode.hpp"

struct B70Raw {
    uint32_t r[40];
};

inline uint32_t b70_raw_u8(uint32_t w, int shift) { return (w >> shift) & 0xffu; }

template <int TYPE>
inline uint32_t b70_grid32(uint32_t i) {
    if constexpr (TYPE == B70_IQ3_S) return iq3s_grid[i];
    else return iq3xxs_grid[i];
}

template <int TYPE>
inline uint64_t b70_grid64(uint32_t i) {
    if constexpr (TYPE == B70_IQ2_XXS) return iq2xxs_grid[i];
    else if constexpr (TYPE == B70_IQ2_XS) return iq2xs_grid[i];
    else return iq2s_grid[i];
}

// ibs = flat >> 8, ib = (flat & 255) >> 5, flat = weight_row * K + k0.
// K % block_elems == 0 (enforced by b70_row_bytes) keeps rows on a block boundary,
// so this flat index matches per-row packing.
template <int TYPE>
inline B70Raw b70_load_raw(const uint8_t* vx, int64_t ibs, int ib) {
    B70Raw w{};
    if constexpr (TYPE == B70_IQ3_S) {
        const uint8_t* x = vx + ibs * 110;
        w.r[0] = b70_ld16(x);
        for (int i = 0; i < 4; ++i) w.r[1 + i] = b70_ld16(x + 2 + 8 * ib + 2 * i);
        w.r[5] = b70_ld16(x + 74 + 4 * ib);
        w.r[6] = b70_ld16(x + 76 + 4 * ib);
        w.r[7] = x[66 + ib];
        w.r[8] = x[106 + ib / 2];
    } else if constexpr (TYPE == B70_IQ2_XXS) {
        const uint8_t* x = vx + ibs * 66;
        w.r[0] = b70_ld16(x);
        for (int i = 0; i < 4; ++i) w.r[1 + i] = b70_ld16(x + 2 + 8 * ib + 2 * i);
    } else if constexpr (TYPE == B70_IQ3_XXS) {
        const uint8_t* x = vx + ibs * 98;
        w.r[0] = b70_ld16(x);
        for (int i = 0; i < 4; ++i) w.r[1 + i] = b70_ld16(x + 2 + 8 * ib + 2 * i);
        w.r[5] = b70_ld16(x + 66 + 4 * ib);
        w.r[6] = b70_ld16(x + 68 + 4 * ib);
    } else if constexpr (TYPE == B70_IQ2_XS) {
        const uint8_t* x = vx + ibs * 74;
        w.r[0] = b70_ld16(x);
        for (int i = 0; i < 4; ++i) w.r[1 + i] = b70_ld16(x + 2 + 8 * ib + 2 * i);
        w.r[5] = x[66 + ib];
    } else if constexpr (TYPE == B70_IQ2_S) {
        const uint8_t* x = vx + ibs * 82;
        w.r[0] = b70_ld16(x);
        w.r[1] = b70_ld16(x + 2 + 4 * ib);
        w.r[2] = b70_ld16(x + 4 + 4 * ib);
        w.r[3] = b70_ld16(x + 34 + 4 * ib);
        w.r[4] = b70_ld16(x + 36 + 4 * ib);
        w.r[5] = x[66 + ib];
        w.r[6] = x[74 + ib];
    } else if constexpr (TYPE == B70_Q2_0) {
        const uint8_t* x = vx + (ibs * 4 + ib / 2) * 18;
        w.r[0] = b70_ld16(x);
        for (int i = 0; i < 4; ++i) w.r[1 + i] = b70_ld16(x + 2 + 8 * (ib % 2) + 2 * i);
    } else if constexpr (TYPE == B70_IQ4_NL) {
        const uint8_t* x = vx + (ibs * 8 + ib) * 18;
        w.r[0] = b70_ld16(x);
        for (int i = 0; i < 8; ++i) w.r[1 + i] = b70_ld16(x + 2 + 2 * i);
    } else if constexpr (TYPE == B70_Q4_K) {
        const uint8_t* b = vx + ibs * 144;
        w.r[0] = b70_ld16(b);
        w.r[1] = b70_ld16(b + 2);
        for (int i = 0; i < 6; ++i) w.r[2 + i] = b70_ld16(b + 4 + 2 * i);
        const uint8_t* q = b + 16 + 32 * (ib / 2);
        for (int i = 0; i < 16; ++i) w.r[8 + i] = b70_ld16(q + 2 * i);
    } else if constexpr (TYPE == B70_Q5_K) {
        const uint8_t* b = vx + ibs * 176;
        w.r[0] = b70_ld16(b);
        w.r[1] = b70_ld16(b + 2);
        for (int i = 0; i < 6; ++i) w.r[2 + i] = b70_ld16(b + 4 + 2 * i);
        for (int i = 0; i < 16; ++i) w.r[8 + i] = b70_ld16(b + 16 + 2 * i);
        const uint8_t* ql = b + 48 + 32 * (ib / 2);
        for (int i = 0; i < 16; ++i) w.r[24 + i] = b70_ld16(ql + 2 * i);
    } else if constexpr (TYPE == B70_Q6_K) {
        const uint8_t* b = vx + ibs * 210;
        const int n = ib / 4;
        const int qu = ib % 4;
        const uint8_t* ql = b + 64 * n + ((qu == 1 || qu == 3) ? 32 : 0);
        const uint8_t* qh = b + 128 + 32 * n;
        const uint8_t* sc = b + 192 + 8 * n;
        w.r[0] = b70_ld16(b + 208);
        w.r[1] = (uint32_t)qu;
        for (int i = 0; i < 16; ++i) w.r[2 + i] = b70_ld16(ql + 2 * i);
        for (int i = 0; i < 16; ++i) w.r[18 + i] = b70_ld16(qh + 2 * i);
        for (int i = 0; i < 4; ++i) w.r[34 + i] = b70_ld16(sc + 2 * i);
    } else if constexpr (TYPE == B70_IQ4_XS) {
        const uint8_t* b = vx + ibs * 136;
        w.r[0] = b70_ld16(b);
        w.r[1] = b70_ld16(b + 2);
        w.r[2] = b70_ld16(b + 4);
        w.r[3] = b70_ld16(b + 6);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int i = 0; i < 8; ++i) w.r[4 + i] = b70_ld16(qs + 2 * i);
    }
    return w;
}

template <int TYPE>
inline void b70_decode_raw(const B70Raw& w, int ib, float* v) {
    if constexpr (TYPE == B70_IQ3_S) {
        const float d = b70_fp16_to_fp32((uint16_t)w.r[0]) * (float)(1 + 2 * (int)((w.r[8] >> (4 * (ib % 2))) & 0xfu));
        const uint32_t qh = w.r[7];
        for (int il = 0; il < 4; ++il) {
            const uint32_t g1 = b70_grid32<B70_IQ3_S>((w.r[1 + il] & 0xffu) | ((qh << (8 - 2 * il)) & 256u));
            const uint32_t g2 = b70_grid32<B70_IQ3_S>((w.r[1 + il] >> 8) | ((qh << (7 - 2 * il)) & 256u));
            const uint32_t signs = b70_raw_u8(w.r[5 + il / 2], 8 * (il % 2));
            for (int j = 0; j < 4; ++j) {
                v[8 * il + j] = d * (float)b70_raw_u8(g1, 8 * j) * ((signs >> j) & 1 ? -1.f : 1.f);
                v[8 * il + 4 + j] = d * (float)b70_raw_u8(g2, 8 * j) * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
            }
        }
    } else if constexpr (TYPE == B70_IQ2_XXS) {
        const uint32_t aux32 = w.r[3] | (w.r[4] << 16);
        const float d = b70_fp16_to_fp32((uint16_t)w.r[0]) * (0.5f + (float)(aux32 >> 28)) * 0.25f;
        const uint32_t idx[4] = {w.r[1] & 0xffu, w.r[1] >> 8, w.r[2] & 0xffu, w.r[2] >> 8};
        for (int il = 0; il < 4; ++il) {
            const uint64_t g = b70_grid64<B70_IQ2_XXS>(idx[il]);
            const uint32_t signs = b70_ksign7((aux32 >> (7 * il)) & 127u);
            for (int j = 0; j < 8; ++j)
                v[8 * il + j] = d * b70_grid_byte_u64(g, j) * ((signs >> j) & 1 ? -1.f : 1.f);
        }
    } else if constexpr (TYPE == B70_IQ3_XXS) {
        const uint32_t aux32 = w.r[5] | (w.r[6] << 16);
        const float d = b70_fp16_to_fp32((uint16_t)w.r[0]) * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        for (int il = 0; il < 4; ++il) {
            const uint32_t g1 = b70_grid32<B70_IQ3_XXS>(w.r[1 + il] & 0xffu);
            const uint32_t g2 = b70_grid32<B70_IQ3_XXS>(w.r[1 + il] >> 8);
            const uint32_t signs = b70_ksign7((aux32 >> (7 * il)) & 127u);
            for (int j = 0; j < 4; ++j) {
                v[8 * il + j] = d * (float)b70_raw_u8(g1, 8 * j) * ((signs >> j) & 1 ? -1.f : 1.f);
                v[8 * il + 4 + j] = d * (float)b70_raw_u8(g2, 8 * j) * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
            }
        }
    } else if constexpr (TYPE == B70_IQ2_XS) {
        const float db = b70_fp16_to_fp32((uint16_t)w.r[0]);
        for (int il = 0; il < 4; ++il) {
            const uint32_t q = w.r[1 + il];
            const uint64_t g = b70_grid64<B70_IQ2_XS>(q & 511u);
            const float d = db * (0.5f + (float)((w.r[5] >> (4 * (il / 2))) & 0xfu)) * 0.25f;
            const uint32_t signs = b70_ksign7(q >> 9);
            for (int j = 0; j < 8; ++j)
                v[8 * il + j] = d * b70_grid_byte_u64(g, j) * ((signs >> j) & 1 ? -1.f : 1.f);
        }
    } else if constexpr (TYPE == B70_IQ2_S) {
        const float db = b70_fp16_to_fp32((uint16_t)w.r[0]);
        const uint32_t qh = w.r[5];
        for (int il = 0; il < 4; ++il) {
            const uint32_t qb = b70_raw_u8(w.r[1 + il / 2], 8 * (il % 2));
            const uint64_t g = b70_grid64<B70_IQ2_S>(qb | ((qh << (8 - 2 * il)) & 0x300u));
            const float d = db * (0.5f + (float)((w.r[6] >> (4 * (il / 2))) & 0xfu)) * 0.25f;
            const uint32_t signs = b70_raw_u8(w.r[3 + il / 2], 8 * (il % 2));
            for (int j = 0; j < 8; ++j)
                v[8 * il + j] = d * b70_grid_byte_u64(g, j) * ((signs >> j) & 1 ? -1.f : 1.f);
        }
    } else if constexpr (TYPE == B70_Q2_0) {
        const float d = b70_fp16_to_fp32((uint16_t)w.r[0]);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 8; ++j)
                v[8 * i + j] = d * (float)((int)((w.r[1 + i] >> (2 * j)) & 3) - 1);
    } else if constexpr (TYPE == B70_IQ4_NL) {
        const float d = b70_fp16_to_fp32((uint16_t)w.r[0]);
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = w.r[1 + i];
            v[2 * i] = d * (float)b70_iq4nl((int)(q & 0xf));
            v[2 * i + 1] = d * (float)b70_iq4nl((int)((q >> 8) & 0xf));
            v[16 + 2 * i] = d * (float)b70_iq4nl((int)((q >> 4) & 0xf));
            v[16 + 2 * i + 1] = d * (float)b70_iq4nl((int)((q >> 12) & 0xf));
        }
    } else if constexpr (TYPE == B70_Q4_K || TYPE == B70_Q5_K) {
        uint8_t scb[12];
        for (int i = 0; i < 6; ++i) {
            scb[2 * i] = (uint8_t)(w.r[2 + i] & 0xffu);
            scb[2 * i + 1] = (uint8_t)((w.r[2 + i] >> 8) & 0xffu);
        }
        int sc = 0, mn = 0;
        b70_scale_min_k4(ib, scb, sc, mn);
        const float d = b70_fp16_to_fp32((uint16_t)w.r[0]);
        const float dmin = b70_fp16_to_fp32((uint16_t)w.r[1]);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)mn;
        const int hi = ib & 1;
        if constexpr (TYPE == B70_Q4_K) {
            for (int l = 0; l < 32; ++l) {
                const uint32_t byte = b70_raw_u8(w.r[8 + l / 2], 8 * (l & 1));
                const int nib = hi ? (int)(byte >> 4) : (int)(byte & 0xf);
                v[l] = d1 * (float)nib - m1;
            }
        } else {
            const uint8_t bit = (uint8_t)(1u << ib);
            for (int l = 0; l < 32; ++l) {
                const uint32_t ql = b70_raw_u8(w.r[24 + l / 2], 8 * (l & 1));
                const uint32_t qh = b70_raw_u8(w.r[8 + l / 2], 8 * (l & 1));
                const int nib = hi ? (int)(ql >> 4) : (int)(ql & 0xf);
                v[l] = d1 * (float)(nib + ((qh & bit) ? 16 : 0)) - m1;
            }
        }
    } else if constexpr (TYPE == B70_Q6_K) {
        const int qu = (int)w.r[1];
        const float d = b70_fp16_to_fp32((uint16_t)w.r[0]);
        uint8_t scb[8];
        for (int i = 0; i < 4; ++i) {
            scb[2 * i] = (uint8_t)(w.r[34 + i] & 0xffu);
            scb[2 * i + 1] = (uint8_t)((w.r[34 + i] >> 8) & 0xffu);
        }
        for (int l = 0; l < 32; ++l) {
            const uint32_t ql = b70_raw_u8(w.r[2 + l / 2], 8 * (l & 1));
            const uint32_t qh = b70_raw_u8(w.r[18 + l / 2], 8 * (l & 1));
            int q;
            if (qu == 0) q = (int)(ql & 0xfu) | (int)(((qh >> 0) & 3u) << 4);
            else if (qu == 1) q = (int)(ql & 0xfu) | (int)(((qh >> 2) & 3u) << 4);
            else if (qu == 2) q = (int)(ql >> 4) | (int)(((qh >> 4) & 3u) << 4);
            else q = (int)(ql >> 4) | (int)(((qh >> 6) & 3u) << 4);
            const int is = l / 16;
            const int sc = (int)(int8_t)scb[is + 2 * qu];
            v[l] = d * (float)sc * (float)(q - 32);
        }
    } else if constexpr (TYPE == B70_IQ4_XS) {
        const float d = b70_fp16_to_fp32((uint16_t)w.r[0]);
        const uint32_t scales_h = w.r[1];
        // b[4 + ib/2]: r[2] holds bytes 4..5, r[3] holds bytes 6..7.
        const uint32_t word = (ib / 2) < 2 ? w.r[2] : w.r[3];
        const int byte_shift = 8 * ((ib / 2) & 1);
        const int ls = (int)((word >> (byte_shift + 4 * (ib & 1))) & 0xfu) |
                        (int)(((scales_h >> (2 * ib)) & 3u) << 4);
        const float dl = d * (float)(ls - 32);
        for (int j = 0; j < 16; ++j) {
            const uint32_t q = b70_raw_u8(w.r[4 + j / 2], 8 * (j & 1));
            v[j] = dl * (float)b70_iq4nl((int)(q & 0xf));
            v[j + 16] = dl * (float)b70_iq4nl((int)(q >> 4));
        }
    }
}

template <int TYPE>
inline void b70_raw_decode_gi(const uint8_t* row, int gi, float* v) {
    const int64_t flat = (int64_t)gi * 32;
    b70_decode_raw<TYPE>(b70_load_raw<TYPE>(row, flat >> 8, (int)((flat & 255) >> 5)), (int)((flat & 255) >> 5), v);
}
