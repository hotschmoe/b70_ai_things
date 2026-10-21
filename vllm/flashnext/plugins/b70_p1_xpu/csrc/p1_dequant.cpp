// C13: bandwidth-oriented dequant for the expert prefill types.
// A subgroup of 16 owns 512 outputs. Packed bytes are staged in SLM.
// Each lane decodes one 32-wide group and stores 32 bf16.
// The generic fallback is b70_decode32, one work-item per 32-wide group,
// used for other types and for shapes that are not a multiple of 512.
#include "p1_dequant.hpp"
#include "decode.hpp"

#include <sycl/sycl.hpp>

#include <c10/xpu/XPUStream.h>

#include <cstdint>
#include <stdexcept>

namespace b70_p1_dequant {

static sycl::queue& q_of() { return c10::xpu::getCurrentXPUStream().queue(); }

static void check_xpu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_xpu(), name, " must be an XPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

inline uint16_t c13_ld16(const uint8_t* p) {
    uint16_t v;
    __builtin_memcpy(&v, p, 2);
    return v;
}

inline uint32_t c13_ksign7(uint32_t s7) {
    // Same parity as b70_ksign7. Callers pass a 7-bit field.
    uint32_t x = s7;
    uint32_t c = 0;
    while (x) {
        c += x & 1u;
        x >>= 1;
    }
    return s7 | ((c & 1u) << 7);
}

inline void c13_store32(uint16_t* dst, const float* v) {
    alignas(32) uint16_t tmp[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) tmp[j] = b70_fp32_to_bf16(v[j]);
    // One 64-byte store. dst is 64-byte aligned: the subgroup base is a
    // multiple of 512 elements and each lane adds 32 elements.
    sycl::vec<uint32_t, 16> packed;
    __builtin_memcpy(&packed, tmp, 64);
    *reinterpret_cast<sycl::vec<uint32_t, 16>*>(dst) = packed;
}

inline int64_t c13_map(int64_t g, int64_t cols, int mode, int pair) {
    if (mode == 0) return g;
    const int64_t row = g / cols;
    const int64_t col = g - row * cols;
    const int64_t e = row / pair;
    const int64_t f = row - e * pair;
    const int64_t dr = (mode == 1) ? (e * (int64_t)(pair * 2) + f) : (e * (int64_t)(pair * 2) + pair + f);
    return dr * cols + col;
}

inline void c13_dec_q2_0(const uint8_t* b, int gi, float* v) {
    const float d = b70_fp16_to_fp32(c13_ld16(b));
    const int e0 = gi * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int e = e0 + j;
        const int q = (b[2 + e / 4] >> ((e & 3) * 2)) & 3;
        v[j] = (float)(q - 1) * d;
    }
}

inline void c13_dec_iq4_nl(const uint8_t* b, float* v) {
    const float d = b70_fp16_to_fp32(c13_ld16(b));
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint8_t byte = b[2 + j];
        v[j] = d * (float)b70_iq4nl(byte & 15);
        v[j + 16] = d * (float)b70_iq4nl(byte >> 4);
    }
}

inline void c13_dec_iq3_xxs(const uint8_t* b, int ib, float* v) {
    uint32_t aux32;
    __builtin_memcpy(&aux32, b + 66 + 4 * ib, 4);
    const float d = b70_fp16_to_fp32(c13_ld16(b)) * (0.5f + (float)(aux32 >> 28)) * 0.5f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint16_t qw = c13_ld16(b + 2 + 8 * ib + 2 * il);
        const uint32_t g1 = iq3xxs_grid[qw & 0xffu];
        const uint32_t g2 = iq3xxs_grid[qw >> 8];
        const uint32_t signs = c13_ksign7((aux32 >> (7 * il)) & 127u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * il + j] = d * (float)((g1 >> (8 * j)) & 0xffu) * ((signs >> j) & 1 ? -1.f : 1.f);
            v[8 * il + 4 + j] = d * (float)((g2 >> (8 * j)) & 0xffu) * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
        }
    }
}

inline void c13_dec_iq3_s(const uint8_t* b, int ib, float* v) {
    const uint32_t scb = b[106 + ib / 2];
    const float d = b70_fp16_to_fp32(c13_ld16(b)) * (float)(1 + 2 * (int)((scb >> (4 * (ib & 1))) & 0xfu));
    const uint32_t qh = b[66 + ib];
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint16_t qw = c13_ld16(b + 2 + 8 * ib + 2 * il);
        const uint32_t signs = b[74 + 4 * ib + il];
        const uint32_t i1 = (qw & 0xffu) | ((qh << (8 - 2 * il)) & 256u);
        const uint32_t i2 = (uint32_t)(qw >> 8) | ((qh << (7 - 2 * il)) & 256u);
        const uint32_t g1 = iq3s_grid[i1];
        const uint32_t g2 = iq3s_grid[i2];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * il + j] = d * (float)((g1 >> (8 * j)) & 0xffu) * ((signs >> j) & 1 ? -1.f : 1.f);
            v[8 * il + 4 + j] = d * (float)((g2 >> (8 * j)) & 0xffu) * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
        }
    }
}

inline void c13_dec_iq2_xxs(const uint8_t* b, int ib, float* v) {
    const uint16_t q0 = c13_ld16(b + 2 + 8 * ib);
    const uint16_t q1 = c13_ld16(b + 4 + 8 * ib);
    const uint32_t aux32 = (uint32_t)c13_ld16(b + 6 + 8 * ib) | ((uint32_t)c13_ld16(b + 8 + 8 * ib) << 16);
    const float d = b70_fp16_to_fp32(c13_ld16(b)) * (0.5f + (float)(aux32 >> 28)) * 0.25f;
    const uint32_t idx[4] = {(uint32_t)q0 & 0xffu, (uint32_t)q0 >> 8, (uint32_t)q1 & 0xffu, (uint32_t)q1 >> 8};
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint64_t g = iq2xxs_grid[idx[il]];
        const uint32_t signs = c13_ksign7((aux32 >> (7 * il)) & 127u);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            v[8 * il + j] = d * (float)((g >> (8 * j)) & 0xffu) * ((signs >> j) & 1 ? -1.f : 1.f);
    }
}

inline void c13_dec_iq2_xs(const uint8_t* b, int ib, float* v) {
    const float db = b70_fp16_to_fp32(c13_ld16(b));
    const uint32_t scale = b[66 + ib];
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint16_t q = c13_ld16(b + 2 + 8 * ib + 2 * il);
        const uint64_t g = iq2xs_grid[q & 511u];
        const float d = db * (0.5f + (float)((scale >> (4 * (il / 2))) & 0xfu)) * 0.25f;
        const uint32_t signs = c13_ksign7((uint32_t)q >> 9);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            v[8 * il + j] = d * (float)((g >> (8 * j)) & 0xffu) * ((signs >> j) & 1 ? -1.f : 1.f);
    }
}

inline void c13_dec_iq2_s(const uint8_t* b, int ib, float* v) {
    const float db = b70_fp16_to_fp32(c13_ld16(b));
    const uint32_t qh = b[66 + ib];
    const uint32_t scale = b[74 + ib];
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint32_t qs = b[2 + 4 * ib + il];
        const uint32_t signs = b[34 + 4 * ib + il];
        const uint32_t idx = qs | ((qh << (8 - 2 * il)) & 0x300u);
        const uint64_t g = iq2s_grid[idx];
        const float d = db * (0.5f + (float)((scale >> (4 * (il / 2))) & 0xfu)) * 0.25f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            v[8 * il + j] = d * (float)((g >> (8 * j)) & 0xffu) * ((signs >> j) & 1 ? -1.f : 1.f);
    }
}

template <int TYPE>
inline void c13_dec(const uint8_t* b, int gi, float* v) {
    if constexpr (TYPE == B70_Q2_0) c13_dec_q2_0(b, gi, v);
    else if constexpr (TYPE == B70_IQ4_NL) c13_dec_iq4_nl(b, v);
    else if constexpr (TYPE == B70_IQ3_XXS) c13_dec_iq3_xxs(b, gi, v);
    else if constexpr (TYPE == B70_IQ3_S) c13_dec_iq3_s(b, gi, v);
    else if constexpr (TYPE == B70_IQ2_XXS) c13_dec_iq2_xxs(b, gi, v);
    else if constexpr (TYPE == B70_IQ2_XS) c13_dec_iq2_xs(b, gi, v);
    else if constexpr (TYPE == B70_IQ2_S) c13_dec_iq2_s(b, gi, v);
}

template <int TYPE>
struct C13Wide {
    static constexpr int BE = TYPE == B70_Q2_0 ? 64 : TYPE == B70_IQ4_NL ? 32 : 256;
    static constexpr int BB = TYPE == B70_Q2_0 ? 18
        : TYPE == B70_IQ4_NL                      ? 18
        : TYPE == B70_IQ3_XXS                     ? 98
        : TYPE == B70_IQ3_S                       ? 110
        : TYPE == B70_IQ2_XXS                     ? 66
        : TYPE == B70_IQ2_XS                      ? 74
                                                  : 82;
};

constexpr int C13_SPAN = 512;
constexpr int C13_WG = 256;
constexpr int C13_SLM = 320;

inline void c13_copy16(uint8_t* d, const uint8_t* s, int n) {
    int i = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (i + 4 <= n) {
            uint32_t v;
            __builtin_memcpy(&v, s + i, 4);
            __builtin_memcpy(d + i, &v, 4);
            i += 4;
        }
    }
    for (; i < n; ++i) d[i] = s[i];
}

template <int TYPE>
class b70_c13_wide;

template <int TYPE>
void launch_wide(sycl::queue& q, const uint8_t* src, uint16_t* dst, int64_t rows, int64_t cols, int dst_mode, int pair, const int32_t* selected = nullptr, int64_t expert_rows = 0) {
    constexpr int BE = C13Wide<TYPE>::BE;
    constexpr int BB = C13Wide<TYPE>::BB;
    constexpr int GPB = BE / 32;
    constexpr int NBLOCKS = C13_SPAN / BE;
    constexpr int NBYTES = NBLOCKS * BB;
    static_assert(NBYTES <= C13_SLM, "SLM stride");
    static_assert(C13_SPAN % BE == 0, "span");
    const int64_t nsg = (rows * cols) / C13_SPAN;
    constexpr int subgroups_per_wg = C13_WG / 16;
    const int64_t nwg = (nsg + subgroups_per_wg - 1) / subgroups_per_wg;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> slm(subgroups_per_wg * C13_SLM, h);
        h.parallel_for<b70_c13_wide<TYPE>>(
            sycl::nd_range<1>((size_t)nwg * C13_WG, C13_WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_linear_id();
                const int sg_in = (int)sg.get_group_linear_id();
                const int64_t sg_id = (int64_t)it.get_group(0) * subgroups_per_wg + sg_in;
                const bool active = sg_id < nsg;
                uint8_t* sm = slm.get_multi_ptr<sycl::access::decorated::no>().get() + sg_in * C13_SLM;
                if (active) {
                    const int64_t block0 = (sg_id * C13_SPAN) / BE;
                    const int64_t blocks_per_expert = selected ? expert_rows * cols / BE : 0;
                    const int64_t local_expert = selected ? block0 / blocks_per_expert : 0;
                    const int64_t source_block = selected ? (int64_t)selected[local_expert] * blocks_per_expert + block0 % blocks_per_expert : block0;
                    const uint8_t* in = src + source_block * BB;
                    for (int off = lane * 16; off < NBYTES; off += 16 * 16) {
                        int n = NBYTES - off;
                        if (n > 16) n = 16;
                        c13_copy16(sm + off, in + off, n);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (!active) return;
                // One lane decodes one 32-wide group and stores 32 bf16.
                // A striped block write was bit-exact but slower (c13-bench4).
                const int block_local = lane / GPB;
                const int gi = lane - block_local * GPB;
                float v[32];
                c13_dec<TYPE>(sm + block_local * BB, gi, v);
                const int64_t dst0 = c13_map(sg_id * C13_SPAN, cols, dst_mode, pair);
                c13_store32(dst + dst0 + (int64_t)lane * 32, v);
            });
    });
}

template <int TYPE, bool FP16>
class b70_c13_generic;

template <int TYPE, bool FP16>
void launch_generic(sycl::queue& q, const uint8_t* src, uint16_t* dst, int64_t rows, int64_t cols, int row_bytes,
                    int dst_mode, int pair, const int32_t* selected = nullptr, int64_t expert_rows = 0) {
    const int gpr = (int)(cols / 32);
    const int64_t total = rows * (int64_t)gpr;
    constexpr int WG = 256;
    const int64_t cap = (int64_t)WG * 4096;
    int64_t nthreads = total < cap ? total : cap;
    if (nthreads < 1) nthreads = 1;
    const size_t grid = (size_t)(((nthreads + WG - 1) / WG) * WG);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_c13_generic<TYPE, FP16>>(
            sycl::nd_range<1>(grid, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int64_t stride = (int64_t)it.get_global_range(0);
                for (int64_t g = (int64_t)it.get_global_id(0); g < total; g += stride) {
                    const int64_t row = g / gpr;
                    const int gi = (int)(g - row * gpr);
                    float v[32];
                    const int64_t source_row = selected ? (int64_t)selected[row / expert_rows] * expert_rows + row % expert_rows : row;
                    b70_decode32<TYPE>(src + source_row * (int64_t)row_bytes, gi, v);
                    const int64_t elem = row * cols + (int64_t)gi * 32;
                    const int64_t de = c13_map(elem, cols, dst_mode, pair);
                    uint16_t* outp = dst + de;
                    if constexpr (FP16) {
                        alignas(32) uint16_t tmp[32];
#pragma unroll
                        for (int j = 0; j < 32; ++j) tmp[j] = sycl::bit_cast<uint16_t>(sycl::half(v[j]));
                        auto* d = reinterpret_cast<sycl::vec<uint16_t, 16>*>(outp);
                        auto* s = reinterpret_cast<const sycl::vec<uint16_t, 16>*>(tmp);
                        d[0] = s[0];
                        d[1] = s[1];
                    } else {
                        c13_store32(outp, v);
                    }
                }
            });
    });
}

template <typename WideFn, typename GenFn>
void dispatch_wide_type(int type, bool tuned_shape, WideFn wide, GenFn gen) {
    switch (type) {
    case B70_IQ3_XXS:
        if (tuned_shape) wide.template operator()<B70_IQ3_XXS>();
        else gen.template operator()<B70_IQ3_XXS>();
        break;
    case B70_IQ3_S:
        if (tuned_shape) wide.template operator()<B70_IQ3_S>();
        else gen.template operator()<B70_IQ3_S>();
        break;
    case B70_IQ2_XXS:
        if (tuned_shape) wide.template operator()<B70_IQ2_XXS>();
        else gen.template operator()<B70_IQ2_XXS>();
        break;
    case B70_IQ2_XS:
        if (tuned_shape) wide.template operator()<B70_IQ2_XS>();
        else gen.template operator()<B70_IQ2_XS>();
        break;
    case B70_IQ2_S:
        if (tuned_shape) wide.template operator()<B70_IQ2_S>();
        else gen.template operator()<B70_IQ2_S>();
        break;
    case B70_Q2_0:
        if (tuned_shape) wide.template operator()<B70_Q2_0>();
        else gen.template operator()<B70_Q2_0>();
        break;
    case B70_IQ4_NL:
        if (tuned_shape) wide.template operator()<B70_IQ4_NL>();
        else gen.template operator()<B70_IQ4_NL>();
        break;
    case B70_Q4_K: gen.template operator()<B70_Q4_K>(); break;
    case B70_Q5_K: gen.template operator()<B70_Q5_K>(); break;
    case B70_Q6_K: gen.template operator()<B70_Q6_K>(); break;
    case B70_IQ4_XS: gen.template operator()<B70_IQ4_XS>(); break;
    case B70_Q2_K: gen.template operator()<B70_Q2_K>(); break;
    case B70_Q3_K: gen.template operator()<B70_Q3_K>(); break;
    case B70_Q8_0: gen.template operator()<B70_Q8_0>(); break;
    default: throw std::runtime_error("b70_gguf: prefill_dequant_wide unsupported type");
    }
}

at::Tensor b70_prefill_dequant_wide_op(at::Tensor q, int64_t ggml_type, int64_t rows, int64_t cols, at::Tensor out,
                                       int64_t dst_mode, int64_t pair) {
    check_xpu(q, "q");
    check_xpu(out, "out");
    TORCH_CHECK(q.scalar_type() == at::kByte, "q must be uint8");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kHalf, "out bf16 or fp16");
    TORCH_CHECK(rows > 0 && cols > 0 && cols % 32 == 0, "prefill_dequant_wide shape");
    TORCH_CHECK(dst_mode >= 0 && dst_mode <= 2, "dst_mode");
    const int rb = b70_row_bytes((int)ggml_type, (int)cols);
    TORCH_CHECK(rb > 0, "bad type or cols");
    TORCH_CHECK(q.numel() >= rows * (int64_t)rb, "q is shorter than rows*row_bytes");
    int64_t out_rows = rows;
    if (dst_mode != 0) {
        TORCH_CHECK(pair > 0 && rows % pair == 0, "pair");
        out_rows = (rows / pair) * (pair * 2);
    }
    TORCH_CHECK(out.dim() == 2 && out.size(0) == out_rows && out.size(1) == cols, "out shape");
    const bool fp16 = out.scalar_type() == at::kHalf;
    const int64_t total = rows * cols;
    // Wide kernel is bf16 only. fp16 and odd shapes take the generic path.
    // dst_mode 1/2 also needs each expert tile to be a multiple of the span
    // so a subgroup never crosses the up/gate gap.
    bool tuned = !fp16 && total % C13_SPAN == 0 && cols % 32 == 0;
    if (dst_mode != 0) tuned = tuned && ((int64_t)pair * cols) % C13_SPAN == 0;
    auto& queue = q_of();
    const uint8_t* src = q.data_ptr<uint8_t>();
    uint16_t* dst = static_cast<uint16_t*>(out.data_ptr());
    const int mode = (int)dst_mode;
    const int pr = (int)pair;
    dispatch_wide_type(
        (int)ggml_type, tuned,
        [&]<int TYPE>() { launch_wide<TYPE>(queue, src, dst, rows, cols, mode, pr); },
        [&]<int TYPE>() {
            if (fp16) launch_generic<TYPE, true>(queue, src, dst, rows, cols, rb, mode, pr);
            else launch_generic<TYPE, false>(queue, src, dst, rows, cols, rb, mode, pr);
        });
    return out;
}

at::Tensor b70_prefill_dequant_selected_op(at::Tensor q, int64_t ggml_type, int64_t rows, int64_t cols, at::Tensor out,
                                       int64_t dst_mode, int64_t pair, at::Tensor sel, int64_t expert_rows) {
    check_xpu(q, "q");
    check_xpu(sel, "sel");
    TORCH_CHECK(sel.scalar_type() == at::kInt && sel.dim() == 1, "sel i32 vector");
    TORCH_CHECK(expert_rows > 0 && rows == sel.numel() * expert_rows, "selected expert rows");
    TORCH_CHECK(sel.device() == q.device(), "selected device");
    // sel comes from the existing validated host bucket plan (0 <= sel < E).
    const int32_t* selected = sel.data_ptr<int32_t>();
    check_xpu(out, "out");
    TORCH_CHECK(q.scalar_type() == at::kByte, "q must be uint8");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kHalf, "out bf16 or fp16");
    TORCH_CHECK(rows > 0 && cols > 0 && cols % 32 == 0, "prefill_dequant_wide shape");
    TORCH_CHECK(dst_mode >= 0 && dst_mode <= 2, "dst_mode");
    const int rb = b70_row_bytes((int)ggml_type, (int)cols);
    TORCH_CHECK(rb > 0, "bad type or cols");
    TORCH_CHECK(q.numel() % (expert_rows * rb) == 0, "source expert pack shape");
    int64_t out_rows = rows;
    if (dst_mode != 0) {
        TORCH_CHECK(pair > 0 && rows % pair == 0, "pair");
        out_rows = (rows / pair) * (pair * 2);
    }
    TORCH_CHECK(out.dim() == 2 && out.size(0) == out_rows && out.size(1) == cols, "out shape");
    const bool fp16 = out.scalar_type() == at::kHalf;
    const int64_t total = rows * cols;
    // Wide kernel is bf16 only. fp16 and odd shapes take the generic path.
    // dst_mode 1/2 also needs each expert tile to be a multiple of the span
    // so a subgroup never crosses the up/gate gap.
    bool tuned = !fp16 && total % C13_SPAN == 0 && cols % 32 == 0;
    if (dst_mode != 0) tuned = tuned && ((int64_t)pair * cols) % C13_SPAN == 0;
    tuned = tuned && (expert_rows * cols) % C13_SPAN == 0;
    auto& queue = q_of();
    const uint8_t* src = q.data_ptr<uint8_t>();
    uint16_t* dst = static_cast<uint16_t*>(out.data_ptr());
    const int mode = (int)dst_mode;
    const int pr = (int)pair;
    dispatch_wide_type(
        (int)ggml_type, tuned,
        [&]<int TYPE>() { launch_wide<TYPE>(queue, src, dst, rows, cols, mode, pr, selected, expert_rows); },
        [&]<int TYPE>() {
            if (fp16) launch_generic<TYPE, true>(queue, src, dst, rows, cols, rb, mode, pr, selected, expert_rows);
            else launch_generic<TYPE, false>(queue, src, dst, rows, cols, rb, mode, pr, selected, expert_rows);
        });
    return out;
}

} // namespace b70_p1_dequant
