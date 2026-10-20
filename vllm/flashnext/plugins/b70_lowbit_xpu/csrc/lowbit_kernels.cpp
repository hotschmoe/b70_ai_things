// b70 low-bit WNA16 / WXA8 kernels for Intel B70 (BMG G31).
//
// Weight layout is b70lb1 (plugins/LOWBIT_FORMAT.md):
//   plane2  : low 2 bits, 4 codes/byte, 32 bytes per group of 128
//   plane_hi: int3 = 1 bit (16 B/group), int4 = 2 bits (32 B/group)
//   scale   : bf16 [N, K/128]
// Groups are packed tight. The row stride is 64-byte aligned; the kernel
// indexes group g at byte g*32 (plane2) or g*16 (int3 hi), not g*stride.
//
// Decode GEMV is ESIMD. A work-group of kGemvWg owns one activation row
// in SLM (the WG=1 path measured ~30-100 GB/s, the same band as the
// NVFP4 WG=1 prototype). Each work-item then streams one output row.
// Prefill uses ESIMD XMX DPAS (bf16 tile 8x16x16, or int8 tile 8x16x32
// for WXA8). No queue.wait(); tensors are recordStream'd.

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>
#include <torch/library.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace esimd = sycl::ext::intel::esimd;
using bf16_t = sycl::ext::oneapi::bfloat16;

static void record_use(const torch::Tensor& t) {
  if (!t.defined() || t.numel() == 0 || !t.is_xpu()) {
    return;
  }
  c10::xpu::XPUCachingAllocator::recordStream(
      t.storage().data_ptr(), c10::xpu::getCurrentXPUStream());
}

static int plane2_stride_of(int k) {
  return (k / 4 + 63) & ~63;
}

static int plane_hi_stride_of(int k, int bits) {
  if (bits == 2) {
    return 0;
  }
  int payload = bits == 3 ? k / 8 : k / 4;
  return (payload + 63) & ~63;
}

template <int N>
SYCL_ESIMD_FUNCTION esimd::simd<float, N> bf16_to_f32(esimd::simd<uint16_t, N> u) {
  esimd::simd<uint32_t, N> x = esimd::convert<uint32_t>(u);
  x = x << 16;
  return x.template bit_cast_view<float>();
}

SYCL_ESIMD_FUNCTION float hsum128(esimd::simd<float, 128> acc) {
  acc.template select<64, 1>(0) += acc.template select<64, 1>(64);
  acc.template select<32, 1>(0) += acc.template select<32, 1>(32);
  acc.template select<16, 1>(0) += acc.template select<16, 1>(16);
  acc.template select<8, 1>(0) += acc.template select<8, 1>(8);
  acc.template select<4, 1>(0) += acc.template select<4, 1>(4);
  acc.template select<2, 1>(0) += acc.template select<2, 1>(2);
  return (float)acc[0] + (float)acc[1];
}

SYCL_ESIMD_FUNCTION uint16_t f32_to_bf16_u16(float f) {
  esimd::simd<float, 1> sf(f);
  uint32_t bits = sf.template bit_cast_view<uint32_t>()[0];
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

template <int N>
SYCL_ESIMD_FUNCTION esimd::simd<uint16_t, N> f32_to_bf16_u16v(
    esimd::simd<float, N> f) {
  esimd::simd<uint32_t, N> bits = f.template bit_cast_view<uint32_t>();
  esimd::simd<uint32_t, N> lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  return esimd::convert<uint16_t>(bits >> 16);
}

static inline float bf16_to_f32_scalar(uint16_t b) {
  uint32_t bits = (uint32_t)b << 16;
  return sycl::bit_cast<float>(bits);
}

SYCL_ESIMD_FUNCTION float bf16_u16_to_f32(uint16_t b) {
  esimd::simd<uint16_t, 1> u(b);
  return bf16_to_f32<1>(u)[0];
}

static inline uint16_t f32_to_bf16_scalar(float f) {
  uint32_t bits = sycl::bit_cast<uint32_t>(f);
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

// Work-group width for decode GEMV. WG=1 left the kernel near 100 GB/s.
static constexpr int kGemvWg = 16;
// M<=8, K<=2560 bf16, the Flash-Next square and the Ornith lm_head K.
static constexpr int kSlmBytes = 8 * 2560 * (int)sizeof(uint16_t);
static constexpr int kSlmRowBytes = 2560 * (int)sizeof(uint16_t);

// 128 signed codes for one group, as float values times the group scale.
// ph is the high plane for this group. int3 uses the low 16 bytes.
template <int BITS>
SYCL_ESIMD_FUNCTION esimd::simd<float, 128> dequant_regs(
    esimd::simd<uint8_t, 32> p2,
    esimd::simd<uint8_t, 32> ph,
    float scale) {
  esimd::simd<uint32_t, 32> p2u = esimd::convert<uint32_t>(p2);
  esimd::simd<int32_t, 128> codes(0);
  codes.template select<32, 4>(0) = esimd::convert<int32_t>(p2u & 3u);
  codes.template select<32, 4>(1) = esimd::convert<int32_t>((p2u >> 2) & 3u);
  codes.template select<32, 4>(2) = esimd::convert<int32_t>((p2u >> 4) & 3u);
  codes.template select<32, 4>(3) = esimd::convert<int32_t>((p2u >> 6) & 3u);

  if constexpr (BITS == 3) {
    esimd::simd<uint8_t, 16> ph16 = ph.template select<16, 1>(0);
    esimd::simd<uint32_t, 16> phu = esimd::convert<uint32_t>(ph16);
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      esimd::simd<int32_t, 16> bit =
          esimd::convert<int32_t>((phu >> b) & 1u);
      auto cur = codes.template select<16, 8>(b);
      codes.template select<16, 8>(b) = cur | (bit << 2);
    }
  } else if constexpr (BITS == 4) {
    esimd::simd<uint32_t, 32> phu = esimd::convert<uint32_t>(ph);
    codes.template select<32, 4>(0) =
        codes.template select<32, 4>(0) |
        (esimd::convert<int32_t>(phu & 3u) << 2);
    codes.template select<32, 4>(1) =
        codes.template select<32, 4>(1) |
        (esimd::convert<int32_t>((phu >> 2) & 3u) << 2);
    codes.template select<32, 4>(2) =
        codes.template select<32, 4>(2) |
        (esimd::convert<int32_t>((phu >> 4) & 3u) << 2);
    codes.template select<32, 4>(3) =
        codes.template select<32, 4>(3) |
        (esimd::convert<int32_t>((phu >> 6) & 3u) << 2);
  }

  constexpr int maxq = 1 << (BITS - 1);
  constexpr int span = maxq << 1;
  esimd::simd<int32_t, 128> neg = codes - span;
  codes.merge(neg, codes >= maxq);
  return esimd::convert<float>(codes) * scale;
}

template <int BITS>
SYCL_ESIMD_FUNCTION esimd::simd<float, 128> dequant_group(
    const uint8_t* row2,
    const uint8_t* rowhi,
    int g,
    float scale) {
  esimd::simd<uint8_t, 32> p2 = esimd::block_load<uint8_t, 32>(
      row2 + g * 32, esimd::overaligned<32>);
  esimd::simd<uint8_t, 32> ph(0);
  if constexpr (BITS == 3) {
    esimd::simd<uint8_t, 16> ph16 = esimd::block_load<uint8_t, 16>(
        rowhi + g * 16, esimd::overaligned<16>);
    ph.template select<16, 1>(0) = ph16;
  } else if constexpr (BITS == 4) {
    ph = esimd::block_load<uint8_t, 32>(
        rowhi + g * 32, esimd::overaligned<32>);
  }
  return dequant_regs<BITS>(p2, ph, scale);
}

SYCL_ESIMD_FUNCTION float dot_slm(int byte_off, esimd::simd<float, 128> wf);

// Two groups, one 64-byte plane2 load. g is even. Writes into acc[m].
template <int BITS>
SYCL_ESIMD_FUNCTION void accum_pair(
    const uint8_t* row2,
    const uint8_t* rowhi,
    const uint16_t* scale_row,
    int g,
    int K,
    int nrows,
    float* acc) {
  esimd::simd<uint8_t, 64> p2 = esimd::block_load<uint8_t, 64>(
      row2 + g * 32, esimd::overaligned<64>);
  esimd::simd<uint8_t, 32> ph0(0);
  esimd::simd<uint8_t, 32> ph1(0);
  if constexpr (BITS == 3) {
    esimd::simd<uint8_t, 32> ph = esimd::block_load<uint8_t, 32>(
        rowhi + g * 16, esimd::overaligned<32>);
    ph0.template select<16, 1>(0) = ph.template select<16, 1>(0);
    ph1.template select<16, 1>(0) = ph.template select<16, 1>(16);
  } else if constexpr (BITS == 4) {
    esimd::simd<uint8_t, 64> ph = esimd::block_load<uint8_t, 64>(
        rowhi + g * 32, esimd::overaligned<64>);
    ph0 = ph.template select<32, 1>(0);
    ph1 = ph.template select<32, 1>(32);
  }
  float sc0 = bf16_u16_to_f32(scale_row[g]);
  float sc1 = bf16_u16_to_f32(scale_row[g + 1]);
  esimd::simd<float, 128> wf = dequant_regs<BITS>(
      p2.template select<32, 1>(0), ph0, sc0);
  for (int m = 0; m < nrows; ++m) {
    acc[m] += dot_slm((m * K + g * 128) * (int)sizeof(uint16_t), wf);
  }
  wf = dequant_regs<BITS>(p2.template select<32, 1>(32), ph1, sc1);
  for (int m = 0; m < nrows; ++m) {
    acc[m] += dot_slm((m * K + (g + 1) * 128) * (int)sizeof(uint16_t), wf);
  }
}

SYCL_ESIMD_FUNCTION void slm_store_group(
    const uint16_t* src, int byte_off) {
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    esimd::simd<uint16_t, 32> part = esimd::block_load<uint16_t, 32>(
        src + s * 32, esimd::overaligned<64>);
    esimd::slm_block_store<uint16_t, 32>(byte_off + s * 64, part);
  }
}

SYCL_ESIMD_FUNCTION float dot_slm(
    int byte_off, esimd::simd<float, 128> wf) {
  esimd::simd<float, 128> xf;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    esimd::simd<uint16_t, 32> xu =
        esimd::slm_block_load<uint16_t, 32>(byte_off + s * 64);
    xf.template select<32, 1>(s * 32) = bf16_to_f32<32>(xu);
  }
  return hsum128(xf * wf);
}

// One work-group broadcasts x into SLM, then one work-item per output row.
template <int BITS>
struct GemvSlmKernel {
  const uint16_t* x;
  const uint8_t* plane2;
  const uint8_t* plane_hi;
  const uint16_t* scale;
  uint16_t* y;
  int M;
  int N;
  int K;
  int groups;
  int stride2;
  int stride_hi;

  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlmBytes>();
    const int n = (int)it.get_global_id(0);
    const int lid = (int)it.get_local_id(0);
    for (int m = 0; m < M; ++m) {
      for (int g = lid; g < groups; g += kGemvWg) {
        const uint16_t* src =
            x + ((size_t)m * (size_t)K + (size_t)g * 128);
        slm_store_group(src, (m * K + g * 128) * (int)sizeof(uint16_t));
      }
    }
    esimd::barrier();
    if (n >= N) {
      return;
    }
    const uint8_t* row2 = plane2 + (size_t)n * (size_t)stride2;
    const uint8_t* rowhi =
        plane_hi ? plane_hi + (size_t)n * (size_t)stride_hi : row2;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint16_t* scale_row = scale + (size_t)n * (size_t)groups;
    int g = 0;
    for (; g + 1 < groups; g += 2) {
      accum_pair<BITS>(row2, rowhi, scale_row, g, K, M, acc);
    }
    if (g < groups) {
      float sc = bf16_u16_to_f32(scale_row[g]);
      esimd::simd<float, 128> wf = dequant_group<BITS>(row2, rowhi, g, sc);
      for (int m = 0; m < M; ++m) {
        acc[m] += dot_slm((m * K + g * 128) * (int)sizeof(uint16_t), wf);
      }
    }
    for (int m = 0; m < M; ++m) {
      y[(size_t)m * (size_t)N + n] = f32_to_bf16_u16(acc[m]);
    }
  }
};

// One work-group per expert slot. x_per_slot selects the activation row.
template <int BITS>
struct IndexedGemvSlmKernel {
  const uint16_t* x;
  const int32_t* ids;
  const uint8_t* plane2;
  const uint8_t* plane_hi;
  const uint16_t* scale;
  uint16_t* y;
  int slots;
  int N;
  int K;
  int groups;
  int stride2;
  int stride_hi;
  int E;
  int topk;
  int x_per_slot;

  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlmRowBytes>();
    const int slot = (int)it.get_global_id(0);
    const int n = (int)it.get_global_id(1);
    const int lid = (int)it.get_local_id(1);
    if (slot < slots) {
      const int x_row = x_per_slot ? slot : slot / topk;
      for (int g = lid; g < groups; g += kGemvWg) {
        const uint16_t* src =
            x + ((size_t)x_row * (size_t)K + (size_t)g * 128);
        slm_store_group(src, g * 128 * (int)sizeof(uint16_t));
      }
    }
    esimd::barrier();
    if (slot >= slots || n >= N) {
      return;
    }
    const int e = ids[slot];
    if (e < 0 || e >= E) {
      y[(size_t)slot * (size_t)N + n] = 0;
      return;
    }
    const size_t row_base = ((size_t)e * (size_t)N + (size_t)n);
    const uint8_t* row2 = plane2 + row_base * (size_t)stride2;
    const uint8_t* rowhi =
        plane_hi ? plane_hi + row_base * (size_t)stride_hi : row2;
    float acc_box[1] = {0.f};
    const uint16_t* scale_row = scale + row_base * (size_t)groups;
    int g = 0;
    for (; g + 1 < groups; g += 2) {
      accum_pair<BITS>(row2, rowhi, scale_row, g, K, 1, acc_box);
    }
    if (g < groups) {
      float sc = bf16_u16_to_f32(scale_row[g]);
      esimd::simd<float, 128> wf = dequant_group<BITS>(row2, rowhi, g, sc);
      acc_box[0] += dot_slm(g * 128 * (int)sizeof(uint16_t), wf);
    }
    float acc = acc_box[0];
    y[(size_t)slot * (size_t)N + n] = f32_to_bf16_u16(acc);
  }
};

struct SiluMulKernel {
  const uint16_t* gu;
  uint16_t* hid;
  int I;
  void operator()(sycl::id<2> id) const {
    const int slot = (int)id[0];
    const int i = (int)id[1];
    const size_t base = (size_t)slot * (size_t)(I * 2);
    float g = bf16_to_f32_scalar(gu[base + i]);
    float u = bf16_to_f32_scalar(gu[base + I + i]);
    float h = g / (1.f + sycl::exp(-g)) * u;
    hid[(size_t)slot * (size_t)I + i] = f32_to_bf16_scalar(h);
  }
};

struct RouterReduceKernel {
  const uint16_t* down;
  const float* router;
  uint16_t* y;
  int topk;
  int H;
  void operator()(sycl::id<2> id) const {
    const int t = (int)id[0];
    const int h = (int)id[1];
    float acc = 0.f;
    for (int k = 0; k < topk; ++k) {
      int slot = t * topk + k;
      float w = router[slot];
      float v = bf16_to_f32_scalar(down[(size_t)slot * (size_t)H + h]);
      acc += w * v;
    }
    y[(size_t)t * (size_t)H + h] = f32_to_bf16_scalar(acc);
  }
};

// Round-trip through bf16 so a fused epilogue matches the 4-launch store/load.
SYCL_ESIMD_FUNCTION float bf16_roundtrip(float v) {
  return bf16_u16_to_f32(f32_to_bf16_u16(v));
}

// One output row. x of length K is already in SLM at byte 0.
template <int BITS>
SYCL_ESIMD_FUNCTION float gemv_row_slm(
    const uint8_t* row2,
    const uint8_t* rowhi,
    const uint16_t* scale_row,
    int K,
    int groups) {
  float acc_box[1] = {0.f};
  int g = 0;
  for (; g + 1 < groups; g += 2) {
    accum_pair<BITS>(row2, rowhi, scale_row, g, K, 1, acc_box);
  }
  if (g < groups) {
    float sc = bf16_u16_to_f32(scale_row[g]);
    esimd::simd<float, 128> wf = dequant_group<BITS>(row2, rowhi, g, sc);
    acc_box[0] += dot_slm(g * 128 * (int)sizeof(uint16_t), wf);
  }
  return acc_box[0];
}

SYCL_ESIMD_FUNCTION float gemv_row_slm_dyn(
    int bits,
    const uint8_t* row2,
    const uint8_t* rowhi,
    const uint16_t* scale_row,
    int K,
    int groups) {
  if (bits == 2) {
    return gemv_row_slm<2>(row2, rowhi, scale_row, K, groups);
  }
  if (bits == 4) {
    return gemv_row_slm<4>(row2, rowhi, scale_row, K, groups);
  }
  return gemv_row_slm<3>(row2, rowhi, scale_row, K, groups);
}

// Launch 1 of 2: gate and up for hidden index i, then SiLU, one submit.
// A work-group owns 16 consecutive i values of one expert slot and shares x.
template <int BITS>
struct GateUpSiluKernel {
  const uint16_t* x;
  const int32_t* ids;
  const uint8_t* plane2;
  const uint8_t* plane_hi;
  const uint16_t* scale;
  uint16_t* hid;
  int slots;
  int I;
  int H;
  int groups;
  int stride2;
  int stride_hi;
  int E;
  int topk;
  const int32_t* ebits;

  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlmRowBytes>();
    const int slot = (int)it.get_global_id(0);
    const int i = (int)it.get_global_id(1);
    const int lid = (int)it.get_local_id(1);
    if (slot < slots) {
      const int x_row = slot / topk;
      for (int g = lid; g < groups; g += kGemvWg) {
        const uint16_t* src =
            x + ((size_t)x_row * (size_t)H + (size_t)g * 128);
        slm_store_group(src, g * 128 * (int)sizeof(uint16_t));
      }
    }
    esimd::barrier();
    if (slot >= slots || i >= I) {
      return;
    }
    const int e = ids[slot];
    if (e < 0 || e >= E) {
      hid[(size_t)slot * (size_t)I + i] = 0;
      return;
    }
    const int N13 = I * 2;
    const size_t gate_row = (size_t)e * (size_t)N13 + (size_t)i;
    const size_t up_row = gate_row + (size_t)I;
    const uint8_t* grow2 = plane2 + gate_row * (size_t)stride2;
    const uint8_t* growhi =
        plane_hi ? plane_hi + gate_row * (size_t)stride_hi : grow2;
    const uint8_t* urow2 = plane2 + up_row * (size_t)stride2;
    const uint8_t* urowhi =
        plane_hi ? plane_hi + up_row * (size_t)stride_hi : urow2;
    const uint16_t* gsc = scale + gate_row * (size_t)groups;
    const uint16_t* usc_row = scale + up_row * (size_t)groups;
    float gacc;
    float uacc;
    if (ebits) {
      const int use_bits = ebits[e];
      gacc = gemv_row_slm_dyn(use_bits, grow2, growhi, gsc, H, groups);
      uacc = gemv_row_slm_dyn(use_bits, urow2, urowhi, usc_row, H, groups);
    } else {
      gacc = gemv_row_slm<BITS>(grow2, growhi, gsc, H, groups);
      uacc = gemv_row_slm<BITS>(urow2, urowhi, usc_row, H, groups);
    }
    gacc = bf16_roundtrip(gacc);
    uacc = bf16_roundtrip(uacc);
    float h = gacc / (1.f + sycl::exp(-gacc)) * uacc;
    hid[(size_t)slot * (size_t)I + i] = f32_to_bf16_u16(h);
  }
};

// Launch 2 of 2: down GEMV for every expert slot of one token, weighted sum.
// One work-group owns 16 consecutive output rows of one token.
template <int BITS>
struct DownReduceKernel {
  const uint16_t* hid;
  const int32_t* ids;
  const float* router;
  const uint8_t* plane2;
  const uint8_t* plane_hi;
  const uint16_t* scale;
  uint16_t* y;
  int T;
  int H;
  int I;
  int groups;
  int stride2;
  int stride_hi;
  int E;
  int topk;
  const int32_t* ebits;

  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlmRowBytes>();
    const int t = (int)it.get_global_id(0);
    const int h = (int)it.get_global_id(1);
    const int lid = (int)it.get_local_id(1);
    float acc = 0.f;
    for (int k = 0; k < topk; ++k) {
      const int slot = t * topk + k;
      if (t < T) {
        for (int g = lid; g < groups; g += kGemvWg) {
          const uint16_t* src =
              hid + ((size_t)slot * (size_t)I + (size_t)g * 128);
          slm_store_group(src, g * 128 * (int)sizeof(uint16_t));
        }
      }
      esimd::barrier();
      if (t < T && h < H) {
        const int e = ids[slot];
        if (e >= 0 && e < E) {
          const size_t row = (size_t)e * (size_t)H + (size_t)h;
          const uint8_t* row2 = plane2 + row * (size_t)stride2;
          const uint8_t* rowhi =
              plane_hi ? plane_hi + row * (size_t)stride_hi : row2;
          const uint16_t* srow = scale + row * (size_t)groups;
          float d = ebits
              ? gemv_row_slm_dyn(ebits[e], row2, rowhi, srow, I, groups)
              : gemv_row_slm<BITS>(row2, rowhi, srow, I, groups);
          d = bf16_roundtrip(d);
          acc += router[slot] * d;
        }
      }
      esimd::barrier();
    }
    if (t < T && h < H) {
      y[(size_t)t * (size_t)H + h] = f32_to_bf16_u16(acc);
    }
  }
};

// One launch, one work-group per token. The group loops experts itself.
// Local barriers only, so a short grid (T work-groups) cannot deadlock.
// The running sum stays in fp32. Occupancy is one work-group per token.
template <int BITS>
struct MoeTokenKernel {
  const uint16_t* x;
  const int32_t* ids;
  const float* router;
  const uint8_t* up2;
  const uint8_t* uph;
  const uint16_t* usc;
  const uint8_t* dp2;
  const uint8_t* dph;
  const uint16_t* dsc;
  uint16_t* hid;
  float* acc;
  uint16_t* y;
  int H;
  int I;
  int up_groups;
  int dn_groups;
  int up_stride2;
  int up_stride_hi;
  int dn_stride2;
  int dn_stride_hi;
  int E;
  int topk;

  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlmRowBytes>();
    const int t = (int)it.get_global_id(0);
    const int lid = (int)it.get_local_id(1);
    for (int h = lid; h < H; h += kGemvWg) {
      acc[(size_t)t * (size_t)H + h] = 0.f;
    }
    esimd::barrier();
    for (int k = 0; k < topk; ++k) {
      for (int g = lid; g < up_groups; g += kGemvWg) {
        const uint16_t* src = x + ((size_t)t * (size_t)H + (size_t)g * 128);
        slm_store_group(src, g * 128 * (int)sizeof(uint16_t));
      }
      esimd::barrier();
      const int slot = t * topk + k;
      const int e = ids[slot];
      const bool live = e >= 0 && e < E;
      for (int i = lid; i < I; i += kGemvWg) {
        float hval = 0.f;
        if (live) {
          const int n13 = I * 2;
          const size_t gate_row = (size_t)e * (size_t)n13 + (size_t)i;
          const size_t up_row = gate_row + (size_t)I;
          const uint8_t* grow2 = up2 + gate_row * (size_t)up_stride2;
          const uint8_t* growhi =
              uph ? uph + gate_row * (size_t)up_stride_hi : grow2;
          const uint8_t* urow2 = up2 + up_row * (size_t)up_stride2;
          const uint8_t* urowhi =
              uph ? uph + up_row * (size_t)up_stride_hi : urow2;
          float gacc = gemv_row_slm<BITS>(
              grow2, growhi, usc + gate_row * (size_t)up_groups, H, up_groups);
          float uacc = gemv_row_slm<BITS>(
              urow2, urowhi, usc + up_row * (size_t)up_groups, H, up_groups);
          gacc = bf16_roundtrip(gacc);
          uacc = bf16_roundtrip(uacc);
          hval = gacc / (1.f + sycl::exp(-gacc)) * uacc;
        }
        hid[(size_t)t * (size_t)I + i] = f32_to_bf16_u16(hval);
      }
      esimd::barrier();
      for (int g = lid; g < dn_groups; g += kGemvWg) {
        const uint16_t* src = hid + ((size_t)t * (size_t)I + (size_t)g * 128);
        slm_store_group(src, g * 128 * (int)sizeof(uint16_t));
      }
      esimd::barrier();
      const float w = live ? router[slot] : 0.f;
      for (int h = lid; h < H; h += kGemvWg) {
        float d = 0.f;
        if (live) {
          const size_t row = (size_t)e * (size_t)H + (size_t)h;
          const uint8_t* row2 = dp2 + row * (size_t)dn_stride2;
          const uint8_t* rowhi =
              dph ? dph + row * (size_t)dn_stride_hi : row2;
          d = gemv_row_slm<BITS>(
              row2, rowhi, dsc + row * (size_t)dn_groups, I, dn_groups);
          d = bf16_roundtrip(d);
        }
        acc[(size_t)t * (size_t)H + h] += w * d;
      }
      esimd::barrier();
    }
    for (int h = lid; h < H; h += kGemvWg) {
      y[(size_t)t * (size_t)H + h] =
          f32_to_bf16_u16(acc[(size_t)t * (size_t)H + h]);
    }
  }
};

static int moe_launch_count() {
  const char* env = std::getenv("B70_LOWBIT_MOE_LAUNCHES");
  if (env != nullptr && env[0] == '1' && env[1] == '\0') {
    return 1;
  }
  if (env != nullptr && env[0] == '2' && env[1] == '\0') {
    return 2;
  }
  return 4;
}

// 16 signed codes starting at k0 (multiple of 16).
SYCL_ESIMD_FUNCTION esimd::simd<int32_t, 16> load_codes16(
    const uint8_t* row2, const uint8_t* rowhi, int k0, int bits) {
  esimd::simd<int32_t, 16> codes(0);
  const uint8_t* p2 = row2 + (k0 >> 2);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t b = p2[i];
    codes[i * 4 + 0] = (int)(b & 3u);
    codes[i * 4 + 1] = (int)((b >> 2) & 3u);
    codes[i * 4 + 2] = (int)((b >> 4) & 3u);
    codes[i * 4 + 3] = (int)((b >> 6) & 3u);
  }
  if (bits == 3) {
    const uint8_t* ph = rowhi + (k0 >> 3);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint32_t b = ph[i];
#pragma unroll
      for (int bit = 0; bit < 8; ++bit) {
        codes[i * 8 + bit] =
            codes[i * 8 + bit] | (int)(((b >> bit) & 1u) << 2);
      }
    }
  } else if (bits == 4) {
    const uint8_t* ph = rowhi + (k0 >> 2);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t b = ph[i];
      codes[i * 4 + 0] |= (int)((b & 3u) << 2);
      codes[i * 4 + 1] |= (int)(((b >> 2) & 3u) << 2);
      codes[i * 4 + 2] |= (int)(((b >> 4) & 3u) << 2);
      codes[i * 4 + 3] |= (int)(((b >> 6) & 3u) << 2);
    }
  }
  int maxq = 1 << (bits - 1);
  int span = maxq << 1;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int c = (int)codes[i];
    if (c >= maxq) {
      c -= span;
    }
    codes[i] = c;
  }
  return codes;
}

SYCL_ESIMD_FUNCTION esimd::simd<int32_t, 32> load_codes32(
    const uint8_t* row2, const uint8_t* rowhi, int k0, int bits) {
  esimd::simd<int32_t, 32> codes(0);
  auto lo = load_codes16(row2, rowhi, k0, bits);
  auto hi = load_codes16(row2, rowhi, k0 + 16, bits);
  codes.template select<16, 1>(0) = lo;
  codes.template select<16, 1>(16) = hi;
  return codes;
}

// bf16 DPAS. A is 8x16 row-major bf16. B is VNNI-2, matching
// kernels/prefill_grouped_f16_esimd_v1.hpp: b[(kk/2)*32 + c*2 + (kk%2)].
struct PrefillBf16Kernel {
  const uint16_t* x;
  const uint8_t* plane2;
  const uint8_t* plane_hi;
  const uint16_t* scale;
  uint16_t* y;
  int M;
  int N;
  int K;
  int groups;
  int stride2;
  int stride_hi;
  int bits;
  int tiles_n;

  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    const int tile = (int)it.get_global_id(0);
    const int tm = tile / tiles_n;
    const int tn = tile - tm * tiles_n;
    const int m0 = tm * 8;
    const int n0 = tn * 16;
    esimd::simd<float, 128> acc(0.f);
    for (int g = 0; g < groups; ++g) {
      float wsc[16];
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        wsc[c] = bf16_u16_to_f32(
            scale[((size_t)(n0 + c) * (size_t)groups) + g]);
      }
#pragma unroll 1
      for (int step = 0; step < 8; ++step) {
        const int k0 = g * 128 + step * 16;
        esimd::simd<uint16_t, 128> az(0);
        esimd::simd<bf16_t, 128> a = az.template bit_cast_view<bf16_t>();
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          if (m0 + r < M) {
            esimd::simd<uint16_t, 16> xu = esimd::block_load<uint16_t, 16>(
                x + ((size_t)(m0 + r) * (size_t)K + k0), esimd::overaligned<32>);
            esimd::simd<bf16_t, 16> xb = xu.template bit_cast_view<bf16_t>();
            a.template select<16, 1>(r * 16) = xb;
          }
        }
        esimd::simd<uint16_t, 256> bz(0);
        esimd::simd<bf16_t, 256> b = bz.template bit_cast_view<bf16_t>();
#pragma unroll
        for (int c = 0; c < 16; ++c) {
          const uint8_t* row2 =
              plane2 + (size_t)(n0 + c) * (size_t)stride2;
          const uint8_t* rowhi =
              plane_hi ? plane_hi + (size_t)(n0 + c) * (size_t)stride_hi
                       : row2;
          esimd::simd<int32_t, 16> codes =
              load_codes16(row2, rowhi, k0, bits);
          esimd::simd<float, 16> wf = esimd::convert<float>(codes) * wsc[c];
          esimd::simd<uint16_t, 16> hb = f32_to_bf16_u16v<16>(wf);
          esimd::simd<bf16_t, 16> hbb = hb.template bit_cast_view<bf16_t>();
#pragma unroll
          for (int kk = 0; kk < 16; ++kk) {
            b[(kk / 2) * 32 + c * 2 + (kk % 2)] = hbb[kk];
          }
        }
        acc = esimd::xmx::dpas<8, 8, float>(acc, b, a);
      }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (m0 + r < M) {
        esimd::simd<float, 16> row = acc.template select<16, 1>(r * 16);
        esimd::simd<uint16_t, 16> hb = f32_to_bf16_u16v<16>(row);
        esimd::block_store<uint16_t, 16>(
            y + ((size_t)(m0 + r) * (size_t)N + n0), hb, esimd::overaligned<32>);
      }
    }
  }
};

struct ActQuantKernel {
  const uint16_t* x;
  int8_t* xq;
  float* ascale;
  int K;
  int groups;
  void operator()(sycl::id<1> id) const {
    const int mg = (int)id[0];
    const int m = mg / groups;
    const int g = mg - m * groups;
    const uint16_t* src = x + ((size_t)m * (size_t)K + (size_t)g * 128);
    float tmp[128];
    float amax = 0.f;
#pragma unroll 1
    for (int i = 0; i < 128; ++i) {
      float v = bf16_to_f32_scalar(src[i]);
      tmp[i] = v;
      float av = v < 0.f ? -v : v;
      if (av > amax) {
        amax = av;
      }
    }
    float sc = amax > 0.f ? amax / 127.f : 1.f;
    ascale[mg] = sc;
    int8_t* dst = xq + ((size_t)m * (size_t)K + (size_t)g * 128);
#pragma unroll 1
    for (int i = 0; i < 128; ++i) {
      int qv = (int)sycl::rint(tmp[i] / sc);
      if (qv > 127) {
        qv = 127;
      }
      if (qv < -127) {
        qv = -127;
      }
      dst[i] = (int8_t)qv;
    }
  }
};

// int8 DPAS. A is 8x32 row-major signed char. B is VNNI-4:
// b[(kk/4)*64 + c*4 + (kk%4)], K=32, N=16, 512 elements.
// Integer partials are scaled by w_scale * a_scale every group of 128.
struct PrefillI8Kernel {
  const int8_t* xq;
  const float* ascale;
  const uint8_t* plane2;
  const uint8_t* plane_hi;
  const uint16_t* scale;
  uint16_t* y;
  int M;
  int N;
  int K;
  int groups;
  int stride2;
  int stride_hi;
  int bits;
  int tiles_n;

  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    const int tile = (int)it.get_global_id(0);
    const int tm = tile / tiles_n;
    const int tn = tile - tm * tiles_n;
    const int m0 = tm * 8;
    const int n0 = tn * 16;
    esimd::simd<float, 128> facc(0.f);
    for (int g = 0; g < groups; ++g) {
      float wsc[16];
      float asc[8];
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        wsc[c] = bf16_u16_to_f32(
            scale[((size_t)(n0 + c) * (size_t)groups) + g]);
      }
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        asc[r] = (m0 + r < M) ? ascale[(size_t)(m0 + r) * (size_t)groups + g]
                              : 0.f;
      }
      esimd::simd<int32_t, 128> iacc(0);
#pragma unroll 1
      for (int step = 0; step < 4; ++step) {
        const int k0 = g * 128 + step * 32;
        esimd::simd<signed char, 256> a(0);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          if (m0 + r < M) {
            esimd::simd<signed char, 32> chunk =
                esimd::block_load<signed char, 32>(
                    xq + ((size_t)(m0 + r) * (size_t)K + k0),
                    esimd::overaligned<32>);
            a.template select<32, 1>(r * 32) = chunk;
          }
        }
        esimd::simd<signed char, 512> b(0);
#pragma unroll
        for (int c = 0; c < 16; ++c) {
          const uint8_t* row2 =
              plane2 + (size_t)(n0 + c) * (size_t)stride2;
          const uint8_t* rowhi =
              plane_hi ? plane_hi + (size_t)(n0 + c) * (size_t)stride_hi
                       : row2;
          esimd::simd<int32_t, 32> codes =
              load_codes32(row2, rowhi, k0, bits);
#pragma unroll
          for (int kk = 0; kk < 32; ++kk) {
            b[(kk / 4) * 64 + c * 4 + (kk % 4)] = (signed char)codes[kk];
          }
        }
        iacc = esimd::xmx::dpas<8, 8, int32_t>(iacc, b, a);
      }
      esimd::simd<float, 128> part = esimd::convert<float>(iacc);
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        float as = asc[r];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
          const int idx = r * 16 + c;
          facc[idx] = (float)facc[idx] + (float)part[idx] * wsc[c] * as;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (m0 + r < M) {
        esimd::simd<float, 16> row = facc.template select<16, 1>(r * 16);
        esimd::simd<uint16_t, 16> hb = f32_to_bf16_u16v<16>(row);
        esimd::block_store<uint16_t, 16>(
            y + ((size_t)(m0 + r) * (size_t)N + n0), hb, esimd::overaligned<32>);
      }
    }
  }
};

static void check_bits(int bits) {
  TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2, 3, or 4");
}

static void check_planes(
    const torch::Tensor& plane2,
    const torch::Tensor& plane_hi,
    const torch::Tensor& scale,
    int64_t n,
    int64_t k,
    int bits) {
  check_bits(bits);
  TORCH_CHECK(k > 0 && (k % 128) == 0, "K must be a multiple of 128");
  TORCH_CHECK(plane2.is_xpu() && plane2.is_contiguous(), "plane2");
  TORCH_CHECK(plane2.scalar_type() == at::kByte, "plane2 uint8");
  TORCH_CHECK(plane2.size(0) == n, "plane2 N");
  TORCH_CHECK(plane2.size(1) == plane2_stride_of((int)k), "plane2 stride");
  TORCH_CHECK(scale.is_xpu() && scale.is_contiguous(), "scale");
  TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale bf16");
  TORCH_CHECK(scale.size(0) == n && scale.size(1) == k / 128, "scale shape");
  if (bits == 2) {
    TORCH_CHECK(plane_hi.numel() == 0, "int2 has no plane_hi");
  } else {
    TORCH_CHECK(plane_hi.is_xpu() && plane_hi.is_contiguous(), "plane_hi");
    TORCH_CHECK(plane_hi.scalar_type() == at::kByte, "plane_hi uint8");
    TORCH_CHECK(
        plane_hi.size(0) == n &&
            plane_hi.size(1) == plane_hi_stride_of((int)k, bits),
        "plane_hi stride");
  }
}

static const uint8_t* hi_ptr(const torch::Tensor& plane_hi) {
  return plane_hi.numel() ? plane_hi.data_ptr<uint8_t>() : nullptr;
}

static torch::Tensor gemv(
    const torch::Tensor& x,
    const torch::Tensor& plane2,
    const torch::Tensor& plane_hi,
    const torch::Tensor& scale,
    int64_t bits) {
  TORCH_CHECK(x.is_xpu() && x.is_contiguous(), "x");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x [M,K] bf16");
  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  const int N = (int)plane2.size(0);
  TORCH_CHECK(M >= 1 && M <= 8, "gemv M is 1..8");
  TORCH_CHECK(
      (int64_t)M * K * (int)sizeof(uint16_t) <= kSlmBytes,
      "gemv SLM holds M*K bf16 up to 8*2560");
  check_planes(plane2, plane_hi, scale, N, K, (int)bits);
  const at::DeviceGuard guard(x.device());
  auto y = torch::empty({M, N}, x.options());
  record_use(x);
  record_use(plane2);
  record_use(plane_hi);
  record_use(scale);
  record_use(y);
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const size_t gN = ((size_t)N + kGemvWg - 1) / kGemvWg * kGemvWg;
  const auto nd = sycl::nd_range<1>(
      sycl::range<1>(gN), sycl::range<1>((size_t)kGemvWg));
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint8_t* p2 = plane2.data_ptr<uint8_t>();
  const uint8_t* ph = hi_ptr(plane_hi);
  const uint16_t* sc = reinterpret_cast<const uint16_t*>(scale.data_ptr());
  uint16_t* yp = reinterpret_cast<uint16_t*>(y.data_ptr());
  const int groups = K / 128;
  const int stride2 = (int)plane2.size(1);
  const int stride_hi = bits == 2 ? 0 : (int)plane_hi.size(1);
  if (bits == 2) {
    q.parallel_for(
        nd, GemvSlmKernel<2>{xp, p2, ph, sc, yp, M, N, K, groups, stride2, stride_hi});
  } else if (bits == 3) {
    q.parallel_for(
        nd, GemvSlmKernel<3>{xp, p2, ph, sc, yp, M, N, K, groups, stride2, stride_hi});
  } else {
    q.parallel_for(
        nd, GemvSlmKernel<4>{xp, p2, ph, sc, yp, M, N, K, groups, stride2, stride_hi});
  }
  return y;
}

// B70_LOWBIT_MOE_LAUNCHES=2 fuses SiLU into gate/up and the router sum into
// down (two submits, full N grid). =1 is one submit with one work-group per
// token. Unset stays the four-submit path.
static torch::Tensor launch_reduced_moe(
    const torch::Tensor& x,
    const torch::Tensor& ids,
    const torch::Tensor& router_w,
    const torch::Tensor& w13_plane2,
    const torch::Tensor& w13_plane_hi,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2_plane2,
    const torch::Tensor& w2_plane_hi,
    const torch::Tensor& w2_scale,
    int64_t bits,
    int which,
    const int32_t* ebits = nullptr) {
  const int T = (int)x.size(0);
  const int H = (int)x.size(1);
  const int topk = (int)ids.size(1);
  const int E = (int)w13_plane2.size(0);
  const int I = (int)w13_plane2.size(1) / 2;
  const int slots = T * topk;
  const at::DeviceGuard guard(x.device());
  auto y = torch::empty({T, H}, x.options());
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const int32_t* idp = ids.data_ptr<int32_t>();
  const float* rp = router_w.data_ptr<float>();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint8_t* up2 = w13_plane2.data_ptr<uint8_t>();
  const uint8_t* uph = hi_ptr(w13_plane_hi);
  const uint16_t* usc = reinterpret_cast<const uint16_t*>(w13_scale.data_ptr());
  const uint8_t* dp2 = w2_plane2.data_ptr<uint8_t>();
  const uint8_t* dph = hi_ptr(w2_plane_hi);
  const uint16_t* dsc = reinterpret_cast<const uint16_t*>(w2_scale.data_ptr());
  const int up_stride2 = (int)w13_plane2.size(2);
  const int up_stride_hi = bits == 2 ? 0 : (int)w13_plane_hi.size(2);
  const int dn_stride2 = (int)w2_plane2.size(2);
  const int dn_stride_hi = bits == 2 ? 0 : (int)w2_plane_hi.size(2);
  uint16_t* yp = reinterpret_cast<uint16_t*>(y.data_ptr());
  auto run = [&](auto tag) {
    constexpr int BITS = decltype(tag)::value;
    if (which == 1) {
      TORCH_CHECK(ebits == nullptr, "mixed bits requires MOE_LAUNCHES=2");
      auto hid = torch::empty({T, I}, x.options());
      auto acc = torch::empty({T, H}, x.options().dtype(at::kFloat));
      for (const torch::Tensor& t :
           {x, ids, router_w, w13_plane2, w13_plane_hi, w13_scale, w2_plane2,
            w2_plane_hi, w2_scale, hid, acc, y}) {
        record_use(t);
      }
      const auto nd = sycl::nd_range<2>(
          sycl::range<2>((size_t)T, (size_t)kGemvWg),
          sycl::range<2>(1, (size_t)kGemvWg));
      q.parallel_for(
          nd,
          MoeTokenKernel<BITS>{
              xp, idp, rp, up2, uph, usc, dp2, dph, dsc,
              reinterpret_cast<uint16_t*>(hid.data_ptr()),
              acc.data_ptr<float>(), yp, H, I, H / 128, I / 128, up_stride2,
              up_stride_hi, dn_stride2, dn_stride_hi, E, topk});
      return;
    }
    auto hid = torch::empty({slots, I}, x.options());
    for (const torch::Tensor& t :
         {x, ids, router_w, w13_plane2, w13_plane_hi, w13_scale, w2_plane2,
          w2_plane_hi, w2_scale, hid, y}) {
      record_use(t);
    }
    uint16_t* hp = reinterpret_cast<uint16_t*>(hid.data_ptr());
    const size_t gI = ((size_t)I + kGemvWg - 1) / kGemvWg * kGemvWg;
    const size_t gH = ((size_t)H + kGemvWg - 1) / kGemvWg * kGemvWg;
    q.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>((size_t)slots, gI),
            sycl::range<2>(1, (size_t)kGemvWg)),
        GateUpSiluKernel<BITS>{
            xp, idp, up2, uph, usc, hp, slots, I, H, H / 128, up_stride2,
            up_stride_hi, E, topk, ebits});
    q.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>((size_t)T, gH),
            sycl::range<2>(1, (size_t)kGemvWg)),
        DownReduceKernel<BITS>{
            hp, idp, rp, dp2, dph, dsc, yp, T, H, I, I / 128, dn_stride2,
            dn_stride_hi, E, topk, ebits});
  };
  if (bits == 2) {
    run(std::integral_constant<int, 2>{});
  } else if (bits == 3) {
    run(std::integral_constant<int, 3>{});
  } else {
    run(std::integral_constant<int, 4>{});
  }
  return y;
}

static torch::Tensor moe_gemv(
    const torch::Tensor& x,
    const torch::Tensor& ids,
    const torch::Tensor& router_w,
    const torch::Tensor& w13_plane2,
    const torch::Tensor& w13_plane_hi,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2_plane2,
    const torch::Tensor& w2_plane_hi,
    const torch::Tensor& w2_scale,
    int64_t bits) {
  TORCH_CHECK(x.is_xpu() && x.is_contiguous(), "x");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x [T,H]");
  TORCH_CHECK(ids.is_xpu() && ids.is_contiguous(), "ids");
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 2, "ids [T,topk]");
  TORCH_CHECK(router_w.is_xpu() && router_w.is_contiguous(), "router");
  TORCH_CHECK(router_w.scalar_type() == at::kFloat, "router fp32");
  TORCH_CHECK(ids.sizes() == router_w.sizes(), "router shape");
  const int T = (int)x.size(0);
  const int H = (int)x.size(1);
  const int topk = (int)ids.size(1);
  TORCH_CHECK(ids.size(0) == T, "ids T");
  TORCH_CHECK(T >= 1 && T <= 8, "moe_gemv T is 1..8");
  TORCH_CHECK(topk >= 1, "topk");
  TORCH_CHECK(w13_plane2.dim() == 3 && w2_plane2.dim() == 3, "expert rank");
  const int E = (int)w13_plane2.size(0);
  const int N13 = (int)w13_plane2.size(1);
  TORCH_CHECK(N13 % 2 == 0, "w13 N is 2*I");
  const int I = N13 / 2;
  TORCH_CHECK(w2_plane2.size(0) == E, "E");
  TORCH_CHECK(w2_plane2.size(1) == H, "down N");
  check_planes(
      w13_plane2.reshape({(int64_t)E * N13, w13_plane2.size(2)}),
      bits == 2 ? w13_plane_hi
                : w13_plane_hi.reshape({(int64_t)E * N13, w13_plane_hi.size(2)}),
      w13_scale.reshape({(int64_t)E * N13, w13_scale.size(2)}),
      (int64_t)E * N13,
      H,
      (int)bits);
  check_planes(
      w2_plane2.reshape({(int64_t)E * H, w2_plane2.size(2)}),
      bits == 2 ? w2_plane_hi
                : w2_plane_hi.reshape({(int64_t)E * H, w2_plane_hi.size(2)}),
      w2_scale.reshape({(int64_t)E * H, w2_scale.size(2)}),
      (int64_t)E * H,
      I,
      (int)bits);

  TORCH_CHECK(H <= 2560 && I <= 2560, "moe SLM activation row cap is 2560");
  if (moe_launch_count() != 4) {
    return launch_reduced_moe(
        x, ids, router_w, w13_plane2, w13_plane_hi, w13_scale, w2_plane2,
        w2_plane_hi, w2_scale, bits, moe_launch_count());
  }

  const int slots = T * topk;
  const at::DeviceGuard guard(x.device());
  auto gu = torch::empty({slots, N13}, x.options());
  auto hid = torch::empty({slots, I}, x.options());
  auto down = torch::empty({slots, H}, x.options());
  auto y = torch::empty({T, H}, x.options());
  for (const torch::Tensor& t :
       {x, ids, router_w, w13_plane2, w13_plane_hi, w13_scale, w2_plane2,
        w2_plane_hi, w2_scale, gu, hid, down, y}) {
    record_use(t);
  }
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const int32_t* idp = ids.data_ptr<int32_t>();
  const size_t gN13 = ((size_t)N13 + kGemvWg - 1) / kGemvWg * kGemvWg;
  const size_t gH = ((size_t)H + kGemvWg - 1) / kGemvWg * kGemvWg;
  const auto nd_up = sycl::nd_range<2>(
      sycl::range<2>((size_t)slots, gN13),
      sycl::range<2>(1, (size_t)kGemvWg));
  const auto nd_dn = sycl::nd_range<2>(
      sycl::range<2>((size_t)slots, gH),
      sycl::range<2>(1, (size_t)kGemvWg));
  auto launch_indexed = [&](auto kernel, sycl::nd_range<2> nd) {
    q.parallel_for(nd, kernel);
  };
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint8_t* up2 = w13_plane2.data_ptr<uint8_t>();
  const uint8_t* uph = hi_ptr(w13_plane_hi);
  const uint16_t* usc = reinterpret_cast<const uint16_t*>(w13_scale.data_ptr());
  uint16_t* gup = reinterpret_cast<uint16_t*>(gu.data_ptr());
  const int up_stride2 = (int)w13_plane2.size(2);
  const int up_stride_hi = bits == 2 ? 0 : (int)w13_plane_hi.size(2);
  if (bits == 2) {
    launch_indexed(
        IndexedGemvSlmKernel<2>{
            xp, idp, up2, uph, usc, gup, slots, N13, H, H / 128, up_stride2,
            up_stride_hi, E, topk, 0},
        nd_up);
  } else if (bits == 3) {
    launch_indexed(
        IndexedGemvSlmKernel<3>{
            xp, idp, up2, uph, usc, gup, slots, N13, H, H / 128, up_stride2,
            up_stride_hi, E, topk, 0},
        nd_up);
  } else {
    launch_indexed(
        IndexedGemvSlmKernel<4>{
            xp, idp, up2, uph, usc, gup, slots, N13, H, H / 128, up_stride2,
            up_stride_hi, E, topk, 0},
        nd_up);
  }
  q.parallel_for(
      sycl::range<2>((size_t)slots, (size_t)I),
      SiluMulKernel{
          reinterpret_cast<const uint16_t*>(gu.data_ptr()),
          reinterpret_cast<uint16_t*>(hid.data_ptr()),
          I});
  const uint16_t* hp = reinterpret_cast<const uint16_t*>(hid.data_ptr());
  const uint8_t* dp2 = w2_plane2.data_ptr<uint8_t>();
  const uint8_t* dph = hi_ptr(w2_plane_hi);
  const uint16_t* dsc = reinterpret_cast<const uint16_t*>(w2_scale.data_ptr());
  uint16_t* dnp = reinterpret_cast<uint16_t*>(down.data_ptr());
  const int dn_stride2 = (int)w2_plane2.size(2);
  const int dn_stride_hi = bits == 2 ? 0 : (int)w2_plane_hi.size(2);
  if (bits == 2) {
    launch_indexed(
        IndexedGemvSlmKernel<2>{
            hp, idp, dp2, dph, dsc, dnp, slots, H, I, I / 128, dn_stride2,
            dn_stride_hi, E, topk, 1},
        nd_dn);
  } else if (bits == 3) {
    launch_indexed(
        IndexedGemvSlmKernel<3>{
            hp, idp, dp2, dph, dsc, dnp, slots, H, I, I / 128, dn_stride2,
            dn_stride_hi, E, topk, 1},
        nd_dn);
  } else {
    launch_indexed(
        IndexedGemvSlmKernel<4>{
            hp, idp, dp2, dph, dsc, dnp, slots, H, I, I / 128, dn_stride2,
            dn_stride_hi, E, topk, 1},
        nd_dn);
  }
  q.parallel_for(
      sycl::range<2>((size_t)T, (size_t)H),
      RouterReduceKernel{
          reinterpret_cast<const uint16_t*>(down.data_ptr()),
          router_w.data_ptr<float>(),
          reinterpret_cast<uint16_t*>(y.data_ptr()),
          topk,
          H});
  return y;
}

static int tiles_n_of(int n) {
  TORCH_CHECK(n > 0 && (n % 16) == 0, "prefill N must be a multiple of 16");
  return n / 16;
}

static torch::Tensor prefill_wna16(
    const torch::Tensor& x,
    const torch::Tensor& plane2,
    const torch::Tensor& plane_hi,
    const torch::Tensor& scale,
    int64_t bits) {
  TORCH_CHECK(x.is_xpu() && x.is_contiguous(), "x");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x [M,K] bf16");
  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  const int N = (int)plane2.size(0);
  TORCH_CHECK(M >= 1, "M");
  check_planes(plane2, plane_hi, scale, N, K, (int)bits);
  const int tiles_n = tiles_n_of(N);
  const int tiles_m = (M + 7) / 8;
  const at::DeviceGuard guard(x.device());
  auto y = torch::empty({M, N}, x.options());
  record_use(x);
  record_use(plane2);
  record_use(plane_hi);
  record_use(scale);
  record_use(y);
  PrefillBf16Kernel k{
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      plane2.data_ptr<uint8_t>(),
      hi_ptr(plane_hi),
      reinterpret_cast<const uint16_t*>(scale.data_ptr()),
      reinterpret_cast<uint16_t*>(y.data_ptr()),
      M,
      N,
      K,
      K / 128,
      (int)plane2.size(1),
      bits == 2 ? 0 : (int)plane_hi.size(1),
      (int)bits,
      tiles_n};
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const size_t ntiles = (size_t)tiles_m * (size_t)tiles_n;
  q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(ntiles), sycl::range<1>(1)), k);
  return y;
}

static torch::Tensor prefill_wxa8(
    const torch::Tensor& x,
    const torch::Tensor& plane2,
    const torch::Tensor& plane_hi,
    const torch::Tensor& scale,
    int64_t bits) {
  TORCH_CHECK(x.is_xpu() && x.is_contiguous(), "x");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x [M,K] bf16");
  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  const int N = (int)plane2.size(0);
  TORCH_CHECK(M >= 1, "M");
  check_planes(plane2, plane_hi, scale, N, K, (int)bits);
  const int groups = K / 128;
  const int tiles_n = tiles_n_of(N);
  const int tiles_m = (M + 7) / 8;
  const at::DeviceGuard guard(x.device());
  auto xq = torch::empty({M, K}, x.options().dtype(torch::kInt8));
  auto ascale = torch::empty(
      {M, groups}, x.options().dtype(torch::kFloat32));
  auto y = torch::empty({M, N}, x.options());
  record_use(x);
  record_use(plane2);
  record_use(plane_hi);
  record_use(scale);
  record_use(xq);
  record_use(ascale);
  record_use(y);
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(
      sycl::range<1>((size_t)M * (size_t)groups),
      ActQuantKernel{
          reinterpret_cast<const uint16_t*>(x.data_ptr()),
          xq.data_ptr<int8_t>(),
          ascale.data_ptr<float>(),
          K,
          groups});
  PrefillI8Kernel k{
      xq.data_ptr<int8_t>(),
      ascale.data_ptr<float>(),
      plane2.data_ptr<uint8_t>(),
      hi_ptr(plane_hi),
      reinterpret_cast<const uint16_t*>(scale.data_ptr()),
      reinterpret_cast<uint16_t*>(y.data_ptr()),
      M,
      N,
      K,
      groups,
      (int)plane2.size(1),
      bits == 2 ? 0 : (int)plane_hi.size(1),
      (int)bits,
      tiles_n};
  const size_t ntiles = (size_t)tiles_m * (size_t)tiles_n;
  q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(ntiles), sycl::range<1>(1)), k);
  return y;
}

// One work-group owns one expert and one 16-wide N tile. B for the whole
// K is unpacked into SLM once (K<=2048, 16 columns, bf16 VNNI: 64 KiB),
// then each work-item DPAS-walks its own 8-row chunks. Empty experts
// return after slm_init so the group stays convergent.
static constexpr int kGroupedLocal = 8;
static constexpr int kGroupedSlm = 65536;

template <int BITS>
struct GroupedPrefillKernel {
  const uint16_t* x;
  const int32_t* order;
  const int32_t* offsets;
  const uint8_t* plane2;
  const uint8_t* plane_hi;
  const uint16_t* scale;
  uint16_t* y;
  int K;
  int N;
  int groups;
  int stride2;
  int stride_hi;
  int E;
  int topk;
  int tiles_n;
  int x_per_slot;

  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kGroupedSlm>();
    const int wg = (int)it.get_group(0);
    const int lid = (int)it.get_local_id(0);
    const int e = wg / tiles_n;
    const int tn = wg - e * tiles_n;
    const int begin = e < E ? offsets[e] : 0;
    const int end = e < E ? offsets[e + 1] : 0;
    const int count = end - begin;
    if (e >= E || count <= 0) {
      return;
    }
    const int n0 = tn * 16;
    const int steps = K >> 4;
    for (int step = lid; step < steps; step += kGroupedLocal) {
      const int k0 = step << 4;
      const int g = k0 >> 7;
      esimd::simd<uint16_t, 256> raw(0);
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        const size_t row = (size_t)e * (size_t)N + (size_t)(n0 + c);
        const uint8_t* row2 = plane2 + row * (size_t)stride2;
        const uint8_t* rowhi =
            plane_hi ? plane_hi + row * (size_t)stride_hi : row2;
        esimd::simd<int32_t, 16> codes =
            load_codes16(row2, rowhi, k0, BITS);
        float wsc = bf16_u16_to_f32(scale[row * (size_t)groups + (size_t)g]);
        esimd::simd<float, 16> wf = esimd::convert<float>(codes) * wsc;
        esimd::simd<uint16_t, 16> hb = f32_to_bf16_u16v<16>(wf);
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          raw[(kk / 2) * 32 + c * 2 + (kk % 2)] = hb[kk];
        }
      }
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        esimd::slm_block_store<uint16_t, 32>(
            step * 512 + s * 64, raw.template select<32, 1>(s * 32));
      }
    }
    esimd::barrier();
    const int n_row_tiles = (count + 7) >> 3;
    for (int tm = lid; tm < n_row_tiles; tm += kGroupedLocal) {
      const int m0 = tm << 3;
      esimd::simd<float, 128> acc(0.f);
      for (int step = 0; step < steps; ++step) {
        const int k0 = step << 4;
        esimd::simd<uint16_t, 128> az(0);
        esimd::simd<bf16_t, 128> a = az.template bit_cast_view<bf16_t>();
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          if (m0 + r < count) {
            const int slot = order[begin + m0 + r];
            const int x_row = x_per_slot ? slot : slot / topk;
            esimd::simd<uint16_t, 16> xu = esimd::block_load<uint16_t, 16>(
                x + ((size_t)x_row * (size_t)K + (size_t)k0),
                esimd::overaligned<32>);
            a.template select<16, 1>(r * 16) =
                xu.template bit_cast_view<bf16_t>();
          }
        }
        esimd::simd<uint16_t, 256> brow(0);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          brow.template select<32, 1>(s * 32) =
              esimd::slm_block_load<uint16_t, 32>(step * 512 + s * 64);
        }
        esimd::simd<bf16_t, 256> b = brow.template bit_cast_view<bf16_t>();
        acc = esimd::xmx::dpas<8, 8, float>(acc, b, a);
      }
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        if (m0 + r < count) {
          const int slot = order[begin + m0 + r];
          esimd::simd<float, 16> row = acc.template select<16, 1>(r * 16);
          esimd::simd<uint16_t, 16> hb = f32_to_bf16_u16v<16>(row);
          esimd::block_store<uint16_t, 16>(
              y + ((size_t)slot * (size_t)N + (size_t)n0),
              hb,
              esimd::overaligned<32>);
        }
      }
    }
  }
};

template <int BITS>
static void launch_grouped(
    sycl::queue& q,
    const torch::Tensor& x,
    const torch::Tensor& order,
    const torch::Tensor& offsets,
    const torch::Tensor& plane2,
    const torch::Tensor& plane_hi,
    const torch::Tensor& scale,
    const torch::Tensor& y,
    int K,
    int N,
    int E,
    int topk,
    int x_per_slot) {
  const int tiles_n = N / 16;
  const int steps = K / 16;
  TORCH_CHECK(steps * 512 <= kGroupedSlm, "grouped prefill K exceeds SLM");
  GroupedPrefillKernel<BITS> k{
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      order.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>(),
      plane2.data_ptr<uint8_t>(),
      hi_ptr(plane_hi),
      reinterpret_cast<const uint16_t*>(scale.data_ptr()),
      reinterpret_cast<uint16_t*>(y.data_ptr()),
      K,
      N,
      K / 128,
      (int)plane2.size(2),
      BITS == 2 ? 0 : (int)plane_hi.size(2),
      E,
      topk,
      tiles_n,
      x_per_slot};
  const size_t local = (size_t)kGroupedLocal;
  const size_t ngrp = (size_t)E * (size_t)tiles_n;
  q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(ngrp * local), sycl::range<1>(local)),
      k);
}

// Eager prefill. One histogram, one sort, then one DPAS submit for gate/up
// across every expert, SiLU, one DPAS submit for down, and the router sum.
// Counts stay on device.
static torch::Tensor moe_prefill(
    const torch::Tensor& x,
    const torch::Tensor& ids,
    const torch::Tensor& router_w,
    const torch::Tensor& w13_plane2,
    const torch::Tensor& w13_plane_hi,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2_plane2,
    const torch::Tensor& w2_plane_hi,
    const torch::Tensor& w2_scale,
    int64_t bits) {
  TORCH_CHECK(x.is_xpu() && x.is_contiguous(), "x");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x [T,H]");
  TORCH_CHECK(ids.is_xpu() && ids.is_contiguous(), "ids");
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 2, "ids");
  TORCH_CHECK(router_w.is_xpu() && router_w.is_contiguous(), "router");
  TORCH_CHECK(router_w.scalar_type() == at::kFloat, "router fp32");
  TORCH_CHECK(ids.sizes() == router_w.sizes(), "router shape");
  TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits");
  const int T = (int)x.size(0);
  const int H = (int)x.size(1);
  const int topk = (int)ids.size(1);
  TORCH_CHECK(ids.size(0) == T && T >= 1 && topk >= 1, "topk");
  TORCH_CHECK(w13_plane2.dim() == 3 && w2_plane2.dim() == 3, "expert rank");
  const int E = (int)w13_plane2.size(0);
  const int N13 = (int)w13_plane2.size(1);
  TORCH_CHECK(N13 % 2 == 0 && (N13 % 16) == 0 && (H % 16) == 0, "tile");
  const int I = N13 / 2;
  TORCH_CHECK(w2_plane2.size(0) == E && w2_plane2.size(1) == H, "down");
  check_planes(
      w13_plane2.reshape({(int64_t)E * N13, w13_plane2.size(2)}),
      bits == 2 ? w13_plane_hi
                : w13_plane_hi.reshape({(int64_t)E * N13, w13_plane_hi.size(2)}),
      w13_scale.reshape({(int64_t)E * N13, w13_scale.size(2)}),
      (int64_t)E * N13,
      H,
      (int)bits);
  check_planes(
      w2_plane2.reshape({(int64_t)E * H, w2_plane2.size(2)}),
      bits == 2 ? w2_plane_hi
                : w2_plane_hi.reshape({(int64_t)E * H, w2_plane_hi.size(2)}),
      w2_scale.reshape({(int64_t)E * H, w2_scale.size(2)}),
      (int64_t)E * H,
      I,
      (int)bits);
  const int slots = T * topk;
  const at::DeviceGuard guard(x.device());
  auto flat = ids.reshape({-1});
  auto key = torch::where(flat >= 0, flat, torch::full_like(flat, E));
  auto counts =
      torch::bincount(key.to(at::kLong), std::nullopt, E + 1).to(at::kInt);
  auto order =
      torch::argsort(key, /*stable=*/true, /*dim=*/-1, /*descending=*/false)
          .to(at::kInt);
  // cumsum of an integer tensor is int64. The kernel reads int32 offsets.
  auto csum = torch::cumsum(counts.slice(0, 0, E), 0);
  auto offsets =
      torch::cat({torch::zeros({1}, counts.options()), csum}, 0).to(at::kInt);
  auto gu = torch::zeros({slots, N13}, x.options());
  auto hid = torch::zeros({slots, I}, x.options());
  auto down = torch::zeros({slots, H}, x.options());
  auto y = torch::empty({T, H}, x.options());
  for (const torch::Tensor& t :
       {x, ids, router_w, w13_plane2, w13_plane_hi, w13_scale, w2_plane2,
        w2_plane_hi, w2_scale, key, counts, order, offsets, gu, hid, down, y}) {
    record_use(t);
  }
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  auto launch = [&](auto tag, const torch::Tensor& act, const torch::Tensor& p2,
                    const torch::Tensor& phi, const torch::Tensor& sc,
                    const torch::Tensor& out, int K, int N, int x_per_slot) {
    constexpr int BITS = decltype(tag)::value;
    launch_grouped<BITS>(
        q, act, order, offsets, p2, phi, sc, out, K, N, E, topk, x_per_slot);
  };
  if (bits == 2) {
    launch(std::integral_constant<int, 2>{}, x, w13_plane2, w13_plane_hi,
           w13_scale, gu, H, N13, 0);
  } else if (bits == 3) {
    launch(std::integral_constant<int, 3>{}, x, w13_plane2, w13_plane_hi,
           w13_scale, gu, H, N13, 0);
  } else {
    launch(std::integral_constant<int, 4>{}, x, w13_plane2, w13_plane_hi,
           w13_scale, gu, H, N13, 0);
  }
  q.parallel_for(
      sycl::range<2>((size_t)slots, (size_t)I),
      SiluMulKernel{
          reinterpret_cast<const uint16_t*>(gu.data_ptr()),
          reinterpret_cast<uint16_t*>(hid.data_ptr()),
          I});
  if (bits == 2) {
    launch(std::integral_constant<int, 2>{}, hid, w2_plane2, w2_plane_hi,
           w2_scale, down, I, H, 1);
  } else if (bits == 3) {
    launch(std::integral_constant<int, 3>{}, hid, w2_plane2, w2_plane_hi,
           w2_scale, down, I, H, 1);
  } else {
    launch(std::integral_constant<int, 4>{}, hid, w2_plane2, w2_plane_hi,
           w2_scale, down, I, H, 1);
  }
  q.parallel_for(
      sycl::range<2>((size_t)T, (size_t)H),
      RouterReduceKernel{
          reinterpret_cast<const uint16_t*>(down.data_ptr()),
          router_w.data_ptr<float>(),
          reinterpret_cast<uint16_t*>(y.data_ptr()),
          topk,
          H});
  return y;
}

// Same two-submit decode as moe_gemv, but each expert reads its own bit
// width. `bits` is the storage width (hi-plane stride), not the traffic.
// Uniform checkpoints keep the templated moe_gemv.
static torch::Tensor moe_gemv_mixed(
    const torch::Tensor& x,
    const torch::Tensor& ids,
    const torch::Tensor& router_w,
    const torch::Tensor& w13_plane2,
    const torch::Tensor& w13_plane_hi,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w2_plane2,
    const torch::Tensor& w2_plane_hi,
    const torch::Tensor& w2_scale,
    const torch::Tensor& expert_bits,
    int64_t bits) {
  TORCH_CHECK(x.is_xpu() && x.is_contiguous(), "x");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x [T,H]");
  TORCH_CHECK(ids.is_xpu() && ids.is_contiguous(), "ids");
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 2, "ids [T,topk]");
  TORCH_CHECK(router_w.is_xpu() && router_w.is_contiguous(), "router");
  TORCH_CHECK(router_w.scalar_type() == at::kFloat, "router fp32");
  TORCH_CHECK(ids.sizes() == router_w.sizes(), "router shape");
  TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits");
  const int T = (int)x.size(0);
  const int H = (int)x.size(1);
  const int topk = (int)ids.size(1);
  TORCH_CHECK(ids.size(0) == T && T >= 1 && T <= 8, "T");
  TORCH_CHECK(w13_plane2.dim() == 3 && w2_plane2.dim() == 3, "expert rank");
  const int E = (int)w13_plane2.size(0);
  const int N13 = (int)w13_plane2.size(1);
  TORCH_CHECK(N13 % 2 == 0 && w2_plane2.size(0) == E, "shape");
  TORCH_CHECK(w2_plane2.size(1) == H, "down N");
  TORCH_CHECK(
      expert_bits.is_xpu() && expert_bits.is_contiguous() &&
          expert_bits.scalar_type() == at::kInt && expert_bits.numel() == E,
      "expert_bits [E] int32");
  check_planes(
      w13_plane2.reshape({(int64_t)E * N13, w13_plane2.size(2)}),
      bits == 2 ? w13_plane_hi
                : w13_plane_hi.reshape({(int64_t)E * N13, w13_plane_hi.size(2)}),
      w13_scale.reshape({(int64_t)E * N13, w13_scale.size(2)}),
      (int64_t)E * N13,
      H,
      (int)bits);
  check_planes(
      w2_plane2.reshape({(int64_t)E * H, w2_plane2.size(2)}),
      bits == 2 ? w2_plane_hi
                : w2_plane_hi.reshape({(int64_t)E * H, w2_plane_hi.size(2)}),
      w2_scale.reshape({(int64_t)E * H, w2_scale.size(2)}),
      (int64_t)E * H,
      N13 / 2,
      (int)bits);
  record_use(expert_bits);
  return launch_reduced_moe(
      x, ids, router_w, w13_plane2, w13_plane_hi, w13_scale, w2_plane2,
      w2_plane_hi, w2_scale, bits, /*which=*/2, expert_bits.data_ptr<int32_t>());
}

TORCH_LIBRARY(b70_lowbit, m) {
  m.def(
      "gemv(Tensor x, Tensor plane2, Tensor plane_hi, Tensor scale, int bits) "
      "-> Tensor");
  m.def(
      "moe_gemv(Tensor x, Tensor ids, Tensor router_w, Tensor w13_plane2, "
      "Tensor w13_plane_hi, Tensor w13_scale, Tensor w2_plane2, "
      "Tensor w2_plane_hi, Tensor w2_scale, int bits) -> Tensor");
  m.def(
      "prefill_wna16(Tensor x, Tensor plane2, Tensor plane_hi, Tensor scale, "
      "int bits) -> Tensor");
  m.def(
      "prefill_wxa8(Tensor x, Tensor plane2, Tensor plane_hi, Tensor scale, "
      "int bits) -> Tensor");
  m.def(
      "moe_prefill(Tensor x, Tensor ids, Tensor router_w, Tensor w13_plane2, "
      "Tensor w13_plane_hi, Tensor w13_scale, Tensor w2_plane2, "
      "Tensor w2_plane_hi, Tensor w2_scale, int bits) -> Tensor");
  m.def(
      "moe_gemv_mixed(Tensor x, Tensor ids, Tensor router_w, Tensor w13_plane2, "
      "Tensor w13_plane_hi, Tensor w13_scale, Tensor w2_plane2, "
      "Tensor w2_plane_hi, Tensor w2_scale, Tensor expert_bits, int bits) "
      "-> Tensor");
}

TORCH_LIBRARY_IMPL(b70_lowbit, XPU, m) {
  m.impl("gemv", &gemv);
  m.impl("moe_gemv", &moe_gemv);
  m.impl("prefill_wna16", &prefill_wna16);
  m.impl("prefill_wxa8", &prefill_wxa8);
  m.impl("moe_prefill", &moe_prefill);
  m.impl("moe_gemv_mixed", &moe_gemv_mixed);
}
