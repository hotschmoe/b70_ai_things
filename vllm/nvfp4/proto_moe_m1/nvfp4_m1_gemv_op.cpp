// nvfp4_m1_gemv_op.cpp -- O4c torch XPU op for LOOP 12 WG=1 1D NVFP4 GEMV.
// Separate .so. Do not replace the live serve _xpu_C. Loads via
// torch.ops.load_library + env B70_NVFP4_M1_SO.
//
// y[1,N] = gemv(x[1,K] bf16, w[N,K/2] uint8 packed E2M1, scale[K/16,N] bf16)
// K % 256 == 0. Uses c10 current XPU stream (GRAPH-safe: no queue.wait).

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <torch/library.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstring>

namespace esimd = sycl::ext::intel::esimd;

static constexpr int VL = 256;
static constexpr int GRP = 16;

template <int N>
SYCL_ESIMD_FUNCTION inline esimd::simd<float, N> bf16_to_f32(
    esimd::simd<uint16_t, N> u) {
  esimd::simd<uint32_t, N> x = esimd::convert<uint32_t>(u);
  x = x << 16;
  return x.template bit_cast_view<float>();
}

SYCL_ESIMD_FUNCTION inline float bf16_u16_to_f32(uint16_t b) {
  esimd::simd<uint16_t, 1> u(b);
  return bf16_to_f32<1>(u)[0];
}

SYCL_ESIMD_FUNCTION inline uint16_t f32_to_bf16_u16(float f) {
  esimd::simd<float, 1> sf(f);
  uint32_t bits = sf.template bit_cast_view<uint32_t>()[0];
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

template <int N>
SYCL_ESIMD_FUNCTION inline esimd::simd<float, N> decode_e2m1(
    esimd::simd<uint8_t, N> nib) {
  esimd::simd<int32_t, N> idx = esimd::convert<int32_t>(nib);
  esimd::simd<int32_t, N> sign = idx & 8;
  idx = idx & 7;
  esimd::simd<int32_t, N> exp = idx >> 1;
  esimd::simd<int32_t, N> mant = idx & 1;
  esimd::simd<int32_t, N> sh =
      esimd::max(exp - 1, esimd::simd<int32_t, N>(0));
  esimd::simd<int32_t, N> magi = (mant + 2) << sh;
  magi.merge(mant, exp == 0);
  esimd::simd<float, N> mag = esimd::convert<float>(magi) * 0.5f;
  esimd::simd<float, N> neg = -mag;
  mag.merge(neg, sign != 0);
  return mag;
}

template <int N>
SYCL_ESIMD_FUNCTION inline float hsum(esimd::simd<float, N> acc) {
  static_assert(N == 256, "tree reduce is for VL=256");
  acc.template select<128, 1>(0) += acc.template select<128, 1>(128);
  acc.template select<64, 1>(0) += acc.template select<64, 1>(64);
  acc.template select<32, 1>(0) += acc.template select<32, 1>(32);
  acc.template select<16, 1>(0) += acc.template select<16, 1>(16);
  acc.template select<8, 1>(0) += acc.template select<8, 1>(8);
  acc.template select<4, 1>(0) += acc.template select<4, 1>(4);
  acc.template select<2, 1>(0) += acc.template select<2, 1>(2);
  return (float)acc[0] + (float)acc[1];
}

struct Gemv1DNt {
  const uint16_t* x;
  const uint8_t* w;
  const uint16_t* scale;
  uint16_t* y;
  int N;
  int K;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    const int n = (int)it.get_global_id(0);
    if (n >= N) return;
    const uint8_t* wrow = w + (size_t)n * (K / 2);
    float acc = 0.f;
    for (int k = 0; k < K; k += VL) {
      constexpr int NB = VL / 2;
      constexpr int NG = VL / GRP;
      esimd::simd<uint8_t, NB> pk =
          esimd::block_load<uint8_t, NB>(wrow + k / 2);
      esimd::simd<uint16_t, VL> xu =
          esimd::block_load<uint16_t, VL>(x + k);
      esimd::simd<float, VL> xf = bf16_to_f32<VL>(xu);
      esimd::simd<uint8_t, VL> nib;
      nib.template select<NB, 2>(0) = pk & uint8_t(0xF);
      nib.template select<NB, 2>(1) = pk >> 4;
      esimd::simd<float, VL> wf = decode_e2m1<VL>(nib);
      esimd::simd<float, VL> sc;
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        uint16_t sb = scale[(size_t)(k / GRP + g) * (size_t)N + (size_t)n];
        sc.template select<GRP, 1>(g * GRP) =
            esimd::simd<float, GRP>(bf16_u16_to_f32(sb));
      }
      acc += hsum<VL>(xf * wf * sc);
    }
    y[n] = f32_to_bf16_u16(acc);
  }
};

static torch::Tensor nvfp4_m1_gemv(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& scale) {
  TORCH_CHECK(x.is_xpu() && w.is_xpu() && scale.is_xpu(), "m1_gemv: XPU");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x bf16");
  TORCH_CHECK(w.scalar_type() == at::ScalarType::Byte, "w uint8");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::BFloat16, "scale bf16");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x [1,K]");
  TORCH_CHECK(w.dim() == 2, "w [N,K/2]");
  TORCH_CHECK(scale.dim() == 2, "scale [K/16,N]");
  const int64_t K = x.size(1);
  const int64_t N = w.size(0);
  TORCH_CHECK(K > 0 && (K % VL) == 0, "K multiple of 256");
  TORCH_CHECK(w.size(1) == K / 2, "w K/2");
  TORCH_CHECK(scale.size(0) == K / GRP && scale.size(1) == N, "scale NT");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && scale.is_contiguous());

  const at::DeviceGuard guard(x.device());
  auto y = at::empty({1, N}, x.options());
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  Gemv1DNt k{
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      w.data_ptr<uint8_t>(),
      reinterpret_cast<const uint16_t*>(scale.data_ptr()),
      reinterpret_cast<uint16_t*>(y.data_ptr()),
      (int)N,
      (int)K};
  q.parallel_for(
      sycl::nd_range<1>{sycl::range<1>((size_t)N), sycl::range<1>(1)}, k);
  return y;
}

TORCH_LIBRARY(b70_nvfp4_m1, m) {
  m.def("gemv(Tensor x, Tensor w, Tensor scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(b70_nvfp4_m1, XPU, m) {
  m.impl("gemv", &nvfp4_m1_gemv);
}
