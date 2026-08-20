// nvfp4_m1_layerlet.cpp -- O4e: one-launch fused up GEMV + silu_and_mul + down GEMV.
//
// O4d: sidecar M=1 GEMV was 2.36x oneDNN isolated but e2e 32.2 vs hold 34.9
// because T x top_k slot loops launch 16 GEMVs. Fuse per-expert:
//   up N=1024 K=2048 -> silu_and_mul I=512 -> down N=2048 K=512
// One WG=1024, SLM holds gu/h. S=8 experts = 8 WGs, one launch.
// WG=1 GEMV path stays the proto; this is launch fusion, not occupancy.
//
// Card 1 only. Do not stop ornith_o1. No live _xpu_C swap.
// Build: bash vllm/nvfp4/proto_moe_m1/build_layerlet.sh
// Run:   ./bin/gpu-run --card 1 bash vllm/nvfp4/proto_moe_m1/run_layerlet.sh

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;

static constexpr int VL = 256;
static constexpr int GRP = 16;
static constexpr int H = 2048;
static constexpr int I = 512;
static constexpr int NUP = 2 * I;
// ESIMD on this BMG image: "work-items in a work-group cannot exceed 64".
static constexpr int WG = 64;
static constexpr int SLM_BYTES =
    NUP * (int)sizeof(float) + I * (int)sizeof(float) + I * (int)sizeof(uint16_t);
static constexpr int OFF_GU = 0;
static constexpr int OFF_H = NUP * (int)sizeof(float);
static constexpr int OFF_HB = OFF_H + I * (int)sizeof(float);

static constexpr float E2M1_LUT[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static uint16_t f32_to_bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u = (u + 0x8000u) & 0xFFFF0000u;
  return static_cast<uint16_t>(u >> 16);
}

SYCL_ESIMD_FUNCTION inline uint16_t f32_to_bf16_u16(float f) {
  esimd::simd<float, 1> sf(f);
  uint32_t bits = sf.template bit_cast_view<uint32_t>()[0];
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

static float bf16_bits_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static float decode_e2m1_u8(uint8_t nib) {
  float mag = E2M1_LUT[nib & 7];
  return (nib & 8) ? -mag : mag;
}

template <int N>
SYCL_ESIMD_FUNCTION inline esimd::simd<float, N> bf16_to_f32(
    esimd::simd<uint16_t, N> u) {
  esimd::simd<uint32_t, N> x = esimd::convert<uint32_t>(u);
  x = x << 16;
  return x.template bit_cast_view<float>();
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

SYCL_ESIMD_FUNCTION inline float hsum256(esimd::simd<float, 256> acc) {
  acc.template select<128, 1>(0) += acc.template select<128, 1>(128);
  acc.template select<64, 1>(0) += acc.template select<64, 1>(64);
  acc.template select<32, 1>(0) += acc.template select<32, 1>(32);
  acc.template select<16, 1>(0) += acc.template select<16, 1>(16);
  acc.template select<8, 1>(0) += acc.template select<8, 1>(8);
  acc.template select<4, 1>(0) += acc.template select<4, 1>(4);
  acc.template select<2, 1>(0) += acc.template select<2, 1>(2);
  return (float)acc[0] + (float)acc[1];
}

SYCL_ESIMD_FUNCTION inline float gemv_row(
    const uint16_t* x, const uint8_t* wrow, const float* srow, int K) {
  float acc = 0.f;
  for (int k = 0; k < K; k += VL) {
    constexpr int NB = VL / 2;
    constexpr int NG = VL / GRP;
    esimd::simd<uint8_t, NB> pk = esimd::block_load<uint8_t, NB>(wrow + k / 2);
    esimd::simd<uint16_t, VL> xu = esimd::block_load<uint16_t, VL>(x + k);
    esimd::simd<float, VL> xf = bf16_to_f32<VL>(xu);
    esimd::simd<uint8_t, VL> nib;
    nib.template select<NB, 2>(0) = pk & uint8_t(0xF);
    nib.template select<NB, 2>(1) = pk >> 4;
    esimd::simd<float, VL> wf = decode_e2m1<VL>(nib);
    esimd::simd<float, NG> sg;
    sg.copy_from(srow + k / GRP);
    esimd::simd<float, VL> sc;
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      sc.template select<GRP, 1>(g * GRP) = esimd::simd<float, GRP>(sg[g]);
    }
    acc += hsum256(xf * wf * sc);
  }
  return acc;
}

SYCL_ESIMD_FUNCTION inline float gemv_row_slm_x(
    const uint8_t* wrow, const float* srow, int K, int x_off) {
  float acc = 0.f;
  for (int k = 0; k < K; k += VL) {
    constexpr int NB = VL / 2;
    constexpr int NG = VL / GRP;
    esimd::simd<uint8_t, NB> pk = esimd::block_load<uint8_t, NB>(wrow + k / 2);
    esimd::simd<uint16_t, VL> xu =
        esimd::slm_block_load<uint16_t, VL>(x_off + k * (int)sizeof(uint16_t));
    esimd::simd<float, VL> xf = bf16_to_f32<VL>(xu);
    esimd::simd<uint8_t, VL> nib;
    nib.template select<NB, 2>(0) = pk & uint8_t(0xF);
    nib.template select<NB, 2>(1) = pk >> 4;
    esimd::simd<float, VL> wf = decode_e2m1<VL>(nib);
    esimd::simd<float, NG> sg;
    sg.copy_from(srow + k / GRP);
    esimd::simd<float, VL> sc;
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      sc.template select<GRP, 1>(g * GRP) = esimd::simd<float, GRP>(sg[g]);
    }
    acc += hsum256(xf * wf * sc);
  }
  return acc;
}

struct Layerlet {
  const uint16_t* x;
  const uint8_t* w1;
  const float* s1;
  const uint8_t* w2;
  const float* s2;
  float* y;
  int S;
  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<SLM_BYTES>();
    const int s = (int)it.get_global_id(0);
    const int lid = (int)it.get_local_id(1);
    if (s >= S) return;

    const uint8_t* w1s = w1 + (size_t)s * NUP * (H / 2);
    const float* s1s = s1 + (size_t)s * NUP * (H / GRP);
    const uint8_t* w2s = w2 + (size_t)s * H * (I / 2);
    const float* s2s = s2 + (size_t)s * H * (I / GRP);

    for (int t = 0; t < NUP / WG; ++t) {
      int n = lid + t * WG;
      float acc = gemv_row(x, w1s + (size_t)n * (H / 2),
                           s1s + (size_t)n * (H / GRP), H);
      esimd::simd<float, 1> v(acc);
      esimd::slm_block_store<float, 1>(OFF_GU + n * (int)sizeof(float), v);
    }
    esimd::barrier();

    for (int t = 0; t < I / WG; ++t) {
      int n = lid + t * WG;
      esimd::simd<float, 1> g =
          esimd::slm_block_load<float, 1>(OFF_GU + n * (int)sizeof(float));
      esimd::simd<float, 1> u = esimd::slm_block_load<float, 1>(
          OFF_GU + (I + n) * (int)sizeof(float));
      float gv = (float)g[0];
      float uv = (float)u[0];
      float hv = (gv / (1.f + sycl::exp(-gv))) * uv;
      esimd::simd<float, 1> hs(hv);
      esimd::slm_block_store<float, 1>(OFF_H + n * (int)sizeof(float), hs);
      uint16_t hb = f32_to_bf16_u16(hv);
      esimd::simd<uint16_t, 1> hbs(hb);
      esimd::slm_block_store<uint16_t, 1>(OFF_HB + n * (int)sizeof(uint16_t),
                                          hbs);
    }
    esimd::barrier();

    for (int t = 0; t < H / WG; ++t) {
      int n = lid + t * WG;
      float acc = gemv_row_slm_x(w2s + (size_t)n * (I / 2),
                                 s2s + (size_t)n * (I / GRP), I, OFF_HB);
      y[(size_t)s * H + n] = acc;
    }
  }
};

struct Gemv1D {
  const uint16_t* x;
  const uint8_t* w;
  const float* scale;
  float* y;
  int N;
  int K;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    const int n = (int)it.get_global_id(0);
    if (n >= N) return;
    y[n] = gemv_row(x, w + (size_t)n * (K / 2), scale + (size_t)n * (K / GRP),
                    K);
  }
};

static void cpu_gemv(const std::vector<float>& x, const std::vector<uint8_t>& w,
                     const std::vector<float>& scale, std::vector<float>& y,
                     int N, int K) {
  y.assign(N, 0.f);
  for (int n = 0; n < N; ++n) {
    double acc = 0.0;
    const uint8_t* wrow = w.data() + (size_t)n * (K / 2);
    const float* srow = scale.data() + (size_t)n * (K / GRP);
    for (int k = 0; k < K; ++k) {
      uint8_t packed = wrow[k / 2];
      uint8_t nib = (k & 1) ? (packed >> 4) : (packed & 0xF);
      acc += (double)x[k] * (double)decode_e2m1_u8(nib) * (double)srow[k / GRP];
    }
    y[n] = (float)acc;
  }
}

static void fill_problem(std::mt19937& rng, int N, int K,
                         std::vector<float>& x_f, std::vector<uint16_t>& x_b,
                         std::vector<uint8_t>& w, std::vector<float>& scale) {
  std::normal_distribution<float> dx(0.f, 0.1f);
  std::uniform_real_distribution<float> ds(0.02f, 0.4f);
  std::uniform_int_distribution<int> dw(0, 255);
  x_f.resize(K);
  x_b.resize(K);
  for (int k = 0; k < K; ++k) {
    float f = dx(rng);
    x_b[k] = f32_to_bf16_bits(f);
    x_f[k] = bf16_bits_to_f32(x_b[k]);
  }
  w.resize((size_t)N * (K / 2));
  for (auto& v : w) v = (uint8_t)dw(rng);
  scale.resize((size_t)N * (K / GRP));
  for (auto& v : scale) v = ds(rng);
}

struct Stats {
  double max_abs = 0;
  double max_rel = 0;
  double ref_abs = 0;
};

static Stats compare(const std::vector<float>& got,
                     const std::vector<float>& ref) {
  Stats s;
  for (size_t i = 0; i < ref.size(); ++i)
    s.ref_abs = std::max(s.ref_abs, (double)std::fabs(ref[i]));
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = std::fabs((double)got[i] - (double)ref[i]);
    s.max_abs = std::max(s.max_abs, d);
    s.max_rel = std::max(s.max_rel, d / (s.ref_abs + 1e-30));
  }
  return s;
}

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  auto max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("max_wg=%zu SLM_BYTES=%d WG=%d H=%d I=%d NUP=%d\n", (size_t)max_wg,
              SLM_BYTES, WG, H, I, NUP);
  if ((size_t)max_wg < (size_t)WG) {
    std::printf("FAIL max_wg < %d\n", WG);
    return 2;
  }

  const int S = 8;
  const int warm = 8;
  const int iters = 30;
  std::mt19937 rng(99);
  std::vector<float> x_f, s1, s2;
  std::vector<uint16_t> x_b;
  std::vector<uint8_t> w1, w2;
  fill_problem(rng, S * NUP, H, x_f, x_b, w1, s1);
  std::vector<float> dummy_x;
  std::vector<uint16_t> dummy_xb;
  fill_problem(rng, S * H, I, dummy_x, dummy_xb, w2, s2);

  std::vector<float> y_ref((size_t)S * H, 0.f);
  for (int s = 0; s < S; ++s) {
    std::vector<uint8_t> w1s(w1.begin() + (size_t)s * NUP * (H / 2),
                             w1.begin() + (size_t)(s + 1) * NUP * (H / 2));
    std::vector<float> s1s(s1.begin() + (size_t)s * NUP * (H / GRP),
                           s1.begin() + (size_t)(s + 1) * NUP * (H / GRP));
    std::vector<float> gu, h, dn;
    cpu_gemv(x_f, w1s, s1s, gu, NUP, H);
    h.resize(I);
    for (int i = 0; i < I; ++i) {
      float g = gu[i];
      float u = gu[I + i];
      h[i] = (g / (1.f + std::exp(-g))) * u;
    }
    std::vector<uint8_t> w2s(w2.begin() + (size_t)s * H * (I / 2),
                             w2.begin() + (size_t)(s + 1) * H * (I / 2));
    std::vector<float> s2s(s2.begin() + (size_t)s * H * (I / GRP),
                           s2.begin() + (size_t)(s + 1) * H * (I / GRP));
    cpu_gemv(h, w2s, s2s, dn, H, I);
    for (int n = 0; n < H; ++n) y_ref[(size_t)s * H + n] = dn[n];
  }

  uint16_t* dX = sycl::aligned_alloc_device<uint16_t>(256, H, q);
  uint8_t* dW1 = sycl::aligned_alloc_device<uint8_t>(256, w1.size(), q);
  uint8_t* dW2 = sycl::aligned_alloc_device<uint8_t>(256, w2.size(), q);
  float* dS1 = sycl::aligned_alloc_device<float>(256, s1.size(), q);
  float* dS2 = sycl::aligned_alloc_device<float>(256, s2.size(), q);
  float* dY = sycl::aligned_alloc_device<float>(256, (size_t)S * H, q);
  uint16_t* dHb = sycl::aligned_alloc_device<uint16_t>(256, I, q);
  float* dGU = sycl::aligned_alloc_device<float>(256, NUP, q);
  q.copy(x_b.data(), dX, H);
  q.copy(w1.data(), dW1, w1.size());
  q.copy(w2.data(), dW2, w2.size());
  q.copy(s1.data(), dS1, s1.size());
  q.copy(s2.data(), dS2, s2.size()).wait();

  Layerlet kl{dX, dW1, dS1, dW2, dS2, dY, S};
  auto nd = sycl::nd_range<2>{sycl::range<2>((size_t)S, (size_t)WG),
                              sycl::range<2>(1, (size_t)WG)};
  q.memset(dY, 0, sizeof(float) * (size_t)S * H).wait();
  q.parallel_for(nd, kl).wait();
  std::vector<float> y_got((size_t)S * H);
  q.copy(dY, y_got.data(), (size_t)S * H).wait();
  Stats st = compare(y_got, y_ref);
  std::printf("layerlet numerics  max_abs=%.3e max_rel=%.3e ref_abs=%.4g\n",
              st.max_abs, st.max_rel, st.ref_abs);

  for (int i = 0; i < warm; ++i) q.parallel_for(nd, kl);
  q.wait();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) q.parallel_for(nd, kl);
  q.wait();
  auto t1 = std::chrono::steady_clock::now();
  double ms_f =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  std::printf("layerlet fused S=%d  %.3f ms  (1 launch)\n", S, ms_f);

  auto seq = [&]() {
    for (int s = 0; s < S; ++s) {
      Gemv1D kup{dX, dW1 + (size_t)s * NUP * (H / 2),
                 dS1 + (size_t)s * NUP * (H / GRP), dGU, NUP, H};
      q.parallel_for(sycl::nd_range<1>{sycl::range<1>((size_t)NUP),
                                       sycl::range<1>(1)},
                     kup);
      q.submit([&](sycl::handler& hnd) {
        hnd.parallel_for(sycl::range<1>((size_t)I), [=](sycl::id<1> id) {
          int i = (int)id[0];
          float g = dGU[i];
          float u = dGU[I + i];
          float v = (g / (1.f + sycl::exp(-g))) * u;
          uint32_t bits;
          std::memcpy(&bits, &v, 4);
          bits = (bits + 0x8000u) & 0xFFFF0000u;
          dHb[i] = (uint16_t)(bits >> 16);
        });
      });
      Gemv1D kdn{dHb, dW2 + (size_t)s * H * (I / 2),
                 dS2 + (size_t)s * H * (I / GRP), dY + (size_t)s * H, H, I};
      q.parallel_for(
          sycl::nd_range<1>{sycl::range<1>((size_t)H), sycl::range<1>(1)}, kdn);
    }
  };
  for (int i = 0; i < warm; ++i) seq();
  q.wait();
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) seq();
  q.wait();
  t1 = std::chrono::steady_clock::now();
  double ms_s =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  std::printf("seq 8x(up+silu+down)  %.3f ms  (24 launches)\n", ms_s);
  std::printf("speedup fused/seq = %.3fx\n", ms_s / ms_f);

  sycl::free(dX, q);
  sycl::free(dW1, q);
  sycl::free(dW2, q);
  sycl::free(dS1, q);
  sycl::free(dS2, q);
  sycl::free(dY, q);
  sycl::free(dHb, q);
  sycl::free(dGU, q);
  if (st.max_rel > 5e-3) {
    std::printf("FAIL layerlet\n");
    return 1;
  }
  std::printf("PASS layerlet\n");
  return 0;
}
