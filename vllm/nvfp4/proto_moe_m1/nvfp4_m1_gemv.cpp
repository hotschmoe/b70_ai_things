// nvfp4_m1_gemv.cpp -- O4 proto: M=1 NVFP4 GEMV with 1D block_load along K.
//
// llm-scaler #491 (moe_decode_gemv.h): DPAS lsc_load_2d<uint8_t,16,16,1> is
// 16B-wide and fills 1/4 of a BMG 64B cacheline (~316 GB/s). 1D
// block_load<uint8_t,VL> along K restored ~528 GB/s. Python sticky/M1 (L63/L64)
// was 33.1/33.3 vs hold 34.9 -- extra host copy, not this load.
//
// This is NVFP4, not their FP8: packed E2M1 (2 nibbles/byte, low first) plus
// per-16-K group scale. Activations are bf16. No DPAS (group=16 vs s8 K=32
// is the proto_blockscale dead-end). One work-item = one output N.
//
// Ornith-1.5 35B-A3B fused expert: H=2048 I=512 top_k=8.
//   up   N=1024 K=2048  (w13 row-major [2I, H/2] packed)
//   down N=2048 K=512
// Scale proto layout is [N, K/16] fp32 (contiguous per output). Serve NT is
// [K/16, N]; a later torch op can gather or stride-load.
//
// Build: bash vllm/nvfp4/proto_moe_m1/build.sh
// Run on card 1 (leave overnight serve on card 0):
//   ./bin/gpu-run --card 1 bash vllm/nvfp4/proto_moe_m1/run.sh

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;

#ifndef VL_K
#define VL_K 256
#endif

static constexpr int GRP = 16;
static constexpr float E2M1_LUT[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static uint16_t f32_to_bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u = (u + 0x8000u) & 0xFFFF0000u;
  return static_cast<uint16_t>(u >> 16);
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

template <int VL>
SYCL_ESIMD_FUNCTION inline float hsum(esimd::simd<float, VL> acc) {
  static_assert(VL == 256, "tree reduce is for VL=256");
  acc.template select<128, 1>(0) += acc.template select<128, 1>(128);
  acc.template select<64, 1>(0) += acc.template select<64, 1>(64);
  acc.template select<32, 1>(0) += acc.template select<32, 1>(32);
  acc.template select<16, 1>(0) += acc.template select<16, 1>(16);
  acc.template select<8, 1>(0) += acc.template select<8, 1>(8);
  acc.template select<4, 1>(0) += acc.template select<4, 1>(4);
  acc.template select<2, 1>(0) += acc.template select<2, 1>(2);
  return (float)acc[0] + (float)acc[1];
}

enum class LoadKind { Block1D, CopyFrom };

template <int VL, LoadKind LK>
SYCL_ESIMD_FUNCTION inline void load_chunk(
    const uint8_t* wrow,
    const uint16_t* x,
    const float* srow,
    int k,
    esimd::simd<float, VL>& xf,
    esimd::simd<float, VL>& wf,
    esimd::simd<float, VL>& sc) {
  constexpr int NB = VL / 2;
  constexpr int NG = VL / GRP;
  esimd::simd<uint8_t, NB> pk;
  esimd::simd<uint16_t, VL> xu;
  if constexpr (LK == LoadKind::Block1D) {
    pk = esimd::block_load<uint8_t, NB>(wrow + k / 2);
    xu = esimd::block_load<uint16_t, VL>(x + k);
  } else {
    pk.copy_from(wrow + k / 2);
    xu.copy_from(x + k);
  }
  xf = bf16_to_f32<VL>(xu);
  esimd::simd<uint8_t, VL> nib;
  nib.template select<NB, 2>(0) = pk & uint8_t(0xF);
  nib.template select<NB, 2>(1) = pk >> 4;
  wf = decode_e2m1<VL>(nib);
  esimd::simd<float, NG> sg;
  sg.copy_from(srow + k / GRP);
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    sc.template select<GRP, 1>(g * GRP) = esimd::simd<float, GRP>(sg[g]);
  }
}

template <int VL, LoadKind LK>
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
    const uint8_t* wrow = w + (size_t)n * (K / 2);
    const float* srow = scale + (size_t)n * (K / GRP);
    float acc = 0.f;
    for (int k = 0; k < K; k += VL) {
      esimd::simd<float, VL> xf, wf, sc;
      load_chunk<VL, LK>(wrow, x, srow, k, xf, wf, sc);
      acc += hsum<VL>(xf * wf * sc);
    }
    y[n] = acc;
  }
};

template <int VL, LoadKind LK>
struct GroupedGemv1D {
  const uint16_t* x;
  const uint8_t* w;
  const float* scale;
  float* y;
  int S;
  int N;
  int K;
  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    const int s = (int)it.get_global_id(0);
    const int n = (int)it.get_global_id(1);
    if (s >= S || n >= N) return;
    const size_t row = ((size_t)s * N + n);
    const uint8_t* wrow = w + row * (K / 2);
    const float* srow = scale + row * (K / GRP);
    float acc = 0.f;
    for (int k = 0; k < K; k += VL) {
      esimd::simd<float, VL> xf, wf, sc;
      load_chunk<VL, LK>(wrow, x, srow, k, xf, wf, sc);
      acc += hsum<VL>(xf * wf * sc);
    }
    y[row] = acc;
  }
};

static void cpu_gemv(
    const std::vector<float>& x,
    const std::vector<uint8_t>& w,
    const std::vector<float>& scale,
    std::vector<float>& y,
    int N,
    int K) {
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

static void fill_problem(
    std::mt19937& rng,
    int N,
    int K,
    std::vector<float>& x_f,
    std::vector<uint16_t>& x_b,
    std::vector<uint8_t>& w,
    std::vector<float>& scale) {
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

static Stats compare(const std::vector<float>& got, const std::vector<float>& ref) {
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

template <typename Kernel>
static double time_ms(sycl::queue& q, Kernel k, sycl::nd_range<1> nd, int warm, int iters) {
  for (int i = 0; i < warm; ++i) q.parallel_for(nd, k);
  q.wait();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) q.parallel_for(nd, k);
  q.wait();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

template <typename Kernel>
static double time_ms2(sycl::queue& q, Kernel k, sycl::nd_range<2> nd, int warm, int iters) {
  for (int i = 0; i < warm; ++i) q.parallel_for(nd, k);
  q.wait();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) q.parallel_for(nd, k);
  q.wait();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static int run_gemv(sycl::queue& q, const char* tag, int N, int K, int warm, int iters) {
  std::printf("\n== gemv %s N=%d K=%d VL=%d ==\n", tag, N, K, VL_K);
  if (K % VL_K != 0) {
    std::printf("FAIL K not multiple of VL\n");
    return 1;
  }
  std::mt19937 rng(12345);
  std::vector<float> x_f, scale, y_ref;
  std::vector<uint16_t> x_b;
  std::vector<uint8_t> w;
  fill_problem(rng, N, K, x_f, x_b, w, scale);
  cpu_gemv(x_f, w, scale, y_ref, N, K);

  uint16_t* dX = sycl::aligned_alloc_device<uint16_t>(256, K, q);
  uint8_t* dW = sycl::aligned_alloc_device<uint8_t>(256, w.size(), q);
  float* dS = sycl::aligned_alloc_device<float>(256, scale.size(), q);
  float* dY = sycl::aligned_alloc_device<float>(256, N, q);
  q.copy(x_b.data(), dX, K);
  q.copy(w.data(), dW, w.size());
  q.copy(scale.data(), dS, scale.size()).wait();

  auto launch = [&](auto kind_tag, auto kernel, std::vector<float>& y_got, double* ms_out) {
    q.parallel_for(
         sycl::nd_range<1>{sycl::range<1>((size_t)N), sycl::range<1>(1)}, kernel)
        .wait();
    y_got.resize(N);
    q.copy(dY, y_got.data(), N).wait();
    Stats st = compare(y_got, y_ref);
    *ms_out = time_ms(
        q, kernel, sycl::nd_range<1>{sycl::range<1>((size_t)N), sycl::range<1>(1)},
        warm, iters);
    double bytes = (double)N * (K / 2) + (double)K * 2.0 + (double)N * (K / GRP) * 4.0;
    double gbs = bytes / (*ms_out * 1e-3) / 1e9;
    std::printf(
        "%s  max_abs=%.3e max_rel=%.3e ref_abs=%.4g  %.3f ms  %.1f GB/s\n",
        kind_tag, st.max_abs, st.max_rel, st.ref_abs, *ms_out, gbs);
    return st.max_rel < 2e-3 && st.max_abs < 5e-2;
  };

  std::vector<float> y1, y0;
  double ms1 = 0, ms0 = 0;
  Gemv1D<VL_K, LoadKind::Block1D> k1{dX, dW, dS, dY, N, K};
  Gemv1D<VL_K, LoadKind::CopyFrom> k0{dX, dW, dS, dY, N, K};
  bool p1 = launch("block1d", k1, y1, &ms1);
  bool p0 = launch("copyfrm", k0, y0, &ms0);
  std::printf("speedup block1d/copyfrm = %.3fx\n", ms0 / ms1);
  sycl::free(dX, q);
  sycl::free(dW, q);
  sycl::free(dS, q);
  sycl::free(dY, q);
  if (!p1 || !p0) {
    std::printf("FAIL gemv %s\n", tag);
    return 1;
  }
  std::printf("PASS gemv %s\n", tag);
  return 0;
}

static int run_grouped(sycl::queue& q, int S, int N, int K, int warm, int iters) {
  std::printf("\n== grouped S=%d N=%d K=%d ==\n", S, N, K);
  std::mt19937 rng(7);
  std::vector<float> x_f, scale, y_ref;
  std::vector<uint16_t> x_b;
  std::vector<uint8_t> w;
  fill_problem(rng, S * N, K, x_f, x_b, w, scale);
  cpu_gemv(x_f, w, scale, y_ref, S * N, K);

  uint16_t* dX = sycl::aligned_alloc_device<uint16_t>(256, K, q);
  uint8_t* dW = sycl::aligned_alloc_device<uint8_t>(256, w.size(), q);
  float* dS = sycl::aligned_alloc_device<float>(256, scale.size(), q);
  float* dY = sycl::aligned_alloc_device<float>(256, S * N, q);
  q.copy(x_b.data(), dX, K);
  q.copy(w.data(), dW, w.size());
  q.copy(scale.data(), dS, scale.size()).wait();

  GroupedGemv1D<VL_K, LoadKind::Block1D> k{
      dX, dW, dS, dY, S, N, K};
  auto nd = sycl::nd_range<2>{
      sycl::range<2>((size_t)S, (size_t)N), sycl::range<2>(1, 1)};
  q.parallel_for(nd, k).wait();
  std::vector<float> y_got(S * N);
  q.copy(dY, y_got.data(), S * N).wait();
  Stats st = compare(y_got, y_ref);
  double ms = time_ms2(q, k, nd, warm, iters);
  double bytes =
      (double)S * N * (K / 2) + (double)K * 2.0 + (double)S * N * (K / GRP) * 4.0;
  std::printf(
      "block1d  max_abs=%.3e max_rel=%.3e  %.3f ms  %.1f GB/s\n",
      st.max_abs, st.max_rel, ms, bytes / (ms * 1e-3) / 1e9);
  sycl::free(dX, q);
  sycl::free(dW, q);
  sycl::free(dS, q);
  sycl::free(dY, q);
  if (st.max_rel > 2e-3) {
    std::printf("FAIL grouped\n");
    return 1;
  }
  std::printf("PASS grouped\n");
  return 0;
}

static int run_apply(sycl::queue& q, int S, int H, int I, int warm, int iters) {
  std::printf("\n== apply T=1 topk=%d H=%d I=%d ==\n", S, H, I);
  const int Nup = 2 * I;
  std::mt19937 rng(99);
  std::vector<float> x_f, s1, s2, wts(S);
  std::vector<uint16_t> x_b, h_b(I);
  std::vector<uint8_t> w1, w2;
  fill_problem(rng, S * Nup, H, x_f, x_b, w1, s1);
  std::vector<float> dummy_x;
  std::vector<uint16_t> dummy_xb;
  fill_problem(rng, S * H, I, dummy_x, dummy_xb, w2, s2);
  std::uniform_real_distribution<float> dw(0.05f, 1.f);
  float wsum = 0.f;
  for (int s = 0; s < S; ++s) {
    wts[s] = dw(rng);
    wsum += wts[s];
  }
  for (int s = 0; s < S; ++s) wts[s] /= wsum;

  std::vector<float> y_ref(H, 0.f);
  std::vector<float> gu, h, dn;
  for (int s = 0; s < S; ++s) {
    std::vector<uint8_t> w1s(w1.begin() + (size_t)s * Nup * (H / 2),
                             w1.begin() + (size_t)(s + 1) * Nup * (H / 2));
    std::vector<float> s1s(s1.begin() + (size_t)s * Nup * (H / GRP),
                           s1.begin() + (size_t)(s + 1) * Nup * (H / GRP));
    cpu_gemv(x_f, w1s, s1s, gu, Nup, H);
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
    for (int n = 0; n < H; ++n) y_ref[n] += wts[s] * dn[n];
  }

  uint16_t* dX = sycl::aligned_alloc_device<uint16_t>(256, H, q);
  uint8_t* dW1 = sycl::aligned_alloc_device<uint8_t>(256, w1.size(), q);
  uint8_t* dW2 = sycl::aligned_alloc_device<uint8_t>(256, w2.size(), q);
  float* dS1 = sycl::aligned_alloc_device<float>(256, s1.size(), q);
  float* dS2 = sycl::aligned_alloc_device<float>(256, s2.size(), q);
  float* dGU = sycl::aligned_alloc_device<float>(256, S * Nup, q);
  float* dH = sycl::aligned_alloc_device<float>(256, S * I, q);
  uint16_t* dHb = sycl::aligned_alloc_device<uint16_t>(256, S * I, q);
  float* dDN = sycl::aligned_alloc_device<float>(256, S * H, q);
  float* dWts = sycl::aligned_alloc_device<float>(256, S, q);
  float* dY = sycl::aligned_alloc_device<float>(256, H, q);
  q.copy(x_b.data(), dX, H);
  q.copy(w1.data(), dW1, w1.size());
  q.copy(w2.data(), dW2, w2.size());
  q.copy(s1.data(), dS1, s1.size());
  q.copy(s2.data(), dS2, s2.size());
  q.copy(wts.data(), dWts, S).wait();

  GroupedGemv1D<VL_K, LoadKind::Block1D> kup{dX, dW1, dS1, dGU, S, Nup, H};
  auto nd_up = sycl::nd_range<2>{
      sycl::range<2>((size_t)S, (size_t)Nup), sycl::range<2>(1, 1)};
  q.parallel_for(nd_up, kup).wait();

  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::range<1>((size_t)S * I), [=](sycl::id<1> id) {
       int idx = (int)id[0];
       int s = idx / I;
       int i = idx % I;
       float g = dGU[s * Nup + i];
       float u = dGU[s * Nup + I + i];
       float v = (g / (1.f + sycl::exp(-g))) * u;
       dH[idx] = v;
       uint32_t bits;
       std::memcpy(&bits, &v, 4);
       bits = (bits + 0x8000u) & 0xFFFF0000u;
       dHb[idx] = (uint16_t)(bits >> 16);
     });
   }).wait();

  int rc_down = 0;
  {
    std::vector<float> h_host(S * I);
    q.copy(dH, h_host.data(), S * I).wait();
    std::vector<float> dn_ref;
    cpu_gemv(h_host, w2, s2, dn_ref, S * H, I);
    (void)dn_ref;
  }

  for (int s = 0; s < S; ++s) {
    Gemv1D<VL_K, LoadKind::Block1D> kdn{
        dHb + s * I, dW2 + (size_t)s * H * (I / 2),
        dS2 + (size_t)s * H * (I / GRP), dDN + s * H, H, I};
    q.parallel_for(
         sycl::nd_range<1>{sycl::range<1>((size_t)H), sycl::range<1>(1)}, kdn)
        .wait();
  }

  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::range<1>((size_t)H), [=](sycl::id<1> id) {
       int n = (int)id[0];
       float acc = 0.f;
       for (int s = 0; s < S; ++s) acc += dWts[s] * dDN[s * H + n];
       dY[n] = acc;
     });
   }).wait();

  std::vector<float> y_got(H);
  q.copy(dY, y_got.data(), H).wait();
  Stats st = compare(y_got, y_ref);

  auto once = [&]() {
    q.parallel_for(nd_up, kup);
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<1>((size_t)S * I), [=](sycl::id<1> id) {
        int idx = (int)id[0];
        int s = idx / I;
        int i = idx % I;
        float g = dGU[s * Nup + i];
        float u = dGU[s * Nup + I + i];
        float v = (g / (1.f + sycl::exp(-g))) * u;
        dH[idx] = v;
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        bits = (bits + 0x8000u) & 0xFFFF0000u;
        dHb[idx] = (uint16_t)(bits >> 16);
      });
    });
    for (int s = 0; s < S; ++s) {
      Gemv1D<VL_K, LoadKind::Block1D> kdn{
          dHb + s * I, dW2 + (size_t)s * H * (I / 2),
          dS2 + (size_t)s * H * (I / GRP), dDN + s * H, H, I};
      q.parallel_for(
          sycl::nd_range<1>{sycl::range<1>((size_t)H), sycl::range<1>(1)}, kdn);
    }
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::range<1>((size_t)H), [=](sycl::id<1> id) {
        int n = (int)id[0];
        float acc = 0.f;
        for (int s = 0; s < S; ++s) acc += dWts[s] * dDN[s * H + n];
        dY[n] = acc;
      });
    });
  };
  for (int i = 0; i < warm; ++i) once();
  q.wait();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) once();
  q.wait();
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  std::printf(
      "apply  max_abs=%.3e max_rel=%.3e  %.3f ms  (S up launches fused, %d down)\n",
      st.max_abs, st.max_rel, ms, S);

  sycl::free(dX, q);
  sycl::free(dW1, q);
  sycl::free(dW2, q);
  sycl::free(dS1, q);
  sycl::free(dS2, q);
  sycl::free(dGU, q);
  sycl::free(dH, q);
  sycl::free(dHb, q);
  sycl::free(dDN, q);
  sycl::free(dWts, q);
  sycl::free(dY, q);
  (void)rc_down;
  if (st.max_rel > 5e-3) {
    std::printf("FAIL apply\n");
    return 1;
  }
  std::printf("PASS apply\n");
  return 0;
}

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("VL_K=%d  (1D block_load NVFP4 GEMV, no DPAS)\n", VL_K);

  const int warm = 8;
  const int iters = 30;
  int rc = 0;
  rc |= run_gemv(q, "up", 1024, 2048, warm, iters);
  rc |= run_gemv(q, "down", 2048, 512, warm, iters);
  rc |= run_grouped(q, 8, 1024, 2048, warm, iters);
  rc |= run_apply(q, 8, 2048, 512, warm, iters);
  std::printf("\n%s: nvfp4 m1 1D gemv proto\n", rc ? "FAIL" : "PASS");
  return rc;
}
