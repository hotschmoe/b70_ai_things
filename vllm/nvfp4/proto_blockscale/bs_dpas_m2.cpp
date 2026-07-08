// bs_dpas_m2.cpp -- MILESTONE 2: block-scaled s8xs8 DPAS + IN-REGISTER f4_e2m1 decode.
//
// Same block-scaled GEMM tile as M1, but the WEIGHT is now stored 4-bit-resident
// (E2M1 nibbles, VNNI-packed) and decoded to s8 IN-REGISTER on the DPAS load path:
//   nibble = sign(bit3) | mag_idx(bits0-2);  s8 = (+/-) E2M1x2[mag_idx]
// via the arithmetic decode  mag = (exp==0 ? mant : (2+mant) << (exp-1)),
// exp=idx>>1, mant=idx&1  (== {0,1,2,3,4,6,8,12}). Low-nibble-first.
//
// Weight read is 4 bits/code (uint16 holds 4 nibbles = 4 K for one N). Re-validates
// BIT-EXACT vs the SAME matched-order CPU reference as M1 (decoded s8 must reproduce
// the original E2M1*2 codes exactly, so the GEMM result is identical to M1).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

static constexpr int SDEPTH = 8;
static constexpr int M      = 8;
static constexpr int N      = 16;
static constexpr int DPK    = 32;
static constexpr int GRP    = 16;
static constexpr int EPD    = 4;

#ifndef KDIM
#define KDIM 5120
#endif
static constexpr int K    = KDIM;
static constexpr int NBLK = K / GRP;

static constexpr int A_INTS = (M * DPK) / EPD;   // 64
static constexpr int B_INTS = (DPK * N) / EPD;   // 128
static constexpr int C_INTS = M * N;             // 128
static constexpr int W4_PER_BLK = (GRP / 4) * N; // 4 kg * 16 n = 64 uint16 (real half of B)

static const int E2M1x2[8] = {0, 1, 2, 3, 4, 6, 8, 12};

static void pack_A_block(int32_t* dst, const int8_t* Ablk) {
  const int kd = DPK / EPD;
  for (int i = 0; i < A_INTS; ++i) dst[i] = 0;
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < GRP; ++k) {
      int idx = m * kd + (k / EPD);
      int sh  = (k % EPD) * 8;
      dst[idx] |= (int32_t)((uint32_t)((uint8_t)Ablk[m * GRP + k]) << sh);
    }
}
static uint16_t encode_nib(int8_t s8) {
  int mag = std::abs((int)s8), idx = 0;
  for (int i = 0; i < 8; ++i) if (E2M1x2[i] == mag) { idx = i; break; }
  int sign = (s8 < 0) ? 8 : 0;
  return (uint16_t)(sign | idx);
}
static void pack_W4_block(uint16_t* dst, const int8_t* Bblk /*GRP x N s8*/) {
  for (int kg = 0; kg < GRP / 4; ++kg)
    for (int n = 0; n < N; ++n) {
      uint16_t v = 0;
      for (int j = 0; j < 4; ++j) {
        int k = kg * 4 + j;
        v |= (uint16_t)(encode_nib(Bblk[k * N + n]) << (j * 4));
      }
      dst[kg * N + n] = v;
    }
}

template <int W>
static ESIMD_INLINE esimd::simd<int32_t, W> decode_e2m1x2(esimd::simd<int32_t, W> nib) {
  esimd::simd<int32_t, W> sign = nib & 0x8;
  esimd::simd<int32_t, W> idx  = nib & 0x7;
  esimd::simd<int32_t, W> exp  = idx >> 1;
  esimd::simd<int32_t, W> mant = idx & 0x1;
  esimd::simd<int32_t, W> sh   = esimd::max(exp - 1, esimd::simd<int32_t, W>(0));
  esimd::simd<int32_t, W> mag  = (mant + 2) << sh;   // exp>=1 branch
  mag.merge(mant, exp == 0);                          // exp==0 -> subnormal = mant
  esimd::simd<int32_t, W> neg = -mag;
  mag.merge(neg, sign != 0);
  return mag;
}

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("M=%d N=%d K=%d NBLK=%d  (weight 4-bit resident, in-register decode)\n", M, N, K, NBLK);

  std::mt19937 rng(12345);
  std::vector<int8_t> w_s8((size_t)K * N);
  std::uniform_int_distribution<int> dcode(0, 7), dsign(0, 1);
  for (size_t i = 0; i < w_s8.size(); ++i) {
    int mag = E2M1x2[dcode(rng)];
    w_s8[i] = (int8_t)((dsign(rng) && mag) ? -mag : mag);
  }
  auto to_bf16 = [](float f) {
    uint32_t u; std::memcpy(&u, &f, 4); u = (u + 0x8000u) & 0xFFFF0000u;
    float r; std::memcpy(&r, &u, 4); return r;
  };
  std::vector<float> g((size_t)NBLK * N);
  std::uniform_real_distribution<float> dg(0.01f, 0.5f);
  for (auto& v : g) v = to_bf16(dg(rng));
  std::vector<float> x((size_t)M * K);
  std::normal_distribution<float> dx(0.f, 0.1f);
  for (auto& v : x) v = dx(rng);
  std::vector<float> act_scale(M);
  std::vector<int8_t> a_s8((size_t)M * K);
  for (int m = 0; m < M; ++m) {
    float amax = 1e-8f;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(x[m * K + k]));
    float s = amax / 127.0f; act_scale[m] = s;
    for (int k = 0; k < K; ++k) {
      int qv = (int)std::lround(x[m * K + k] / s);
      a_s8[m * K + k] = (int8_t)std::max(-127, std::min(127, qv));
    }
  }

  std::vector<float>  ref_f32((size_t)M * N);
  std::vector<double> ref_f64((size_t)M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float fa = 0.f; double da = 0.0;
      for (int b = 0; b < NBLK; ++b) {
        int32_t ip = 0;
        for (int kk = 0; kk < GRP; ++kk) {
          int k = b * GRP + kk;
          ip += (int32_t)a_s8[m * K + k] * (int32_t)w_s8[(size_t)k * N + n];
        }
        fa += g[(size_t)b * N + n] * (float)ip;
        da += (double)g[(size_t)b * N + n] * (double)ip;
      }
      ref_f32[m * N + n] = act_scale[m] * fa;
      ref_f64[m * N + n] = (double)act_scale[m] * da;
    }

  std::vector<int32_t>  Apk((size_t)NBLK * A_INTS);
  std::vector<uint16_t> W4((size_t)NBLK * W4_PER_BLK);
  std::vector<float>    Gpk((size_t)NBLK * N);
  std::vector<int8_t>   Ablk(M * GRP), Bblk(GRP * N);
  for (int b = 0; b < NBLK; ++b) {
    for (int m = 0; m < M; ++m)
      for (int kk = 0; kk < GRP; ++kk)
        Ablk[m * GRP + kk] = a_s8[m * K + b * GRP + kk];
    for (int kk = 0; kk < GRP; ++kk)
      for (int n = 0; n < N; ++n)
        Bblk[kk * N + n] = w_s8[(size_t)(b * GRP + kk) * N + n];
    pack_A_block(&Apk[(size_t)b * A_INTS], Ablk.data());
    pack_W4_block(&W4[(size_t)b * W4_PER_BLK], Bblk.data());
    for (int n = 0; n < N; ++n) Gpk[(size_t)b * N + n] = g[(size_t)b * N + n];
  }

  int32_t*  dA = sycl::malloc_device<int32_t>(Apk.size(), q);
  uint16_t* dW = sycl::malloc_device<uint16_t>(W4.size(), q);
  float*    dG = sycl::malloc_device<float>(Gpk.size(), q);
  float*    dS = sycl::malloc_device<float>(M, q);
  float*    dY = sycl::malloc_device<float>(C_INTS, q);
  q.copy(Apk.data(), dA, Apk.size());
  q.copy(W4.data(), dW, W4.size());
  q.copy(Gpk.data(), dG, Gpk.size());
  q.copy(act_scale.data(), dS, M).wait();

  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
       [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
         esimd::simd<float, C_INTS> acc = 0.0f;
         for (int b = 0; b < NBLK; ++b) {
           esimd::simd<int32_t, A_INTS> a; a.copy_from(dA + (size_t)b * A_INTS);
           // ---- 4-bit weight load + in-register E2M1 decode -> s8 VNNI B tile ----
           esimd::simd<uint16_t, W4_PER_BLK> w4;
           w4.copy_from(dW + (size_t)b * W4_PER_BLK);        // 64 uint16 = 4 bit/code
           esimd::simd<int32_t, W4_PER_BLK> w4i = w4;        // widen to int lanes
           esimd::simd<int32_t, W4_PER_BLK> s0 = decode_e2m1x2<W4_PER_BLK>( w4i        & 0xF);
           esimd::simd<int32_t, W4_PER_BLK> s1 = decode_e2m1x2<W4_PER_BLK>((w4i >> 4)  & 0xF);
           esimd::simd<int32_t, W4_PER_BLK> s2 = decode_e2m1x2<W4_PER_BLK>((w4i >> 8)  & 0xF);
           esimd::simd<int32_t, W4_PER_BLK> s3 = decode_e2m1x2<W4_PER_BLK>((w4i >> 12) & 0xF);
           esimd::simd<int32_t, W4_PER_BLK> dw =
               (s0 & 0xFF) | ((s1 & 0xFF) << 8) | ((s2 & 0xFF) << 16) | ((s3 & 0xFF) << 24);
           esimd::simd<int32_t, B_INTS> bb = 0;              // upper 16 K stay zero
           bb.select<W4_PER_BLK, 1>(0) = dw;                 // real 16 K = dword 0..63
           esimd::simd<int32_t, C_INTS> c = 0;
           esimd::simd<int32_t, C_INTS> r =
               xmx::dpas<SDEPTH, M, int32_t, int32_t, int32_t, int32_t,
                         xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(c, bb, a);
           esimd::simd<float, N> gvec; gvec.copy_from(dG + (size_t)b * N);
           esimd::simd<float, C_INTS> grow = gvec.replicate<M>();
           acc += grow * r;
         }
         esimd::simd<float, M> as; as.copy_from(dS);
         for (int m = 0; m < M; ++m) { float sm = as[m]; acc.select<N, 1>(m * N) *= sm; }
         acc.copy_to(dY);
       });
   }).wait();

  std::vector<float> Y(C_INTS);
  q.copy(dY, Y.data(), C_INTS).wait();

  double max_abs_f32 = 0, max_rel_f64 = 0, ref_absmax = 0;
  int exact_f32 = 0;
  for (int i = 0; i < C_INTS; ++i) {
    ref_absmax = std::max(ref_absmax, std::fabs(ref_f64[i]));
    max_abs_f32 = std::max(max_abs_f32, std::fabs((double)Y[i] - (double)ref_f32[i]));
    if (Y[i] == ref_f32[i]) ++exact_f32;
  }
  for (int i = 0; i < C_INTS; ++i)
    max_rel_f64 = std::max(max_rel_f64, std::fabs((double)Y[i] - ref_f64[i]) / (ref_absmax + 1e-30));
  std::printf("bit-exact vs matched-order fp32 CPU ref: %d / %d elems identical\n", exact_f32, C_INTS);
  std::printf("max |gpu - fp32ref| = %.3e\n", max_abs_f32);
  std::printf("max rel-err vs fp64 truth = %.3e   (ref_absmax=%.4g)\n", max_rel_f64, ref_absmax);
  std::printf("sample: gpu[0]=%.6f ref[0]=%.6f  gpu[1]=%.6f ref[1]=%.6f\n",
              Y[0], ref_f32[0], Y[1], ref_f32[1]);
  bool pass = (exact_f32 == C_INTS) && (max_rel_f64 < 1e-3);
  std::printf("%s: block-scaled s8 DPAS + in-register f4 decode\n", pass ? "PASS" : "FAIL");

  sycl::free(dA, q); sycl::free(dW, q); sycl::free(dG, q); sycl::free(dS, q); sycl::free(dY, q);
  return pass ? 0 : 1;
}
