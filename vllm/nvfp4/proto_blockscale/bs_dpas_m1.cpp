// bs_dpas_m1.cpp -- MILESTONE 1: block-scaled s8xs8 DPAS, one output tile.
//
// Computes one [M=8 x N=16] output tile of an NVFP4-as-int8 block-scaled GEMM:
//
//   y[m,n] = act_scale[m] * sum_{b=0}^{NBLK-1} g[b,n] * ( sum_{k in block b} a_s8[m,k] * w_s8[k,n] )
//
// where each block b is GRP=16 contiguous K, a_s8 is per-token-int8 activation,
// w_s8 is the E2M1*2 signed int8 weight code, g[b,n] is the per-16-K-group weight
// scale (bf16-rounded, = e4m3_block_scale * global_scale / 2), and act_scale[m] is
// the per-token activation scale.
//
// THE CRUX: ESIMD s8 DPAS has SystolicDepth==8 (hard static_assert) and OpsPerChannel=4
// -> reduces K=32 per instruction. NVFP4's group is 16. To rescale at group granularity
// we must reduce EXACTLY 16 K into an s32 partial before applying g[b,n]. We do this by
// issuing one K=32 DPAS per 16-group with the upper 16 K ZERO-PADDED. This wastes half
// the DPAS K-slots (the fundamental cost of block<DPAS-K on HW with no native block-scale
// accumulate). Milestone 3 measures whether the remaining win still beats the current path.
//
// Validates BIT-EXACT (integer partials) + reports fp32/fp64 rel-err vs a matched-order
// CPU reference. Build AOT for BMG-G31 (see build.sh). Run on card 0.

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

static constexpr int SDEPTH = 8;   // hardware-fixed
static constexpr int M      = 8;   // RepeatCount
static constexpr int N      = 16;  // ExecutionSize (int DPAS)
static constexpr int DPK    = 32;  // s8 DPAS K = SDEPTH*OpsPerChannel(4)
static constexpr int GRP    = 16;  // NVFP4 K-group
static constexpr int EPD    = 4;   // s8 per dword (VNNI density)

#ifndef KDIM
#define KDIM 5120                  // real gate_proj K
#endif
static constexpr int K    = KDIM;
static constexpr int NBLK = K / GRP;         // 320

// per-block packed sizes (K=32 DPAS tile, upper 16 zeroed)
static constexpr int A_INTS = (M * DPK) / EPD;   // 8*32/4 = 64
static constexpr int B_INTS = (DPK * N) / EPD;   // 32*16/4 = 128
static constexpr int C_INTS = M * N;             // 128

// E2M1*2 signed int8 code magnitudes (proven exact): {0,1,2,3,4,6,8,12}
static const int E2M1x2[8] = {0, 1, 2, 3, 4, 6, 8, 12};

// pack one [M x DPK] s8 A-block into VNNI dwords; cols>=16 are zero.
static void pack_A_block(int32_t* dst, const int8_t* Ablk /*M x GRP*/) {
  const int kd = DPK / EPD;  // 8 dwords per row
  for (int i = 0; i < A_INTS; ++i) dst[i] = 0;
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < GRP; ++k) {  // only 16 real K, rest 0
      int idx = m * kd + (k / EPD);
      int sh  = (k % EPD) * 8;
      dst[idx] |= (int32_t)((uint32_t)((uint8_t)Ablk[m * GRP + k]) << sh);
    }
}
// pack one [DPK x N] s8 B-block (VNNI); rows>=16 zero.
static void pack_B_block(int32_t* dst, const int8_t* Bblk /*GRP x N*/) {
  for (int i = 0; i < B_INTS; ++i) dst[i] = 0;
  for (int k = 0; k < GRP; ++k)
    for (int n = 0; n < N; ++n) {
      int idx = (k / EPD) * N + n;
      int sh  = (k % EPD) * 8;
      dst[idx] |= (int32_t)((uint32_t)((uint8_t)Bblk[k * N + n]) << sh);
    }
}

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("M=%d N=%d K=%d NBLK=%d  A_INTS=%d B_INTS=%d\n", M, N, K, NBLK, A_INTS, B_INTS);

  std::mt19937 rng(12345);
  // synthetic weight in the exact E2M1*2 value set, per-block group scales, per-token acts.
  std::vector<int8_t> w_s8((size_t)K * N);
  std::uniform_int_distribution<int> dcode(0, 7);
  std::uniform_int_distribution<int> dsign(0, 1);
  for (size_t i = 0; i < w_s8.size(); ++i) {
    int mag = E2M1x2[dcode(rng)];
    w_s8[i] = (int8_t)((dsign(rng) && mag) ? -mag : mag);
  }
  // group scales g[b][n], bf16-rounded to match the real bf16-scale path.
  auto to_bf16 = [](float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    u = (u + 0x8000u) & 0xFFFF0000u;   // round-to-nearest-even-ish truncate
    float r; std::memcpy(&r, &u, 4); return r;
  };
  std::vector<float> g((size_t)NBLK * N);
  std::uniform_real_distribution<float> dg(0.01f, 0.5f);
  for (auto& x : g) x = to_bf16(dg(rng));
  // activations x[M][K] bf16-ish, per-token symmetric int8 quant.
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
      int q = (int)std::lround(x[m * K + k] / s);
      q = std::max(-127, std::min(127, q));
      a_s8[m * K + k] = (int8_t)q;
    }
  }

  // ---- CPU reference in the EXACT device order (per-block int32, fp32 accumulate) ----
  std::vector<float>  ref_f32((size_t)M * N);
  std::vector<double> ref_f64((size_t)M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float  fa = 0.f; double da = 0.0;
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

  // ---- host pack per-block A/B tiles ----
  std::vector<int32_t> Apk((size_t)NBLK * A_INTS), Bpk((size_t)NBLK * B_INTS);
  std::vector<float>   Gpk((size_t)NBLK * N);
  std::vector<int8_t>  Ablk(M * GRP), Bblk(GRP * N);
  for (int b = 0; b < NBLK; ++b) {
    for (int m = 0; m < M; ++m)
      for (int kk = 0; kk < GRP; ++kk)
        Ablk[m * GRP + kk] = a_s8[m * K + b * GRP + kk];
    for (int kk = 0; kk < GRP; ++kk)
      for (int n = 0; n < N; ++n)
        Bblk[kk * N + n] = w_s8[(size_t)(b * GRP + kk) * N + n];
    pack_A_block(&Apk[(size_t)b * A_INTS], Ablk.data());
    pack_B_block(&Bpk[(size_t)b * B_INTS], Bblk.data());
    for (int n = 0; n < N; ++n) Gpk[(size_t)b * N + n] = g[(size_t)b * N + n];
  }

  int32_t* dA = sycl::malloc_device<int32_t>(Apk.size(), q);
  int32_t* dB = sycl::malloc_device<int32_t>(Bpk.size(), q);
  float*   dG = sycl::malloc_device<float>(Gpk.size(), q);
  float*   dS = sycl::malloc_device<float>(M, q);
  float*   dY = sycl::malloc_device<float>(C_INTS, q);
  q.copy(Apk.data(), dA, Apk.size());
  q.copy(Bpk.data(), dB, Bpk.size());
  q.copy(Gpk.data(), dG, Gpk.size());
  q.copy(act_scale.data(), dS, M).wait();

  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
       [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
         esimd::simd<float, C_INTS> acc = 0.0f;
         for (int b = 0; b < NBLK; ++b) {
           esimd::simd<int32_t, A_INTS> a; a.copy_from(dA + (size_t)b * A_INTS);
           esimd::simd<int32_t, B_INTS> bb; bb.copy_from(dB + (size_t)b * B_INTS);
           esimd::simd<int32_t, C_INTS> c = 0;
           esimd::simd<int32_t, C_INTS> r =
               xmx::dpas<SDEPTH, M, int32_t, int32_t, int32_t, int32_t,
                         xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(c, bb, a);
           // rescale block partial by g[b,n], broadcast g across the M rows.
           esimd::simd<float, N> gvec; gvec.copy_from(dG + (size_t)b * N);
           esimd::simd<float, C_INTS> grow = gvec.replicate<M>();  // [g(16) x M] = m-major
           acc += grow * r;   // int32 -> float convert on the multiply
         }
         // per-token act scale (per row m).
         esimd::simd<float, M> as; as.copy_from(dS);
         for (int m = 0; m < M; ++m) { float sm = as[m]; acc.select<N, 1>(m * N) *= sm; }
         acc.copy_to(dY);
       });
   }).wait();

  std::vector<float> Y(C_INTS);
  q.copy(dY, Y.data(), C_INTS).wait();

  // ---- compare ----
  double max_abs_f32 = 0, max_rel_f64 = 0, ref_absmax = 0;
  int exact_f32 = 0;
  for (int i = 0; i < C_INTS; ++i) {
    ref_absmax = std::max(ref_absmax, std::fabs(ref_f64[i]));
    double d32 = std::fabs((double)Y[i] - (double)ref_f32[i]);
    max_abs_f32 = std::max(max_abs_f32, d32);
    if (Y[i] == ref_f32[i]) ++exact_f32;
  }
  for (int i = 0; i < C_INTS; ++i) {
    double d = std::fabs((double)Y[i] - ref_f64[i]);
    max_rel_f64 = std::max(max_rel_f64, d / (ref_absmax + 1e-30));
  }
  std::printf("bit-exact vs matched-order fp32 CPU ref: %d / %d elems identical\n", exact_f32, C_INTS);
  std::printf("max |gpu - fp32ref| = %.3e   (fp32 accumulate-order noise)\n", max_abs_f32);
  std::printf("max rel-err vs fp64 truth = %.3e   (ref_absmax=%.4g)\n", max_rel_f64, ref_absmax);
  std::printf("sample: gpu[0]=%.6f ref[0]=%.6f  gpu[1]=%.6f ref[1]=%.6f\n",
              Y[0], ref_f32[0], Y[1], ref_f32[1]);

  bool pass = (max_rel_f64 < 1e-3);
  std::printf("%s: block-scaled s8 DPAS tile numerics\n", pass ? "PASS" : "FAIL");

  sycl::free(dA, q); sycl::free(dB, q); sycl::free(dG, q);
  sycl::free(dS, q); sycl::free(dY, q);
  return pass ? 0 : 1;
}
