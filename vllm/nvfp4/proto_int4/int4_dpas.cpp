// int4_dpas.cpp -- minimal ESIMD DPAS microkernel to test whether a NATIVE
// int4 (s4xs4->s32) DPAS instruction is reachable + correct on Intel Arc B70
// (Xe2 / Battlemage). Parametrized so the SAME register layout + host packing
// can build an s8 control (PREC=8) and the s4 target (PREC=4).
//
// Build: see build.sh. Precision selected by -DPREC={8,4,2}.
//
// One ESIMD work-item computes a single DPAS tile:
//   Result[M x N] (s32) = A[M x K] (sN) * B[K x N] (sN, VNNI-encoded)
// with SystolicDepth=8, RepeatCount(M)=8, ExecutionSize(N)=16.
//
// K depth depends on precision: K = SystolicDepth * OpsPerChannel where
// OpsPerChannel = 32/PREC (8 for s4, 4 for s8). Container simd sizes are the
// SAME for s4 and s8 (A=64 ints, B=128 ints, C=128 ints) -- only the precision
// enum, K, and the host nibble/byte packing differ.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <random>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

#ifndef PREC
#define PREC 4
#endif

#if PREC == 8
  #define APREC xmx::dpas_argument_type::s8
  #define BPREC xmx::dpas_argument_type::s8
  static constexpr int ELEMBITS = 8;
  static constexpr const char* PREC_NAME = "s8";
#elif PREC == 4
  #define APREC xmx::dpas_argument_type::s4
  #define BPREC xmx::dpas_argument_type::s4
  static constexpr int ELEMBITS = 4;
  static constexpr const char* PREC_NAME = "s4";
#elif PREC == 2
  #define APREC xmx::dpas_argument_type::s2
  #define BPREC xmx::dpas_argument_type::s2
  static constexpr int ELEMBITS = 2;
  static constexpr const char* PREC_NAME = "s2";
#else
  #error "PREC must be 8, 4 or 2"
#endif

static constexpr int SDEPTH  = 8;
static constexpr int RCOUNT  = 8;    // M
static constexpr int EXECSZ  = 16;   // N
// OpsPerChannel: elements of PREC packed per dword, capped at 8 (per ESIMD hdr).
static constexpr int OPC_RAW = 32 / ELEMBITS;
static constexpr int OPC     = OPC_RAW > 8 ? 8 : OPC_RAW;
static constexpr int KDEPTH  = SDEPTH * OPC;     // s4:64 s8:32 s2:64
// packing density per dword (how many PREC-elements fit in 32 bits)
static constexpr int EPD     = 32 / ELEMBITS;    // s4:8 s8:4 s2:16

static constexpr int M = RCOUNT;   // 8
static constexpr int N = EXECSZ;   // 16
static constexpr int K = KDEPTH;   // 64 (s4) / 32 (s8)

// simd container element counts (in 32-bit ints)
static constexpr int A_INTS = (M * K * ELEMBITS) / 32;   // 8*64*4/32 = 64
static constexpr int B_INTS = (K * N * ELEMBITS) / 32;   // 64*16*4/32 = 128
static constexpr int C_INTS = M * N;                     // 128 (s32)

// ---- host packing helpers -------------------------------------------------
// A is M x K row-major. Dword index = m*(K/EPD) + k/EPD; field = k%EPD.
static inline void pack_A(std::vector<int32_t>& dst,
                          const std::vector<int>& A /*M*K*/) {
  dst.assign(A_INTS, 0);
  const int kd = K / EPD;
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) {
      int idx = m * kd + (k / EPD);
      int sh  = (k % EPD) * ELEMBITS;
      uint32_t field = (uint32_t)(A[m*K + k] & ((1u<<ELEMBITS)-1));
      dst[idx] |= (int32_t)(field << sh);
    }
}
// B is K x N, VNNI: dword index = (k/EPD)*N + n; field = k%EPD.
static inline void pack_B(std::vector<int32_t>& dst,
                          const std::vector<int>& B /*K*N*/) {
  dst.assign(B_INTS, 0);
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n) {
      int idx = (k / EPD) * N + n;
      int sh  = (k % EPD) * ELEMBITS;
      uint32_t field = (uint32_t)(B[k*N + n] & ((1u<<ELEMBITS)-1));
      dst[idx] |= (int32_t)(field << sh);
    }
}

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("precision=%s  M=%d N=%d K=%d  A_INTS=%d B_INTS=%d C_INTS=%d\n",
              PREC_NAME, M, N, K, A_INTS, B_INTS, C_INTS);

  // deterministic small operands in signed range [-lim, lim-1]
  const int lim = 1 << (ELEMBITS - 1);          // s4:8 s8:128 s2:2
  std::vector<int> A(M*K), B(K*N);
  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> d(-lim, lim - 1);
  for (auto& x : A) x = d(rng);
  for (auto& x : B) x = d(rng);

  // CPU reference (full precision int accumulate)
  std::vector<int32_t> ref(M*N, 0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int64_t acc = 0;
      for (int k = 0; k < K; ++k) acc += (int64_t)A[m*K+k] * (int64_t)B[k*N+n];
      ref[m*N + n] = (int32_t)acc;
    }

  std::vector<int32_t> Apk, Bpk;
  pack_A(Apk, A);
  pack_B(Bpk, B);

  int32_t* dA = sycl::malloc_device<int32_t>(A_INTS, q);
  int32_t* dB = sycl::malloc_device<int32_t>(B_INTS, q);
  int32_t* dC = sycl::malloc_device<int32_t>(C_INTS, q);
  q.copy(Apk.data(), dA, A_INTS).wait();
  q.copy(Bpk.data(), dB, B_INTS).wait();

  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
       [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
         esimd::simd<int32_t, A_INTS> a;
         esimd::simd<int32_t, B_INTS> b;
         a.copy_from(dA);
         b.copy_from(dB);
         esimd::simd<int32_t, C_INTS> c = 0;
         esimd::simd<int32_t, C_INTS> r =
             xmx::dpas<SDEPTH, RCOUNT, int32_t, int32_t, int32_t, int32_t,
                       BPREC, APREC>(c, b, a);
         r.copy_to(dC);
       });
   }).wait();

  std::vector<int32_t> C(C_INTS);
  q.copy(dC, C.data(), C_INTS).wait();

  int mism = 0, first = -1;
  for (int i = 0; i < C_INTS; ++i)
    if (C[i] != ref[i]) { if (first < 0) first = i; ++mism; }

  std::printf("mismatches: %d / %d\n", mism, C_INTS);
  if (mism) {
    std::printf("first mismatch idx=%d (m=%d n=%d): gpu=%d ref=%d\n",
                first, first / N, first % N, C[first], ref[first]);
    // dump a few for diagnosis
    for (int i = 0; i < 8; ++i)
      std::printf("  [%2d] gpu=%d ref=%d\n", i, C[i], ref[i]);
  } else {
    std::printf("PASS: native %s DPAS matches CPU int reference. sample C[0]=%d C[1]=%d\n",
                PREC_NAME, C[0], C[1]);
  }

  sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
  return mism ? 1 : 0;
}
