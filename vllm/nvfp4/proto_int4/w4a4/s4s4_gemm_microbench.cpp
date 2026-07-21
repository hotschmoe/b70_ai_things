// s4s4_gemm_microbench.cpp -- a REAL tiled symmetric s4 x s4 -> s32 DPAS GEMM
// mainloop on the qwen3.6-27b gate/up prefill shape, for Intel Arc B70
// (Xe2 / Battlemage). This is the W4A4 prefill kernel: int4 activations x int4
// weights, fp32 accumulate via the native ESIMD dpas s4/s4 atom that
// proto_int4/int4_dpas.cpp already proved BIT-EXACT and ~2x the int8 MAC rate.
//
// vs the single-tile proof kernel this adds: (1) a tiled K-mainloop that walks
// the full K=5120 in DPAS steps of 64, (2) 4-way accumulator ILP to hide DPAS
// latency and approach the compute ceiling (mirrors bench.cpp's 4 chains),
// (3) per-token int4 ACTIVATION scale + per-channel int4 WEIGHT scale dequant
// epilogue, (4) in-file correctness (relerr vs a CPU int-exact reference AND vs
// the original fp32 GEMM = the quant-error signal), (5) TOPS timing.
//
// Shape (default): M sweepable, N=34816 (fused gate_up), K=5120.
//   out[M,N] (f32) = dequant( sum_k Aq[M,k] * Wq[N,k] )
//   with Aq,Wq symmetric signed int4 in [-8,7]; act scale per-token (per row m),
//   weight scale per-channel (per output col n).
//
// DPAS atom: SystolicDepth=8, RepeatCount(M)=8, ExecSize(N)=16, s4 K-depth=64.
// Each ESIMD work-item computes an 8 x (16*NSUB) output tile; the same A tile is
// reused across NSUB independent B subtiles (A-reuse + ILP).
//
// Build:  see build_gemm.sh   (AOT intel_gpu_bmg_g31, ESIMD)
// Params: -DGEMM_M=<M> -DNSUB=<n>   (defaults M=512, NSUB=4)
//
// NO GPU is touched at compile time; the coordinator runs the built binary.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

// ---- shape ---------------------------------------------------------------
#ifndef GEMM_M
#define GEMM_M 512            // prefill token count; sweep {512,1024,2048}
#endif
#ifndef GEMM_N
#define GEMM_N 34816          // fused gate_up output channels
#endif
#ifndef GEMM_K
#define GEMM_K 5120           // hidden
#endif
#ifndef NSUB
#define NSUB 4                // N subtiles per work-item (independent DPAS chains)
#endif

static constexpr int M = GEMM_M, N = GEMM_N, K = GEMM_K;

// ---- DPAS tile geometry (s4) --------------------------------------------
static constexpr int SDEPTH = 8;    // systolic depth
static constexpr int RCOUNT = 8;    // M per DPAS tile
static constexpr int EXECSZ = 16;   // N per DPAS tile
static constexpr int ELEMBITS = 4;  // s4
static constexpr int OPC    = 32 / ELEMBITS > 8 ? 8 : 32 / ELEMBITS; // 8
static constexpr int KT     = SDEPTH * OPC;   // 64  (DPAS K depth for s4)
static constexpr int EPD    = 32 / ELEMBITS;  // 8   int4 packed per dword

static constexpr int MT = RCOUNT;             // 8   rows per tile
static constexpr int NT = EXECSZ;             // 16  cols per DPAS subtile
static constexpr int NTILE = NT * NSUB;       // cols per work-item

// simd container element counts (int32) for one DPAS tile
static constexpr int A_INTS = (MT * KT * ELEMBITS) / 32;  // 8*64*4/32 = 64
static constexpr int B_INTS = (KT * NT * ELEMBITS) / 32;  // 64*16*4/32 = 128
static constexpr int C_INTS = MT * NT;                    // 128 (s32)

static constexpr int numKt = K / KT;   // 80
static constexpr int numNt = N / NT;   // 2176
static constexpr int globalM = M / MT; // rows of tiles
static constexpr int globalN = N / NTILE;

// ---- host packing --------------------------------------------------------
// Activations Aq[M,K] -> blocked layout Atile[(kt*M + m)*8 + (kk/8)] so the
// 8-row A operand for a tile is 64 contiguous int32.
static void pack_A(std::vector<int32_t>& dst, const std::vector<int8_t>& Aq) {
  dst.assign((size_t)numKt * M * (KT / EPD), 0);
  for (int kt = 0; kt < numKt; ++kt)
    for (int m = 0; m < M; ++m)
      for (int kk = 0; kk < KT; ++kk) {
        int k = kt * KT + kk;
        size_t idx = ((size_t)kt * M + m) * (KT / EPD) + (kk / EPD);
        uint32_t field = (uint32_t)(Aq[(size_t)m * K + k] & 0xF);
        dst[idx] |= (int32_t)(field << ((kk % EPD) * ELEMBITS));
      }
}
// Weights Wq[N,K] (out,in) -> VNNI B tiles: B[k,n]=Wq[n,k], tile 64x16.
// Btile[(kt*numNt + nt)*128 + (kk/8)*16 + nn].
static void pack_B(std::vector<int32_t>& dst, const std::vector<int8_t>& Wq) {
  dst.assign((size_t)numKt * numNt * B_INTS, 0);
  for (int kt = 0; kt < numKt; ++kt)
    for (int nt = 0; nt < numNt; ++nt) {
      size_t base = ((size_t)kt * numNt + nt) * B_INTS;
      for (int kk = 0; kk < KT; ++kk)
        for (int nn = 0; nn < NT; ++nn) {
          int k = kt * KT + kk, n = nt * NT + nn;
          size_t idx = base + (kk / EPD) * NT + nn;
          uint32_t field = (uint32_t)(Wq[(size_t)n * K + k] & 0xF);
          dst[idx] |= (int32_t)(field << ((kk % EPD) * ELEMBITS));
        }
    }
}

static inline int8_t q4(float v) { // symmetric signed int4 round+clamp
  int q = (int)std::lround(v);
  return (int8_t)std::max(-8, std::min(7, q));
}

int main(int argc, char** argv) {
  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("s4s4 GEMM  M=%d N=%d K=%d  NSUB=%d  tiles(gM=%d gN=%d numKt=%d)\n",
              M, N, K, NSUB, globalM, globalN, numKt);
  static_assert(M % MT == 0, "M must be divisible by 8");
  static_assert(N % NTILE == 0, "N must be divisible by 16*NSUB");
  static_assert(K % KT == 0, "K must be divisible by 64");

  // ---- synthetic fp32 operands with realistic activation outliers --------
  std::mt19937 rng(1234);
  std::normal_distribution<float> gn(0.f, 1.f);
  std::vector<float> Af((size_t)M * K), Wf((size_t)N * K);
  for (auto& x : Wf) x = 0.02f * gn(rng);              // weights: tight Gaussian
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  for (size_t i = 0; i < Af.size(); ++i) {
    float v = gn(rng);
    if (u01(rng) < 0.01f) v *= 12.f;                   // 1% activation outliers
    Af[i] = v;
  }

  // ---- symmetric quant: per-token act scale, per-channel weight scale -----
  std::vector<float> as(M), ws(N);
  std::vector<int8_t> Aq((size_t)M * K), Wq((size_t)N * K);
  for (int m = 0; m < M; ++m) {
    float mx = 1e-8f;
    for (int k = 0; k < K; ++k) mx = std::max(mx, std::fabs(Af[(size_t)m*K+k]));
    as[m] = mx / 7.f;
    for (int k = 0; k < K; ++k) Aq[(size_t)m*K+k] = q4(Af[(size_t)m*K+k]/as[m]);
  }
  for (int n = 0; n < N; ++n) {
    float mx = 1e-8f;
    for (int k = 0; k < K; ++k) mx = std::max(mx, std::fabs(Wf[(size_t)n*K+k]));
    ws[n] = mx / 7.f;
    for (int k = 0; k < K; ++k) Wq[(size_t)n*K+k] = q4(Wf[(size_t)n*K+k]/ws[n]);
  }

  std::vector<int32_t> Apk, Bpk;
  pack_A(Apk, Aq);
  pack_B(Bpk, Wq);

  int32_t* dA  = sycl::malloc_device<int32_t>(Apk.size(), q);
  int32_t* dB  = sycl::malloc_device<int32_t>(Bpk.size(), q);
  float*   dAS = sycl::malloc_device<float>(M, q);
  float*   dWS = sycl::malloc_device<float>(N, q);
  float*   dO  = sycl::malloc_device<float>((size_t)M * N, q);
  q.copy(Apk.data(), dA, Apk.size()).wait();
  q.copy(Bpk.data(), dB, Bpk.size()).wait();
  q.copy(as.data(), dAS, M).wait();
  q.copy(ws.data(), dWS, N).wait();

  const int total = globalM * globalN;
  const int LOCAL = 32;
  const int gpad  = ((total + LOCAL - 1) / LOCAL) * LOCAL;

  auto run = [&]() {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{(size_t)gpad}, sycl::range<1>{(size_t)LOCAL}},
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          int wi = it.get_global_id(0);
          if (wi >= total) return;
          int r = wi / globalN;          // row-block of 8
          int c = wi % globalN;          // col-block of NTILE
          int m0 = r * MT;

          esimd::simd<int32_t, C_INTS> acc[NSUB];
          #pragma unroll
          for (int s = 0; s < NSUB; ++s) acc[s] = 0;

          for (int kt = 0; kt < numKt; ++kt) {
            esimd::simd<int32_t, A_INTS> a;
            a.copy_from(dA + ((size_t)kt * M + m0) * (KT / EPD));
            #pragma unroll
            for (int s = 0; s < NSUB; ++s) {
              int nt = c * NSUB + s;
              esimd::simd<int32_t, B_INTS> b;
              b.copy_from(dB + ((size_t)kt * numNt + nt) * B_INTS);
              acc[s] = xmx::dpas<SDEPTH, RCOUNT, int32_t, int32_t, int32_t,
                                 int32_t, xmx::dpas_argument_type::s4,
                                 xmx::dpas_argument_type::s4>(acc[s], b, a);
            }
          }

          // dequant epilogue: out = acc * act_scale[row] * wt_scale[col]
          esimd::simd<float, MT> asv;
          asv.copy_from(dAS + m0);                     // 8 per-token scales
          #pragma unroll
          for (int s = 0; s < NSUB; ++s) {
            int nt = c * NSUB + s;
            esimd::simd<float, NT> wsv;
            wsv.copy_from(dWS + nt * NT);              // 16 per-channel scales
            esimd::simd<float, C_INTS> outf = acc[s];  // int32 -> float
            #pragma unroll
            for (int m = 0; m < MT; ++m) {
              float a_m = asv[m];
              esimd::simd<float, NT> row = outf.template select<NT, 1>(m * NT);
              row = row * wsv * a_m;
              row.copy_to(dO + (size_t)(m0 + m) * N + nt * NT);
            }
          }
        });
    }).wait();
  };

  run();                                    // warmup
  double best = 1e30;
  for (int r = 0; r < 5; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    run();
    auto t1 = std::chrono::high_resolution_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }
  double macs = (double)M * N * K;
  std::printf("time_best=%.4f ms  TOPS(2*MAC)=%.1f  (int8 ref 367, bf16 ref 183)\n",
              best * 1e3, 2.0 * macs / best / 1e12);

  // ---- correctness: pull GPU output, compare to two references -----------
  std::vector<float> O((size_t)M * N);
  q.copy(dO, O.data(), O.size()).wait();

  // sample a subset of (m,n) to bound host cost (K=5120 * full N is large)
  std::mt19937 srng(7);
  std::uniform_int_distribution<int> dm(0, M - 1), dn(0, N - 1);
  int NS = 4096;
  double num_q = 0, den_q = 0;   // GPU vs CPU int-exact-quant  (kernel correctness)
  double num_f = 0, den_f = 0;   // GPU vs original fp32 GEMM   (quant error)
  int worst_idx = -1; double worst = 0;
  for (int t = 0; t < NS; ++t) {
    int m = dm(srng), n = dn(srng);
    int64_t iacc = 0; double facc = 0;
    for (int k = 0; k < K; ++k) {
      iacc += (int64_t)Aq[(size_t)m*K+k] * (int64_t)Wq[(size_t)n*K+k];
      facc += (double)Af[(size_t)m*K+k] * (double)Wf[(size_t)n*K+k];
    }
    double ref_q = (double)iacc * as[m] * ws[n];   // dequantized int result
    double gpu   = O[(size_t)m*N + n];
    num_q += (gpu-ref_q)*(gpu-ref_q); den_q += ref_q*ref_q;
    num_f += (gpu-facc)*(gpu-facc);   den_f += facc*facc;
    double e = std::fabs(gpu-ref_q)/(std::fabs(ref_q)+1e-6);
    if (e > worst) { worst = e; worst_idx = t; }
  }
  double rel_q = std::sqrt(num_q/den_q), rel_f = std::sqrt(num_f/den_f);
  std::printf("KERNEL correctness  relerr(GPU vs CPU int-exact) = %.3e  worst_pt=%.3e  [target < 1e-3]\n",
              rel_q, worst);
  std::printf("QUANT accuracy      relerr(GPU vs fp32 GEMM)      = %.3e  "
              "[W4A4 no-rotation signal; see w4a4_accuracy_probe.py]\n", rel_f);
  bool ok = rel_q < 1e-3;
  std::printf("%s  (kernel %s)\n", ok ? "PASS" : "FAIL",
              ok ? "computes the s4xs4 GEMM correctly" : "MISMATCH -- check packing/layout");
  (void)worst_idx;

  sycl::free(dA,q); sycl::free(dB,q); sycl::free(dAS,q);
  sycl::free(dWS,q); sycl::free(dO,q);
  return ok ? 0 : 1;
}
