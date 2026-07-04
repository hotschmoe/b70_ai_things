// bench.cpp -- compute-bound DPAS throughput microbench for Intel Arc B70 (Xe2).
// Measures native s8 vs s4 vs s2 DPAS instruction/MAC rate to test the
// "int4 = 2x int8, int2 = 4x int8" hardware claim (vs the arXiv 2508.06753
// "low-bit int DPAS same throughput as int8" rumor).
//
// Each work-item runs a dependent chain of NITER dpas.8x8 ops accumulating in
// registers. With a large grid the machine is throughput-bound; the dependent
// chain also exposes per-instruction latency differences directly. We report
// dpas-instructions/s and MAC/s (= instr/s * M*N*K, where K differs by precision:
// s8 K=32, s4/s2 K=64). Build: -DPREC={8,4,2}.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdio>
#include <chrono>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

#ifndef PREC
#define PREC 4
#endif
#if PREC == 8
  #define APREC xmx::dpas_argument_type::s8
  #define BPREC xmx::dpas_argument_type::s8
  static constexpr int ELEMBITS = 8; static constexpr const char* PN="s8";
#elif PREC == 4
  #define APREC xmx::dpas_argument_type::s4
  #define BPREC xmx::dpas_argument_type::s4
  static constexpr int ELEMBITS = 4; static constexpr const char* PN="s4";
#elif PREC == 2
  #define APREC xmx::dpas_argument_type::s2
  #define BPREC xmx::dpas_argument_type::s2
  static constexpr int ELEMBITS = 2; static constexpr const char* PN="s2";
#endif

static constexpr int SDEPTH=8, RCOUNT=8, EXECSZ=16;
static constexpr int OPC_RAW=32/ELEMBITS; static constexpr int OPC=OPC_RAW>8?8:OPC_RAW;
static constexpr int K=SDEPTH*OPC;              // s8:32 s4:64 s2:64
static constexpr int M=RCOUNT, N=EXECSZ;
static constexpr int A_INTS=(M*K*ELEMBITS)/32;  // 64/64/32
static constexpr int B_INTS=(K*N*ELEMBITS)/32;  // 128/128/64
static constexpr int C_INTS=M*N;                // 128

#ifndef NITER
#define NITER 2048
#endif
#ifndef NGROUPS
#define NGROUPS 8192
#endif

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  int eu = dev.get_info<sycl::info::device::max_compute_units>();
  std::printf("precision=%s K=%d  max_compute_units=%d  NITER=%d NGROUPS=%d\n",
              PN, K, eu, NITER, NGROUPS);

  const int WI = NGROUPS;               // one ESIMD thread per work-item
  int32_t* out = sycl::malloc_device<int32_t>(WI, q);

  auto once = [&](int iters)->double {
    auto t0 = std::chrono::high_resolution_clock::now();
    q.submit([&](sycl::handler& h){
      h.parallel_for(sycl::nd_range<1>{sycl::range<1>{(size_t)WI}, sycl::range<1>{64}},
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          int gid = it.get_global_id(0);
          esimd::simd<int32_t, A_INTS> a(gid + 1);
          esimd::simd<int32_t, B_INTS> b0(gid + 3), b1(gid + 5),
                                       b2(gid + 7), b3(gid + 9);
          // 4 independent accumulator chains -> hide DPAS latency, approach
          // throughput peak (not latency-bound like a single chain).
          esimd::simd<int32_t, C_INTS> c0(0), c1(0), c2(0), c3(0);
          #pragma unroll 8
          for (int i=0;i<iters;++i) {
            c0 = xmx::dpas<SDEPTH,RCOUNT,int32_t,int32_t,int32_t,int32_t,BPREC,APREC>(c0,b0,a);
            c1 = xmx::dpas<SDEPTH,RCOUNT,int32_t,int32_t,int32_t,int32_t,BPREC,APREC>(c1,b1,a);
            c2 = xmx::dpas<SDEPTH,RCOUNT,int32_t,int32_t,int32_t,int32_t,BPREC,APREC>(c2,b2,a);
            c3 = xmx::dpas<SDEPTH,RCOUNT,int32_t,int32_t,int32_t,int32_t,BPREC,APREC>(c3,b3,a);
          }
          out[gid] = (c0[0]^c0[63]) + (c1[0]^c1[63]) + (c2[0]^c2[63]) + (c3[0]^c3[63]);
        });
    }).wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1-t0).count();
  };

  once(64);                     // warmup
  double best = 1e30;
  for (int r=0;r<5;++r) best = std::min(best, once(NITER));

  double instr = (double)WI * NITER * 4.0;   // 4 chains/iter
  double macs  = instr * (double)M * N * K;  // MACs
  std::printf("%s: best=%.4f ms  dpas_instr=%.3e  instr/s=%.3e  MAC/s=%.3e  TOPS(2*MAC)=%.1f\n",
              PN, best*1e3, instr, instr/best, macs/best, 2.0*macs/best/1e12);
  sycl::free(out, q);
  return 0;
}
