// bs_dpas_m3.cpp -- MILESTONE 3: tiled block-scaled INT8-XMX GEMM over the real
// 27B gate/up shape (N=17408, K=5120), with in-register f4_e2m1 decode, and TIMING.
//
// Each ESIMD work-item computes an [TM=8 x TN=64] output tile (4 N-subtiles of 16),
// reducing K in 16-wide NVFP4 groups. Per group: one s8 DPAS per N-subtile with the
// upper 16 K zero-padded (block<DPAS-K), an fp32 rescale by the per-16-K bf16 group
// scale, into 4 independent fp32 accumulators (4-way ILP to hide DPAS latency). Weight
// is 4-bit resident (E2M1 nibbles) decoded in-register. Per-token act scale applied at
// the end; bf16 output.
//
// Modes (compile -DBSMODE):
//   1 = block-scaled (the real kernel): 320 groups of 16, zero-pad to K=32, per-group rescale.
//   0 = per-channel CEILING of the SAME tiling: 160 groups of full K=32 (no pad, no per-group
//       rescale, single per-N scale at the end) -- measures how close this ESIMD tiling gets
//       to oneDNN's per-channel int8 0.35/1.29ms, isolating the block-scale penalty.
//
// Correctness: host reference for work-item 0's [8x64] tile, compared bit-exact/relerr.
// Timing: best-of over ITERS launches of the full grid. Env: MM (M), ITERS.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

#ifndef BSMODE
#define BSMODE 1
#endif

static constexpr int SDEPTH = 8;
static constexpr int TM  = 8;     // rows per tile (RepeatCount)
static constexpr int N16 = 16;    // DPAS ExecutionSize
static constexpr int NSUB = 4;    // N-subtiles per tile
static constexpr int TN  = N16 * NSUB;  // 64 cols per tile
static constexpr int DPK = 32;    // s8 DPAS K
static constexpr int GRP = 16;    // NVFP4 group
static constexpr int EPD = 4;

#ifndef KDIM
#define KDIM 5120
#endif
#ifndef NDIM
#define NDIM 17408
#endif
static constexpr int K = KDIM;
static constexpr int N = NDIM;

#if BSMODE == 1
static constexpr int RGRP = GRP;          // reduce 16 real K per DPAS
static constexpr int NG   = K / GRP;       // 320 groups
#else
static constexpr int RGRP = DPK;          // reduce full 32 real K per DPAS
static constexpr int NG   = K / DPK;       // 160 groups
#endif

static constexpr int A_INTS = (TM * DPK) / EPD;      // 64 (zero-padded tile)
static constexpr int B_INTS = (DPK * N16) / EPD;     // 128
static constexpr int C_INTS = TM * N16;              // 128
static constexpr int W4_PER_SUB = (RGRP / 4) * N16;  // BS:64  PC:128 uint16 (per subtile,group)

static const int E2M1x2[8] = {0, 1, 2, 3, 4, 6, 8, 12};

static uint16_t encode_nib(int8_t s8) {
  int mag = std::abs((int)s8), idx = 0;
  for (int i = 0; i < 8; ++i) if (E2M1x2[i] == mag) { idx = i; break; }
  return (uint16_t)(((s8 < 0) ? 8 : 0) | idx);
}

template <int W>
static ESIMD_INLINE esimd::simd<int32_t, W> decode_e2m1x2(esimd::simd<int32_t, W> nib) {
  esimd::simd<int32_t, W> sign = nib & 0x8;
  esimd::simd<int32_t, W> idx  = nib & 0x7;
  esimd::simd<int32_t, W> exp  = idx >> 1;
  esimd::simd<int32_t, W> mant = idx & 0x1;
  esimd::simd<int32_t, W> sh   = esimd::max(exp - 1, esimd::simd<int32_t, W>(0));
  esimd::simd<int32_t, W> mag  = (mant + 2) << sh;
  mag.merge(mant, exp == 0);
  esimd::simd<int32_t, W> neg = -mag;
  mag.merge(neg, sign != 0);
  return mag;
}

int main() {
  const int MM = std::getenv("MM") ? atoi(std::getenv("MM")) : 512;
  const int ITERS = std::getenv("ITERS") ? atoi(std::getenv("ITERS")) : 30;
  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("MODE=%s  M=%d N=%d K=%d  tile=%dx%d  groups=%d (RGRP=%d)  W4/sub=%d\n",
              BSMODE ? "block-scaled" : "per-channel-ceiling", MM, N, K, TM, TN, NG, RGRP, W4_PER_SUB);
  if (MM % TM || N % TN) { std::printf("bad dims\n"); return 2; }
  const int GM = MM / TM, GN = N / TN, NSG = N / N16;

  std::mt19937 rng(777);
  // weight [K][N] E2M1*2 codes
  std::vector<int8_t> w_s8((size_t)K * N);
  { std::uniform_int_distribution<int> dc(0,7), ds(0,1);
    for (auto& v : w_s8){ int mg=E2M1x2[dc(rng)]; v=(int8_t)((ds(rng)&&mg)?-mg:mg);} }
  auto to_bf16=[](float f){uint32_t u;std::memcpy(&u,&f,4);u=(u+0x8000u)&0xFFFF0000u;float r;std::memcpy(&r,&u,4);return r;};
  // group scales per (block16, n): BS uses per-16 scale; PC uses a single per-N scale.
  const int NBLK16 = K / GRP;
  std::vector<float> gsrc((size_t)NBLK16 * N);
  { std::uniform_real_distribution<float> dg(0.01f,0.5f); for (auto& v: gsrc) v=to_bf16(dg(rng)); }
  // per-N single scale (PC): use block 0's scale as representative (timing is data-independent)
  std::vector<float> gpc(N); for (int n=0;n<N;++n) gpc[n]=gsrc[n];
  // activations
  std::vector<float> xf((size_t)MM*K); { std::normal_distribution<float> dx(0.f,0.1f); for(auto&v:xf)v=dx(rng);}
  std::vector<float> act_scale(MM); std::vector<int8_t> a_s8((size_t)MM*K);
  for (int m=0;m<MM;++m){ float amax=1e-8f; for(int k=0;k<K;++k)amax=std::max(amax,std::fabs(xf[m*K+k]));
    float s=amax/127.f; act_scale[m]=s; for(int k=0;k<K;++k){int qv=(int)std::lround(xf[m*K+k]/s); a_s8[m*K+k]=(int8_t)std::max(-127,std::min(127,qv));}}

  // ---- pack A: per (mtile, group) zero-padded [TM x DPK] (64 ints). RGRP real K, rest 0 ----
  std::vector<int32_t> A((size_t)GM * NG * A_INTS, 0);
  for (int mt=0; mt<GM; ++mt)
    for (int gp=0; gp<NG; ++gp) {
      int32_t* d = &A[((size_t)mt*NG+gp)*A_INTS];
      const int kd = DPK/EPD;
      for (int m=0;m<TM;++m)
        for (int kk=0; kk<RGRP; ++kk) {
          int k = gp*RGRP + kk;
          int idx = m*kd + (kk/EPD), sh=(kk%EPD)*8;
          d[idx] |= (int32_t)((uint32_t)((uint8_t)a_s8[(size_t)(mt*TM+m)*K+k])<<sh);
        }
    }
  // ---- pack W4: per (nsub_global, group) 4-bit VNNI. RGRP K, 16 cols ----
  std::vector<uint16_t> W((size_t)NSG * NG * W4_PER_SUB, 0);
  for (int ns=0; ns<NSG; ++ns)
    for (int gp=0; gp<NG; ++gp) {
      uint16_t* d = &W[((size_t)ns*NG+gp)*W4_PER_SUB];
      for (int kg=0; kg<RGRP/4; ++kg)
        for (int n=0;n<N16;++n){
          uint16_t v=0;
          for(int j=0;j<4;++j){int k=gp*RGRP+kg*4+j; v|=(uint16_t)(encode_nib(w_s8[(size_t)k*N + ns*N16+n])<<(j*4));}
          d[kg*N16+n]=v;
        }
    }
  // ---- scales ----
#if BSMODE == 1
  std::vector<float> G((size_t)NG * N);  // [group][n]
  for (int gp=0; gp<NG; ++gp) for (int n=0;n<N;++n) G[(size_t)gp*N+n]=gsrc[(size_t)gp*N+n];
#else
  std::vector<float> G(N); for (int n=0;n<N;++n) G[n]=gpc[n];
#endif

  int32_t*  dA = sycl::malloc_device<int32_t>(A.size(), q);
  uint16_t* dW = sycl::malloc_device<uint16_t>(W.size(), q);
  float*    dG = sycl::malloc_device<float>(G.size(), q);
  float*    dS = sycl::malloc_device<float>(MM, q);
  bf16*     dY = sycl::malloc_device<bf16>((size_t)MM*N, q);
  q.copy(A.data(),dA,A.size()); q.copy(W.data(),dW,W.size());
  q.copy(G.data(),dG,G.size()); q.copy(act_scale.data(),dS,MM).wait();

  const int TOTAL = GM * GN;
  auto launch = [&](){
    return q.submit([&](sycl::handler& h){
      h.parallel_for(sycl::nd_range<1>{sycl::range<1>{(size_t)TOTAL}, sycl::range<1>{16}},
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          const int gid = it.get_global_id(0);
          const int mt = gid / GN, tn = gid % GN;   // tile (row-block, col-block)
          esimd::simd<float, NSUB*C_INTS> acc = 0.f;   // 4 subtiles x [8x16]
          for (int gp=0; gp<NG; ++gp) {
            esimd::simd<int32_t,A_INTS> a; a.copy_from(dA + ((size_t)mt*NG+gp)*A_INTS);
#if BSMODE == 1
            esimd::simd<float,TN> gvec; gvec.copy_from(dG + (size_t)gp*N + (size_t)tn*TN);
#endif
            #pragma unroll
            for (int sub=0; sub<NSUB; ++sub) {
              int ns = tn*NSUB + sub;
              esimd::simd<uint16_t,W4_PER_SUB> w4; w4.copy_from(dW + ((size_t)ns*NG+gp)*W4_PER_SUB);
              esimd::simd<int32_t,W4_PER_SUB> w4i = w4;
              esimd::simd<int32_t,B_INTS> bb = 0;
#if BSMODE == 1
              // 64 uint16 -> 64 dwords (16K x 16N), upper 16K zero
              esimd::simd<int32_t,W4_PER_SUB> s0=decode_e2m1x2<W4_PER_SUB>( w4i     &0xF);
              esimd::simd<int32_t,W4_PER_SUB> s1=decode_e2m1x2<W4_PER_SUB>((w4i>>4) &0xF);
              esimd::simd<int32_t,W4_PER_SUB> s2=decode_e2m1x2<W4_PER_SUB>((w4i>>8) &0xF);
              esimd::simd<int32_t,W4_PER_SUB> s3=decode_e2m1x2<W4_PER_SUB>((w4i>>12)&0xF);
              bb.select<W4_PER_SUB,1>(0) = (s0&0xFF)|((s1&0xFF)<<8)|((s2&0xFF)<<16)|((s3&0xFF)<<24);
#else
              // full 32K: 128 uint16 -> 128 dwords (32K x 16N)
              esimd::simd<int32_t,B_INTS> s0=decode_e2m1x2<B_INTS>( w4i     &0xF);
              esimd::simd<int32_t,B_INTS> s1=decode_e2m1x2<B_INTS>((w4i>>4) &0xF);
              esimd::simd<int32_t,B_INTS> s2=decode_e2m1x2<B_INTS>((w4i>>8) &0xF);
              esimd::simd<int32_t,B_INTS> s3=decode_e2m1x2<B_INTS>((w4i>>12)&0xF);
              bb = (s0&0xFF)|((s1&0xFF)<<8)|((s2&0xFF)<<16)|((s3&0xFF)<<24);
#endif
              esimd::simd<int32_t,C_INTS> c=0;
              esimd::simd<int32_t,C_INTS> r =
                xmx::dpas<SDEPTH,TM,int32_t,int32_t,int32_t,int32_t,
                          xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(c, bb, a);
#if BSMODE == 1
              esimd::simd<float,N16> gsub = gvec.select<N16,1>(sub*N16);
              esimd::simd<float,C_INTS> grow = gsub.replicate<TM>();
              esimd::simd<float,C_INTS> rf = r * grow;
#else
              esimd::simd<float,C_INTS> rf = r;   // accumulate raw s32; scale once at end
#endif
              acc.select<C_INTS,1>(sub*C_INTS) += rf;
            }
          }
          esimd::simd<float,TM> as; as.copy_from(dS + (size_t)mt*TM);
#if BSMODE == 0
          esimd::simd<float,TN> gpcv; gpcv.copy_from(dG + (size_t)tn*TN);
#endif
          #pragma unroll
          for (int sub=0; sub<NSUB; ++sub) {
            esimd::simd<float,C_INTS> a4 = acc.select<C_INTS,1>(sub*C_INTS);
#if BSMODE == 0
            esimd::simd<float,N16> gsub = gpcv.select<N16,1>(sub*N16);
            esimd::simd<float,C_INTS> grow = gsub.replicate<TM>();
            a4 *= grow;
#endif
            for (int m=0;m<TM;++m){ float sm=as[m]; a4.select<N16,1>(m*N16)*=sm; }
            esimd::simd<bf16,C_INTS> ob = a4;
            for (int m=0;m<TM;++m){
              esimd::simd<bf16,N16> row = ob.select<N16,1>(m*N16);
              row.copy_to(dY + (size_t)(mt*TM+m)*N + (size_t)tn*TN + sub*N16);
            }
          }
        });
    });
  };

  launch().wait();  // warmup + fills dY for correctness check

  // ---- correctness: host ref for tile (mt=0,tn=0) = rows 0..7, cols 0..63 ----
  std::vector<bf16> Y((size_t)MM*N);
  q.copy(dY, Y.data(), (size_t)MM*N).wait();
  double maxrel=0, refabs=0; int nchk=0, exact=0;
  for (int m=0;m<TM;++m) for (int col=0; col<TN; ++col) {
    double da=0.0;
#if BSMODE == 1
    for (int b=0;b<NBLK16;++b){ int32_t ip=0; for(int kk=0;kk<GRP;++kk){int k=b*GRP+kk; ip+=(int32_t)a_s8[m*K+k]*(int32_t)w_s8[(size_t)k*N+col];}
      da += (double)gsrc[(size_t)b*N+col]*(double)ip; }
    da *= act_scale[m];
#else
    { int64_t ip=0; for(int k=0;k<K;++k) ip+=(int64_t)a_s8[m*K+k]*(int64_t)w_s8[(size_t)k*N+col];
      da = (double)gpc[col]*(double)ip*(double)act_scale[m]; }
#endif
    double gv=(double)(float)Y[(size_t)m*N+col];
    refabs=std::max(refabs,std::fabs(da)); maxrel=std::max(maxrel,std::fabs(gv-da)); ++nchk;
    if ((float)gv==(float)da) ++exact;
  }
  double relerr = maxrel/(refabs+1e-30);
  std::printf("correctness (tile 0, %d elems): rel-err vs fp64 = %.3e  bf16-exact=%d/%d  (refabs=%.4g)\n",
              nchk, relerr, exact, nchk, refabs);

  // ---- timing ----
  auto once=[&](){ auto t0=std::chrono::high_resolution_clock::now(); launch().wait();
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count(); };
  for (int w=0; w<3; ++w) once();
  double best=1e30; for (int i=0;i<ITERS;++i) best=std::min(best, once());
  double gmac = (double)MM*N*K, tops = 2.0*gmac/best/1e12;
  std::printf("TIME M=%d : best=%.4f ms  (%.1f TOPS eff, %.1f GMAC)\n", MM, best*1e3, tops, gmac/1e9);
  std::printf("RESULT_LINE mode=%s M=%d ms=%.4f relerr=%.3e\n", BSMODE?"BS":"PC", MM, best*1e3, relerr);

  sycl::free(dA,q);sycl::free(dW,q);sycl::free(dG,q);sycl::free(dS,q);sycl::free(dY,q);
  return relerr<1e-2 ? 0 : 1;
}
