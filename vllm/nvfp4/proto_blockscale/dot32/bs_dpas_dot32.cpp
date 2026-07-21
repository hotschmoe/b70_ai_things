// bs_dpas_dot32.cpp -- the documented-but-unbuilt NVFP4 prefill "dot32 + correction"
// block-scaled INT8-XMX GEMM, tiled over the real 27B gate/up shape (N=17408, K=5120,
// group_size=16), with in-register f4_e2m1 decode, fp32 accumulate, and TIMING.
//
// -------------------------------------------------------------------------------------
// THE dot32 + correction IDENTITY (exact, worked out)
// -------------------------------------------------------------------------------------
// NVFP4 applies a per-16-K weight scale g[b][n] that VARIES within the K reduction. On
// Xe2 the s8 DPAS reduces a FIXED K=32 per instruction (SystolicDepth=8, OpsPerChannel=4;
// there is no K=16 s8 DPAS). So the "naive block-scaled" (BS) kernel issues ONE K=32 DPAS
// per 16-group with the upper 16 K zero-padded -> 2 DPAS per adjacent group PAIR.
//
// The dot32 trick, for an adjacent pair of groups g0 (scale bg0) and g1 (scale bg1):
//   exact_pair = bg0*dot16_g0 + bg1*dot16_g1
//              = bg1*(dot16_g0 + dot16_g1) + (bg0 - bg1)*dot16_g0
//              = bg1*dot32           + (bg0 - bg1)*dot16_g0
// where dot32 = dot16_g0 + dot16_g1 is ONE FULL-efficiency K=32 s8 DPAS over BOTH groups,
// and dot16_g0 is a K=16 (zero-padded to K=32) correction DPAS over the FIRST group only.
// This is EXACT (algebraic rearrangement, no approximation).
//
// dot16_g0 is obtained for free from the same operands: DPAS(A16, W32) where A16 is the
// full-32 A tile with its upper 16 K zeroed -> the K16..31 products vanish -> pure g0
// partial. No extra weight decode, no extra load.
//
// -------------------------------------------------------------------------------------
// THE DPAS-COUNT REALITY (why this is a NO-GO -- verified empirically by this file)
// -------------------------------------------------------------------------------------
// The README framed this as "1.5 DPAS-equiv per 2 groups (vs 2.0 for naive BS)". That
// counts a full K=32 DPAS as 1.0 and a K=16 DPAS as 0.5 EFFICIENCY-units. But wall-clock
// is governed by DPAS ISSUE COUNT, and on fixed-SystolicDepth-8 hardware a K=16 DPAS
// consumes a FULL issue slot (it is a K=32 DPAS with zeros). So per group pair:
//     naive BS      : 2 DPAS (K=16 padded + K=16 padded)
//     dot32+corr    : 2 DPAS (K=32 full   + K=16 padded)
// IDENTICAL issue count. The "0.5" correction still costs a whole instruction. Extracting
// TWO independently-scaled K=16 partials from a K=32 span requires TWO linearly-independent
// DPAS measurements -- fundamental, cannot be < 2. So dot32 CANNOT beat naive BS.
//
// This file builds all three so a SINGLE coordinator GPU run settles it on real silicon:
//   -DKMODE=32  DOT32 : dot32 + correction (exact). DPAS/tile = 160 pairs * 4 sub * 2 = 1280
//   -DKMODE=16  BS    : naive block-scaled (m3-equivalent). 320 grp * 4 sub * 1 = 1280
//   -DKMODE=0   PC    : per-channel full-K32 ceiling.       160 grp * 4 sub * 1 =  640
// Prediction: time(DOT32) ~= time(BS) (both 1280 DPAS), and both ~= 1.52x time(PC).
// If the box shows DOT32 materially faster than BS, the analysis is WRONG -- measure it.
//
// Correctness: host fp64 reference for work-item 0's [8x64] tile, relerr target < 5e-3.
// Env: MM (M in {512,1024,2048}), ITERS.  Build: see build.sh (AOT intel_gpu_bmg_g31).

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

#ifndef KMODE
#define KMODE 32          // 32=DOT32(scheme) 16=BS(naive) 0=PC(ceiling)
#endif

static constexpr int SDEPTH = 8;
static constexpr int TM  = 8;      // rows per tile (RepeatCount)
static constexpr int N16 = 16;     // DPAS ExecutionSize
static constexpr int NSUB = 4;     // N-subtiles per tile
static constexpr int TN  = N16 * NSUB;   // 64 cols per tile
static constexpr int DPK = 32;     // s8 DPAS K
static constexpr int GRP = 16;     // NVFP4 group

#ifndef KDIM
#define KDIM 5120
#endif
#ifndef NDIM
#define NDIM 17408
#endif
static constexpr int K = KDIM;
static constexpr int N = NDIM;

static constexpr int A_INTS = (TM * DPK) / 4;    // 64  (full-32 A tile, VNNI)
static constexpr int B_INTS = (DPK * N16) / 4;   // 128 (full-32 B tile, VNNI)
static constexpr int C_INTS = TM * N16;          // 128 (DPAS result [8x16])

// group granularity of the OUTER loop
#if KMODE == 16
static constexpr int RGRP = GRP;      // 16 real K per outer step (naive BS)
static constexpr int NG   = K / GRP;  // 320
#else
static constexpr int RGRP = DPK;      // 32 real K per outer step (DOT32 pair, or PC)
static constexpr int NG   = K / DPK;  // 160
#endif
// 4-bit weight uint16 per (subtile, outer-group): (RGRP/4)*16
static constexpr int W4_PER_SUB = (RGRP / 4) * N16;   // BS:64  DOT32/PC:128

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

#define DP(C, B, A) \
  xmx::dpas<SDEPTH, TM, int32_t, int32_t, int32_t, int32_t, \
            xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(C, B, A)

int main() {
  const int MM = std::getenv("MM") ? atoi(std::getenv("MM")) : 512;
  const int ITERS = std::getenv("ITERS") ? atoi(std::getenv("ITERS")) : 30;
  sycl::queue q{sycl::gpu_selector_v};
  auto dev = q.get_device();
  const char* mn = (KMODE == 32) ? "DOT32(scheme)" : (KMODE == 16) ? "BS(naive)" : "PC(ceiling)";
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("MODE=%s  M=%d N=%d K=%d  tile=%dx%d  outer-groups=%d (RGRP=%d)  W4/sub=%d\n",
              mn, MM, N, K, TM, TN, NG, RGRP, W4_PER_SUB);
#if KMODE == 32
  std::printf("  DOT32: 2 DPAS/pair/sub (dot32 full-K32 + dot16 correction). DPAS/tile=%d\n",
              NG * NSUB * 2);
#else
  std::printf("  %s: 1 DPAS/group/sub. DPAS/tile=%d\n", mn, NG * NSUB);
#endif
  if (MM % TM || N % TN) { std::printf("bad dims\n"); return 2; }
  const int GM = MM / TM, GN = N / TN, NSG = N / N16;

  std::mt19937 rng(777);
  // weight [K][N] E2M1*2 codes
  std::vector<int8_t> w_s8((size_t)K * N);
  { std::uniform_int_distribution<int> dc(0,7), ds(0,1);
    for (auto& v : w_s8){ int mg=E2M1x2[dc(rng)]; v=(int8_t)((ds(rng)&&mg)?-mg:mg);} }
  auto to_bf16=[](float f){uint32_t u;std::memcpy(&u,&f,4);u=(u+0x8000u)&0xFFFF0000u;float r;std::memcpy(&r,&u,4);return r;};
  // per-16-K group scales [320][N] (bf16-rounded)
  const int NBLK16 = K / GRP;   // 320
  std::vector<float> gsrc((size_t)NBLK16 * N);
  { std::uniform_real_distribution<float> dg(0.01f,0.5f); for (auto& v: gsrc) v=to_bf16(dg(rng)); }
  std::vector<float> gpc(N); for (int n=0;n<N;++n) gpc[n]=gsrc[n];   // PC representative
  // activations, per-token int8 quant
  std::vector<float> xf((size_t)MM*K); { std::normal_distribution<float> dx(0.f,0.1f); for(auto&v:xf)v=dx(rng);}
  std::vector<float> act_scale(MM); std::vector<int8_t> a_s8((size_t)MM*K);
  for (int m=0;m<MM;++m){ float amax=1e-8f; for(int k=0;k<K;++k)amax=std::max(amax,std::fabs(xf[m*K+k]));
    float s=amax/127.f; act_scale[m]=s; for(int k=0;k<K;++k){int qv=(int)std::lround(xf[m*K+k]/s); a_s8[m*K+k]=(int8_t)std::max(-127,std::min(127,qv));}}

  // ---- pack A: per (mtile, outer-group) [TM x RGRP] VNNI, dwords beyond RGRP zeroed ----
  std::vector<int32_t> A((size_t)GM * NG * A_INTS, 0);
  for (int mt=0; mt<GM; ++mt)
    for (int gp=0; gp<NG; ++gp) {
      int32_t* d = &A[((size_t)mt*NG+gp)*A_INTS];
      const int kd = DPK/4;   // 8 dwords per row (always full-32 stride)
      for (int m=0;m<TM;++m)
        for (int kk=0; kk<RGRP; ++kk) {
          int k = gp*RGRP + kk;
          int idx = m*kd + (kk/4), sh=(kk%4)*8;
          d[idx] |= (int32_t)((uint32_t)((uint8_t)a_s8[(size_t)(mt*TM+m)*K+k])<<sh);
        }
    }
  // ---- pack W4: per (nsub_global, outer-group) 4-bit VNNI. RGRP K, 16 cols ----
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
#if KMODE == 0
  std::vector<float> G(N); for (int n=0;n<N;++n) G[n]=gpc[n];
#elif KMODE == 16
  std::vector<float> G((size_t)NG * N);              // [group16][n]
  for (int gp=0; gp<NG; ++gp) for (int n=0;n<N;++n) G[(size_t)gp*N+n]=gsrc[(size_t)gp*N+n];
#else  // DOT32: need both scales of each pair -> [pair][2][n], pair=gp
  std::vector<float> G((size_t)NG * 2 * N);          // [pair][{bg0,bg1}][n]
  for (int p=0; p<NG; ++p) for (int n=0;n<N;++n){
    G[((size_t)p*2+0)*N+n]=gsrc[(size_t)(2*p+0)*N+n];   // bg0
    G[((size_t)p*2+1)*N+n]=gsrc[(size_t)(2*p+1)*N+n];   // bg1
  }
#endif

  int32_t*  dA = sycl::malloc_device<int32_t>(A.size(), q);
  uint16_t* dW = sycl::malloc_device<uint16_t>(W.size(), q);
  float*    dG = sycl::malloc_device<float>(G.size(), q);
  float*    dS = sycl::malloc_device<float>(MM, q);
  bf16*     dY = sycl::malloc_device<bf16>((size_t)MM*N, q);
  q.copy(A.data(),dA,A.size()); q.copy(W.data(),dW,W.size());
  q.copy(G.data(),dG,G.size()); q.copy(act_scale.data(),dS,MM).wait();

  const int TOTAL = GM * GN;
  // upper-16-K mask for the DOT32 correction A16 (lanes with (idx%8)>=4 -> 0)
  auto launch = [&](){
    return q.submit([&](sycl::handler& h){
      h.parallel_for(sycl::nd_range<1>{sycl::range<1>{(size_t)TOTAL}, sycl::range<1>{16}},
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          const int gid = it.get_global_id(0);
          const int mt = gid / GN, tn = gid % GN;
          esimd::simd<float, NSUB*C_INTS> acc = 0.f;

#if KMODE == 32
          // low-16-K keep mask over the 64-dword A tile: keep dwords where (i%8)<4
          esimd::simd<uint32_t, A_INTS> lane(0, 1);
          esimd::simd_mask<A_INTS> keeplo = (lane & 7) < 4;
#endif
          for (int gp=0; gp<NG; ++gp) {
            esimd::simd<int32_t,A_INTS> a; a.copy_from(dA + ((size_t)mt*NG+gp)*A_INTS);
#if KMODE == 32
            esimd::simd<int32_t,A_INTS> a16 = 0; a16.merge(a, keeplo);   // upper-16-K zeroed
            esimd::simd<float,TN> bg0v; bg0v.copy_from(dG + ((size_t)gp*2+0)*N + (size_t)tn*TN);
            esimd::simd<float,TN> bg1v; bg1v.copy_from(dG + ((size_t)gp*2+1)*N + (size_t)tn*TN);
#elif KMODE == 16
            esimd::simd<float,TN> gvec; gvec.copy_from(dG + (size_t)gp*N + (size_t)tn*TN);
#endif
            #pragma unroll
            for (int sub=0; sub<NSUB; ++sub) {
              int ns = tn*NSUB + sub;
              esimd::simd<uint16_t,W4_PER_SUB> w4; w4.copy_from(dW + ((size_t)ns*NG+gp)*W4_PER_SUB);
              esimd::simd<int32_t,W4_PER_SUB> w4i = w4;
              esimd::simd<int32_t,B_INTS> bb = 0;
#if KMODE == 16
              // 16 real K -> 64 dwords, upper 16 K zero
              esimd::simd<int32_t,W4_PER_SUB> s0=decode_e2m1x2<W4_PER_SUB>( w4i     &0xF);
              esimd::simd<int32_t,W4_PER_SUB> s1=decode_e2m1x2<W4_PER_SUB>((w4i>>4) &0xF);
              esimd::simd<int32_t,W4_PER_SUB> s2=decode_e2m1x2<W4_PER_SUB>((w4i>>8) &0xF);
              esimd::simd<int32_t,W4_PER_SUB> s3=decode_e2m1x2<W4_PER_SUB>((w4i>>12)&0xF);
              bb.select<W4_PER_SUB,1>(0) = (s0&0xFF)|((s1&0xFF)<<8)|((s2&0xFF)<<16)|((s3&0xFF)<<24);
#else
              // full 32 K -> 128 dwords
              esimd::simd<int32_t,B_INTS> s0=decode_e2m1x2<B_INTS>( w4i     &0xF);
              esimd::simd<int32_t,B_INTS> s1=decode_e2m1x2<B_INTS>((w4i>>4) &0xF);
              esimd::simd<int32_t,B_INTS> s2=decode_e2m1x2<B_INTS>((w4i>>8) &0xF);
              esimd::simd<int32_t,B_INTS> s3=decode_e2m1x2<B_INTS>((w4i>>12)&0xF);
              bb = (s0&0xFF)|((s1&0xFF)<<8)|((s2&0xFF)<<16)|((s3&0xFF)<<24);
#endif
              esimd::simd<int32_t,C_INTS> z=0;
#if KMODE == 32
              // dot32 (full efficiency) + dot16_g0 correction (A16 upper zero)
              esimd::simd<int32_t,C_INTS> r32 = DP(z, bb, a);
              esimd::simd<int32_t,C_INTS> r16 = DP(z, bb, a16);
              esimd::simd<float,N16> bg0 = bg0v.select<N16,1>(sub*N16);
              esimd::simd<float,N16> bg1 = bg1v.select<N16,1>(sub*N16);
              esimd::simd<float,C_INTS> bg1r = bg1.replicate<TM>();
              esimd::simd<float,C_INTS> dltr = (bg0 - bg1).replicate<TM>();
              esimd::simd<float,C_INTS> rf = bg1r * r32 + dltr * r16;
#elif KMODE == 16
              esimd::simd<int32_t,C_INTS> r = DP(z, bb, a);
              esimd::simd<float,N16> gsub = gvec.select<N16,1>(sub*N16);
              esimd::simd<float,C_INTS> grow = gsub.replicate<TM>();
              esimd::simd<float,C_INTS> rf = grow * r;
#else
              esimd::simd<int32_t,C_INTS> r = DP(z, bb, a);
              esimd::simd<float,C_INTS> rf = r;   // raw s32, scale once at end
#endif
              acc.select<C_INTS,1>(sub*C_INTS) += rf;
            }
          }
          esimd::simd<float,TM> as; as.copy_from(dS + (size_t)mt*TM);
#if KMODE == 0
          esimd::simd<float,TN> gpcv; gpcv.copy_from(dG + (size_t)tn*TN);
#endif
          #pragma unroll
          for (int sub=0; sub<NSUB; ++sub) {
            esimd::simd<float,C_INTS> a4 = acc.select<C_INTS,1>(sub*C_INTS);
#if KMODE == 0
            esimd::simd<float,N16> gsub = gpcv.select<N16,1>(sub*N16);
            a4 *= gsub.replicate<TM>();
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

  launch().wait();  // warmup + fill dY for correctness

  // ---- correctness: host fp64 ref for tile (0,0) = rows 0..7, cols 0..63 ----
  std::vector<bf16> Y((size_t)MM*N);
  q.copy(dY, Y.data(), (size_t)MM*N).wait();
  double maxrel=0, refabs=0; int nchk=0, exact=0;
  for (int m=0;m<TM;++m) for (int col=0; col<TN; ++col) {
    double da=0.0;
#if KMODE == 0
    { int64_t ip=0; for(int k=0;k<K;++k) ip+=(int64_t)a_s8[m*K+k]*(int64_t)w_s8[(size_t)k*N+col];
      da = (double)gpc[col]*(double)ip*(double)act_scale[m]; }
#else
    // exact per-16-block reference (both BS and DOT32 must reproduce this)
    for (int b=0;b<NBLK16;++b){ int32_t ip=0; for(int kk=0;kk<GRP;++kk){int k=b*GRP+kk; ip+=(int32_t)a_s8[m*K+k]*(int32_t)w_s8[(size_t)k*N+col];}
      da += (double)gsrc[(size_t)b*N+col]*(double)ip; }
    da *= act_scale[m];
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
  std::printf("RESULT_LINE mode=%s M=%d ms=%.4f relerr=%.3e\n", mn, MM, best*1e3, relerr);

  sycl::free(dA,q);sycl::free(dW,q);sycl::free(dG,q);sycl::free(dS,q);sycl::free(dY,q);
  return relerr<5e-3 ? 0 : 1;
}
