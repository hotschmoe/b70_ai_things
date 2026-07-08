// bench_penalty.cpp -- COMPUTE-BOUND isolation of the block-scale DPAS penalty.
//
// Both modes do the SAME useful MAC work for a K=5120 tile reduction (4 N-subtiles,
// M=8, N=16 each) with 4-way ILP, register-resident (no memory loads in the hot loop),
// thousands of threads -> throughput-bound. The ONLY difference is the block-scale
// structure that the NVFP4 group=16 forces onto the s8 DPAS (SystolicDepth==8 -> K=32):
//
//   PC (per-channel, no block scale): 160 groups x 4 sub, one full-K=32 s8 DPAS each,
//       accumulate s32 in a chain. 640 DPAS. == oneDNN's per-channel int8 structure.
//   BS (block-scaled, NVFP4 group=16): 320 groups x 4 sub, one s8 DPAS reducing a
//       16-K group each (SystolicDepth still 8 -> half the K slots useful), rescale the
//       s32 partial by an fp32 group scale into an fp32 accumulator. 1280 DPAS + 1280
//       fp32 rescales. This is the price of block<DPAS-K with no native block-scale MMA.
//   BSD = BS + the in-register E2M1 f4->s8 decode on every group (measures decode cost).
//
// Reports ms and the BS/PC (and BSD/PC) ratio = the fundamental compute penalty of
// block-scaling on Xe2 DPAS. Build -DMODE={0 PC,1 BS,2 BSD}.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <algorithm>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

#ifndef MODE
#define MODE 1
#endif
static constexpr int SDEPTH=8, M=8, N=16, DPK=32, GRP=16;
static constexpr int A_INTS=(M*DPK)/4;   // 64
static constexpr int B_INTS=(DPK*N)/4;   // 128
static constexpr int C_INTS=M*N;         // 128
static constexpr int K=5120;
static constexpr int NSUB=4;
#if MODE==0
static constexpr int NG=K/DPK;   // 160 -- PC chained (dependent accumulator)
static const char* MN="PC(per-channel full-K32, chained)";
#elif MODE==3
static constexpr int NG=K/DPK;   // 160 -- PC INDEPENDENT DPAS (true ceiling)
static const char* MN="PCI(per-channel full-K32, independent)";
#else
static constexpr int NG=K/GRP;   // 320
static const char* MN= (MODE==2)?"BSD(block-scaled+decode)":"BS(block-scaled)";
#endif

#ifndef NGROUPS
#define NGROUPS 16384
#endif
#ifndef REPS
#define REPS 8
#endif

template <int W>
static ESIMD_INLINE esimd::simd<int32_t,W> decode_e2m1x2(esimd::simd<int32_t,W> nib){
  esimd::simd<int32_t,W> sign=nib&0x8, idx=nib&0x7, exp=idx>>1, mant=idx&1;
  esimd::simd<int32_t,W> sh=esimd::max(exp-1,esimd::simd<int32_t,W>(0));
  esimd::simd<int32_t,W> mag=(mant+2)<<sh; mag.merge(mant,exp==0);
  esimd::simd<int32_t,W> neg=-mag; mag.merge(neg,sign!=0); return mag;
}

int main(){
  sycl::queue q{sycl::gpu_selector_v};
  auto dev=q.get_device();
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("MODE=%s  NG=%d NSUB=%d  DPAS/thread=%d  NGROUPS=%d REPS=%d\n",
              MN, NG, NSUB, NG*NSUB, (int)NGROUPS, (int)REPS);
  const int WI=NGROUPS;
  float* out=sycl::malloc_device<float>(WI,q);

  auto once=[&](){
    auto t0=std::chrono::high_resolution_clock::now();
    q.submit([&](sycl::handler& h){
      h.parallel_for(sycl::nd_range<1>{sycl::range<1>{(size_t)WI}, sycl::range<1>{64}},
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          int gid=it.get_global_id(0);
          // 4 GENUINELY INDEPENDENT subtiles (distinct weights + acts), like the real
          // kernel -- prevents the compiler from deduplicating the 4 DPAS.
          esimd::simd<int32_t,A_INTS> a0(gid+1),a1(gid+5),a2(gid+9),a3(gid+13);
          esimd::simd<int32_t,B_INTS> b0(gid+3),b1(gid+7),b2(gid+11),b3(gid+17);
#if MODE!=0
          esimd::simd<float,N> gv0(0.03f),gv1(0.05f),gv2(0.02f),gv3(0.04f);
          esimd::simd<float,C_INTS> g0=gv0.replicate<M>(),g1=gv1.replicate<M>(),
                                    g2=gv2.replicate<M>(),g3=gv3.replicate<M>();
          esimd::simd<uint16_t,64> w0((uint16_t)(gid|0x1111)),w1((uint16_t)(gid|0x2222)),
                                   w2((uint16_t)(gid|0x3333)),w3((uint16_t)(gid|0x4444));
#endif
          #define DP(C,B,A) xmx::dpas<SDEPTH,M,int32_t,int32_t,int32_t,int32_t,\
              xmx::dpas_argument_type::s8,xmx::dpas_argument_type::s8>(C,B,A)
#if MODE==0
          // PC: 4 independent chained s32 accumulators, full-K=32 DPAS (no rescale).
          // a_s += 1 each group == reading a new K-slice (faithful + defeats DCE/hoist).
          esimd::simd<int32_t,C_INTS> c0(0),c1(0),c2(0),c3(0);
          for (int gp=0; gp<NG*REPS; ++gp) {
            c0=DP(c0,b0,a0); c1=DP(c1,b1,a1); c2=DP(c2,b2,a2); c3=DP(c3,b3,a3);
            a0+=1; a1+=1; a2+=1; a3+=1;
          }
          out[gid]=(float)(c0[0]^c1[63]^c2[7]^c3[100]);
#elif MODE==3
          // PCI: full-K=32 DPAS but INDEPENDENT (fresh z, sum into s32 acc) -> true
          // per-channel int8 throughput ceiling (no rescale waste, no chain latency).
          esimd::simd<int32_t,C_INTS> f0(0),f1(0),f2(0),f3(0),z(0);
          for (int gp=0; gp<NG*REPS; ++gp) {
            esimd::simd<int32_t,C_INTS> r0=DP(z,b0,a0),r1=DP(z,b1,a1),r2=DP(z,b2,a2),r3=DP(z,b3,a3);
            f0+=r0; f1+=r1; f2+=r2; f3+=r3;
            a0+=1; a1+=1; a2+=1; a3+=1;
          }
          out[gid]=(float)(f0[0]^f1[63]^f2[7]^f3[100]);
#else
          // BS: 4 independent fp32 accumulators, per-group s32 partial * group scale.
          esimd::simd<float,C_INTS> f0(0),f1(0),f2(0),f3(0);
          esimd::simd<int32_t,C_INTS> z(0);
          for (int gp=0; gp<NG*REPS; ++gp) {
            esimd::simd<int32_t,B_INTS> B0=b0,B1=b1,B2=b2,B3=b3;
  #if MODE==2
            // in-register E2M1 decode of 4 distinct 4-bit weight tiles -> s8 VNNI (real cost)
            #define DEC(BB,WW) { esimd::simd<int32_t,64> wi=WW; \
              esimd::simd<int32_t,64> d0=decode_e2m1x2<64>( wi     &0xF), \
                d1=decode_e2m1x2<64>((wi>>4)&0xF), d2=decode_e2m1x2<64>((wi>>8)&0xF), \
                d3=decode_e2m1x2<64>((wi>>12)&0xF); BB=0; \
                BB.select<64,1>(0)=(d0&0xFF)|((d1&0xFF)<<8)|((d2&0xFF)<<16)|((d3&0xFF)<<24); }
            DEC(B0,w0) DEC(B1,w1) DEC(B2,w2) DEC(B3,w3)
            w0+=(uint16_t)1; w1+=(uint16_t)1; w2+=(uint16_t)1; w3+=(uint16_t)1;
  #endif
            esimd::simd<int32_t,C_INTS> r0=DP(z,B0,a0), r1=DP(z,B1,a1),
                                       r2=DP(z,B2,a2), r3=DP(z,B3,a3);
            f0+=g0*r0; f1+=g1*r1; f2+=g2*r2; f3+=g3*r3;
            a0+=1; a1+=1; a2+=1; a3+=1;
          }
          out[gid]=f0[0]+f1[63]+f2[7]+f3[100];
#endif
        });
    }).wait();
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
  };
  once();
  double best=1e30; for(int i=0;i<6;++i) best=std::min(best,once());
  // useful MACs = WI * REPS * NSUB * (per-tile useful K reduction) * M*N
  double useful = (double)WI*REPS*NSUB*(double)M*N*K;
  std::printf("%s: best=%.4f ms  useful TOPS=%.1f  (WIxREPS=%d)\n",
              MN, best*1e3, 2.0*useful/best/1e12, WI*REPS);
  std::printf("PENALTY_LINE mode=%d ms=%.4f\n", MODE, best*1e3);
  sycl::free(out,q); return 0;
}
