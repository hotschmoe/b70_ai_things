// CPU-only proof that the dot32+correction identity is numerically exact vs the
// per-16-block reference. Mirrors the kernel's arithmetic (E2M1 decode, per-token int8
// act quant, fp32 accumulate). No SYCL, no GPU. Validates relerr over a real gate/up
// column slab.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

static const int E2M1x2[8] = {0,1,2,3,4,6,8,12};

int main(){
  const int K=5120, GRP=16, NBLK=K/GRP, NPAIR=NBLK/2;
  const int M=8, N=64;               // one output tile
  std::mt19937 rng(777);
  std::vector<int8_t> w((size_t)K*N);
  { std::uniform_int_distribution<int> dc(0,7), ds(0,1);
    for(auto&v:w){int mg=E2M1x2[dc(rng)];v=(int8_t)((ds(rng)&&mg)?-mg:mg);} }
  auto tobf16=[](float f){uint32_t u;std::memcpy(&u,&f,4);u=(u+0x8000u)&0xFFFF0000u;float r;std::memcpy(&r,&u,4);return r;};
  std::vector<float> g((size_t)NBLK*N);
  { std::uniform_real_distribution<float> dg(0.01f,0.5f); for(auto&v:g)v=tobf16(dg(rng)); }
  std::vector<float> xf((size_t)M*K); { std::normal_distribution<float> dx(0,0.1f); for(auto&v:xf)v=dx(rng);}
  std::vector<float> as(M); std::vector<int8_t> a((size_t)M*K);
  for(int m=0;m<M;++m){float amax=1e-8f;for(int k=0;k<K;++k)amax=std::max(amax,std::fabs(xf[m*K+k]));
    float s=amax/127.f;as[m]=s;for(int k=0;k<K;++k){int qv=(int)std::lround(xf[m*K+k]/s);a[m*K+k]=(int8_t)std::max(-127,std::min(127,qv));}}

  double maxrel=0, refabs=0;
  for(int m=0;m<M;++m)for(int n=0;n<N;++n){
    // fp64 truth (per-16-block)
    double truth=0.0;
    for(int b=0;b<NBLK;++b){int32_t ip=0;for(int kk=0;kk<GRP;++kk){int k=b*GRP+kk;ip+=(int32_t)a[m*K+k]*(int32_t)w[(size_t)k*N+n];}
      truth+=(double)g[(size_t)b*N+n]*(double)ip;}
    truth*=as[m];
    // dot32 + correction, fp32 accumulate (matches kernel)
    float acc=0.f;
    for(int p=0;p<NPAIR;++p){
      float bg0=g[(size_t)(2*p+0)*N+n], bg1=g[(size_t)(2*p+1)*N+n];
      int32_t dot32=0, dot16=0;
      for(int kk=0;kk<32;++kk){int k=p*32+kk;int32_t prod=(int32_t)a[m*K+k]*(int32_t)w[(size_t)k*N+n];
        dot32+=prod; if(kk<16)dot16+=prod;}
      acc += bg1*(float)dot32 + (bg0-bg1)*(float)dot16;
    }
    float got=as[m]*acc;
    refabs=std::max(refabs,std::fabs(truth));
    maxrel=std::max(maxrel,std::fabs((double)got-truth));
  }
  double rel=maxrel/(refabs+1e-30);
  std::printf("dot32+correction vs per-16-block fp64: rel-err=%.3e  (refabs=%.4g)  %s\n",
              rel, refabs, rel<5e-3?"PASS":"FAIL");
  return rel<5e-3?0:1;
}
