// Opt-in CPU DSO. Frozen dot32 body is reused; no SDK/model/device dependency.
#define main hc35_unused_scalar_main
#include "hc_f32_arithmetic35_host_v1.cpp"
#undef main
namespace {
bool environment_ok() noexcept {
 if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__)
 if(_mm_getcsr()&((unsigned(1)<<15)|(unsigned(1)<<6)))return false;
#else
 return false;
#endif
 volatile float small=std::ldexp(1.0f,-126),sub=std::ldexp(1.0f,-149);
 return ::fmaf(small,0.5f,0.0f)==std::ldexp(1.0f,-127)&&::fmaf(sub,std::ldexp(1.0f,24),0.0f)==std::ldexp(1.0f,-125);
}
}
extern "C" unsigned hc_rows_dot32_abi_v1() noexcept {return 1;}
extern "C" int hc_rows_dot32_v1(const float* weights,const float* input,float* output,
                               uint32_t K,uint32_t M,int paired) noexcept {
 try {
  if(!weights||!input||!output||K<32||K>10240||K%32||M<1||M>10240||size_t(M)*K*4>(64u<<20)||(paired!=0&&paired!=1))return 1;
  if(!environment_ok())return 2;
  for(size_t i=0;i<size_t(M)*K;++i)if(!std::isfinite(weights[i]))return 3;
  for(size_t i=0;i<(paired?size_t(M)*K:K);++i)if(!std::isfinite(input[i]))return 3;
  std::vector<float> w(K),x(K);
  if(!paired)std::copy(input,input+K,x.begin());
  for(size_t row=0;row<M;++row){
   std::copy(weights+row*K,weights+(row+1)*K,w.begin());
   if(paired)std::copy(input+row*K,input+(row+1)*K,x.begin());
   output[row]=dot32(w,x);if(!std::isfinite(output[row]))return 3;
  }
  return environment_ok()?0:2;
 }catch(...){return 4;}
}
