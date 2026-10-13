#pragma once
// Default OFF. Original typed weights, decoder F32->Q8_1/MMVQ and AB primitive.
// Does NOT change whole-T GDN conv/recurrence or claim their numerical parity.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/sycl_queue.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::kernels::pf_gdn_decoder_proj {
inline bool enabled(){static const bool on=[](){const char* e=std::getenv("STRATA_PF_GDN_DECODER_PROJ");if(!e || !std::strcmp(e,"0"))return false;if(!std::strcmp(e,"1"))return true;throw std::runtime_error("STRATA_PF_GDN_DECODER_PROJ must be absent/0/1");}();return on;}
inline constexpr size_t scratch_bytes=8u*(6144u/32u)*36u;
struct Weights {
 int qkv_type=-1,z_type=-1,out_type=-1;
 const void *qkv=nullptr,*z=nullptr,*out=nullptr;
 const uint16_t *alpha=nullptr,*beta=nullptr;
 const float *dt=nullptr,*ssm_a=nullptr;
};
inline void validate(const Weights&w,int64_t T,void*scratch,size_t bytes,void*stream){
 if(T<1 || T>4096 || !scratch || bytes<scratch_bytes || w.qkv_type!=8 || w.z_type!=8 || !native_mmvq_supported(w.out_type) || !native_mmvq_multi_exact() || !w.qkv || !w.z || !w.out || !w.alpha || !w.beta || !w.dt || !w.ssm_a)throw std::runtime_error("Prefill decoder projection exact original Q8_0 QKV/Z, supported output, BF16 AB and bounded scratch required");
 auto* q=strata::q_of(stream);if(!q || !q->has_property<sycl::property::queue::in_order>())throw std::runtime_error("Prefill decoder projection ordered owner stream required");
}
inline void input(const Weights&w,const float*mixed,float*qkv,float*z,float*decay,float*beta,int64_t T,void*scratch,size_t bytes,void*stream){
 validate(w,T,scratch,bytes,stream);if(!mixed || !qkv || !z || !decay || !beta)throw std::runtime_error("Prefill decoder projection complete input/output required");
 for(int64_t t=0;t<T;t+=8){const int n=int(T-t<8?T-t:8);
  native_quantize_q8_1(mixed+t*2560,scratch,2560,n,stream);
  native_mmvq(w.qkv_type,w.qkv,scratch,qkv+t*10240,2560,10240,n,stream);
  native_mmvq(w.z_type,w.z,scratch,z+t*6144,2560,6144,n,stream);
  // The exact decoder reduction/gate routine consumes F32 mixed, not BF16 hi/lo.
  gdn_ab_multi(mixed+t*2560,w.alpha,w.beta,w.dt,w.ssm_a,decay+t*48,beta+t*48,2560,48,n,stream);
 }
}
inline void output(const Weights&w,const float*normalized_gated_y,float*block,int64_t T,void*scratch,size_t bytes,void*stream){
 validate(w,T,scratch,bytes,stream);if(!normalized_gated_y || !block)throw std::runtime_error("Prefill decoder projection normalized+gated F32 output required");
 // F32 y is [T,6144]. Existing ld16 pads ONLY the unused half precision view.
 for(int64_t t=0;t<T;t+=8){const int n=int(T-t<8?T-t:8);
  native_quantize_q8_1(normalized_gated_y+t*6144,scratch,6144,n,stream);
  native_mmvq(w.out_type,w.out,scratch,block+t*2560,6144,2560,n,stream);
 }
}
}
