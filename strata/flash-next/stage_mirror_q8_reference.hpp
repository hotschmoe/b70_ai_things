// Independent host Q8_1 representation contract for stage mirror arithmetic.
// No SYCL operations or calls to backend quantization helpers.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace mirror_reference {
inline float half_value(uint16_t u){int e=(u>>10)&31,f=u&1023;if(e==31) throw std::runtime_error("nonfinite half reference");return float((u&0x8000 ? -1.0:1.0)*std::ldexp(double(e ? 1024+f:f),e ? e-25:-24));}
inline uint16_t half_bits(float f){uint32_t x;std::memcpy(&x,&f,4);uint16_t sign=(x>>16)&0x8000;int e=int((x>>23)&255)-112;uint32_t m=x&0x7fffff,h;if(e<=0){if(e < -10) return sign;m|=0x800000;unsigned shift=unsigned(14-e);h=m>>shift;uint32_t rem=m&((1u<<shift)-1),mid=1u<<(shift-1);if(rem>mid || (rem==mid && (h&1))) ++h;}else{if(e>=31) throw std::runtime_error("nonfinite half conversion reference");h=(uint32_t(e)<<10)|(m>>13);uint32_t rem=m&8191;if(rem>4096 || (rem==4096 && (h&1))) ++h;}return uint16_t(sign|h);}
inline uint16_t load_half(const uint8_t* p){return uint16_t(p[0])|(uint16_t(p[1])<<8);}
inline std::vector<uint8_t> q8_1(const std::vector<float>& input){if(input.size()%32) throw std::runtime_error("Q8 reference block extent");std::vector<uint8_t> out(input.size()/32*36);for(size_t b=0;b<input.size()/32;++b){float amax=0,lanes[32];for(int l=0;l<32;++l){float x=input[b*32+l];if(!std::isfinite(x)) throw std::runtime_error("nonfinite Q8 input reference");amax=std::max(amax,std::abs(x));lanes[l]=x;}for(int mask=16;mask;mask>>=1){float old[32];std::memcpy(old,lanes,sizeof(old));for(int l=0;l<32;++l) lanes[l]=old[l]+old[l^mask];}float scale=std::min(65504.0f,amax/127.0f),sum=std::clamp(lanes[0],-65504.0f,65504.0f);uint16_t d=half_bits(scale),s=half_bits(sum);uint8_t* dst=out.data()+b*36;dst[0]=d&255;dst[1]=d>>8;dst[2]=s&255;dst[3]=s>>8;for(int l=0;l<32;++l){float raw=amax==0 ? 0:std::round(input[b*32+l]/scale);int code=int(std::clamp(raw,-127.0f,127.0f));dst[4+l]=uint8_t(code&255);}}return out;}
inline double dot(const float* weights,const uint8_t* original_row,int weight_type,const uint8_t* activation,size_t n){double result=0;for(size_t b=0;b<n/32;++b){const uint8_t* block=activation+b*36;double scale=half_value(load_half(block)),sum_decoded=0;for(size_t l=0;l<32;++l){int raw=block[4+l];double x=scale*(raw<128 ? raw:raw-256);result+=double(weights[b*32+l])*x;sum_decoded+=x;}if(weight_type==6){double delta=half_value(load_half(original_row+b*22));result+=16*delta*(sum_decoded-double(half_value(load_half(block+2))));}}return result;}
struct Metric{double nmse=0,linf=0;bool finite=true;};
inline Metric metric(const std::vector<float>& values,const std::vector<float>& reference){if(values.size()!=reference.size()) throw std::runtime_error("numeric reference extent");Metric m;double e2=0,r2=0,me=0,mr=0;for(size_t i=0;i<values.size();++i){m.finite=m.finite && std::isfinite(values[i]) && std::isfinite(reference[i]);double e=double(values[i])-reference[i];e2+=e*e;r2+=double(reference[i])*reference[i];me=std::max(me,std::abs(e));mr=std::max(mr,std::abs(double(reference[i])));}m.nmse=e2/std::max(1e-30,r2);m.linf=me/std::max(1e-6,mr);return m;}
}
