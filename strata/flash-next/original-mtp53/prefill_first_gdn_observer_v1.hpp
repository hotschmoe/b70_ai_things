#pragma once
// Default-OFF output-only first-row diagnostic. Extra host copies/waits are diagnostic cost.
#include "strata/sycl_queue.hpp"
#include "strata/core/prefix_lifecycle.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
namespace strata::core::prefill_first_gdn {
inline void observe(int64_t layer,int half,int64_t pos,int64_t T,int64_t n,const int64_t* tokens,int device,bool f16_io,int bf16x2_mode,
 const float* residual,const float* mixed,const uint16_t* f16,const uint16_t* bf16,const uint16_t* bf16_lo,
 const float* qkv,const float* z,const float* ab,const float* decay,const float* beta,const float* state,const float* conv,
 int qkv_type,int64_t qkv_ne0,int64_t qkv_ne1,int z_type,int64_t z_ne0,int64_t z_ne1,void* stream){
 const char* dir=std::getenv("STRATA_PREFILL_FIRST_GDN_DIR");if(!dir || !*dir)return;
 if(layer!=0 || half!=0 || pos!=0)return;
 if(device!=0 || T<1 || T>64 || !(n==53 || n==116) || !tokens || tokens[0]!=248045 || !residual || !mixed || !f16 || !bf16 || !qkv || !z || !ab || !decay || !beta || !state || !conv || qkv_ne0!=2560 || qkv_ne1!=10240 || z_ne0!=2560 || z_ne1!=6144)throw std::runtime_error("First GDN exact source/prefix/shape diagnostic required");
 const char* binding=std::getenv("STRATA_PREFILL_FIRST_GDN_BINDING_SHA256");if(!binding || std::strlen(binding)!=64)throw std::runtime_error("First GDN actual source binding required");for(const char*p=binding;*p;++p)if(!std::strchr("0123456789abcdef",*p))throw std::runtime_error("First GDN binding hex");
 static unsigned frames=0;if(++frames>2)throw std::runtime_error("First GDN bounded two-frame quota");
 namespace fs=std::filesystem;fs::path parent(dir);if(!parent.is_absolute() || fs::canonical(parent)!=parent || fs::is_symlink(parent))throw std::runtime_error("First GDN canonical existing output required");const auto out=parent/(std::to_string(getpid())+"-"+std::to_string(frames));if(!fs::create_directory(out))throw std::runtime_error("First GDN exclusive frame directory");
 auto& queue=*strata::q_of(stream);if(!queue.has_property<sycl::property::queue::in_order>())throw std::runtime_error("First GDN ordered owner queue required");
 std::ostringstream meta;meta<<"{\"schema\":1,\"producer_pid\":"<<getpid()<<",\"frame\":"<<frames<<",\"source_binding_sha256\":\""<<binding<<"\",\"layer\":0,\"position\":0,\"first_token\":248045,\"prefill_n\":"<<n<<",\"chunk_T\":"<<T<<",\"device\":0,\"f16_io\":"<<(f16_io?"true":"false")<<",\"bf16x2_mode\":"<<bf16x2_mode<<",\"BF16_low_present\":"<<(bf16_lo?"true":"false")<<",\"qkv_original_quant_type\":"<<qkv_type<<",\"qkv_shape\":[2560,10240],\"z_original_quant_type\":"<<z_type<<",\"z_shape\":[2560,6144],\"before_any_L0_recurrence\":true,\"before_L0_conv\":true,\"added_diagnostic_host_copies_and_waits\":true,\"weights_payload_exported\":false,\"fields\":[";bool comma=false;
 auto output=[&](const char* name,const void* pointer,size_t bytes,const char* type,const char* shape,bool host=false){if(!pointer)throw std::runtime_error("First GDN complete field missing");std::vector<uint8_t> raw(bytes);if(host)std::memcpy(raw.data(),pointer,bytes);else try{queue.memcpy(raw.data(),pointer,bytes).wait_and_throw();}catch(...){std::fprintf(stderr,"PREFILL_FIRST_GDN_FATAL copy_not_retired=1\n");std::fflush(stderr);std::_Exit(86);}
  int fd=::open((out/(std::string(name)+".bin")).c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("First GDN exclusive raw file");try{size_t at=0;while(at<bytes){ssize_t k=::write(fd,raw.data()+at,bytes-at);if(k<=0)throw std::runtime_error("First GDN raw write");at+=size_t(k);}int rc=::close(fd);fd=-1;if(rc)throw std::runtime_error("First GDN raw close");}catch(...){if(fd>=0)::close(fd);throw;}
  if(std::strstr(name,"before"))for(uint8_t c:raw)if(c!=0)throw std::runtime_error("First GDN requires exact bytezero initial state/conv");
  strata::core::prefix_lifecycle::Sha256 hash;hash.update(raw.data(),raw.size());char digest[65];hash.finish(digest);if(comma)meta<<",";comma=true;meta<<"{\"name\":\""<<name<<"\",\"bytes\":"<<bytes<<",\"dtype\":\""<<type<<"\",\"shape\":"<<shape<<",\"sha256\":\""<<digest<<"\"}";
 };
 output("actual_prefill_input_tokens",tokens,size_t(n)*8,"i64","[]",true);
 output("residual_input",residual,40960,"f32","[10240]");output("attn_mixed",mixed,10240,"f32","[2560]");
 output("attn_mixed_f16",f16,5120,"f16","[2560]");output("attn_mixed_bf16_high",bf16,5120,"bf16","[2560]");if(bf16_lo)output("attn_mixed_bf16_low",bf16_lo,5120,"bf16","[2560]");
 output("gdn_qkv",qkv,40960,"f32","[10240]");output("gdn_z",z,24576,"f32","[6144]");output("gdn_alpha_beta_raw",ab,384,"f32","[2,48]");output("gdn_decay",decay,192,"f32","[48]");output("gdn_beta",beta,192,"f32","[48]");
 output("gdn_state_before",state,3145728,"f32","[128,48,128]");output("gdn_conv_before",conv,122880,"f32","[10240,3]");meta<<"],\"math_qualified\":false,\"speed_qualified\":false}\n";
 const auto text=meta.str();const auto manifest=out/"frame.json";int fd=::open(manifest.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("First GDN exclusive metadata");size_t at=0;while(at<text.size()){ssize_t k=::write(fd,text.data()+at,text.size()-at);if(k<=0){::close(fd);throw std::runtime_error("First GDN metadata write");}at+=size_t(k);}if(::close(fd))throw std::runtime_error("First GDN metadata close");std::fprintf(stderr,"PREFILL_FIRST_GDN_OBSERVED pid=%ld frame=%u n=%lld T=%lld token0=248045 before_conv=1 before_recurrence=1\n",long(getpid()),frames,(long long)n,(long long)T);
}
}
