#pragma once
// Output-only source candidate observer. No raw AB fabricated for direct decoder AB.
#include "strata/sycl_queue.hpp"
#include "strata/core/prefix_lifecycle.hpp"
#include "strata/kernels/prefill_gdn_decoder_projection.hpp"
#include <filesystem>
#include <sstream>
#include <vector>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
namespace strata::core::prefill_first_gdn_candidate {
struct Settings {const char*dir=nullptr;const char*binding=nullptr;Settings(){dir=std::getenv("STRATA_PREFILL_FIRST_GDN_DIR");binding=std::getenv("STRATA_PREFILL_FIRST_GDN_BINDING_SHA256");}bool enabled()const{return dir&&*dir;}};
inline const Settings& settings(){static const Settings s;return s;}
class Frame final {
 std::filesystem::path path_;std::ostringstream json_;bool comma_=false,candidate_;int n_,T_,pid_;unsigned frame_;size_t total_=0;
public:
 const float *R=nullptr,*mixed=nullptr,*qkv=nullptr,*z=nullptr,*ab=nullptr,*decay=nullptr,*beta=nullptr,*state=nullptr,*conv=nullptr,*y=nullptr,*bo=nullptr;
 const uint16_t*f16=nullptr,*bf16=nullptr,*bf16_lo=nullptr,*y_h=nullptr;
 Frame(int n,int T,const int64_t*tokens,bool f16_io,int bf16x2,const strata::kernels::pf_gdn_decoder_proj::Weights&w,bool candidate,bool low_present):candidate_(candidate),n_(n),T_(T),pid_(int(getpid())){
  static unsigned frames=0;frame_=++frames;const auto&s=settings();if(!s.enabled() || !s.binding || std::strlen(s.binding)!=64 || !tokens || tokens[0]!=248045 || frame_>2 || !(n==53||n==116) || T<1||T>64)throw std::runtime_error("First candidate GDN source/first-token/route required");for(const char*p=s.binding;*p;++p)if(!std::strchr("0123456789abcdef",*p))throw std::runtime_error("First candidate binding hex");
  namespace fs=std::filesystem;fs::path root(s.dir);if(!root.is_absolute()||fs::canonical(root)!=root||fs::is_symlink(root))throw std::runtime_error("First candidate canonical owned output required");path_=root/(std::to_string(pid_)+"-"+std::to_string(frame_));if(!fs::create_directory(path_))throw std::runtime_error("First candidate exclusive frame");
  json_<<"{\"schema\":2,\"producer_pid\":"<<pid_<<",\"frame\":"<<frame_<<",\"source_binding_sha256\":\""<<s.binding<<"\",\"layer\":0,\"position\":0,\"first_token\":248045,\"prefill_n\":"<<n<<",\"chunk_T\":"<<T<<",\"device\":0,\"candidate_projection\":"<<(candidate?"true":"false")<<",\"f16_io\":"<<(f16_io?"true":"false")<<",\"bf16x2_mode\":"<<bf16x2<<",\"timing_flags_apply_to\":\"input_fields_only\",\"before_L0_conv\":true,\"before_any_L0_recurrence\":true,\"BF16_low_present\":"<<(low_present?"true":"false")<<",\"raw_AB_observed\":"<<(candidate?"false":"true")<<",\"F32_normalized_gated_y_observed\":"<<(candidate?"true":"false")<<",\"qkv_original_quant_type\":"<<w.qkv_type<<",\"z_original_quant_type\":"<<w.z_type<<",\"out_original_quant_type\":"<<w.out_type<<",\"Q81_packet_scope\":\"first row of first <=8 projection chunk BEFORE scratch reuse\",\"added_host_copies_waits\":true,\"fields\":[";
  field("actual_prefill_input_tokens",tokens,size_t(n)*8,"i64",true,nullptr);
 }
 bool candidate()const{return candidate_;}
 void field(const char*name,const void*ptr,size_t bytes,const char*type,bool host,void*stream){
  if(!ptr||bytes>(4u<<20)||total_+bytes>(8u<<20))throw std::runtime_error("First candidate exact bounded field required");total_+=bytes;std::vector<uint8_t>raw(bytes);
  if(host)std::memcpy(raw.data(),ptr,bytes);else try{strata::q_of(stream)->memcpy(raw.data(),ptr,bytes).wait_and_throw();}catch(...){std::fprintf(stderr,"PREFILL_FIRST_GDN_FATAL unretired_copy=1\n");std::fflush(stderr);std::_Exit(86);}
  if(std::strstr(name,"before"))for(uint8_t b:raw)if(b)throw std::runtime_error("First candidate bytezero state/conv required");
  save(name,raw);strata::core::prefix_lifecycle::Sha256 hash;hash.update(raw.data(),raw.size());char digest[65];hash.finish(digest);const bool after=std::strstr(name,"output") || !std::strcmp(name,"gdn_block_output") || !std::strcmp(name,"gdn_normalized_gated_y");const char* point=after?"after_recurrence_and_postnorm":(!std::strcmp(name,"gdn_normalized_qkv")?"after_conv_before_recurrence":"before_conv_and_recurrence");if(comma_)json_<<",";comma_=true;json_<<"{\"name\":\""<<name<<"\",\"bytes\":"<<bytes<<",\"dtype\":\""<<type<<"\",\"point\":\""<<point<<"\",\"sha256\":\""<<digest<<"\"}";
 }
 void save(const char*name,const std::vector<uint8_t>&raw){
  int fd=::open((path_/(std::string(name)+".bin")).c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("First candidate exclusive raw");try{size_t at=0;while(at<raw.size()){ssize_t n=::write(fd,raw.data()+at,raw.size()-at);if(n<=0)throw std::runtime_error("First candidate raw write");at+=size_t(n);}int rc=::close(fd);fd=-1;if(rc)throw std::runtime_error("First candidate raw close");}catch(...){if(fd>=0)::close(fd);throw;}
 }
 void before(void*stream,const void*input_q81){
  field("residual_input",R,40960,"f32",false,stream);field("attn_mixed",mixed,10240,"f32",false,stream);
  field("attn_mixed_f16",f16,5120,"f16_VIEW",false,stream);field("attn_mixed_bf16_high",bf16,5120,"bf16_VIEW",false,stream);if(bf16_lo)field("attn_mixed_bf16_low",bf16_lo,5120,"bf16_VIEW",false,stream);
  if(candidate_)field("attn_input_q81",input_q81,2880,"Q8_1_LE[80,36]",false,stream);else field("gdn_alpha_beta_raw",ab,384,"f32[alpha48,beta48]",false,stream);
  field("gdn_qkv",qkv,40960,"f32",false,stream);field("gdn_z",z,24576,"f32",false,stream);field("gdn_decay",decay,192,"f32",false,stream);field("gdn_beta",beta,192,"f32",false,stream);
  field("gdn_state_before",state,3145728,"f32",false,stream);field("gdn_conv_before",conv,122880,"f32",false,stream);
 }
 void after(void*stream,const void*output_q81){
  if(candidate_){field("gdn_normalized_gated_y",y,24576,"f32",false,stream);field("gdn_output_q81",output_q81,6912,"Q8_1_LE[192,36]",false,stream);}
  field("gdn_output_f16",y_h,12288,"f16_VIEW",false,stream);field("gdn_block_output",bo,10240,"f32",false,stream);
 }
 void finish(){json_<<"],\"bytes\":"<<total_<<",\"complete\":true,\"math_qualified\":false,\"speed_qualified\":false}\n";const std::string text=json_.str();const auto p=path_/"frame.json";int fd=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("First candidate exclusive metadata");size_t at=0;while(at<text.size()){ssize_t n=::write(fd,text.data()+at,text.size()-at);if(n<=0){::close(fd);throw std::runtime_error("First candidate metadata write");}at+=size_t(n);}if(::close(fd))throw std::runtime_error("First candidate metadata close");std::fprintf(stderr,"PREFILL_FIRST_GDN_CANDIDATE_OBSERVED pid=%d frame=%u n=%d T=%d candidate=%d raw_AB_observed=%d complete=1\n",pid_,frame_,n_,T_,candidate_?1:0,candidate_?0:1);}
 static void input_hook(void*user,const void*packet,int,void*stream){static_cast<Frame*>(user)->before(stream,packet);}
 static void output_hook(void*user,const void*packet,int,void*stream){static_cast<Frame*>(user)->after(stream,packet);}
};
}
