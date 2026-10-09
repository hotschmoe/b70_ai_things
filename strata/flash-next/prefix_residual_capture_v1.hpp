#pragma once
#include "strata/core/prefix_residual_capture_contract.hpp"
#include "strata/core/fidelity_observer.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <string>
namespace strata::core::prefix_residual30 {
struct config {
 bool enabled=false;const char* dir=nullptr;const char* binding=nullptr;
 config(){const char*p=std::getenv("STRATA_PREFIX30");enabled=p&&!std::strcmp(p,"1");if(enabled){
  dir=std::getenv("STRATA_PREFIX30_DIR");binding=std::getenv("STRATA_PREFIX30_BINDING_SHA256");struct stat st;
  const char*native=std::getenv("STRATA_SYCL_NATIVE_HC");
  if(!fidelity_diag::settings().enabled||!fidelity_diag::settings().activations||!native||std::strcmp(native,"1")||!dir||dir[0]!='/'||stat(dir,&st)||!S_ISDIR(st.st_mode)||!binding||std::strlen(binding)!=64)throw std::invalid_argument("P30 requires nativeHC/SFDactivations/existingabsoluteDIR/binding64hex");
  for(int i=0;i<64;++i)if(!std::isxdigit(static_cast<unsigned char>(binding[i])))throw std::invalid_argument("P30 binding nothex");
 }}
};
inline const config& settings(){static const config value;return value;}
inline bool eligible(){const auto&r=fidelity_diag::current();return settings().enabled&&r.active&&r.resume==0&&r.ordinal<=4&&r.ids.size()>=1&&r.ids.size()<=8;}
class snapshot {
 sycl::queue&q_;int stage_,lb_,le_;std::uint8_t*data_=nullptr;std::size_t bytes_;std::vector<bool>roster_;std::uint64_t nonce_=0,zero_=0;int rows_=0;std::int64_t p0_=0;std::string route_;
 std::size_t offset(int layer,int phase)const{return (std::size_t(phase)*(le_-lb_)+std::size_t(layer-lb_))*max_rows*row_floats*4;}
 public:
 snapshot(sycl::queue&q,int stage,int lb,int le):q_(q),stage_(stage),lb_(lb),le_(le),bytes_(stage_bytes(lb,le)),roster_(bytes_/(max_rows*row_floats*4),false){
  if(lb<0||lb>=le||le>layers)throw std::invalid_argument("P30 stagebounds");
  data_=static_cast<std::uint8_t*>(sycl::malloc_device(bytes_+16,q_));if(!data_)throw std::bad_alloc();
  std::fprintf(stderr,"P30 allocation pid=%ld stage=%d lb=%d le=%d pointer=%p bytes=%zu\n",long(getpid()),stage_,lb_,le_,(void*)data_,bytes_+16);
 }
 ~snapshot(){try{release();}catch(...){}}
 void release(){if(!data_)return;q_.wait_and_throw();std::fprintf(stderr,"P30 release_begin pid=%ld stage=%d pointer=%p bytes=%zu\n",long(getpid()),stage_,(void*)data_,bytes_+16);sycl::free(data_,q_);data_=nullptr;std::fprintf(stderr,"P30 release_returned pid=%ld stage=%d\n",long(getpid()),stage_);}
 bool begin(const char*route,std::int64_t p0,int rows,std::string&err){
  if(!eligible())return false;const auto&r=fidelity_diag::current();
  if(!route||(std::strcmp(route,"prefill")&&std::strcmp(route,"verifier"))||rows<1||rows>8||p0<0||p0+rows>std::int64_t(r.ids.size())||(std::strcmp(route,"verifier")==0&&(rows!=1||p0!=std::int64_t(r.ids.size())-1))){err="P30 source token/matrix boundary";return false;}
  route_=route;p0_=p0;rows_=rows;nonce_=(std::uint64_t(std::uint32_t(getpid()))<<32)|r.ordinal;
  if(route_=="prefill")std::fill(roster_.begin(),roster_.end(),false);
  q_.memcpy(data_+bytes_,&nonce_,8);q_.memcpy(data_+bytes_+8,&zero_,8);return true;
 }
 void copy(const char*phase,int layer,const float*source,int rows,sycl::queue&q){
  const int index=phase_index(phase);if(layer<lb_||layer>=le_||!source||&q!=&q_)throw std::invalid_argument("P30 source/stagequeue mismatch");
  roster_[std::size_t(index)*(le_-lb_)+std::size_t(layer-lb_)]=true;q.memcpy(data_+offset(layer,index),source,extent(rows));
 }
 void stamp(sycl::queue&q){if(&q!=&q_||std::find(roster_.begin(),roster_.end(),false)!=roster_.end())throw std::invalid_argument("P30 incomplete original3phase roster");q.memcpy(data_+bytes_+8,data_+bytes_,8);}
 bool dump(std::string&err){
  q_.wait_and_throw();std::uint64_t marker=0;q_.memcpy(&marker,data_+bytes_+8,8).wait_and_throw();if(marker!=nonce_||!nonce_){err="P30 current replay marker absent";return false;}
  const auto&r=fidelity_diag::current();char stem[256];std::snprintf(stem,sizeof stem,"p30-%ld-r%u-s%d-%s-p%lld-n%d",long(getpid()),r.ordinal,stage_,route_.c_str(),(long long)p0_,rows_);std::string base=std::string(settings().dir)+"/"+stem;
  int mfd=::open((base+".json").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(mfd<0){err="P30 metadata exclusiveopen";return false;}FILE*meta=::fdopen(mfd,"w");if(!meta){::close(mfd);err="P30 metadata fdopen";return false;}
  std::fprintf(meta,"{\"schema\":1,\"pid\":%ld,\"request\":%u,\"stage\":%d,\"lb\":%d,\"le\":%d,\"route\":\"%s\",\"native_hc_source\":true,\"first_position\":%lld,\"rows\":%d,\"row_floats\":10240,\"source_extent_bytes\":%zu,\"nonce\":%llu,\"binding_sha256\":\"%s\",\"gen_ids\":[",long(getpid()),r.ordinal,stage_,lb_,le_,route_.c_str(),(long long)p0_,rows_,extent(rows_),(unsigned long long)marker,settings().binding);
  for(std::size_t i=0;i<r.ids.size();++i)std::fprintf(meta,"%s%lld",i?",":"",(long long)r.ids[i]);std::fprintf(meta,"],\"fields\":[");bool ok=true;int count=0;const char*names[]={"input","attention","ffn"};
  for(int phase=0;phase<3&&ok;++phase)for(int layer=lb_;layer<le_;++layer){std::vector<std::uint8_t>raw(extent(rows_));q_.memcpy(raw.data(),data_+offset(layer,phase),raw.size()).wait_and_throw();std::string path=base+"-l"+std::to_string(layer)+"-"+names[phase]+".f32";int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0){ok=false;break;}std::size_t done=0;while(done<raw.size()){ssize_t n=::write(fd,raw.data()+done,raw.size()-done);if(n>0)done+=std::size_t(n);else if(n<0&&errno==EINTR)continue;else{ok=false;break;}}if(::close(fd))ok=false;std::fprintf(meta,"%s{\"layer\":%d,\"phase\":\"%s\",\"bytes\":%zu,\"file\":\"%s\",\"encoding\":\"LE_F32[rows,4,2560]\"}",count++?",":"",layer,names[phase],raw.size(),path.c_str());}
  std::fprintf(meta,"],\"full_model_math_qualified\":false}\n");if(::fclose(meta))ok=false;if(!ok){err="P30 fieldpublication failed";return false;}std::fprintf(stderr,"P30 frame pid=%ld request=%u stage=%d route=%s rows=%d p0=%lld metadata=%s.json\n",long(getpid()),r.ordinal,stage_,route_.c_str(),rows_,(long long)p0_,base.c_str());return true;
 }
};
}
