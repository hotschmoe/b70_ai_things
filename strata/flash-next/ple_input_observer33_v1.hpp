#pragma once
// Bounded host observation only. Never retain borrowed row/input pointers.
#include "strata/core/prefix_lifecycle.hpp"
#include <cmath>
#include <array>
#include <cctype>
#include <cerrno>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
namespace strata::core::ple_input33 {
struct config {
 bool enabled=false;const char*dir=nullptr;const char*binding=nullptr;
 config(){const char*p=std::getenv("STRATA_PLE_INPUT33");if(!p||!*p||!std::strcmp(p,"0"))return;
  if(std::strcmp(p,"1"))throw std::invalid_argument("PLE_INPUT33 flag0|1");enabled=true;
  if(std::getenv("STRATA_VERIFY_EAGER"))throw std::invalid_argument("PLE_INPUT33 requires actual graph submission, notEAGER");
  dir=std::getenv("STRATA_PLE_INPUT33_DIR");binding=std::getenv("STRATA_PLE_INPUT33_BINDING_SHA256");
  struct stat st;if(!dir||dir[0]!='/'||lstat(dir,&st)||!S_ISDIR(st.st_mode)||!binding||std::strlen(binding)!=64)throw std::invalid_argument("PLE_INPUT33 absolute nonsymlinkDIR/binding64hex required");
  for(int i=0;i<64;++i)if(!std::isxdigit(static_cast<unsigned char>(binding[i])))throw std::invalid_argument("PLE_INPUT33 binding hex required");
 }
};
inline const config&settings(){static const config value;return value;}
struct publication {uint64_t epoch=0;unsigned request=0;int stage=0,device=0,T=0;int64_t pos=0;};
struct observation{publication p;bool submitted=false,terminal=false;};
inline std::array<observation,4>&observations(){static std::array<observation,4> value{};return value;}
inline void write_bytes(int dirfd,const char*name,const void*data,size_t bytes){
 int fd=openat(dirfd,name,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);if(fd<0)throw std::runtime_error("PLE_INPUT33 exclusive raw open failed");
 const auto*p=static_cast<const uint8_t*>(data);size_t done=0;while(done<bytes){const auto n=::write(fd,p+done,bytes-done);if(n>0)done+=size_t(n);else if(n<0&&errno==EINTR)continue;else{::close(fd);throw std::runtime_error("PLE_INPUT33 raw write failed");}}if(::close(fd))throw std::runtime_error("PLE_INPUT33 raw close failed");
}
inline publication publish(bool active,unsigned ordinal,int64_t reused,bool SFD,bool activations,bool exact,
 int stage,int device,int lb,int le,int T,int64_t pos,const int64_t*ids,size_t nids,const int32_t*tokens,const int32_t*prev,
 const uint32_t*rows,const float*embedding,bool nohost,bool deviceplan,bool AR) {
 // OFF returns before any data dereference, filesystem operation, hash or output.
 if(!settings().enabled||!active)return {};
 static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559);
 uint32_t one=1;if(*reinterpret_cast<const uint8_t*>(&one)!=1)throw std::invalid_argument("PLE_INPUT33 requires little-endian host");
 if(!SFD||!activations||!exact||reused!=0||ordinal<1||ordinal>4||stage!=0||device<0||lb!=0||le<=1||le>48||T<1||T>8||pos<0||pos+T>int64_t(nids)||!(nids==1||nids==2||nids==4||nids==8)||!ids||!tokens||!prev||!rows||!embedding||AR||!(deviceplan||nohost))throw std::invalid_argument("PLE_INPUT33 unsupported current/prelaunch native window");
 for(size_t i=0;i<nids;++i)if(ids[i]<0||ids[i]>=248320)throw std::invalid_argument("PLE_INPUT33 acceptedGEN token out of modelvocab");
 for(int t=0;t<T;++t)if(tokens[t]!=ids[size_t(pos+t)])throw std::invalid_argument("PLE_INPUT33 actual window tokens differ from accepted GEN");
 for(int t=0;t<T*16;++t)if(rows[t]>=320001536u)throw std::invalid_argument("PLE_INPUT33 actual rowID out of original table");
 for(int t=0;t<T*2560;++t)if(!std::isfinite(embedding[t]))throw std::invalid_argument("PLE_INPUT33 nonfinite actual embedding");
 static std::atomic<unsigned> frames{0};if(++frames>4)throw std::invalid_argument("PLE_INPUT33 frame quota4");
 publication p{(uint64_t(uint32_t(getpid()))<<32)|ordinal,ordinal,stage,device,T,pos};
 char stem[192],rawname[208],rowname[208],metaname[208];std::snprintf(stem,sizeof stem,"ple-input33-%ld-r%u-s%d-p%lld-t%d",long(getpid()),ordinal,stage,(long long)pos,T);
 std::snprintf(rawname,sizeof rawname,"%s.f32",stem);std::snprintf(rowname,sizeof rowname,"%s.u32",stem);std::snprintf(metaname,sizeof metaname,"%s.json",stem);
 char rawhash[65],rowhash[65];prefix_lifecycle::Sha256 rawsha,rowsha;rawsha.update(reinterpret_cast<const uint8_t*>(embedding),size_t(T)*2560*4);rawsha.finish(rawhash);rowsha.update(reinterpret_cast<const uint8_t*>(rows),size_t(T)*16*4);rowsha.finish(rowhash);
 int dirfd=open(settings().dir,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);if(dirfd<0)throw std::runtime_error("PLE_INPUT33 current directory open failed");
 try{write_bytes(dirfd,rawname,embedding,size_t(T)*2560*4);write_bytes(dirfd,rowname,rows,size_t(T)*16*4);
  int fd=openat(dirfd,metaname,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);if(fd<0)throw std::runtime_error("PLE_INPUT33 metadata exclusive open failed");FILE*f=fdopen(fd,"w");if(!f){close(fd);throw std::runtime_error("PLE_INPUT33 fdopen failed");}
  std::fprintf(f,"{\"schema\":1,\"pid\":%ld,\"request\":%u,\"epoch\":%llu,\"stage\":%d,\"device\":%d,\"lb\":%d,\"le\":%d,\"T\":%d,\"pos\":%lld,\"reused\":0,\"source_exact\":true,\"no_host\":%s,\"device_plan\":%s,\"AR\":false,\"prelaunch_host_publication_only\":true,\"binding_sha256\":\"%s\",\"embedding_encoding\":\"LE_F32[T,2560]\",\"row_ids_encoding\":\"LE_U32[T,16]\",\"embedding_file\":\"%s\",\"embedding_sha256\":\"%s\",\"embedding_bytes\":%zu,\"row_ids_file\":\"%s\",\"row_ids_sha256\":\"%s\",\"row_ids_bytes\":%zu,\"prev\":[%d,%d],\"gen_ids\":[",long(getpid()),ordinal,(unsigned long long)p.epoch,stage,device,lb,le,T,(long long)pos,nohost?"true":"false",deviceplan?"true":"false",settings().binding,rawname,rawhash,size_t(T)*2560*4,rowname,rowhash,size_t(T)*16*4,prev[0],prev[1]);
  for(size_t i=0;i<nids;++i)std::fprintf(f,"%s%lld",i?",":"",(long long)ids[i]);std::fprintf(f,"],\"window_tokens\":[");for(int i=0;i<T;++i)std::fprintf(f,"%s%d",i?",":"",tokens[i]);std::fprintf(f,"],\"row_ids\":[");for(int i=0;i<T*16;++i)std::fprintf(f,"%s%u",i?",":"",rows[i]);std::fprintf(f,"],\"full_model_math_qualified\":false}\n");if(fclose(f))throw std::runtime_error("PLE_INPUT33 metadata close failed");
 }catch(...){close(dirfd);throw;}close(dirfd);
 std::fprintf(stderr,"PLE_INPUT33 published pid=%ld request=%u stage=%d device=%d epoch=%llu pos=%lld T=%d metadata=%s\n",long(getpid()),ordinal,stage,device,(unsigned long long)p.epoch,(long long)pos,T,metaname);std::fflush(stderr);if(observations()[ordinal-1].p.epoch)throw std::invalid_argument("PLE_INPUT33 duplicate currentordinal");observations()[ordinal-1].p=p;return p;
}
inline void submit_begin(const publication&p){if(!p.epoch)return;std::fprintf(stderr,"PLE_INPUT33 submit_begin pid=%ld request=%u stage=%d device=%d epoch=%llu pos=%lld T=%d\n",long(getpid()),p.request,p.stage,p.device,(unsigned long long)p.epoch,(long long)p.pos,p.T);std::fflush(stderr);}
inline void submit_returned(const publication&p,int rc){if(!p.epoch)return;observations()[p.request-1].submitted=rc==0;std::fprintf(stderr,"PLE_INPUT33 submit_returned pid=%ld request=%u stage=%d device=%d epoch=%llu pos=%lld T=%d rc=%d\n",long(getpid()),p.request,p.stage,p.device,(unsigned long long)p.epoch,(long long)p.pos,p.T,rc);std::fflush(stderr);}
inline void terminal(bool active,unsigned request,bool cancelled,const char*finish,int64_t generated){
 if(!settings().enabled||!active||request<1||request>4)return;auto&o=observations()[request-1];if(!o.p.epoch)return;
 if(!o.submitted||o.terminal)throw std::invalid_argument("PLE_INPUT33 currentSFD terminal without unique successful submission");o.terminal=true;
 std::fprintf(stderr,"PLE_INPUT33 terminal pid=%ld request=%u stage=%d device=%d epoch=%llu pos=%lld T=%d cancelled=%d generated=%lld finish=%s\n",long(getpid()),request,o.p.stage,o.p.device,(unsigned long long)o.p.epoch,(long long)o.p.pos,o.p.T,int(cancelled),(long long)generated,finish);std::fflush(stderr);
}

}
