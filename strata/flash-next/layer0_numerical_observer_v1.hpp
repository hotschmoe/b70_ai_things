// 0021 default-off bounded numerical observer. No model corrections or RNG.
#pragma once
#include "strata/core/layer0_numerical_contract.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <memory>
namespace strata::core::layer0_diag {
struct config {
 bool enabled=false;const char* arm=nullptr;const char* dir=nullptr;const char* binding=nullptr;
 config(){const char*p=std::getenv("STRATA_LAYER0_Q8_DIAG");enabled=p&&!std::strcmp(p,"1");if(enabled){
  arm=std::getenv("STRATA_LAYER0_Q8_DIAG_ARM");dir=std::getenv("STRATA_LAYER0_Q8_DIAG_DIR");binding=std::getenv("STRATA_LAYER0_Q8_DIAG_BINDING_SHA256");}}
};
inline const config& settings(){static const config c;return c;}
struct request {bool active=false;unsigned ordinal=0;std::vector<std::int64_t> ids;std::int64_t reused=-1;};
inline request& current(){static request r;return r;}
inline void skip(const char* reason){if(settings().enabled)std::fprintf(stderr,"L0Q8 skip pid=%ld request=%u reason=%s math_qualified=0\n",long(getpid()),current().ordinal,reason);}
inline void validate_config(){const auto&c=settings();if(!c.enabled)return;
 if(!c.dir||c.dir[0]!='/'||!c.arm||!c.binding||std::strlen(c.binding)!=64)throw std::invalid_argument("Layer0 diagnostic path/binding missing");
 for(const char*p=c.dir;*p;++p)if(!((*p>='a'&&*p<='z')||(*p>='A'&&*p<='Z')||(*p>='0'&&*p<='9')||std::strchr("/._-",*p)))throw std::invalid_argument("Layer0 diagnostic path characters invalid");
 for(const char*p=c.binding;*p;++p)if(!((*p>='0'&&*p<='9')||(*p>='a'&&*p<='f')))throw std::invalid_argument("Layer0 diagnostic binding is not lowercase SHA256");
 struct stat d;if(stat(c.dir,&d)||!S_ISDIR(d.st_mode))throw std::invalid_argument("Layer0 diagnostic directory missing");}
inline void begin(const std::vector<std::int64_t>&ids,bool supported){if(!settings().enabled)return;auto&r=current();r.active=false;
 struct stat a;if(!settings().arm||stat(settings().arm,&a)||!S_ISREG(a.st_mode))return;validate_config();++r.ordinal;r.reused=-1;
 if(!supported||r.ordinal>maximum_frames||!(ids.size()==1||ids.size()==2||ids.size()==4||ids.size()==8)){skip("unsupported_mode_prefix_or_quota");return;}
 if(std::any_of(ids.begin(),ids.end(),[](std::int64_t id){return id<0||id>=248320;})){skip("invalid_GEN_token");return;}
 r.ids=ids;r.active=true;std::fprintf(stderr,"L0Q8 request pid=%ld request=%u ids=",long(getpid()),r.ordinal);
 for(std::size_t i=0;i<ids.size();++i)std::fprintf(stderr,"%s%lld",i?",":"",(long long)ids[i]);std::fprintf(stderr," prefix=%zu layer=0 first_window_T1=1 full_model_math_qualified=0\n",ids.size());}
inline void resumed(std::int64_t count){if(!settings().enabled||!current().active)return;current().reused=count;if(count!=0){current().active=false;skip("reused_prefix_not_empty_baseline");}}
class snapshot {
 sycl::queue q_;int stage_;std::uint8_t*data_=nullptr;
 std::array<std::array<bool,fields.size()>,4>rosters_{};std::array<bool,4>sealed_{};int building_=-1,armed_key_=-1;
 std::uint64_t nonce_=0,zero_=0,published_=0,invalid_=~std::uint64_t(0);std::array<std::uint64_t,4>keys_{{0,1,2,3}};unsigned armed_request_=0;std::int64_t armed_pos_=-1;int armed_token_=-1;
 void affinity(sycl::queue&q){if(q.get_context()!=q_.get_context()||q.get_device()!=q_.get_device())throw std::invalid_argument("Layer0 snapshot queue context/device changed");}
public:
 snapshot(const sycl::queue&q,int stage):q_(q),stage_(stage){if(!settings().enabled)return;validate_config();if(!q_.has_property<sycl::property::queue::in_order>())throw std::invalid_argument("Layer0 snapshot queue must be in order");data_=sycl::malloc_device<std::uint8_t>(frame_bytes()+56,q_);if(!data_)throw std::bad_alloc();
  std::fprintf(stderr,"L0Q8 allocation stage=%d pointer=%p bytes=%zu owner_queue=verifier_cs\n",stage_,(void*)data_,frame_bytes()+56);try{q_.memcpy(data_+frame_bytes()+24,keys_.data(),32).wait_and_throw();}catch(...){sycl::free(data_,q_);data_=nullptr;throw;}}
 ~snapshot(){try{release();}catch(...) {}}
 void release(){if(!data_)return;q_.wait_and_throw();std::fprintf(stderr,"L0Q8 release_begin stage=%d pointer=%p bytes=%zu\n",stage_,(void*)data_,frame_bytes()+56);sycl::free(data_,q_);data_=nullptr;std::fprintf(stderr,"L0Q8 release_returned stage=%d\n",stage_);}
 void begin_roster(int key){if(key<0||key>=4)throw std::invalid_argument("Layer0 graph key invalid");building_=key;rosters_[key].fill(false);sealed_[key]=false;}
 void seal_roster(int key){if(key!=building_||key<0||key>=4)throw std::invalid_argument("Layer0 graph construction mismatch");for(std::size_t i=0;i<fields.size();++i)if(rosters_[key][i]!=(i<24||i>=31))throw std::invalid_argument("Layer0 graph does not contain exact26 producer hooks");sealed_[key]=true;building_=-1;}
 void arm(int key,int rows,std::int64_t pos,int token,sycl::queue&q){affinity(q);const auto&r=current();if(!r.active||r.reused!=0||!first_position(r.ids.size(),rows,pos,token,r.ids)||key<0||key>=4||!sealed_[key])throw std::invalid_argument("Layer0 current frame/graph not qualified for replay");
  armed_key_=key;armed_request_=r.ordinal;armed_pos_=pos;armed_token_=token;nonce_=(std::uint64_t(std::uint32_t(getpid()))<<32)|r.ordinal;
  // Persistent member backing survives until run completes or owning release drains q.
  q.memcpy(data_+frame_bytes()+8,&nonce_,8);q.memcpy(data_+frame_bytes(),&zero_,8);q.memcpy(data_+frame_bytes()+16,&invalid_,8);
 }
 void stamp(sycl::queue&q){affinity(q);if(building_<0)throw std::invalid_argument("Layer0 stamp outside graph construction");for(std::size_t i=0;i<fields.size();++i)if(rosters_[building_][i]!=(i<24||i>=31))throw std::invalid_argument("Layer0 stamp before exact26 producer copies");q.memcpy(data_+frame_bytes(),data_+frame_bytes()+8,8);q.memcpy(data_+frame_bytes()+16,data_+frame_bytes()+24+building_*8,8);}
 void copy(const char*name,const void*src,std::size_t bytes,sycl::queue&q){if(!settings().enabled)return;auto i=index(name);copy_extent(i,bytes);affinity(q);if(!data_||!src)throw std::invalid_argument("Layer0 null copy descriptor");if(building_<0)throw std::invalid_argument("Layer0 copy outside graph construction");q.memcpy(data_+offset(i),src,bytes);rosters_[building_][i]=true;}
 void pair(const char*name,const void*a,const void*b,std::size_t bytes,sycl::queue&q){if(!settings().enabled)return;auto i=index(name);copy_extent(i,bytes*2);affinity(q);if(!data_||!a||!b)throw std::invalid_argument("Layer0 null pair descriptor");if(building_<0)throw std::invalid_argument("Layer0 copy outside graph construction");q.memcpy(data_+offset(i),a,bytes);q.memcpy(data_+offset(i)+bytes,b,bytes);rosters_[building_][i]=true;}
 bool dump(int key,int rows,std::int64_t pos,int token,std::string&err){const auto&r=current();if(!r.active)return true;
  if(r.reused!=0||!first_position(r.ids.size(),rows,pos,token,r.ids)){skip("wrong_token_position_or_state_boundary");return true;}
  if(key!=armed_key_||armed_request_!=r.ordinal||pos!=armed_pos_||token!=armed_token_||key<0||key>=4||!sealed_[key]||published_==nonce_){err="Layer0 stale/unarmed/duplicate frame";return false;}
  std::array<std::uint64_t,3>marker{};q_.memcpy(marker.data(),data_+frame_bytes(),24).wait_and_throw();const auto completed=marker[0];if(completed!=nonce_||marker[1]!=nonce_||marker[2]!=std::uint64_t(key)){err="Layer0 current graph replay marker absent; no fields publish";return false;}
  const std::uint16_t endian=1;if(!*reinterpret_cast<const std::uint8_t*>(&endian)){err="Layer0 requires little-endian raw source runtime";return false;}
  const std::string stem=std::string(settings().dir)+"/l0q8-"+std::to_string(long(getpid()))+"-r"+std::to_string(r.ordinal)+"-s"+std::to_string(stage_);
  const int meta=::open((stem+".json").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(meta<0){err="Layer0 metadata open failed";return false;}
  FILE*out=fdopen(meta,"w");if(!out){::close(meta);err="Layer0 metadata stream failed";return false;}
  std::unique_ptr<FILE,int(*)(FILE*)>close_meta(out,&std::fclose);
  std::fprintf(out,"{\"schema\":1,\"pid\":%ld,\"request\":%u,\"stage\":%d,\"layer\":0,\"rows\":1,\"position\":%lld,\"token\":%d,\"reused\":%lld,\"binding_sha256\":\"%s\",\"gen_ids\":[",long(getpid()),r.ordinal,stage_,(long long)pos,token,(long long)r.reused,settings().binding);
  for(std::size_t j=0;j<r.ids.size();++j)std::fprintf(out,"%s%lld",j?",":"",(long long)r.ids[j]);
  std::fprintf(out,"],\"graph_key\":%d,\"completed_nonce\":%llu,\"request_replay_marker_verified\":true,",key,(unsigned long long)completed);
  std::fprintf(out,"\"raw_fused_hidden_observed\":false,\"complete_preregistered_layout\":false,\"full_model_math_qualified\":false,\"fields\":[");bool ok=true;
  for(std::size_t i=0;i<fields.size();++i){std::string path=stem+"-"+fields[i].name+".bin";
   if(rosters_[key][i]){std::vector<std::uint8_t>raw(fields[i].bytes);q_.memcpy(raw.data(),data_+offset(i),raw.size()).wait_and_throw();int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0){ok=false;break;}std::size_t done=0;while(done<raw.size()){ssize_t n=::write(fd,raw.data()+done,raw.size()-done);if(n>0)done+=std::size_t(n);else if(n<0&&errno==EINTR)continue;else{ok=false;break;}}if(::close(fd))ok=false;if(!ok)break;}
   std::fprintf(out,"%s{\"name\":\"%s\",\"encoding\":\"%s\",\"bytes\":%zu,\"observed\":%s,\"file\":\"%s\",\"provenance\":\"%s\"}",i?",":"",fields[i].name,fields[i].encoding,fields[i].bytes,rosters_[key][i]?"true":"false",rosters_[key][i]?path.c_str():"",rosters_[key][i]?"actual_buffer":"UNOBSERVED_no_producer_hook");
  }
  std::fprintf(out,"]}\n");if(std::fclose(close_meta.release()))ok=false;if(!ok){err="Layer0 diagnostic output incomplete; preserve partial evidence";return false;}
  published_=nonce_;
  std::fprintf(stderr,"L0Q8 frame pid=%ld request=%u stage=%d layer=0 position=%lld token=%d metadata=%s.json math_qualified=0\n",long(getpid()),r.ordinal,stage_,(long long)pos,token,stem.c_str());return true;
 }
};
}
