#pragma once
// Default-OFF native output targets only. No captured value is a math input.
#include "strata/core/prefix_residual_capture.hpp"
#include "strata/core/layer.hpp"
#include <array>
#include <cerrno>
#include <stdexcept>
namespace strata::core::qsa3_target {
struct field {const char*name;size_t bytes;const char*encoding;bool rows;};
inline const std::array<field,29>& fields(){static const std::array<field,29> value{{
 {"hc_residual",40960,"LE_F32[rows,4,2560]",true},{"hc_mixed",10240,"LE_F32[rows,2560]",true},
 {"mixed_q81",2880,"Q8_1[rows,80,36]",true},{"indexer_key_projected",512,"LE_F32[rows,128]",true},
 {"k_projected",2048,"LE_F32[rows,2,256]",true},{"v_projected",2048,"LE_F32[rows,2,256]",true},
 {"k_fused_RoPE",2048,"LE_F32[rows,2,256]",true},{"q_full_projected",49152,"LE_F32[rows,24,2,256]",true},
 {"q_fused_RoPE",24576,"LE_F32[rows,24,256]",true},{"indexer_query_projected",2048,"LE_F32[rows,4,128]",true},
 {"indexer_query_fused_RoPE",2048,"LE_F32[rows,4,128]",true},{"step",16,"LE_I32[rows,4]",true},
 {"selected_ids_capacity4",16,"LE_I32[rows,4];active_width=step[3]",true},
 {"attention",24576,"LE_F32[rows,24,256]",true},{"gated",24576,"LE_F32[rows,24,256]",true},
 {"gated_q81",6912,"Q8_1[rows,192,36]",true},{"output_projected",10240,"LE_F32[rows,2560]",true},
 {"q_gamma",1024,"LE_F32[256]",false},{"k_gamma",1024,"LE_F32[256]",false},
 {"indexer_q_gamma",512,"LE_F32[128]",false},{"indexer_k_gamma",512,"LE_F32[128]",false},
 {"indexer_before",3076,"LE_BYTES;tail384F32,dead128F32,pooled256F32,blockposI32",false},
 {"indexer_after",3076,"LE_BYTES;tail384F32,dead128F32,pooled256F32,blockposI32",false},
 {"page_before",4,"LE_I32[1]",false},{"page_resolved",4,"LE_I32[1]",false},
 {"K_physical_page0_before",4096,"LE_F16[1,2,4,256]",false},
 {"V_physical_page0_before",4096,"LE_F16[1,2,4,256]",false},
 {"K_physical_page0_after_resolve",4096,"LE_F16[1,2,4,256]",false},
 {"V_physical_page0_after_resolve",4096,"LE_F16[1,2,4,256]",false}}};return value;}
inline int index(const char*name){for(size_t i=0;i<fields().size();++i)if(!std::strcmp(name,fields()[i].name))return int(i);throw std::invalid_argument("QSA3 unknown target field");}
struct config {
 bool enabled=false;const char*dir=nullptr;const char*binding=nullptr;
 config(){const char*p=std::getenv("STRATA_QSA3_TARGET");enabled=p&&!std::strcmp(p,"1");if(!enabled)return;
  dir=std::getenv("STRATA_QSA3_TARGET_DIR");binding=std::getenv("STRATA_QSA3_TARGET_BINDING_SHA256");struct stat st;
  if(!prefix_residual30::settings().enabled||!fidelity_diag::settings().enabled||!fidelity_diag::settings().activations||std::getenv("STRATA_VERIFY_EAGER")||!dir||dir[0]!='/'||stat(dir,&st)||!S_ISDIR(st.st_mode)||!binding||std::strlen(binding)!=64)throw std::invalid_argument("QSA3 requires existing P30/SFDactivations, absoluteDIR, binding64 and EAGERabsent");
  for(int i=0;i<64;++i)if(!std::isxdigit(static_cast<unsigned char>(binding[i])))throw std::invalid_argument("QSA3 binding nothex");
 }
};
inline const config&settings(){static const config c;return c;}
class snapshot {
 sycl::queue&q_;int stage_,lb_,le_;bool owner_;uint8_t*data_=nullptr;size_t bytes_=0;
 std::array<size_t,29> offsets_{};std::array<std::array<bool,2>,29> seen_{};bool sealed_[3]={};int recording_=0,rows_=0;unsigned request_=0;int64_t p0_=0;std::string route_;uint64_t nonce_=0,zero_=0;
 int64_t n_pages_=0,n_slots_=0,max_cells_=0;int fd_=-1;bool released_=false;
 public:
 snapshot(sycl::queue&q,int stage,int lb,int le):q_(q),stage_(stage),lb_(lb),le_(le),owner_(lb<=3&&3<le){
  if(lb<0||lb>=le||le>48||(stage!=0&&stage!=1)||(owner_&&stage!=0))throw std::invalid_argument("QSA3 actual stage bounds/owner");
  fd_=::open(settings().dir,O_RDONLY|O_DIRECTORY|O_NOFOLLOW);if(fd_<0)throw std::invalid_argument("QSA3 output directory open");
  try{if(owner_){for(size_t i=0;i<fields().size();++i){offsets_[i]=bytes_;const size_t extent=fields()[i].bytes*(fields()[i].rows?2:1);if(extent>size_t(-1)-bytes_)throw std::overflow_error("QSA3 bytequota overflow");bytes_+=extent;}data_=static_cast<uint8_t*>(sycl::malloc_device(bytes_+16,q_));if(!data_)throw std::bad_alloc();}}catch(...){::close(fd_);fd_=-1;throw;}
  std::fprintf(stderr,"QSA3 allocation pid=%ld stage=%d lb=%d le=%d owner=%d bytes=%zu target_only=1\n",long(getpid()),stage,lb,le,int(owner_),owner_?bytes_+16:0);
 }
 ~snapshot(){try{release();}catch(...){}if(fd_>=0)::close(fd_);}
 void release(){if(released_)return;if(data_){q_.wait_and_throw();sycl::free(data_,q_);data_=nullptr;}released_=true;std::fprintf(stderr,"QSA3 release_returned pid=%ld stage=%d owner=%d\n",long(getpid()),stage_,int(owner_));}
 void record_begin(int rows){if(rows<1||rows>2)throw std::invalid_argument("QSA3 record rows1/2");recording_=rows;sealed_[rows]=false;for(auto&v:seen_)v={false,false};}
 void copy(const char*name,const void*source,int first,int count,int total,sycl::queue&q){
  if(!owner_||!data_)throw std::invalid_argument("QSA3 nonowner copy");const int f=index(name);const auto&d=fields()[f];if(!source||&q!=&q_||total!=recording_||first<0||count<1||first+count>total||(!d.rows&&(first!=0||count!=1)))throw std::invalid_argument("QSA3 field source/row/queue");
  const int slots=d.rows?count:1;for(int i=0;i<slots;++i){const int row=d.rows?first+i:0;if(seen_[f][row])throw std::invalid_argument("QSA3 duplicate row/field");seen_[f][row]=true;}q.memcpy(data_+offsets_[f]+(d.rows?size_t(first)*d.bytes:0),source,size_t(slots)*d.bytes);
 }
 void state(const char*name,const QsaState&st,sycl::queue&q){
  if(!owner_||!data_||&q!=&q_||st.kv_mode!=0||st.kv_int8||st.kv_q4||st.kv_hybrid||st.kv_rot||st.kv_elastic!=-1||st.n_pages<1||st.n_slots!=st.n_pages||st.max_cells!=2048||st.idx_pooled_rows<2||!st.k_pool||!st.v_pool||!st.page_table||!st.idx_tail||!st.idx_dead||!st.idx_pooled||!st.idx_block_pos)throw std::invalid_argument("QSA3 unsupported actual resident state/capacity");n_pages_=st.n_pages;n_slots_=st.n_slots;max_cells_=st.max_cells;
  const int f=index(name);if(seen_[f][0])throw std::invalid_argument("QSA3 duplicate state");seen_[f][0]=true;uint8_t*out=data_+offsets_[f];q.memcpy(out,st.idx_tail,1536);q.memcpy(out+1536,st.idx_dead,512);q.memcpy(out+2048,st.idx_pooled,1024);q.memcpy(out+3072,st.idx_block_pos,4);
 }
 void record_end(int rows,sycl::queue&q){if(&q!=&q_||recording_!=rows||rows<1||rows>2)throw std::invalid_argument("QSA3 record seal shape");if(owner_){for(size_t f=0;f<fields().size();++f)for(int r=0;r<(fields()[f].rows?rows:1);++r)if(!seen_[f][r])throw std::invalid_argument("QSA3 missing recorded field/row");q.memcpy(data_+bytes_+8,data_+bytes_,8);}sealed_[rows]=true;}
 void begin(const char*route,int64_t p0,int rows){const auto&r=fidelity_diag::current();const int64_t ids[4]={248045,8678,198,15666};if(!r.active||r.resume!=0||r.ids.size()!=4||r.ordinal<1||r.ordinal>4||std::getenv("STRATA_VERIFY_EAGER")||!((p0==0&&rows==2&&!std::strcmp(route,"prompt_verifier"))||(p0==2&&rows==1&&!std::strcmp(route,"prompt_verifier"))||(p0==3&&rows==1&&!std::strcmp(route,"verifier"))))throw std::invalid_argument("QSA3 exact fresh prefix4/2,1,1 windows required");for(int i=0;i<4;++i)if(r.ids[i]!=ids[i])throw std::invalid_argument("QSA3 actual accepted IDs differ");route_=route;p0_=p0;rows_=rows;request_=r.ordinal;nonce_=(uint64_t(uint32_t(getpid()))<<32)|(uint64_t(r.ordinal)<<24)|(uint64_t(p0)<<16)|(uint64_t(rows)<<8)|uint64_t(stage_+1);if(owner_){q_.memcpy(data_+bytes_,&nonce_,8);q_.memcpy(data_+bytes_+8,&zero_,8);}}
 void dump(){const auto&r=fidelity_diag::current();if(!r.active||r.ordinal!=request_||!rows_||!nonce_||!sealed_[rows_])throw std::invalid_argument("QSA3 unsealed current window/request");q_.wait_and_throw();uint64_t marker=nonce_;if(owner_){q_.memcpy(&marker,data_+bytes_+8,8).wait_and_throw();if(marker!=nonce_)throw std::invalid_argument("QSA3 stale graph replay nonce");}
  char stem[192];std::snprintf(stem,sizeof stem,"qsa3-%ld-r%u-s%d-%s-p%lld-n%d",long(getpid()),r.ordinal,stage_,route_.c_str(),(long long)p0_,rows_);std::string base(stem);int mfd=::openat(fd_,(base+".json").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(mfd<0)throw std::runtime_error("QSA3 metadata exclusiveopen");FILE*meta=fdopen(mfd,"w");if(!meta){::close(mfd);throw std::runtime_error("QSA3 metadata fdopen");}
  std::fprintf(meta,"{\"schema\":1,\"pid\":%ld,\"request\":%u,\"stage\":%d,\"lb\":%d,\"le\":%d,\"layer\":3,\"owner\":%s,\"route\":\"%s\",\"first_position\":%lld,\"rows\":%d,\"nonce\":%llu,\"device_nonce_observed\":%s,\"binding_sha256\":\"%s\",\"n_pages\":%lld,\"n_slots\":%lld,\"max_cells\":%lld,\"indexer_before_active_pooled_rows\":%lld,\"indexer_after_active_pooled_rows\":%lld,\"unused_padding_is_native_math_target\":false,\"forward_state_is_committed_state\":false,\"fields\":[",long(getpid()),r.ordinal,stage_,lb_,le_,owner_?"true":"false",route_.c_str(),(long long)p0_,rows_,(unsigned long long)marker,owner_?"true":"false",settings().binding,(long long)n_pages_,(long long)n_slots_,(long long)max_cells_,(long long)(p0_?p0_/4+1:0),(long long)((p0_+rows_)/4+1));
  try{if(owner_){for(size_t f=0;f<fields().size();++f){const auto&d=fields()[f];const size_t bytes=d.bytes*(d.rows?rows_:1);std::vector<uint8_t>raw(bytes);q_.memcpy(raw.data(),data_+offsets_[f],bytes).wait_and_throw();std::string file=base+"-"+d.name+".bin";int fd=::openat(fd_,file.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("QSA3 field exclusiveopen");size_t done=0;while(done<bytes){ssize_t n=::write(fd,raw.data()+done,bytes-done);if(n>0)done+=size_t(n);else if(n<0&&errno==EINTR)continue;else{::close(fd);throw std::runtime_error("QSA3 field shortwrite");}}if(::close(fd))throw std::runtime_error("QSA3 field close");std::fprintf(meta,"%s{\"name\":\"%s\",\"bytes\":%zu,\"file\":\"%s\",\"encoding\":\"%s\"}",f?",":"",d.name,bytes,file.c_str(),d.encoding);}}
   std::fprintf(meta,"],\"internal_norm_argument_or_rsqrt_observed\":false,\"internal_attention_score_softmax_observed\":false,\"captures_are_math_inputs\":false,\"full_model_math_qualified\":false}\n");if(::fclose(meta))throw std::runtime_error("QSA3 metadata close");meta=nullptr;
  }catch(...){if(meta)::fclose(meta);throw;}
  std::fprintf(stderr,"QSA3 frame pid=%ld request=%u stage=%d owner=%d route=%s p0=%lld rows=%d fields=%zu nonce=%llu metadata=%s/%s.json\n",long(getpid()),r.ordinal,stage_,int(owner_),route_.c_str(),(long long)p0_,rows_,owner_?fields().size():0,(unsigned long long)marker,settings().dir,base.c_str());rows_=0;nonce_=0;
 }
};
}
