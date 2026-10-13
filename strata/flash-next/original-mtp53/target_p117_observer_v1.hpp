#pragma once
// Diagnostic only. No arithmetic/state writes; extra snapshot/hash/file IO is explicit.
#include "strata/kernels/original_mtp53_shifted_joint_checkpoint_file.hpp"
#include <fstream>
#include <sstream>
#include <type_traits>
namespace strata::kernels::original_mtp53 {
inline void target_p117_observe53(const OriginalShiftedJointCheckpoint53& image,const char* phase){
 const char* root=std::getenv("STRATA_ORIGINAL_MTP53_TARGET_P117_DIR");
 if(!root || !*root)return;
 if(image.original().target_inputs.size()!=117)return;
 static unsigned serial=0;if(++serial>8)throw std::runtime_error("P117 diagnostic event cap");
 namespace fs=std::filesystem;fs::path parent(root);
 if(!parent.is_absolute() || !fs::is_directory(parent) || fs::is_symlink(parent) || fs::canonical(parent)!=parent)throw std::runtime_error("P117 diagnostic canonical existing output required");
 const auto out=parent/(std::to_string(serial)+"-"+phase);
 if(!fs::create_directory(out))throw std::runtime_error("P117 diagnostic exclusive event directory");
 size_t total=0;std::ostringstream meta;meta<<"{\"schema\":1,\"phase\":\""<<phase<<"\",\"prefix_tokens\":117,\"target_logits_available\":false,\"original_envelope_sha256\":\""<<image.original().envelope_sha<<"\",\"fields\":[";bool first=true;
 auto field=[&](const std::string& name,size_t count,auto visit){
  if(count>(128u<<20) || total+count>(768u<<20))throw std::runtime_error("P117 diagnostic byte cap");total+=count;
  const auto file=out/(name+".bin");int fd=::open(file.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("P117 field open");
  JointHash53 hash;size_t got=0;try{bool ok=visit([&](const uint8_t* p,size_t n,size_t at){if(at!=got || n>count-got)throw std::runtime_error("P117 noncontiguous field");hash.update(p,n);size_t written=0;while(written<n){ssize_t k=::write(fd,p+written,n-written);if(k<=0)throw std::runtime_error("P117 field write");written+=size_t(k);}got+=n;return true;});if(!ok || got!=count)throw std::runtime_error("P117 complete field required");int rc=::close(fd);fd=-1;if(rc)throw std::runtime_error("P117 field close");}catch(...){if(fd>=0)::close(fd);throw;}
  char digest[65];hash.finish(digest);if(!first)meta<<",";first=false;meta<<"{\"name\":\""<<name<<"\",\"bytes\":"<<count<<",\"sha256\":\""<<digest<<"\"}";
 };
 auto vec=[&](const std::string& name,const auto& b){field(name,b.size()*sizeof(typename std::decay_t<decltype(b)>::value_type),[&](auto fn){return fn(reinterpret_cast<const uint8_t*>(b.data()),b.size()*sizeof(typename std::decay_t<decltype(b)>::value_type),0);});};
 for(int s=0;s<2;++s){const auto& t=image.targets()[s];const std::string base="stage"+std::to_string(s)+"-";
  vec(base+"geometry",t.geometry);vec(base+"ids",t.live.ids);vec(base+"gdn_conv",t.live.gdn);vec(base+"ple_history",t.live.ple);vec(base+"indexer_tail",t.live.tails);vec(base+"indexer_dead",t.live.dead);vec(base+"indexer_block_pos",t.live.block_pos);vec(base+"ple_prev",image.ple_previous()[s]);
  const std::array<int64_t,5> info{t.layer_lo,t.layer_hi,int64_t(t.cvec),int64_t(t.live.used),int64_t(t.live.pinned)};vec(base+"meta",info);
  if(!t.checkpoints.empty() || !t.stage_images.empty() || !t.live.imgs.empty())throw std::runtime_error("P117 observer exact unbranched text state required");
  for(size_t k=0;k<t.kv.size();++k){const auto& v=t.kv[k];const auto tag=base+"qsa"+std::to_string(k)+"-";const std::array<int64_t,7> layout{v.format,v.cells,v.heads,v.head_dim,v.page_size,v.pooled_rows,v.idx_dim};vec(tag+"layout",layout);
   unsigned i=0;for(const auto* b:{&v.k,&v.v,&v.k_scale,&v.v_scale,&v.pooled}){const char* names[]={"k","v","k_scale","v_scale","pooled"};field(tag+names[i++],b->size(),[&](auto fn){return b->visit(0,b->size(),fn);});}}
 }
 vec("original-last_target_hidden",image.original().last_target_hidden);
 // Current and pre-last are DISTINCT original state, never target recurrence.
 for(int i=0;i<2;++i){const auto& c=i?image.original().pre_last:image.original().current;const std::string tag=i?"original-pre_last-":"original-current-";vec(tag+"committed",c.committed);vec(tag+"working",c.working);vec(tag+"hidden",c.hidden);vec(tag+"logits",c.logits);vec(tag+"prefix",c.prefix);const std::array<uint64_t,4> m{c.source_rid,c.source_generation,c.cells,uint64_t(c.head_ready)};vec(tag+"owner_counts",m);}
 meta<<"],\"total_bytes\":"<<total<<",\"fresh_target_admission\":false,\"head_restoration_claim\":false}\n";
 const auto text=meta.str();shifted_joint_file53::write_bytes(out/"fields.json",std::vector<uint8_t>(text.begin(),text.end()));
 std::fprintf(stderr,"MTP53_TARGET_P117_OBSERVED event=%u phase=%s fields_complete=1 target_logits_available=0 bytes=%zu\n",serial,phase,total);
}
}
