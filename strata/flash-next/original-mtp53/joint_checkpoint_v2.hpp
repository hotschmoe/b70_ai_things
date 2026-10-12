#pragma once
// Source-only compound target/original ownership. No target math or state writes.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/original_mtp53_checkpoint_contract.hpp"
#include <memory>
#include <set>
#include <deque>
#include <functional>
namespace strata::kernels::original_mtp53 {
using JointHash53=strata::core::prefix_lifecycle::Sha256;
inline void joint_number53(JointHash53& h,uint64_t n){strata::core::prefix_lifecycle::number(h,n);}
inline void joint_bytes53(JointHash53& h,const std::vector<uint8_t>& b){joint_number53(h,b.size());h.update(b.data(),b.size());}
inline void joint_checkpoint_state53(JointHash53& h,const strata::core::ConversationCheckpoint& c,int depth=0){
 if(depth>16 || c.stage_parts.size()>2 || c.ids.size()>4096 || c.imgs.size()>4096)throw std::runtime_error("joint target checkpoint bounds");
 strata::core::prefix_lifecycle::digest_tokens(h,c.ids);joint_number53(h,c.imgs.size());for(const auto& im:c.imgs){joint_number53(h,im.start);joint_number53(h,im.hash);}
 for(const auto* b:{&c.gdn,&c.ple,&c.tails,&c.dead,&c.block_pos})joint_bytes53(h,*b);
 joint_number53(h,c.used);joint_number53(h,c.pinned);joint_number53(h,c.stage_parts.size());for(const auto& part:c.stage_parts)joint_checkpoint_state53(h,part,depth+1);
}
inline void joint_target_bytes53(JointHash53& h,const strata::core::SavedConversation& image){
 if(!image.stage_images.empty() || image.checkpoints.size()>64 || image.kv.size()>48)throw std::runtime_error("joint target stage/row bounds");
 for(auto n:image.geometry)joint_number53(h,n);joint_number53(h,image.layer_lo);joint_number53(h,image.layer_hi);joint_number53(h,image.cvec);
 joint_checkpoint_state53(h,image.live);joint_number53(h,image.checkpoints.size());for(const auto& c:image.checkpoints)joint_checkpoint_state53(h,c);
 joint_number53(h,image.kv.size());for(const auto& kv:image.kv){for(auto n:{int64_t(kv.format),kv.cells,kv.heads,kv.head_dim,kv.page_size,kv.pooled_rows,kv.idx_dim})joint_number53(h,n);
  for(const auto* b:{&kv.k,&kv.v,&kv.k_scale,&kv.v_scale,&kv.pooled}){joint_number53(h,b->size());if(!b->visit(0,b->size(),[&](const uint8_t* p,size_t n,size_t){h.update(p,n);return true;}))throw std::runtime_error("joint complete target buffer traversal failed");}
 }
}
class OriginalJointCheckpoint53 final {
 std::array<strata::core::SavedConversation,2> targets_;std::array<std::array<int32_t,2>,2> ple_previous_;OriginalCheckpoint53 original_;std::vector<std::shared_ptr<const OriginalJointCheckpoint53>> branches_;std::string digest_;size_t depth_=1;
public:
 OriginalJointCheckpoint53(std::array<strata::core::SavedConversation,2>&& targets,OriginalCheckpoint53&& original,std::array<std::array<int32_t,2>,2> ple_previous,std::vector<std::shared_ptr<const OriginalJointCheckpoint53>> branches={})
 :targets_(std::move(targets)),ple_previous_(ple_previous),original_(std::move(original)),branches_(std::move(branches)){
  validate_checkpoint53(original_);if(original_.payload_sha!=checkpoint_digest53(original_))throw std::runtime_error("joint original independently exported bytes changed");
  if(targets_[0].layer_lo!=0 || targets_[0].layer_hi!=24 || targets_[1].layer_lo!=24 || targets_[1].layer_hi!=48 || targets_[0].geometry!=targets_[1].geometry || targets_[0].cvec!=targets_[1].cvec || branches_.size()>64)throw std::runtime_error("joint exact paired target carve/contract differs");
  for(const auto& t:targets_)if(t.live.ids!=original_.prefix || !t.stage_images.empty() || !t.live.imgs.empty())throw std::runtime_error("joint original/target accepted text prefix differs");
  if(targets_[0].checkpoints.size()!=targets_[1].checkpoints.size() || targets_[0].checkpoints.size()!=branches_.size())throw std::runtime_error("joint every older target requires saved FULL original/target branch");
  for(size_t i=0;i<branches_.size();++i){const auto& b=branches_[i];if(b)depth_=std::max(depth_,b->depth_+1);if(depth_>64)throw std::runtime_error("joint branch depth bound");if(!b || b->original().cells>=original_.cells || b->original().draft_contract_sha!=original_.draft_contract_sha || b->original().target_contract_sha!=original_.target_contract_sha)throw std::runtime_error("joint older branch contract differs");
   for(int stage=0;stage<2;++stage){const auto& expected=targets_[stage].checkpoints[i];const auto& actual=b->targets()[stage].live;
    if(expected.ids!=actual.ids || expected.imgs!=actual.imgs || expected.gdn!=actual.gdn || expected.ple!=actual.ple || expected.tails!=actual.tails || expected.dead!=actual.dead || expected.block_pos!=actual.block_pos)throw std::runtime_error("joint older prefix running state differs");}
  }
  for(const auto& p:ple_previous_)for(int32_t t:p)if(t < -1 || t>=248320)throw std::runtime_error("joint PLE previous token invalid");
  JointHash53 h;const char schema[]="ORIGINAL-JOINT58-CP1";h.update(reinterpret_cast<const uint8_t*>(schema),sizeof(schema)-1);for(const auto& t:targets_)joint_target_bytes53(h,t);for(const auto& p:ple_previous_)for(int32_t t:p)joint_number53(h,uint32_t(t));
  h.update(reinterpret_cast<const uint8_t*>(original_.payload_sha.data()),64);joint_number53(h,branches_.size());for(const auto& b:branches_)h.update(reinterpret_cast<const uint8_t*>(b->digest().data()),64);char out[65];h.finish(out);digest_=out;
 }
 OriginalJointCheckpoint53(const OriginalJointCheckpoint53&)=delete;OriginalJointCheckpoint53& operator=(const OriginalJointCheckpoint53&)=delete;
 const std::array<strata::core::SavedConversation,2>& targets()const noexcept{return targets_;}const std::array<std::array<int32_t,2>,2>& ple_previous()const noexcept{return ple_previous_;}const OriginalCheckpoint53& original()const noexcept{return original_;}const std::vector<std::shared_ptr<const OriginalJointCheckpoint53>>& branches()const noexcept{return branches_;}const std::string& digest()const noexcept{return digest_;}
 size_t own_bytes()const{size_t n=sizeof(*this)+digest_.capacity()+branches_.capacity()*sizeof(std::shared_ptr<const OriginalJointCheckpoint53>);for(const auto& t:targets_)n+=t.bytes();n+=original_.publisher_sha.capacity()+original_.draft_contract_sha.capacity()+original_.target_contract_sha.capacity()+original_.payload_sha.capacity()+original_.prefix.capacity()*4+original_.committed.capacity()+original_.working.capacity()+original_.hidden.capacity()+original_.logits.capacity();return n;}
 bool pinned()const{for(const auto& t:targets_)if(t.pinned())return true;return false;}
};
class OriginalJointCache53 final {
 size_t budget_;std::deque<std::shared_ptr<const OriginalJointCheckpoint53>> entries_;
 // Weak ownership registry charges held/borrowed children even AFTER root eviction;
 // never a token-keyed payload side map. Expired payload charges are pruned.
 std::vector<std::weak_ptr<const OriginalJointCheckpoint53>> known_;
 void register_tree(const std::shared_ptr<const OriginalJointCheckpoint53>& p,std::set<const void*>& seen){if(!seen.insert(p.get()).second)return;bool registered=false;for(const auto& w:known_)if(auto k=w.lock();k && k.get()==p.get())registered=true;if(!registered)known_.push_back(p);for(const auto& b:p->branches())register_tree(b,seen);}
public:
 explicit OriginalJointCache53(size_t bytes):budget_(bytes){}
 size_t charged_bytes(){known_.erase(std::remove_if(known_.begin(),known_.end(),[](const auto& w){return w.expired();}),known_.end());size_t n=sizeof(*this)+(entries_.size()+1)*sizeof(std::shared_ptr<const OriginalJointCheckpoint53>)+(1+entries_.size()/16)*1024+known_.capacity()*sizeof(std::weak_ptr<const OriginalJointCheckpoint53>);for(const auto& w:known_)if(auto p=w.lock())n+=p->own_bytes();return n;}
 bool put(std::shared_ptr<const OriginalJointCheckpoint53> p){if(!p)return false;for(const auto& e:entries_)if(e->digest()==p->digest())return true;std::set<const void*> seen;register_tree(p,seen);
  // The pending candidate stays charged while admission tries eviction.
  while(charged_bytes()>budget_){auto i=std::find_if(entries_.begin(),entries_.end(),[](const auto& e){return e.use_count()==1 && !e->pinned();});if(i==entries_.end())return false;entries_.erase(i);}
  entries_.push_back(std::move(p));return true;
 }
 std::shared_ptr<const OriginalJointCheckpoint53> best(const std::vector<int32_t>& request,const std::string& draft,const std::string& target){
  std::shared_ptr<const OriginalJointCheckpoint53> result;std::set<const void*> visited;
  std::function<void(const std::shared_ptr<const OriginalJointCheckpoint53>&)> visit=[&](const auto& p){if(!visited.insert(p.get()).second)return;const auto& c=p->original();if(c.draft_contract_sha==draft && c.target_contract_sha==target && c.prefix.size()<=request.size() && std::equal(c.prefix.begin(),c.prefix.end(),request.begin()) && (!result || c.cells>result->original().cells))result=p;for(const auto& b:p->branches())visit(b);};for(const auto& e:entries_)visit(e);return result;
 }
 void clear(){entries_.clear();}size_t roots()const{return entries_.size();}
};
}
