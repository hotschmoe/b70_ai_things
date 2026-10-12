#include "strata/kernels/original_mtp53_joint_checkpoint.hpp"
#include "strata/kernels/original_mtp53_joint_checkpoint_file.hpp"
#include <cstdio>
using namespace strata::kernels::original_mtp53;
void ok53(bool yes){if(!yes)throw std::runtime_error("joint CPU ownership control failed");}
OriginalCheckpoint53 original53(int cells){OriginalCheckpoint53 c;c.source_rid=9;c.source_generation=2;c.cells=cells;c.prefix={248045};if(cells==2)c.prefix.push_back(198);c.publisher_sha="8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231";c.draft_contract_sha=std::string(64,'a');c.target_contract_sha=std::string(64,'b');c.committed.resize(8915972);c.working.resize(8915972);c.hidden.resize(40960);c.logits.resize(993280);c.payload_sha=checkpoint_digest53(c);return c;}
std::array<strata::core::SavedConversation,2> target53(const OriginalCheckpoint53& c,uint8_t byte=1){std::array<strata::core::SavedConversation,2> t;for(int i=0;i<2;++i){t[i].geometry[0]=48;t[i].layer_lo=i*24;t[i].layer_hi=(i+1)*24;t[i].live.ids=c.prefix;t[i].live.gdn={byte};t[i].live.ple={2};t[i].live.tails={3};t[i].live.dead={4};t[i].live.block_pos={5};strata::core::ConversationKv kv;kv.cells=c.cells;kv.k={byte,7};kv.v={8,9};kv.pooled={10};t[i].kv.push_back(std::move(kv));}return t;}
int main()try{
 const std::array<std::array<int32_t,2>,2> previous{{{248045,198},{-1,-1}}};
 auto c=original53(2);auto base=std::make_shared<OriginalJointCheckpoint53>(target53(c),OriginalCheckpoint53(c),previous);auto equal=std::make_shared<OriginalJointCheckpoint53>(target53(c),OriginalCheckpoint53(c),previous);ok53(base->digest()==equal->digest());
 auto changed=std::make_shared<OriginalJointCheckpoint53>(target53(c,2),OriginalCheckpoint53(c),previous);ok53(base->digest()!=changed->digest());
 auto altered=target53(c);altered[1].kv[0].v={8,10};auto raw_changed=std::make_shared<OriginalJointCheckpoint53>(std::move(altered),OriginalCheckpoint53(c),previous);ok53(raw_changed->digest()!=base->digest());
 auto prev_changed=previous;prev_changed[1][0]=8678;auto ple_changed=std::make_shared<OriginalJointCheckpoint53>(target53(c),OriginalCheckpoint53(c),prev_changed);ok53(ple_changed->digest()!=base->digest());
 auto older=original53(1);auto older_targets=target53(older);auto child=std::make_shared<OriginalJointCheckpoint53>(std::move(older_targets),std::move(older),previous);
 auto with_branch=target53(c);for(int i=0;i<2;++i)with_branch[i].checkpoints.push_back(child->targets()[i].live);
 bool refused=false;try{auto invalid=std::make_shared<OriginalJointCheckpoint53>(std::array<strata::core::SavedConversation,2>(with_branch),OriginalCheckpoint53(c),previous);}catch(const std::runtime_error&){refused=true;}ok53(refused);
 auto parent=std::make_shared<OriginalJointCheckpoint53>(std::move(with_branch),OriginalCheckpoint53(c),previous,std::vector<std::shared_ptr<const OriginalJointCheckpoint53>>{child});
 OriginalJointCache53 tree_cache(parent->own_bytes()+child->own_bytes()+8192);ok53(tree_cache.put(parent));auto prefix=tree_cache.best({248045},c.draft_contract_sha,c.target_contract_sha);ok53(prefix && prefix->digest()==child->digest());ok53(!tree_cache.best({248045},std::string(64,'c'),c.target_contract_sha));
 // Held payload remains charged after cache eviction; no sidecar orphan/credit.
 OriginalJointCache53 cache(base->own_bytes()+8192);ok53(cache.put(base));auto held=cache.best(c.prefix,c.draft_contract_sha,c.target_contract_sha);base.reset();cache.clear();ok53(cache.roots()==0 && cache.charged_bytes()>=held->own_bytes());ok53(!cache.put(changed));held.reset();ok53(cache.put(changed));
 std::printf("MTP53_JOINT_CPU_PASS controls=8 full_target_bytes_digest=1 original_digest_join=1 older_full_branch_required=1 held_eviction_charged=1 model_runtime=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MTP53_JOINT_CPU_ERROR %s\n",e.what());return 2;}
