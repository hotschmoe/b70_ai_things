// Synthetic source-only fixture: original34 pack, no target/main model or math authority.
#include "strata/kernels/original_mtp53_shared_sessions.hpp"
#include <cstdio>
#include <set>
#include <array>
using namespace strata::kernels::original_mtp53;
namespace {
template<class T>bool exact(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
void require(bool yes,const char* error){if(!yes)throw std::runtime_error(error);}
void compare(const OriginalDraftPrivateSnapshot53& a,const OriginalDraftPrivateSnapshot53& b){
 require(a.cells==b.cells && exact(a.committed,b.committed) && exact(a.working,b.working) && exact(a.hidden,b.hidden) && exact(a.logits,b.logits),"interleaved/private serial complete state/head differs");
 require(a.committed.size()==8915972 && a.working.size()==8915972 && a.logits.size()==248320,"full original bank/head layout differs");
}
void distinct(const std::array<OriginalDraftPrivateSnapshot53,4>& views){std::set<uintptr_t> seen;for(const auto& v:views)for(auto p:v.private_allocations)require(p && seen.insert(p).second,"private USM aliases another slot");}
int32_t token(int lane,int step){static constexpr int32_t ids[]={248045,8678,198,15666,760};return ids[(lane+step)%5];}
void feed(OriginalDraftSession53& s,OriginalDraftSlotOwner53 who,int lane,int step,sycl::queue& q,float* device,std::vector<float>& host){
 for(int i=0;i<10240;++i)host[i]=float((i+lane*7+step*3)%31-15)*0.003125f;
 q.memcpy(device,host.data(),host.size()*4).wait_and_throw();s.advance_actual(who,token(lane,step),device);
}
}
int main(int argc,char** argv)try{
 if(argc!=2)throw std::runtime_error("exact original MTP path required");
 const strata::core::OnDevice on(1);auto& q=dpct::get_in_order_queue();
 struct stat st{};require(::stat(argv[1],&st)==0,"original file stat unavailable");
 OriginalDraftIdentity53 identity{argv[1],"8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231",uint64_t(st.st_dev),uint64_t(st.st_ino),uint64_t(st.st_size),int64_t(st.st_mtim.tv_sec)*1000000000+st.st_mtim.tv_nsec,int64_t(st.st_ctim.tv_sec)*1000000000+st.st_ctim.tv_nsec};
 // Root must verify this whole original SHA freshly before/after execution.
 auto pack=std::make_shared<OriginalDraftWeights53>(argv[1],identity,q,1);std::weak_ptr<const OriginalDraftWeights53> lifetime=pack;
 std::array<OriginalDraftSlotOwner53,4> who{{{101,1},{102,1},{103,1},{104,1}}};
 std::array<std::unique_ptr<OriginalDraftSession53>,4> slots;
 for(int i=0;i<4;++i)slots[i]=std::make_unique<OriginalDraftSession53>(pack,who[i],q,1);
 pack.reset();require(!lifetime.expired(),"sessions lost shared weights owner");
 OwnedRuntime53 input(q);auto* hidden=input.take<float>(10240);std::vector<float> host(10240);
 std::array<std::vector<int32_t>,4> proposals;std::array<std::vector<std::vector<float>>,2> prefix_heads;
 for(int step=0;step<5;++step){for(int lane=0;lane<2;++lane){feed(*slots[lane],who[lane],lane,step,q,hidden,host);proposals[lane].push_back(slots[lane]->propose_actual(who[lane]));prefix_heads[lane].push_back(slots[lane]->diagnostic_snapshot(who[lane]).logits);}}
 const auto untouched=slots[0]->diagnostic_snapshot(who[0]);bool stale=false;
 try{slots[0]->advance_actual({101,2},248045,hidden);}catch(const std::runtime_error&){stale=true;}
 require(stale,"stale generation accepted");compare(untouched,slots[0]->diagnostic_snapshot(who[0]));
 stale=false;try{(void)slots[0]->cells({999,1});}catch(const std::runtime_error&){stale=true;}require(stale,"stale RID accepted");
 for(int lane=0;lane<2;++lane)for(int step=0;step<5;++step){feed(*slots[lane+2],who[lane+2],lane,step,q,hidden,host);proposals[lane+2].push_back(slots[lane+2]->propose_actual(who[lane+2]));require(exact(prefix_heads[lane][step],slots[lane+2]->diagnostic_snapshot(who[lane+2]).logits),"complete prefix head differs");}
 std::array<OriginalDraftPrivateSnapshot53,4> views;
 for(int i=0;i<4;++i){views[i]=slots[i]->diagnostic_snapshot(who[i]);require(views[i].cells==5,"private cell count differs");}
 distinct(views);compare(views[0],views[2]);compare(views[1],views[3]);require(proposals[0]==proposals[2] && proposals[1]==proposals[3],"own proposal sequence differs");
 // Reuse used A/B banks in-place; serial references are genuinely NEW private
 // sessions from the SAME surviving weightpack, not four weight reloads.
 const auto old_who=who;auto pack_for_refs=lifetime.lock();require(bool(pack_for_refs),"shared pack absent before reset");
 for(int lane=0;lane<2;++lane){who[lane]={uint64_t(201+lane),2};slots[lane]->reset_fresh(old_who[lane],who[lane]);require(slots[lane]->cells(who[lane])==0,"reset count not empty");
  bool refused=false;try{slots[lane]->advance_actual(old_who[lane],248045,hidden);}catch(const std::runtime_error&){refused=true;}require(refused,"old generation accepted after actual reset");
 }
 for(int lane=0;lane<2;++lane){slots[lane+2]->drain(who[lane+2]);slots[lane+2].reset();who[lane+2]={uint64_t(203+lane),2};slots[lane+2]=std::make_unique<OriginalDraftSession53>(pack_for_refs,who[lane+2],q,1);}
 pack_for_refs.reset();for(auto& v:proposals)v.clear();for(auto& v:prefix_heads)v.clear();
 for(int step=0;step<5;++step)for(int lane=0;lane<2;++lane){feed(*slots[lane],who[lane],lane+2,step,q,hidden,host);proposals[lane].push_back(slots[lane]->propose_actual(who[lane]));prefix_heads[lane].push_back(slots[lane]->diagnostic_snapshot(who[lane]).logits);}
 for(int lane=0;lane<2;++lane)for(int step=0;step<5;++step){feed(*slots[lane+2],who[lane+2],lane+2,step,q,hidden,host);proposals[lane+2].push_back(slots[lane+2]->propose_actual(who[lane+2]));require(exact(prefix_heads[lane][step],slots[lane+2]->diagnostic_snapshot(who[lane+2]).logits),"reset complete prefix head differs");}
 std::array<OriginalDraftPrivateSnapshot53,4> reset_views;for(int i=0;i<4;++i)reset_views[i]=slots[i]->diagnostic_snapshot(who[i]);
 distinct(reset_views);compare(reset_views[0],reset_views[2]);compare(reset_views[1],reset_views[3]);
 for(int lane=0;lane<2;++lane){require(reset_views[lane].private_allocations==views[lane].private_allocations,"reset changed reserved GPU allocations");require(proposals[lane]==proposals[lane+2],"reset own proposals differ");}
 // Complete checkpoint/private branch roundtrip; synthetic prefix witness only.
 std::vector<int32_t> prefix;for(int step=0;step<5;++step)prefix.push_back(token(2,step));
 const std::string draft_contract(64,'a'),target_contract(64,'b');
 auto saved=slots[0]->export_checkpoint(who[0],prefix,draft_contract,target_contract);const auto saved_digest=saved.payload_sha;
 auto wire=serialize_checkpoint53(saved);auto decoded=deserialize_checkpoint53(wire,saved_digest);
 const auto before_branch=slots[0]->diagnostic_snapshot(who[0]);
 for(int step=5;step<7;++step)feed(*slots[0],who[0],2,step,q,hidden,host);
 require(slots[0]->cells(who[0])==7,"checkpoint branch did not advance");
 slots[0]->import_checkpoint(who[0],decoded,saved_digest,prefix,draft_contract,target_contract);
 compare(before_branch,slots[0]->diagnostic_snapshot(who[0]));
 // Shared prefix import into a DIFFERENT private session/current owner.
 const auto destination_alloc=slots[1]->diagnostic_snapshot(who[1]).private_allocations;
 slots[1]->import_checkpoint(who[1],decoded,saved_digest,prefix,draft_contract,target_contract);
 compare(before_branch,slots[1]->diagnostic_snapshot(who[1]));require(destination_alloc==slots[1]->diagnostic_snapshot(who[1]).private_allocations,"checkpoint import changed private ownership");
 bool rejected=false;auto wrong_prefix=prefix;wrong_prefix.back()=(wrong_prefix.back()+1)%248320;
 try{slots[1]->import_checkpoint(who[1],decoded,saved_digest,wrong_prefix,draft_contract,target_contract);}catch(const std::runtime_error&){rejected=true;}require(rejected,"checkpoint mismatched target prefix accepted");compare(before_branch,slots[1]->diagnostic_snapshot(who[1]));
 auto corrupt=decoded;corrupt.working.back()^=1;rejected=false;
 try{slots[1]->import_checkpoint(who[1],corrupt,saved_digest,prefix,draft_contract,target_contract);}catch(const std::runtime_error&){rejected=true;}require(rejected,"checkpoint corrupt full-bank payload accepted");compare(before_branch,slots[1]->diagnostic_snapshot(who[1]));
 // Cached continuation: imported A/B versus genuinely fresh serial prefix5.
 auto continued_pack=lifetime.lock();require(bool(continued_pack),"pack absent at continuation");
 slots[3]->drain(who[3]);slots[3].reset();who[3]={304,3};slots[3]=std::make_unique<OriginalDraftSession53>(continued_pack,who[3],q,1);continued_pack.reset();
 for(int step=0;step<5;++step)feed(*slots[3],who[3],2,step,q,hidden,host);
 compare(before_branch,slots[2]->diagnostic_snapshot(who[2]));compare(before_branch,slots[3]->diagnostic_snapshot(who[3]));
 std::array<std::vector<OriginalDraftPrivateSnapshot53>,2> continuation;std::array<std::vector<int32_t>,2> continuation_proposals;
 for(int step=5;step<9;++step)for(int lane=0;lane<2;++lane){feed(*slots[lane],who[lane],2,step,q,hidden,host);continuation[lane].push_back(slots[lane]->diagnostic_snapshot(who[lane]));continuation_proposals[lane].push_back(slots[lane]->propose_actual(who[lane]));require(continuation[lane].back().cells==step+1,"cached continuation count differs");}
 for(int lane=0;lane<2;++lane)for(int step=5;step<9;++step){feed(*slots[lane+2],who[lane+2],2,step,q,hidden,host);compare(continuation[lane][size_t(step-5)],slots[lane+2]->diagnostic_snapshot(who[lane+2]));require(continuation_proposals[lane][size_t(step-5)]==slots[lane+2]->propose_actual(who[lane+2]),"cached continuation proposal differs");}
 // Divergence at the SAME exact saved5: cached A versus new native prefix5,
 // then different synthetic suffix. No truncation of a latest9 bank.
 slots[0]->import_checkpoint(who[0],decoded,saved_digest,prefix,draft_contract,target_contract);
 continued_pack=lifetime.lock();slots[2]->drain(who[2]);slots[2].reset();who[2]={405,4};slots[2]=std::make_unique<OriginalDraftSession53>(continued_pack,who[2],q,1);continued_pack.reset();
 for(int step=0;step<5;++step)feed(*slots[2],who[2],2,step,q,hidden,host);compare(before_branch,slots[2]->diagnostic_snapshot(who[2]));
 for(int step=5;step<9;++step){feed(*slots[0],who[0],4,step,q,hidden,host);const auto actual=slots[0]->diagnostic_snapshot(who[0]);const auto pick=slots[0]->propose_actual(who[0]);feed(*slots[2],who[2],4,step,q,hidden,host);compare(actual,slots[2]->diagnostic_snapshot(who[2]));require(pick==slots[2]->propose_actual(who[2]) && actual.cells==step+1,"divergent continuation differs");}
 std::array<OriginalDraftPrivateSnapshot53,4> final_views;for(int i=0;i<4;++i){require(slots[i]->cells(who[i])==9,"final continuation count9 required");final_views[i]=slots[i]->diagnostic_snapshot(who[i]);}distinct(final_views);for(int lane=0;lane<2;++lane)require(final_views[lane].private_allocations==views[lane].private_allocations,"cached continuation changed reserved private allocations");
 for(int i=0;i<4;++i){slots[i]->drain(who[i]);slots[i].reset();if(i<3)require(!lifetime.expired(),"weights retired before last session");}
 require(lifetime.expired(),"shared pack did not retire after final private drain");input.retire();
 std::printf("MTP53_CHECKPOINT_CONTINUATION_PASS packs=1 sessions=4 cells_each=9 full_head=248320 bank_bytes=8915972 interleaved_serial_exact=1 stale_owner_refused=1 owner_retired=1 reset_slots=2 fresh_references=2 old_owner_after_reset_refused=1 private_reset_allocations_unchanged=1 checkpoint_branch_shared_roundtrip_exact=1 checkpoint_prefix_hash_negative=1 continuation_5_to9_exact=1 divergent_suffix_exact=1 MAIN_executed=0 math_qualified=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MTP53_SHARED_SLOT_ERROR %s\n",e.what());return 2;}
