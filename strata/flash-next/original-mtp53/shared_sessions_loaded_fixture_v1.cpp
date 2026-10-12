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
 for(int i=0;i<4;++i){slots[i]->drain(who[i]);slots[i].reset();if(i<3)require(!lifetime.expired(),"weights retired before last session");}
 require(lifetime.expired(),"shared pack did not retire after final private drain");input.retire();
 std::printf("MTP53_SHARED_SLOT_PASS packs=1 sessions=4 cells_each=5 full_head=248320 bank_bytes=8915972 interleaved_serial_exact=1 stale_owner_refused=1 owner_retired=1 MAIN_executed=0 math_qualified=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MTP53_SHARED_SLOT_ERROR %s\n",e.what());return 2;}
