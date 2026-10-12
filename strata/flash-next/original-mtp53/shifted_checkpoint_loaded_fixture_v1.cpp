#include "strata/kernels/original_mtp53_shared_sessions.hpp"
#include <cstdio>
#include <array>
using namespace strata::kernels::original_mtp53;
namespace {
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
template<class T>bool exact(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(T)));}
void compare(const OriginalDraftPrivateSnapshot53& a,const OriginalDraftPrivateSnapshot53& b){require(a.cells==b.cells && exact(a.committed,b.committed) && exact(a.working,b.working) && exact(a.hidden,b.hidden) && exact(a.logits,b.logits),"shifted fullstate/head differs");}
int32_t tok(int i){static constexpr int32_t t[]={248045,8678,198,15666,760,2919,10458,369,42};return t[i%9];}
void feed(OriginalDraftSession53& s,OriginalDraftSlotOwner53 who,int step,int32_t input,int32_t next,sycl::queue& q,float* device,std::vector<float>& host){for(int j=0;j<10240;++j)host[j]=float((j+step*7+input%31)%31-15)*0.003125f;q.memcpy(device,host.data(),40960).wait_and_throw();s.advance_actual(who,next,device);}
}
int main(int argc,char** argv)try{
 require(argc==2,"original MTP path required");const strata::core::OnDevice on(1);auto& q=dpct::get_in_order_queue();struct stat st{};require(::stat(argv[1],&st)==0,"original file stat failed");
 OriginalDraftIdentity53 identity{argv[1],"8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231",uint64_t(st.st_dev),uint64_t(st.st_ino),uint64_t(st.st_size),int64_t(st.st_mtim.tv_sec)*1000000000+st.st_mtim.tv_nsec,int64_t(st.st_ctim.tv_sec)*1000000000+st.st_ctim.tv_nsec};auto pack=std::make_shared<OriginalDraftWeights53>(argv[1],identity,q,1);OwnedRuntime53 inputs(q);auto* hidden=inputs.take<float>(10240);std::vector<float> host(10240);const std::string draft(64,'a'),target(64,'b');
 for(int P:{1,4}){
  std::array<OriginalDraftSlotOwner53,3> who{{{uint64_t(P*10+1),1},{uint64_t(P*10+2),1},{uint64_t(P*10+3),1}}};std::array<std::unique_ptr<OriginalDraftSession53>,3> s;
  for(int i=0;i<3;++i){s[i]=std::make_unique<OriginalDraftSession53>(pack,OriginalDraftSlotOwner53{0,0},q,1);s[i]->reset_fresh({0,0},who[i]);}
  std::vector<int32_t> target_prefix;for(int i=0;i<P;++i){target_prefix.push_back(tok(i));feed(*s[0],who[0],i,tok(i),tok(i+1),q,hidden,host);}
  auto c=s[0]->export_shifted(who[0],target_prefix,draft,target);require(c.lookahead==tok(P) && c.pre_last.cells==uint64_t(P-1),"actual shifted envelope metadata differs");if(P==1)require(!c.pre_last.head_ready && c.pre_last.logits.empty(),"empty pre-last faked a head");
  const auto original=s[0]->diagnostic_snapshot(who[0]);s[2]->import_shifted(who[2],c,c.envelope_sha,target_prefix,tok(P),draft,target);compare(original,s[2]->diagnostic_snapshot(who[2]));
  const int32_t changed=42;s[0]->import_shifted(who[0],c,c.envelope_sha,target_prefix,changed,draft,target);
  for(int i=0;i<P;++i)feed(*s[1],who[1],i,tok(i),i+1==P?changed:tok(i+1),q,hidden,host);
  compare(s[0]->diagnostic_snapshot(who[0]),s[1]->diagnostic_snapshot(who[1]));require(s[0]->propose_actual(who[0])==s[1]->propose_actual(who[1]),"shifted repaired proposal differs");
  feed(*s[0],who[0],P,changed,tok(P+1),q,hidden,host);feed(*s[1],who[1],P,changed,tok(P+1),q,hidden,host);compare(s[0]->diagnostic_snapshot(who[0]),s[1]->diagnostic_snapshot(who[1]));
  const auto old=who[0];s[0]->release_owner(old);bool refused=false;try{(void)s[0]->cells(old);}catch(const std::runtime_error&){refused=true;}require(refused,"released old RID/gen accepted");s[0]->reset_fresh({0,0},{uint64_t(P*100),1});require(s[0]->cells({uint64_t(P*100),1})==0,"true firstgen1 rebind failed");
  for(int i=1;i<3;++i)s[i]->drain(who[i]);s[0]->drain({uint64_t(P*100),1});
 }
 inputs.retire();pack.reset();std::printf("MTP53_SHIFTED_SLOT_PASS cases_P1_P4=1 empty_head_absent=1 first_real_gen1=1 matching_full_reuse=1 divergent_repair_fullstate_exact=1 continuation_exact=1 original_pack_count=1 MAIN_executed=0 math_qualified=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MTP53_SHIFTED_SLOT_ERROR %s\n",e.what());return 2;}
