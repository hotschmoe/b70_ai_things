#pragma once
#include "strata/kernels/original_mtp53_loaded_main_adapter.hpp"
#include <chrono>
namespace strata::kernels::original_mtp53 {
inline bool original_greedy53(LoadedMainDraft53* draft,strata::core::Verifier& main,
 const std::vector<int64_t>& prompt,int budget,strata::core::PoolMultiFn pool,void* user)try{
 const char* explicit_commit=std::getenv("STRATA_ONE_TOKEN_COMMIT");
 if(!explicit_commit || std::strcmp(explicit_commit,"0")!=0 || prompt.empty() || budget!=64 || prompt.size()+size_t(budget)>256)throw std::runtime_error("original greedy53 explicit-commit/prompt/budget/capacity differs");
 PhaseTiming53 timings;if(draft)draft->set_timing(&timings);
 using clock=std::chrono::steady_clock;const auto begin=clock::now();
 auto ns=[](){return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();};
 auto eos=[](int32_t id){return id==248044 || id==248046;};
 std::string error;int32_t first=-1;int64_t position=0;
 for(size_t row=0;row<prompt.size();++row){
  if(prompt[row]<0 || prompt[row]>=248320)throw std::runtime_error("original greedy53 prompt token out of vocabulary");
  const int32_t input=int32_t(prompt[row]);
  {PhaseScope53 phase(&timings,"target_run");if(!main.run(1,&input,position,pool,user,&first,error))throw std::runtime_error(error);}
  {PhaseScope53 phase(&timings,"target_commit_wait");if(!main.commit(1,error)||!main.wait_commit(error))throw std::runtime_error(error);}
  ++position;
  // Prompt conditioning uses the actual next prompt token, not MAIN's hypothetical pick.
  if(draft){const int32_t next=row+1<prompt.size()?int32_t(prompt[row+1]):first;draft->advance_actual(next,main.final_R_all());if(draft->cells()!=position)throw std::runtime_error("original greedy53 prompt catch-up position differs");}
 }
 timings.prompt=false;const double counter_wait=main.ms_wait,counter_pool=main.ms_pool,counter_host=main.ms_host,counter_commit=main.ms_commit;
 const auto decode_begin=clock::now();int generated=0,useful=0,windows=0,accepted_windows=0,rejected_windows=0,single_tail_windows=0;
 auto emit=[&](int32_t id,const char* role){++generated;if(!eos(id))++useful;std::printf("MTP53_GREEDY_TOKEN ordinal=%d id=%d eos=%d role=%s host_monotonic_ns=%lld\n",generated,id,int(eos(id)),role,(long long)ns());std::fflush(stdout);};
 emit(first,"target_prompt_pick");int32_t input=first;bool stopped=eos(first);
 while(!stopped && generated<budget){
  const int remaining=budget-generated;std::array<int32_t,2> actual{-1,-1};bool accepted=false;int keep=1;int32_t guess=-1;
  if(draft){guess=draft->propose_actual();if(!draft->verify_actual(main,input,guess,position,pool,user,remaining,actual,accepted,error))throw std::runtime_error(error);keep=accepted&&!eos(actual[0])?2:1;if(remaining<2)++single_tail_windows;else if(accepted)++accepted_windows;else ++rejected_windows;}
  else {{PhaseScope53 phase(&timings,"target_run");if(!main.run(1,&input,position,pool,user,actual.data(),error))throw std::runtime_error(error);}{PhaseScope53 phase(&timings,"target_commit_wait");if(!main.commit(1,error)||!main.wait_commit(error))throw std::runtime_error(error);}}
  ++windows;position+=keep;if(draft && draft->cells()!=position)throw std::runtime_error("original greedy53 target/draft commit position differs");
  std::printf("MTP53_GREEDY_WINDOW window=%d input=%d guess=%d accepted=%d committed=%d position=%lld remaining_before=%d\n",windows,input,guess,int(accepted),keep,(long long)position,remaining);std::fflush(stdout);
  for(int row=0;row<keep;++row){emit(actual[row],draft?"verified_target":"ordinary_target");if(eos(actual[row])){stopped=true;break;}}
  input=actual[keep-1];
 }
 {PhaseScope53 phase(&timings,"final_existing_drain");dpct::get_current_device().queues_wait_and_throw();}const auto done=clock::now();
 std::printf("MTP53_GREEDY_PASS mode=%s prompt_tokens=%zu generated=%d useful=%d natural_eos=%d windows=%d accepted_windows=%d rejected_windows=%d single_tail_windows=%d prompt_serial_ms=%.9f decode_wall_ms=%.9f process_elapsed_ms=%.9f draft_cells=%lld budget=64 full512_qualified=0 draft_numerical_qualified=0 speed_qualified=0\n",draft?"original_mtp_on":"target_off",prompt.size(),generated,useful,int(stopped),windows,accepted_windows,rejected_windows,single_tail_windows,std::chrono::duration<double,std::milli>(decode_begin-begin).count(),std::chrono::duration<double,std::milli>(done-decode_begin).count(),std::chrono::duration<double,std::milli>(done-begin).count(),(long long)(draft?draft->cells():0));std::fflush(stdout);if(draft)draft->set_timing(nullptr);
 if(timings.enabled){std::printf("MTP53_VERIFIER_DECODE_COUNTERS wait_ms=%.9f pool_ms=%.9f host_ms=%.9f commit_ms=%.9f counters_may_overlap=1 additive_qualified=0\n",main.ms_wait-counter_wait,main.ms_pool-counter_pool,main.ms_host-counter_host,main.ms_commit-counter_commit);timings.dump();std::fflush(stdout);}return true;
}catch(const std::exception& e){std::fprintf(stderr,"MTP53_GREEDY_ERROR %s\n",e.what());return false;}
}
