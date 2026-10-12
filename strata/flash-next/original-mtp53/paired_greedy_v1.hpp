#pragma once
#include "strata/kernels/original_mtp53_loaded_main_adapter.hpp"
#include "strata/prefill/prefill.hpp"
#include <chrono>
#include <functional>
namespace strata::kernels::original_mtp53 {
// Batched target prompt, serial independent own-draft recurrence; diagnostic, no serving claim.
inline bool original_pair_greedy53(LoadedMainDraft53& resident,bool enabled,strata::core::Verifier& main,
 strata::prefill::Prefill& first_prefill,strata::prefill::Prefill& last_prefill,
 const std::vector<int64_t>& prompt,int budget,strata::core::PoolMultiFn pool,void* user)try{
 const char* phase_flag=std::getenv("STRATA_ORIGINAL_MTP53_PHASE_TIMING");if(phase_flag && std::strcmp(phase_flag,"0")!=0)throw std::runtime_error("pair53 real shape requires phase timing OFF");
 const char* explicit_commit=std::getenv("STRATA_ONE_TOKEN_COMMIT");
 if(!explicit_commit || std::strcmp(explicit_commit,"0")!=0 || prompt.empty() || prompt.size()!=2048 || budget!=512 || prompt.size()+size_t(budget)>4096 || first_prefill.chunk()!=512)throw std::runtime_error("pair53 actualP2048/GEN512/context4096/pref512 guards differ");
 for(int64_t id:prompt)if(id<0 || id>=248320)throw std::runtime_error("pair53 prompt token out of vocabulary");
 using clock=std::chrono::steady_clock;auto ns=[](){return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();};auto eos=[](int32_t id){return id==248044||id==248046;};PhaseTiming53 timing;if(enabled)resident.set_timing(&timing);
 const auto begin=clock::now();int64_t seen=0;auto saved=std::move(last_prefill.on_chunk);
 using ChunkFn=std::function<bool(const float*,int64_t,int64_t,std::string&)>;bool prompt_joined=false;
 struct Restore {strata::prefill::Prefill& p;ChunkFn& saved;bool& joined;~Restore(){if(!joined){std::fprintf(stderr,"MTP53_PAIR_PREFILL_FAILURE_RETAIN active_callback_or_stage_owner=1\n");std::fflush(nullptr);std::_Exit(86);}p.on_chunk=std::move(saved);}} restore{last_prefill,saved,prompt_joined};
 last_prefill.on_chunk=[&](const float* rows,int64_t T,int64_t p0,std::string& error){
  if(p0!=seen || T<1 || T>512 || p0+T>=int64_t(prompt.size())){error="pair53 actual lastPrefill chunk sequence differs";return false;}
  if(enabled && !resident.prefill_chunk(rows,prompt,int(T),p0,error))return false;
  seen+=T;std::printf("MTP53_PAIR_PREFILL rows=%lld begin=%lld end=%lld target_batched=1 owned_draft_catchup=%d device=1\n",(long long)T,(long long)p0,(long long)seen,int(enabled));std::fflush(stdout);return true;
 };
 std::string error;const bool prompt_ok=first_prefill.run(prompt.data(),int64_t(prompt.size()-1),0,error);prompt_joined=true;if(!prompt_ok || seen!=int64_t(prompt.size()-1))throw std::runtime_error(error.empty()?"pair53 complete target prefill missing":error);
 int64_t position=int64_t(prompt.size()-1);const int32_t last=int32_t(prompt.back());int32_t first=-1;
 if(!main.run(1,&last,position,pool,user,&first,error)||!main.commit(1,error)||!main.wait_commit(error))throw std::runtime_error(error);++position;
 if(enabled){resident.advance_actual(first,main.final_R_all());if(resident.cells()!=position)throw std::runtime_error("pair53 prompt state not complete");}
 timing.prompt=false;const auto decode_begin=clock::now();int generated=0,useful=0,accepted_windows=0,rejected_windows=0;auto emit=[&](int32_t id){++generated;if(!eos(id))++useful;std::printf("MTP53_PAIR_TOKEN ordinal=%d id=%d eos=%d host_monotonic_ns=%lld\n",generated,id,int(eos(id)),(long long)ns());std::fflush(stdout);};emit(first);int32_t input=first;bool stopped=eos(first);
 while(!stopped && generated<budget){const int remaining=budget-generated;std::array<int32_t,2> actual{-1,-1};bool accepted=false;int keep=1;int32_t guess=-1;
  if(enabled){guess=resident.propose_actual();if(!resident.verify_actual(main,input,guess,position,pool,user,remaining,actual,accepted,error))throw std::runtime_error(error);keep=accepted&&!eos(actual[0])?2:1;if(accepted)++accepted_windows;else ++rejected_windows;}
  else if(!main.run(1,&input,position,pool,user,actual.data(),error)||!main.commit(1,error)||!main.wait_commit(error))throw std::runtime_error(error);
  position+=keep;if(enabled && resident.cells()!=position)throw std::runtime_error("pair53 target/own draft positions differ");
  std::printf("MTP53_PAIR_WINDOW input=%d guess=%d accepted=%d committed=%d position=%lld remaining=%d\n",input,guess,int(accepted),keep,(long long)position,remaining);std::fflush(stdout);
  for(int row=0;row<keep;++row){emit(actual[row]);if(eos(actual[row])){stopped=true;break;}}input=actual[keep-1];
 }
 if(!main.wait_commit(error))throw std::runtime_error(error);if(enabled)resident.set_timing(nullptr);const auto end=clock::now();
 std::printf("MTP53_PAIR_PASS mode=%s prompt_tokens=%zu generated=%d useful=%d natural_eos=%d accepted_windows=%d rejected_windows=%d prompt_wall_ms=%.9f decode_wall_ms=%.9f draft_capacity=4096 draft_cells=%lld resident_original_tensor_control_both_arms=1 fullmath_qualified=0 API_qualified=0 speed_qualified=0\n",enabled?"on":"off",prompt.size(),generated,useful,int(stopped),accepted_windows,rejected_windows,std::chrono::duration<double,std::milli>(decode_begin-begin).count(),std::chrono::duration<double,std::milli>(end-decode_begin).count(),(long long)resident.cells());std::fflush(stdout);if(timing.enabled)timing.dump();return true;
 }catch(const std::exception& e){std::fprintf(stderr,"MTP53_PAIR_ERROR %s\n",e.what());return false;}
}
