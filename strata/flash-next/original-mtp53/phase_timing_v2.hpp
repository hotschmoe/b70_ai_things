#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>
namespace strata::kernels::original_mtp53 {
struct PhaseTiming53;
inline thread_local PhaseTiming53* host_phase_context53=nullptr;
struct PhaseTiming53 {
 using Clock=std::chrono::steady_clock;static constexpr size_t capacity=32768;
 struct Row{const char* phase;bool prompt;int64_t begin,end;long tid;const void* owner;int device,lb,le,T;int64_t pos;bool success;};
 std::vector<Row> rows;std::mutex mutex;bool enabled=false,prompt=true,active=false;
 static int64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
 PhaseTiming53(){const char* v=std::getenv("STRATA_ORIGINAL_MTP53_PHASE_TIMING");enabled=v&&std::strcmp(v,"1")==0;if(enabled){rows.reserve(capacity);if(host_phase_context53){std::fprintf(stderr,"MTP53_PHASE_TIMER existing owner refused\n");std::fflush(nullptr);std::_Exit(86);}host_phase_context53=this;}}
 ~PhaseTiming53(){if(host_phase_context53==this)host_phase_context53=nullptr;}
 void append(const char* phase,bool is_prompt,int64_t begin,int64_t end,const void* owner=nullptr,int device=-1,int lb=-1,int le=-1,int T=0,int64_t pos=-1,bool success=true)noexcept{
  try{std::lock_guard<std::mutex> guard(mutex);if(rows.size()>=capacity||end<begin){std::fprintf(stderr,"MTP53_PHASE_TIMER bounded span refused\n");std::fflush(nullptr);std::_Exit(86);}rows.push_back({phase,is_prompt,begin,end,long(syscall(SYS_gettid)),owner,device,lb,le,T,pos,success});}catch(...){std::fprintf(stderr,"MTP53_PHASE_TIMER append failure retained\n");std::fflush(nullptr);std::_Exit(86);}}
 void dump()const{if(!enabled)return;for(size_t i=0;i<rows.size();++i){const auto& r=rows[i];std::printf("MTP53_HOST_PHASE ordinal=%zu scope=%s phase=%s wall_ms=%.9f begin_ns=%lld end_ns=%lld pid=%ld tid=%ld owner=%p device=%d lb=%d le=%d T=%d position=%lld success=%d\n",i+1,r.prompt?"teacher_prompt":"decode",r.phase,double(r.end-r.begin)/1e6,(long long)r.begin,(long long)r.end,long(getpid()),r.tid,r.owner,r.device,r.lb,r.le,r.T,(long long)r.pos,int(r.success));}std::printf("MTP53_HOST_PHASE_SCOPE rows=%zu capacity=32768 additional_GPU_waits=0 additional_GPU_queries=0 additional_GPU_copies=0 stage_spans_nested=1 spans_additive=0 host_only=1 production_speed_qualified=0\n",rows.size());}
};
struct PhaseScope53 {
 PhaseTiming53* timer;const char* name;int64_t begin=0;
 PhaseScope53(PhaseTiming53* p,const char* n):timer(p&&p->enabled?p:nullptr),name(n){if(timer){if(timer->active){std::fprintf(stderr,"MTP53_PHASE_TIMER primitive nested scope refused\n");std::fflush(nullptr);std::_Exit(86);}timer->active=true;begin=PhaseTiming53::ns();}}
 ~PhaseScope53()noexcept{if(timer){const auto end=PhaseTiming53::ns();timer->active=false;timer->append(name,timer->prompt,begin,end);}}
};
// Separate host stage spans intentionally nest inside target_run; never sum with it.
struct TargetHostSpan53 {
 PhaseTiming53* timer=host_phase_context53;const void* owner;int device,lb,le,T;int64_t pos,begin=0,submit_begin=0;bool success=false;const char* region;
 TargetHostSpan53(const void* o,int d,int a,int b,int n,int64_t p,const char* name="target_stage_run"):owner(o),device(d),lb(a),le(b),T(n),pos(p),region(name){if(timer&&timer->enabled)begin=PhaseTiming53::ns();else timer=nullptr;}
 void submitting(){if(timer)submit_begin=PhaseTiming53::ns();}
 void submitted(bool ok){if(timer)timer->append("target_stage_submit",timer->prompt,submit_begin,PhaseTiming53::ns(),owner,device,lb,le,T,pos,ok);}
 void returned(bool ok){success=ok;}
 ~TargetHostSpan53()noexcept{if(timer)timer->append(region,timer->prompt,begin,PhaseTiming53::ns(),owner,device,lb,le,T,pos,success);}
};
}
