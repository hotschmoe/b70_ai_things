#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace strata::kernels::original_mtp53 {
struct PhaseTiming53 {
 using Clock=std::chrono::steady_clock;
 struct Row{const char* phase;bool prompt;double ms;};
 std::vector<Row> rows;bool enabled=false,prompt=true,active=false;
 PhaseTiming53(){const char* v=std::getenv("STRATA_ORIGINAL_MTP53_PHASE_TIMING");enabled=v&&std::strcmp(v,"1")==0;if(enabled)rows.reserve(2048);}
 void dump()const{if(!enabled)return;for(size_t i=0;i<rows.size();++i)std::printf("MTP53_HOST_PHASE ordinal=%zu scope=%s phase=%s wall_ms=%.9f\n",i+1,rows[i].prompt?"serial_prompt":"decode",rows[i].phase,rows[i].ms);std::printf("MTP53_HOST_PHASE_SCOPE rows=%zu additional_GPU_waits=0 additional_GPU_queries=0 timers_nested=0 host_only=1 production_speed_qualified=0\n",rows.size());}
};
struct PhaseScope53 {
 PhaseTiming53* timer;const char* name;PhaseTiming53::Clock::time_point begin{};
 PhaseScope53(PhaseTiming53* p,const char* n):timer(p&&p->enabled?p:nullptr),name(n){if(timer){if(timer->active){std::fprintf(stderr,"MTP53_PHASE_TIMER nested scope refused\n");std::fflush(stderr);std::_Exit(86);}timer->active=true;begin=PhaseTiming53::Clock::now();}}
 ~PhaseScope53()noexcept{if(timer){const auto end=PhaseTiming53::Clock::now();timer->active=false;if(timer->rows.size()>=2048){std::fprintf(stderr,"MTP53_PHASE_TIMER row bound exceeded\n");std::fflush(stderr);std::_Exit(86);}timer->rows.push_back({name,timer->prompt,std::chrono::duration<double,std::milli>(end-begin).count()});}}
};
}
