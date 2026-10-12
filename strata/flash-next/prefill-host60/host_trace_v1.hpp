#pragma once
// Host-only diagnostic. No SYCL/event/queue call, dynamic row storage or GPU operation.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/syscall.h>
#include <unistd.h>
namespace strata::prefill::host60 {
struct Owner { uint64_t rid=0,slotgen=0;int slot=-1; };
inline Owner& request_owner(){static Owner value;return value;}
inline bool enabled(){static const bool value=[] {const char* p=std::getenv("STRATA_PREFILL_HOST60");if(!p||std::strcmp(p,"0")==0)return false;if(std::strcmp(p,"1")==0)return true;std::fprintf(stderr,"STRATA_PREFILL_HOST60=0|1\n");std::abort();}();return value;}
// Original serial request owner sets this before fully drained sp.run calls.
// Stage workers snapshot it after std::async's publication; decode service never changes it.
inline void set_owner(uint64_t rid,uint64_t slotgen,int slot){if(enabled())request_owner()={rid,slotgen,slot};}
struct Row { const char* phase=nullptr;int layer=-1;int64_t ns=0,ne=-1; };
class Trace final {
    using Clock=std::chrono::steady_clock;
    bool on_=false;std::array<Row,64> rows_{};unsigned used_=0;bool overflow_=false;
    Owner owner_{};const void* object_;int device_;int64_t pos_,count_,tid_=0;uint64_t run_=0;
public:
    Trace(const void* object,int device,int64_t pos,int64_t count):object_(object),device_(device),pos_(pos),count_(count){
        if(!enabled()||count!=512||pos!=24)return;
        static std::atomic<uint64_t> runs{0};run_=runs.fetch_add(1,std::memory_order_relaxed)+1;
        if(run_>8){std::fprintf(stderr,"PFH60 bound runs=8 exceeded=1\n");return;}
        owner_=request_owner();tid_=syscall(SYS_gettid);on_=true;mark("run_enter");
    }
    Trace(const Trace&)=delete;Trace& operator=(const Trace&)=delete;
    void mark(const char* phase,int layer=-1,int64_t ne=-1){if(!on_)return;if(used_==rows_.size()){overflow_=true;return;}rows_[used_++]={phase,layer,std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(),ne};}
    bool early(int64_t layer,int64_t begin)const{return on_&&layer<begin+2;}
    ~Trace(){
        if(!on_)return;
        for(unsigned i=0;i<used_;++i)std::fprintf(stderr,"PFH60 row pid=%ld tid=%lld owner=%p rid=%llu slotgen=%llu slot=%d run=%llu device=%d pos=%lld T=%lld seq=%u layer=%d phase=%s ne=%lld host_steady_ns=%lld\n",long(getpid()),(long long)tid_,object_,(unsigned long long)owner_.rid,(unsigned long long)owner_.slotgen,owner_.slot,(unsigned long long)run_,device_,(long long)pos_,(long long)count_,i,rows_[i].layer,rows_[i].phase,(long long)rows_[i].ne,(long long)rows_[i].ns);
        std::fprintf(stderr,"PFH60 end pid=%ld tid=%lld owner=%p rid=%llu run=%llu rows=%u overflow=%d deferred=1 host_only=1\n",long(getpid()),(long long)tid_,object_,(unsigned long long)owner_.rid,(unsigned long long)run_,used_,int(overflow_));
    }
};
}
