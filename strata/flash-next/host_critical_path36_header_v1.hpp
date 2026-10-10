#pragma once
// Source36 H: host clocks only. No SYCL dependency, event query, or device wait.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace strata::program::host_trace36 {
inline constexpr size_t capacity = 32768;
inline constexpr uint64_t max_requests=4;
static_assert(std::atomic<bool>::is_always_lock_free,"Trace violation flag must be lock-free");
enum Kind { Request=1, Reset, PromptSpan, Verify, StageInputs, PlePrefetch, PleGather,
            Capture, CommitCapture, ReplaySubmit, ReplayWait, Sample, LPReadback,
            LPCPU, Commit, CommitWait, TEmit, DoneEmit, GraphCreate, GraphHit, GraphPreexisting, TPrintf, StdoutFlush, GraphFinalize, CaptureWait };
struct Context { int dev=-1, lb=-1, le=-1, T=0; int64_t pos=-1; uintptr_t queue=0; uint64_t graph=0; };
struct Event { uint64_t seq,ns,span; Context ctx; int kind,edge,value; };
struct Graph { uintptr_t ptr,queue; uint64_t generation; };
struct State {
    bool active=false,complete=false,overflow=false,clock_error=false,export_error=false,request_limit=false;
    std::atomic<bool> on{false};
    std::atomic<bool> thread_error{false};
    char dir[4096]{},binding[65]{}; Event* events=nullptr; size_t used=0; uint64_t request=0,span=0;
    uint64_t rid=0,slotgen=0,graph_seq=0; int64_t prompt=0; int fresh=0,produced=0;
    pthread_t owner{}; Context ctx{},last_complete{}; Graph graphs[128]{}; size_t graph_count=0;
};
inline State& state() { static State s; return s; }
inline bool enabled() { return state().on.load(std::memory_order_acquire); }
inline bool writer() { auto&s=state();if(!s.on.load(std::memory_order_acquire))return false; if(!pthread_equal(s.owner,pthread_self())){s.thread_error.store(true,std::memory_order_relaxed);return false;}return true; }
inline uint64_t now() { if(!writer())return 0;timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t)!=0){state().clock_error=true;return 0;}return uint64_t(t.tv_sec)*1000000000ULL+uint64_t(t.tv_nsec); }
inline void event(int kind,int edge,uint64_t span=0,int value=0) {
    auto&s=state();if(!s.on||!writer()||!s.active)return;
    if(s.used==capacity){s.overflow=true;return;}
    s.events[s.used]={uint64_t(s.used),now(),span,s.ctx,kind,edge,value};++s.used;
}
inline uint64_t graph(uintptr_t ptr,uintptr_t queue,bool create=false) {
    auto&s=state();if(!s.on||!writer()||!s.active||!ptr)return 0;
    for(size_t i=0;i<s.graph_count;++i)if(s.graphs[i].ptr==ptr&&s.graphs[i].queue==queue){if(create)s.graphs[i].generation=++s.graph_seq;return s.graphs[i].generation;}
    if(s.graph_count==128){s.overflow=true;return 0;}
    const uint64_t n=++s.graph_seq;s.graphs[s.graph_count++]={ptr,queue,n};return n;
}
inline void export_request() {
    auto&s=state();if(!s.on||!writer()||!s.active)return;
    char path[4352];std::snprintf(path,sizeof(path),"%s/host-trace36-%ld-%llu.jsonl",s.dir,long(getpid()),(unsigned long long)s.request);
    const int fd=::open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0){s.export_error=true;std::fprintf(stderr,"HOST_TRACE36_EXPORT_ERROR request=%llu\n",(unsigned long long)s.request);s.active=false;return;}
    FILE*f=fdopen(fd,"w");if(!f){::close(fd);s.export_error=true;s.active=false;return;}
    std::fprintf(f,"{\"schema\":36,\"clock\":\"CLOCK_MONOTONIC\",\"device_clock\":\"not_requested\",\"binding\":\"%s\",\"pid\":%ld,\"request\":%llu,\"rid\":%llu,\"slotgen\":%llu,\"prompt\":%lld,\"fresh\":%d,\"produced\":%d,\"complete\":%s,\"overflow\":%s,\"thread_error\":%s,\"clock_error\":%s,\"events\":%zu,\"capacity\":%zu}\n",s.binding,long(getpid()),(unsigned long long)s.request,(unsigned long long)s.rid,(unsigned long long)s.slotgen,(long long)s.prompt,s.fresh,s.produced,s.complete?"true":"false",s.overflow?"true":"false",s.thread_error.load(std::memory_order_relaxed)?"true":"false",s.clock_error?"true":"false",s.used,capacity);
    for(size_t i=0;i<s.used;++i){const auto&e=s.events[i];std::fprintf(f,"{\"seq\":%llu,\"ns\":%llu,\"span\":%llu,\"kind\":%d,\"edge\":%d,\"value\":%d,\"dev\":%d,\"lb\":%d,\"le\":%d,\"T\":%d,\"pos\":%lld,\"queue\":%llu,\"graph\":%llu}\n",(unsigned long long)e.seq,(unsigned long long)e.ns,(unsigned long long)e.span,e.kind,e.edge,e.value,e.ctx.dev,e.ctx.lb,e.ctx.le,e.ctx.T,(long long)e.ctx.pos,(unsigned long long)e.ctx.queue,(unsigned long long)e.ctx.graph);}
    const bool write_error=std::ferror(f)!=0;
    std::fprintf(f,"{\"footer\":true,\"records\":%zu,\"write_error\":%s}\n",s.used,write_error?"true":"false");
    const bool final_write_error=std::ferror(f)!=0;
    const int close_rc=std::fclose(f);
    if(write_error||final_write_error||close_rc!=0){s.export_error=true;std::fprintf(stderr,"HOST_TRACE36_EXPORT_ERROR request=%llu\n",(unsigned long long)s.request);}
    std::fprintf(stderr,"HOST_TRACE36_CLOSED request=%llu success=%d\n",(unsigned long long)s.request,int(!s.export_error));
    s.active=false;
}
inline void exit_export() { auto&s=state();if(!enabled()||!writer())return;export_request();std::fprintf(stderr,"HOST_TRACE36_SESSION requests=%llu thread_error=%d export_error=%d request_limit=%d\n",(unsigned long long)s.request,int(s.thread_error.load(std::memory_order_acquire)),int(s.export_error),int(s.request_limit));std::free(s.events);s.events=nullptr; }
inline bool init(bool serial) {
    const char*flag=std::getenv("STRATA_CRITICAL_PATH_TRACE");if(!flag||std::strcmp(flag,"0")==0)return true;
    if(std::strcmp(flag,"1")!=0||!serial||std::getenv("STRATA_VERIFY_EAGER"))return false;
    auto&s=state();const char*dir=std::getenv("STRATA_CRITICAL_PATH_TRACE_DIR");const char*b=std::getenv("STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256");
    struct stat st{};if(!dir||dir[0]!='/'||std::strlen(dir)>=sizeof(s.dir)||lstat(dir,&st)!=0||!S_ISDIR(st.st_mode)||!b||std::strlen(b)!=64)return false;
    for(int i=0;i<64;++i)if(!((b[i]>='0'&&b[i]<='9')||(b[i]>='a'&&b[i]<='f')))return false;
    s.events=static_cast<Event*>(std::calloc(capacity,sizeof(Event)));if(!s.events)return false;
    std::strcpy(s.dir,dir);std::strcpy(s.binding,b);s.owner=pthread_self();
    if(std::atexit(exit_export)!=0){std::free(s.events);s.events=nullptr;return false;}s.on.store(true,std::memory_order_release);return true;
}
struct Scope {
    uint64_t id=0;int kind=0;
    explicit Scope(int k):kind(k){if(enabled()&&writer()&&state().active){id=++state().span;event(kind,1,id);}}
    ~Scope(){if(id)event(kind,2,id);}
};
inline uint64_t begin(int kind){if(!enabled()||!writer()||!state().active)return 0;const uint64_t id=++state().span;event(kind,1,id);return id;}
inline void end(int kind,uint64_t id){if(id){event(kind,2,id);if(kind==ReplayWait&&writer())state().last_complete=state().ctx;}}
struct ContextRestore { Context old{};bool active=false;ContextRestore(){if(enabled()&&writer()&&state().active){old=state().ctx;state().ctx.graph=0;active=true;}}~ContextRestore(){if(active&&writer())state().ctx=old;} };
struct Frame {
    Context old{};bool active=false;
    Frame(int dev,int lb,int le,int T,int64_t pos,uintptr_t queue){if(enabled()&&writer()&&state().active){old=state().ctx;state().ctx={dev,lb,le,T,pos,queue,0};active=true;}}
    ~Frame(){if(active&&writer())state().ctx=old;}
};
struct RequestScope {
    uint64_t id=0;
    RequestScope(int64_t prompt,int fresh,uint64_t rid,uint64_t slotgen,int64_t requested=64){auto&s=state();if(!s.on||!writer())return;
        if(s.active)export_request();if(++s.request>max_requests||prompt<1||prompt>2048||requested<1||requested>64){s.request_limit=true;return;}s.used=0;s.span=0;s.complete=false;s.overflow=false;s.clock_error=false;s.produced=0;s.ctx={};s.last_complete={};s.prompt=prompt;s.fresh=fresh;s.rid=rid;s.slotgen=slotgen;s.active=true;
        id=++s.span;event(Request,1,id);
    }
    ~RequestScope(){if(id){event(Request,2,id);export_request();}}
};
inline void completed(int produced){if(enabled()&&writer()&&state().active){state().produced=produced;state().complete=true;}}
inline void token(int id,int64_t pos){if(enabled()&&writer()&&state().active){auto old=state().ctx;state().ctx=state().last_complete;state().ctx.pos=pos;event(TEmit,0,0,id);state().ctx=old;}}
inline void graph_mark(uintptr_t ptr,bool create){auto&s=state();if(!s.on||!writer()||!s.active)return;bool known=false;for(size_t i=0;i<s.graph_count;++i)known=known||(s.graphs[i].ptr==ptr&&s.graphs[i].queue==s.ctx.queue);s.ctx.graph=graph(ptr,s.ctx.queue,create);event(create?GraphCreate:(known?GraphHit:GraphPreexisting),0);}
} // namespace strata::program::host_trace36
