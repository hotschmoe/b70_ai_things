// ROOT-only future CPU host compile/run; no SYCL, weights, or GPU dependency.
#include "host_critical_path36_header_v2.hpp"
#include <cassert>
#include <csignal>
#include <dirent.h>
#include <sys/resource.h>
#include <string>
#include <thread>
using namespace strata::program::host_trace36;
int descriptors(){DIR*d=opendir("/proc/self/fd");assert(d);int n=0;while(readdir(d))++n;closedir(d);return n;}
int main(int argc,char**argv) {
    assert(argc==3);const std::string mode=argv[1];
    if(mode=="off")unsetenv("STRATA_CRITICAL_PATH_TRACE");else setenv("STRATA_CRITICAL_PATH_TRACE","1",1);
    setenv("STRATA_CRITICAL_PATH_TRACE_DIR",argv[2],1);
    setenv("STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",1);
    if(mode=="invalid"){assert(!init(false));assert(state().events==nullptr);std::puts("HOST_TRACE36_CPU invalid PASS");return 0;}
    assert(init(true));
    const int fd_before=descriptors();
    if(mode=="writefail"){signal(SIGXFSZ,SIG_IGN);rlimit limit{1,1};assert(setrlimit(RLIMIT_FSIZE,&limit)==0);}
    {
        RequestScope request(3,1,0,0);
        if(mode=="off"){Scope scope(Verify);event(TEmit,0,0,16);assert(!enabled()&&state().events==nullptr&&state().used==0&&state().request==0);}
        else {
            Frame frame(0,0,48,1,2,16);Scope verify(Verify);
            const auto reset=begin(Reset);end(Reset,reset);
            {Scope prompt(PromptSpan);}
            graph_mark(32,mode!="preexisting");const auto original=state().ctx.graph;
            if(mode=="reuse"){graph_mark(32,true);assert(state().ctx.graph!=original&&state().graph_count==1);assert(graph(32,16)==state().ctx.graph);}
            if(mode=="foreign"){
                const auto used=state().used,span=state().span,requests=state().request,graphs=state().graph_seq;
                std::thread worker([]{Scope s(Capture);Frame f(1,32,48,2,0,99);RequestScope r(1,1,0,0);graph_mark(99,true);completed(99);token(99,99);now();});worker.join();
                assert(state().thread_error.load()&&state().used==used&&state().span==span&&state().request==requests&&state().graph_seq==graphs&&state().ctx.dev==0);
            }
            if(mode=="overflow")for(size_t i=0;i<capacity+2;++i)event(GraphHit,0);
            const auto submit=begin(ReplaySubmit);end(ReplaySubmit,submit);
            const auto wait=begin(ReplayWait);end(ReplayWait,wait);
            token(16,3);const auto done=begin(DoneEmit);end(DoneEmit,done);completed(1);
            if(mode=="overflow")assert(state().overflow&&state().used==capacity);
        }
    }
    assert(descriptors()==fd_before);
    if(mode=="writefail")assert(state().export_error);
    if(mode=="limit"){for(int i=0;i<4;++i){RequestScope excess(3,1,0,0);}assert(state().request_limit&&state().request==5);}
    std::printf("HOST_TRACE36_CPU %s PASS; synthetic host control only\n",mode.c_str());
}
