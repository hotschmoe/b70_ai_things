// CPU-only header lifetime control. No SYCL/model/device code.
#include "host_critical_path42_header_v1.hpp"
#include <cstdlib>
#include <cstdio>
#include <cstring>
int main(int argc,char** argv) {
    if(argc!=4)return 2;
    const bool on=std::strcmp(argv[1],"on")==0;
    ::setenv("STRATA_CRITICAL_PATH_TRACE",on?"1":"0",1);
    ::setenv("STRATA_CRITICAL_PATH_TRACE_DIR",argv[2],1);
    ::setenv("STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",1);
    namespace h=strata::program::host_trace36;
    if(!h::init(true))return 3;
    if(on){h::RequestScope request(1,1,0,0,1);h::completed(1);}
    h::exit_export();
    h::exit_export(); // explicit repeat and later atexit must not re-emit/delete
    if(h::enabled()||h::state().events!=nullptr)return 4;
    std::fprintf(stderr,"CPU_SYNTHETIC_HEADER_TERMINAL_PASS on=%d\n",int(on));
    std::fflush(stderr);
    if(std::strcmp(argv[3],"fast")==0)std::_Exit(0);
    return 0; // registered original atexit callback executes again, harmless
}
