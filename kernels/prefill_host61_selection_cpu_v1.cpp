// CPU-only real diagnostic-header selection/lifecycle control; no SYCL/model.
#include "host_trace_actual_chunk_v1.hpp"
#include <cassert>
#include <string>
#include <unistd.h>
int main(){
    setenv("STRATA_PREFILL_HOST60","1",1);
    FILE* capture=std::tmpfile();assert(capture);
    const int saved=dup(STDERR_FILENO);assert(saved>=0);
    assert(dup2(fileno(capture),STDERR_FILENO)>=0);
    strata::prefill::host60::set_owner(1,1,0);
    {
        strata::prefill::host60::Trace t(reinterpret_cast<void*>(0x1234),0,24,1512);
        t.mark("preparation");t.chunk(24,512);t.mark("first_actual_chunk");t.finish_chunk();
        t.chunk(536,512);t.mark("MUST_NOT_RECORD_SECOND_CHUNK");
    }
    std::fflush(stderr);assert(dup2(saved,STDERR_FILENO)>=0);close(saved);
    std::rewind(capture);std::string text;char buf[4096];
    while(std::fgets(buf,sizeof(buf),capture))text+=buf;
    std::fclose(capture);
    assert(text.find("T=512 input_n=1512")!=std::string::npos);
    assert(text.find("phase=first_actual_chunk")!=std::string::npos);
    assert(text.find("phase=chunk_layers_submitted")!=std::string::npos);
    assert(text.find("MUST_NOT_RECORD_SECOND_CHUNK")==std::string::npos);
    assert(text.find("rows=4 overflow=0 deferred=1 host_only=1")!=std::string::npos);
    std::puts("PFH61_SELECTION_CPU_PASS outer1512_actual512=1_after_first_roi_ignored=1_deferred_rows=4 no_model_no_GPU=1");
}
