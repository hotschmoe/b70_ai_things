// Root-only CPU clock-writer fixture. /results must be a fresh writable mount.
#include "pti_clock_explicit_finalize_v1.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    ::setenv("UNITRACE_CLOCK_METADATA_PATH","/results/unitrace/clock.{pid}.jsonl",1);
    ::setenv("UNITRACE_HOST_CLOCK_BRIDGE","1",1);
    b70_pti_clock_v1::Emit("host_clock_origin",{{"epoch_origin_ns",1},{"monotonic_raw_minus_boot_i64_bits",0},{"monotonic_raw_minus_tai_i64_bits",0}});
    b70_pti_clock_v1::HostBridge(1);
    if(!b70_pti_clock_v1::CloseOnce()||!b70_pti_clock_v1::CloseOnce())return 3;
    if(std::strcmp(argv[1],"late")==0) {
        b70_pti_clock_v1::Emit("host_clock_origin",{{"epoch_origin_ns",1}});
        return 4; // forbidden emission must terminate with existing code86
    }
    std::puts("CPU_SYNTHETIC_EXPLICIT_CLOCK_CLOSE_ONCE_PASS");std::fflush(stdout);
    if(std::strcmp(argv[1],"fast")==0)std::_Exit(0);
    return 0; // later original destructor must not emit a second final bridge
}
