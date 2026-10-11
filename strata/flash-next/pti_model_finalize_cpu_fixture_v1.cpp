// Root-only standalone CPU guard fixture: no PTI/GPU/model library.
#include "pti_model_finalize_guard_v1.hpp"
#include <cstdio>
#include <stdexcept>
int main() {
    using b70_pti_finalize_v1::Guard;
    const uint64_t pid=uint64_t(::getpid()),tid=uint64_t(::syscall(SYS_gettid));
    int teardown=0,close=0;Guard owned;owned.Initialize();
    auto td=[&]{++teardown;};auto cl=[&]{++close;return true;};
    if(owned.Finalize(1,pid,tid,td,cl)!=0||owned.Finalize(1,pid,tid,td,cl)!=0||teardown!=1||close!=1)return 2;
    Guard foreign;foreign.Initialize();int other=0;
    if(foreign.Finalize(1,pid+1,tid,[&]{++other;},[]{return true;})!=2||other)return 3;
    if(foreign.Finalize(1,pid,tid,[&]{++other;},[]{return true;})!=5||other!=1)return 4;
    Guard partial;partial.Initialize();int tries=0,finished=0;
    auto throws=[&]{if(++tries==1)throw std::runtime_error("partial teardown");};
    if(partial.Finalize(1,pid,tid,throws,[&]{++finished;return true;})!=4)return 5;
    if(partial.Finalize(1,pid,tid,throws,[&]{++finished;return true;})!=5||tries!=2||finished!=1)return 6;
    Guard clock;clock.Initialize();int clocks=0;
    if(clock.Finalize(1,pid,tid,[]{},[&]{return ++clocks>1;})!=3)return 7;
    if(clock.Finalize(1,pid,tid,[]{},[&]{return ++clocks>1;})!=5||clocks!=2)return 8;
    std::puts("CPU_SYNTHETIC_PTI_FINALIZER_OWNED_IDEMPOTENT_PARTIAL_FAILURE_PASS");return 0;
}
