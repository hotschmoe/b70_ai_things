// Standalone CPU-testable owner/idempotence guard; no GPU/model dependencies.
#ifndef B70_PTI_MODEL_FINALIZE_GUARD_V1_H
#define B70_PTI_MODEL_FINALIZE_GUARD_V1_H
#include <atomic>
#include <cstdint>
#include <unistd.h>
#include <sys/syscall.h>
namespace b70_pti_finalize_v1 {
struct Guard {
    std::atomic<bool> ready{false}, failed{false};
    uint64_t owner_pid=0, owner_tid=0;
    bool teardown_done=false, clock_done=false;
    void Initialize() noexcept {
        if(ready.load(std::memory_order_acquire)){failed.store(true,std::memory_order_release);return;}
        owner_pid=uint64_t(::getpid());owner_tid=uint64_t(::syscall(SYS_gettid));
        ready.store(true,std::memory_order_release);
    }
    template<class Teardown,class Close>
    int Finalize(uint32_t version,uint64_t pid,uint64_t tid,Teardown teardown,Close close) noexcept {
        if(!ready.load(std::memory_order_acquire))return 1;
        if(version!=1||pid!=owner_pid||tid!=owner_tid||pid!=uint64_t(::getpid())||tid!=uint64_t(::syscall(SYS_gettid))){failed.store(true,std::memory_order_release);return 2;}
        // Only the immutable owner touches callback completion flags.
        try {
            if(!teardown_done){teardown();teardown_done=true;}
            if(!clock_done){if(!close()){failed.store(true,std::memory_order_release);return 3;}clock_done=true;}
        } catch(...) {failed.store(true,std::memory_order_release);return 4;}
        return failed.load(std::memory_order_acquire)?5:0;
    }
};
inline Guard& state(){static Guard guard;return guard;}
} // namespace b70_pti_finalize_v1
#endif
