// Default-OFF normal native fast exit is unchanged; diagnostic loaded symbol only.
#ifndef STRATA_PTI_NATIVE_FINALIZE43_V1_HPP
#define STRATA_PTI_NATIVE_FINALIZE43_V1_HPP
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/syscall.h>
namespace strata::program {
inline bool diagnostic_pti_finalize43() noexcept {
    const char* flag=std::getenv("STRATA_DIAGNOSTIC_PTI_FINALIZE");
    if(!flag||std::strcmp(flag,"0")==0)return true;
    if(std::strcmp(flag,"1")!=0)return false;
    const char* role=std::getenv("UNITRACE_MODEL_FINALIZE_V1");
    if(!role||std::strcmp(role,"1")!=0)return false;
    using Finalizer=int(*)(uint32_t,uint64_t,uint64_t) noexcept;
    void* symbol=::dlsym(RTLD_DEFAULT,"b70_pti_model_finalize_v1");
    if(!symbol)return false;
    Dl_info info{};char path[PATH_MAX]{};
    if(!::dladdr(symbol,&info)||!info.dli_fname||!::realpath(info.dli_fname,path)||std::strcmp(path,"/pti/lib/libunitrace_tool.so")!=0)return false;
    const auto finalize=reinterpret_cast<Finalizer>(symbol);
    return finalize(1,uint64_t(::getpid()),uint64_t(::syscall(SYS_gettid)))==0;
}
} // namespace strata::program
#endif
