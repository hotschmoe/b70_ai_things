#pragma once
// Source prototype only. Startup/runtime ownership metadata, no device waits.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <variant>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace strata::program::serial_pair_mapping_v1 {
inline bool enabled(){static const bool on=[] {const char*p=std::getenv("STRATA_SERIAL_PAIR_RAW_METADATA");if(!p||!std::strcmp(p,"0"))return false;if(std::strcmp(p,"1"))throw std::invalid_argument("STRATA_SERIAL_PAIR_RAW_METADATA=0|1");return true;}();return on;}
inline void record(const char*role,int logical_device,int lb,int le,sycl::queue&q,uint64_t rid=0,uint64_t slotgen=0,int64_t pos=-1,int64_t T=0){
 if(!enabled())return;
 const char*binding=std::getenv("STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256");if(!binding||std::strlen(binding)!=64||!role||!((lb==0&&le==24)||(lb==24&&le==48))||logical_device!=(lb==0?0:1))throw std::invalid_argument("Exact serial24/24 owner/binding required");
 for(int i=0;i<64;i++)if(!((binding[i]>='0'&&binding[i]<='9')||(binding[i]>='a'&&binding[i]<='f')))throw std::invalid_argument("Mapping binding encoding");
 static std::atomic<unsigned>next{0};const unsigned seq=next.fetch_add(1,std::memory_order_relaxed);if(seq>=32)throw std::invalid_argument("Mapping row cap32 exceeded");
 if(q.get_backend()!=sycl::backend::ext_oneapi_level_zero)throw std::invalid_argument("Actual native LevelZero owner required");
 // Native queue conversion may materialize a backend handle. Prefer startup
 // hooks before timed requests; never interpret this as a device execution query.
 const auto native=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);const char*kind=nullptr;uintptr_t handle=0;
 if(std::holds_alternative<ze_command_list_handle_t>(native)){kind="immediate_CL";handle=reinterpret_cast<uintptr_t>(std::get<ze_command_list_handle_t>(native));}
 else if(std::holds_alternative<ze_command_queue_handle_t>(native)){kind="regular_CQ";handle=reinterpret_cast<uintptr_t>(std::get<ze_command_queue_handle_t>(native));}
 else throw std::invalid_argument("Unrecognized native queue variant");
 const auto dev=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());const auto ctx=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());timespec ts{};if(!handle||!dev||!ctx||clock_gettime(CLOCK_MONOTONIC_RAW,&ts))throw std::invalid_argument("Actual owner handles/RAW clock missing");
 std::fprintf(stderr,"STAGE_NATIVE_RAW schema=1 seq=%u pid=%ld tid=%ld binding=%s role=%s logical_device=%d lb=%d le=%d rid=%llu slotgen=%llu pos=%lld T=%lld sycl_queue=%llu native_kind=%s native_handle=%llu device_handle=%llu context_handle=%llu RAW_ns=%llu no_added_explicit_device_wait=1\n",seq,long(getpid()),long(syscall(SYS_gettid)),binding,role,logical_device,lb,le,(unsigned long long)rid,(unsigned long long)slotgen,(long long)pos,(long long)T,(unsigned long long)reinterpret_cast<uintptr_t>(&q),kind,(unsigned long long)handle,(unsigned long long)reinterpret_cast<uintptr_t>(dev),(unsigned long long)reinterpret_cast<uintptr_t>(ctx),(unsigned long long)(uint64_t(ts.tv_sec)*1000000000ULL+uint64_t(ts.tv_nsec)));
}
}
