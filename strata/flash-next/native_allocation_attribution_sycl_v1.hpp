// Draft actual SYCL owner identity; invoked only by enabled instrumentation.
#pragma once
#include "strata/core/native_allocation_attribution.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
namespace strata::core::allocation_attribution {
inline std::pair<std::string,std::string> usm_identity(sycl::queue& q){
 if(!enabled())return {}; if(q.get_backend()!=sycl::backend::ext_oneapi_level_zero)throw std::invalid_argument("actual LevelZero owning context required");auto device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());ze_device_properties_t properties{};properties.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;if(zeDeviceGetProperties(device,&properties)!=ZE_RESULT_SUCCESS)throw std::runtime_error("actual native device UUID unavailable");char uuid[ZE_MAX_DEVICE_UUID_SIZE*2+1];for(size_t i=0;i<ZE_MAX_DEVICE_UUID_SIZE;++i)std::snprintf(uuid+i*2,3,"%02x",properties.uuid.id[i]);char context[64];std::snprintf(context,sizeof(context),"ctx_%p",(void*)sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context()));return {uuid,context};
}
inline AllocationKey usm_key(const void* pointer,sycl::queue& q){if(!enabled())return {};auto identity=usm_identity(q);return {identity.first,identity.second,reinterpret_cast<uintptr_t>(pointer)};}
inline void usm_ready(const void* pointer,uint64_t bytes,int layer,int expert,sycl::queue& q,const char* boundary){if(!enabled())return;auto identity=usm_identity(q);expert_ready(pointer,bytes,layer,expert,boundary,identity.first,identity.second);}
inline void usm_success(void* pointer,uint64_t bytes,sycl::queue& q,int stage,const char* role){
 if(!enabled())return;if(!pointer||!bytes||stage<0||q.get_backend()!=sycl::backend::ext_oneapi_level_zero)throw std::invalid_argument("actual backed SYCL allocation identity required");auto kind=sycl::get_pointer_type(pointer,q.get_context());const char* space=kind==sycl::usm::alloc::device?"device_usm":kind==sycl::usm::alloc::shared?"shared_usm":kind==sycl::usm::alloc::host?"host_usm":nullptr;if(!space)throw std::invalid_argument("unbacked/foreign allocation refused");
 auto identity=usm_identity(q);allocation_success(pointer,bytes,space,identity.first,identity.second,stage,role);
}
}
