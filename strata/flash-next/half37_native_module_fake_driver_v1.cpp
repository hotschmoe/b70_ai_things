// Synthetic CPU ABI fixture only: no GPU driver or native GPU code.
#include <level_zero/ze_api.h>
#include <level_zero/ze_ddi.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
extern "C" ze_result_t ZE_APICALL zeModuleCreate(ze_context_handle_t,ze_device_handle_t,const ze_module_desc_t*,ze_module_handle_t*out,ze_module_build_log_handle_t*){*out=(ze_module_handle_t)(uintptr_t)0x100;return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeModuleDestroy(ze_module_handle_t){return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeModuleGetNativeBinary(ze_module_handle_t,size_t*size,uint8_t*out){if(std::getenv("HALF37_FAKE_NATIVE_FAIL"))return ZE_RESULT_ERROR_INVALID_ARGUMENT;if(!out){*size=64;return ZE_RESULT_SUCCESS;}std::memset(out,0,64);std::memcpy(out,"\x7f" "ELF",4);*size=64;return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeDeviceGetProperties(ze_device_handle_t,ze_device_properties_t*p){p->vendorId=0xffff;p->deviceId=0xffff;std::strcpy(p->name,"CPU_SYNTHETIC_NO_GPU");return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeKernelCreate(ze_module_handle_t,const ze_kernel_desc_t*,ze_kernel_handle_t*out){static uintptr_t next=0x200;*out=(ze_kernel_handle_t)++next;return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeKernelDestroy(ze_kernel_handle_t){return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeCommandListCreate(ze_context_handle_t,ze_device_handle_t,const ze_command_list_desc_t*,ze_command_list_handle_t*out){static uintptr_t next=0x300;*out=(ze_command_list_handle_t)++next;return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeCommandListCreateImmediate(ze_context_handle_t,ze_device_handle_t,const ze_command_queue_desc_t*,ze_command_list_handle_t*out){static uintptr_t next=0x400;*out=(ze_command_list_handle_t)++next;return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeCommandListDestroy(ze_command_list_handle_t){return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeCommandListReset(ze_command_list_handle_t){return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeCommandListAppendLaunchKernel(ze_command_list_handle_t,ze_kernel_handle_t,const ze_group_count_t*,ze_event_handle_t,uint32_t,ze_event_handle_t*){return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeCommandQueueExecuteCommandLists(ze_command_queue_handle_t,uint32_t,ze_command_list_handle_t*,ze_fence_handle_t){return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeCommandListCreateCloneExp(ze_command_list_handle_t,ze_command_list_handle_t*out){*out=(ze_command_list_handle_t)(uintptr_t)0x500;return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeCommandListImmediateAppendCommandListsExp(ze_command_list_handle_t,uint32_t,ze_command_list_handle_t*,ze_event_handle_t,uint32_t,ze_event_handle_t*){return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeGetModuleProcAddrTable(ze_api_version_t,ze_module_dditable_t*t){*t={};t->pfnCreate=zeModuleCreate;t->pfnDestroy=zeModuleDestroy;t->pfnGetNativeBinary=zeModuleGetNativeBinary;return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeGetKernelProcAddrTable(ze_api_version_t,ze_kernel_dditable_t*t){*t={};t->pfnCreate=zeKernelCreate;t->pfnDestroy=zeKernelDestroy;return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeGetCommandListProcAddrTable(ze_api_version_t,ze_command_list_dditable_t*t){*t={};t->pfnCreate=zeCommandListCreate;t->pfnCreateImmediate=zeCommandListCreateImmediate;t->pfnDestroy=zeCommandListDestroy;t->pfnReset=zeCommandListReset;t->pfnAppendLaunchKernel=zeCommandListAppendLaunchKernel;return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeGetCommandQueueProcAddrTable(ze_api_version_t,ze_command_queue_dditable_t*t){*t={};t->pfnExecuteCommandLists=zeCommandQueueExecuteCommandLists;return ZE_RESULT_SUCCESS;}
extern "C" ze_result_t ZE_APICALL zeGetCommandListExpProcAddrTable(ze_api_version_t,ze_command_list_exp_dditable_t*t){*t={};if(!std::getenv("HALF37_FAKE_NULL_EXP")){t->pfnCreateCloneExp=zeCommandListCreateCloneExp;t->pfnImmediateAppendCommandListsExp=zeCommandListImmediateAppendCommandListsExp;}return ZE_RESULT_SUCCESS;}
