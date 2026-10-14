#include <cstdio>
#include <cstring>
#include <unistd.h>
#include "strata/core/ep_mem.hpp"
using namespace strata::core;
int main() {
  if (dpct::device_count()!=2) { std::puts("FAIL expected two devices"); return 2; }
  auto d0=ep_ze_device(0), d1=ep_ze_device(1);
  for (int owner=0;owner<2;++owner) {
    int other=1-owner;
    auto od=ep_ze_device(owner), pd=ep_ze_device(other);
    ze_bool_t access=0;
    auto z=zeDeviceCanAccessPeer(od,pd,&access);
    ze_device_p2p_properties_t props{}; props.stype=ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
    auto zp=zeDeviceGetP2PProperties(od,pd,&props);
    std::printf("PEER %d->%d can_rc=0x%x access=%u props_rc=0x%x flags=0x%x\n",owner,other,z,access,zp,props.flags);
    auto oc=ep_ze_context(owner), pc=ep_ze_context(other);
    std::printf("CONTEXT %d owner=%p other=%p\n",owner,(void*)oc,(void*)pc);
    for (int warmed=0;warmed<2;++warmed) {
      void* local=nullptr; void* remote=nullptr;
      ze_device_mem_alloc_desc_t desc{}; desc.stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
      z=zeMemAllocDevice(oc,&desc,1048576,4096,od,&local);
      if(z) { std::printf("ALLOC rc=0x%x\n",z); return 3; }
      if(warmed) dpct::dev_mgr::instance().get_device(owner).in_order_queue().memset(local,0,1048576).wait();
      ze_ipc_mem_handle_t handle{};
      z=zeMemGetIpcHandle(oc,local,&handle);
      int fd=-1;std::memcpy(&fd,handle.data,sizeof(fd));
      std::printf("EXPORT owner=%d warm=%d rc=0x%x fd=%d\n",owner,warmed,z,fd);
      z=zeMemOpenIpcHandle(pc,pd,handle,0,&remote);
      std::printf("IMPORT owner=%d warm=%d rc=0x%x\n",owner,warmed,z);
      if(z==ZE_RESULT_SUCCESS) zeMemCloseIpcHandle(pc,remote);
      remote=nullptr;
      z=zeMemOpenIpcHandle(oc,od,handle,0,&remote);
      std::printf("SELF owner=%d warm=%d rc=0x%x\n",owner,warmed,z);
      if(z==ZE_RESULT_SUCCESS) zeMemCloseIpcHandle(oc,remote);
      zeMemFree(oc,local);
    }
  }
  return 0;
}
