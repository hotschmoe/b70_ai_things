// Device regions one card writes and the other reads.
// Allocation follows strata-ab/src-ep/sycl/include/strata/core/ep_mem.hpp:
// a dedicated zeMemAllocDevice (4096 align), never a carve-out of the SYCL
// USM pool. Small pooled mallocs page-faulted on the peer. IPC open when
// the two torch contexts differ. A same-context pointer is tried only when
// both queues share a sycl::context (probe 5).
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace b70ep {

struct Region {
    void* local = nullptr;
    void* remote = nullptr;
    ze_context_handle_t owner_ctx = nullptr;
    ze_context_handle_t other_ctx = nullptr;
    bool ipc = false;
    // path 3: sycl::aligned_alloc_device in the shared torch context.
    bool usm = false;
    std::optional<sycl::context> usm_ctx;
};

inline ze_context_handle_t ze_ctx(const sycl::context& c) {
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c);
}
inline ze_device_handle_t ze_dev(const sycl::device& d) {
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);
}

inline void ze_init_once() {
    static int once = 0;
    if (once) return;
    once = 1;
    zeInit(0);
}

// One extra page past the caller's bytes. A vectorized tail store can
// touch the next page even when the lane is predicated; on a tight
// zeMemAllocDevice that faults the peer and the queue comes back
// DEVICE_LOST. The kernels only index the requested bytes.
inline size_t region_bytes(size_t bytes) {
    const size_t padded = bytes + 4096u;
    return (padded + 4095u) & ~size_t(4095);
}

// path: 1 = IPC, 2 = same-context ze pointer (remote == local),
// 3 = same-context sycl::aligned_alloc_device (J.5 EU push allocator).
inline bool region_alloc(const sycl::queue& owner, const sycl::queue& other, size_t bytes, int path, Region& r,
                         std::string& err) {
    ze_init_once();
    r = Region{};
    r.owner_ctx = ze_ctx(owner.get_context());
    r.other_ctx = ze_ctx(other.get_context());
    const size_t alloc = region_bytes(bytes);
    if (path == 3) {
        void* p = sycl::aligned_alloc_device(4096, alloc, owner);
        if (p == nullptr) {
            err = "aligned_alloc_device failed bytes=" + std::to_string(alloc);
            return false;
        }
        r.local = r.remote = p;
        r.usm = true;
        r.usm_ctx = owner.get_context();
        return true;
    }
    ze_device_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    ze_result_t z = zeMemAllocDevice(r.owner_ctx, &desc, alloc, 4096, ze_dev(owner.get_device()), &r.local);
    if (z != ZE_RESULT_SUCCESS || r.local == nullptr) {
        err = "zeMemAllocDevice failed (" + std::to_string((unsigned)z) + ") bytes=" + std::to_string(bytes);
        return false;
    }
    if (path == 2) {
        r.remote = r.local;
        r.ipc = false;
        return true;
    }
    ze_ipc_mem_handle_t h{};
    z = zeMemGetIpcHandle(r.owner_ctx, r.local, &h);
    if (z == ZE_RESULT_SUCCESS) {
        z = zeMemOpenIpcHandle(r.other_ctx, ze_dev(other.get_device()), h, 0, &r.remote);
    }
    if (z != ZE_RESULT_SUCCESS || r.remote == nullptr) {
        zeMemFree(r.owner_ctx, r.local);
        r.local = nullptr;
        err = "zeMemOpenIpcHandle failed (" + std::to_string((unsigned)z) + ")";
        return false;
    }
    r.ipc = true;
    return true;
}

inline void region_free(Region& r) {
    if (r.usm) {
        if (r.local != nullptr && r.usm_ctx) sycl::free(r.local, *r.usm_ctx);
        r = Region{};
        return;
    }
    if (r.ipc && r.remote != nullptr) zeMemCloseIpcHandle(r.other_ctx, r.remote);
    if (r.local != nullptr && r.owner_ctx != nullptr) zeMemFree(r.owner_ctx, r.local);
    r = Region{};
}

}  // namespace b70ep
