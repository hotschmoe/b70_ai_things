// Doorbell loads and stores for the B70 EP sidecar.
// Ported from strata-ab/src-ep/sycl/include/strata/sycl_doorbell.hpp
// (Strata, Intel GPU cache-bypass handshake). Attribution retained.
// Trimmed: no intel_gpu_driver(), no host-mapped payload helpers that need it.
// Peer flags are device memory. sys_load / sys_store use uncached L1+L3 hints.
// Atomics and volatile do not bypass the Arc L3. A system-scope fence alone
// does not either. See docs/20261015_strata_ep_publication_scope.md.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <cstdint>

namespace b70ep {

using doorbell_uncached_read = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::read_hint<sycl::ext::intel::experimental::cache_control<
        sycl::ext::intel::experimental::cache_mode::uncached,
        sycl::ext::oneapi::experimental::cache_level::L1,
        sycl::ext::oneapi::experimental::cache_level::L3>>));

using doorbell_uncached_write = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::write_hint<sycl::ext::intel::experimental::cache_control<
        sycl::ext::intel::experimental::cache_mode::uncached,
        sycl::ext::oneapi::experimental::cache_level::L1,
        sycl::ext::oneapi::experimental::cache_level::L3>>));

inline uint32_t sys_load(const uint32_t* p) {
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_read> u(
        const_cast<uint32_t*>(p));
    return u[0];
}

inline void sys_store(uint32_t* p, uint32_t v) {
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_write> u(p);
    u[0] = v;
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

template <class T>
inline T sys_load_mapped(const T* p) {
    sycl::ext::oneapi::experimental::annotated_ptr<T, doorbell_uncached_read> u(const_cast<T*>(p));
    return u[0];
}

inline uint32_t load_uncached_u32(const uint32_t* p) {
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_read> u(
        const_cast<uint32_t*>(p));
    return u[0];
}

}  // namespace b70ep
