// Port of strata-ab/src-ep/sycl/src/kernels/cuda/ep_kernels.dp.cpp
// (Strata expert-parallel transport). Bodies of ep_copy_uncached, ep_bump,
// ep_push, ep_wait, ep_wait_value, and ep_send_rows match that file, with
// sycl::queue& in place of q_of(stream). ep_push's barrier is
// memory_scope::system (the S28 one-line fix). ep_send_rows keeps the
// default-scope barrier from ep_kernels.dp.cpp.
#include "ep_kernels.hpp"
#include "doorbell.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace b70ep {
namespace {

using sys_rmw = sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::system>;

}  // namespace

void ep_copy_uncached(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    constexpr size_t WG = 256;
    const size_t words = bytes / 4;
    const size_t groups = std::min<size_t>((words + WG - 1) / WG, 64);
    uint32_t* d = static_cast<uint32_t*>(dst);
    const uint32_t* s = static_cast<const uint32_t*>(src);
    q.parallel_for<class b70_ep_copy_uncached_k>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (size_t i = it.get_global_id(0); i < words; i += groups * WG) d[i] = load_uncached_u32(s + i);
    });
}

void ep_bump(sycl::queue& q, uint32_t* ctr) {
    q.single_task<class b70_ep_bump_k>([=] { *ctr += 1; });
}

void ep_push(sycl::queue& q, const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes, int32_t* peer_ids,
             uint8_t* peer_xq, uint32_t* peer_flag, const uint32_t* ctr) {
    constexpr size_t WG = 256;
    const size_t words = xq_bytes / 4;
    q.parallel_for<class b70_ep_push_k>(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const size_t lid = it.get_local_id(0);
        for (size_t i = lid; i < (size_t)n_ids; i += WG) peer_ids[i] = ids[i];
        const uint32_t* s = reinterpret_cast<const uint32_t*>(xq);
        uint32_t* d = reinterpret_cast<uint32_t*>(peer_xq);
        for (size_t i = lid; i < words; i += WG) d[i] = s[i];
        // Keep cross-device ordering on the barrier itself. A default-scope
        // barrier can absorb the adjacent fence with only work-group scope.
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group(), sycl::memory_scope::system);
        if (lid == 0) sys_store(peer_flag, *ctr);
    });
}

void ep_wait(sycl::queue& q, const uint32_t* flag, const uint32_t* ctr, uint32_t spin_max, uint32_t* err,
             uint32_t code) {
    q.single_task<class b70_ep_wait_k>([=] {
        const uint32_t want = *ctr;
        uint32_t spin = 0;
        while ((int32_t)(sys_load(flag) - want) < 0) {
            if (++spin >= spin_max) {
                if (err != nullptr) sys_store(err, code);
                break;
            }
        }
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    });
}

void ep_wait_value(sycl::queue& q, const uint32_t* flag, uint32_t value, uint32_t spin_max, uint32_t* err,
                   uint32_t code) {
    q.single_task<class b70_ep_wait_value_k>([=] {
        uint32_t spin = 0;
        while ((int32_t)(sys_load(flag) - value) < 0) {
            if (++spin >= spin_max) {
                if (err != nullptr) sys_store(err, code);
                break;
            }
        }
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    });
}

void ep_send_rows(sycl::queue& q, float* peer_rows, const float* rows, const int32_t* ids, const int32_t* res0, int n,
                  int64_t row, uint32_t* done, uint32_t* peer_flag, const uint32_t* ctr) {
    if (n <= 0) return;
    constexpr size_t WG = 256;
    q.parallel_for<class b70_ep_send_rows_k>(sycl::nd_range<1>((size_t)n * WG, WG), [=](sycl::nd_item<1> it) {
        const int r = (int)it.get_group(0);
        const int32_t e = ids[r];
        if (!(e >= 0 && res0[e] >= 0)) {
            const sycl::float4* s = reinterpret_cast<const sycl::float4*>(rows + (size_t)r * row);
            sycl::float4* d = reinterpret_cast<sycl::float4*>(peer_rows + (size_t)r * row);
            for (int64_t i = (int64_t)it.get_local_id(0); i < row / 4; i += (int64_t)WG) d[i] = s[i];
        }
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
        if (it.get_local_id(0) == 0 && sys_rmw(*done).fetch_add(1u) == (uint32_t)n - 1) {
            *done = 0;
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            sys_store(peer_flag, *ctr);
        }
    });
}

void ep_push_ctr(sycl::queue& q, void* peer_dst, const void* src, int64_t n4, uint32_t* done, uint32_t* peer_flag,
                 const uint32_t* ctr) {
    if (n4 <= 0) return;
    constexpr size_t WG = 256;
    const size_t groups = (size_t)std::min<int64_t>((n4 + (int64_t)WG - 1) / (int64_t)WG, 4096);
    sycl::float4* d = static_cast<sycl::float4*>(peer_dst);
    const sycl::float4* s = static_cast<const sycl::float4*>(src);
    q.parallel_for<class b70_ep_push_ctr_k>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (int64_t i = (int64_t)it.get_global_id(0); i < n4; i += (int64_t)(groups * WG)) d[i] = s[i];
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group(), sycl::memory_scope::system);
        if (it.get_local_id(0) == 0 && sys_rmw(*done).fetch_add(1u) == (uint32_t)groups - 1) {
            *done = 0;
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            sys_store(peer_flag, *ctr);
        }
    });
}

void ep_add_uncached(sycl::queue& q, float* y, const float* a, const float* b, int64_t n4) {
    if (n4 <= 0) return;
    constexpr size_t WG = 256;
    const size_t groups = std::min<size_t>((size_t)((n4 + (int64_t)WG - 1) / (int64_t)WG), 64);
    q.parallel_for<class b70_ep_add_uncached_k>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (int64_t i = (int64_t)it.get_global_id(0); i < n4; i += (int64_t)(groups * WG)) {
            const uint32_t* bp = reinterpret_cast<const uint32_t*>(b) + (size_t)i * 4;
            const uint32_t w0 = load_uncached_u32(bp);
            const uint32_t w1 = load_uncached_u32(bp + 1);
            const uint32_t w2 = load_uncached_u32(bp + 2);
            const uint32_t w3 = load_uncached_u32(bp + 3);
            const sycl::float4 bv(sycl::bit_cast<float>(w0), sycl::bit_cast<float>(w1), sycl::bit_cast<float>(w2),
                                  sycl::bit_cast<float>(w3));
            const sycl::float4 av = reinterpret_cast<const sycl::float4*>(a)[i];
            reinterpret_cast<sycl::float4*>(y)[i] = av + bv;
        }
    });
}

// O4: same sum, written as bf16 (RNE). Equal to fp32 add then .to(bf16).
void ep_add_uncached_bf16(sycl::queue& q, uint16_t* y, const float* a, const float* b, int64_t n4) {
    if (n4 <= 0) return;
    constexpr size_t WG = 256;
    const size_t groups = std::min<size_t>((size_t)((n4 + (int64_t)WG - 1) / (int64_t)WG), 64);
    q.parallel_for<class b70_ep_add_uncached_bf16_k>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (int64_t i = (int64_t)it.get_global_id(0); i < n4; i += (int64_t)(groups * WG)) {
            const uint32_t* bp = reinterpret_cast<const uint32_t*>(b) + (size_t)i * 4;
            const sycl::float4 av = reinterpret_cast<const sycl::float4*>(a)[i];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v = av[j] + sycl::bit_cast<float>(load_uncached_u32(bp + j));
                y[(size_t)i * 4 + j] = sycl::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(v));
            }
        }
    });
}

void ep_dispatch(sycl::queue& q, const int32_t* ids, const float* w, const uint16_t* x, int32_t* ids0,
                 int32_t* peer_ids, float* peer_w, uint16_t* peer_x, int n_ids, size_t x_bytes, int n0, int e1,
                 const int32_t* map0, const int32_t* map1, int nexp, uint32_t* done, uint32_t* peer_flag,
                 const uint32_t* ctr) {
    constexpr size_t WG = 256;
    const size_t words = x_bytes / 4;
    size_t groups = (std::max(words, (size_t)std::max(n_ids, 1)) + WG - 1) / WG;
    if (groups < 1) groups = 1;
    if (groups > 64) groups = 64;
    const bool mapped = map0 != nullptr;
    q.parallel_for<class b70_ep_dispatch_k>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        const size_t stride = groups * WG;
        for (size_t i = it.get_global_id(0); i < (size_t)n_ids; i += stride) {
            const int id = ids[i];
            if (mapped) {
                const bool ok = id >= 0 && id < nexp;
                const int s0 = ok ? map0[id] : -1;
                const int s1 = ok ? map1[id] : -1;
                // A slot outside the local pack contributes 0, same as -1.
                ids0[i] = (s0 >= 0 && s0 < n0) ? s0 : -1;
                peer_ids[i] = (s1 >= 0 && s1 < e1) ? s1 : -1;
            } else {
                ids0[i] = (id >= 0 && id < n0) ? id : -1;
                peer_ids[i] = (id >= n0 && id < n0 + e1) ? id - n0 : -1;
            }
            peer_w[i] = w[i];
        }
        const uint32_t* s = reinterpret_cast<const uint32_t*>(x);
        uint32_t* d = reinterpret_cast<uint32_t*>(peer_x);
        for (size_t i = it.get_global_id(0); i < words; i += stride) d[i] = s[i];
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group(), sycl::memory_scope::system);
        if (it.get_local_id(0) == 0 && sys_rmw(*done).fetch_add(1u) == (uint32_t)groups - 1) {
            *done = 0;
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            sys_store(peer_flag, *ctr);
        }
    });
}

template <bool Scoped>
class b70_ep_push_probe_k;

template <bool Scoped>
void ep_push_probe(sycl::queue& q, const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes,
                   int32_t* peer_ids, uint8_t* peer_xq, uint32_t* peer_flag, const uint32_t* ctr) {
    constexpr size_t WG = 256;
    const size_t words = xq_bytes / 4;
    q.parallel_for<b70_ep_push_probe_k<Scoped>>(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const size_t lid = it.get_local_id(0);
        for (size_t i = lid; i < (size_t)n_ids; i += WG) peer_ids[i] = ids[i];
        const uint32_t* s = reinterpret_cast<const uint32_t*>(xq);
        uint32_t* d = reinterpret_cast<uint32_t*>(peer_xq);
        for (size_t i = lid; i < words; i += WG) d[i] = s[i];
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        if constexpr (Scoped) sycl::group_barrier(it.get_group(), sycl::memory_scope::system);
        else sycl::group_barrier(it.get_group());
        if (lid == 0) sys_store(peer_flag, *ctr);
    });
}

template void ep_push_probe<false>(sycl::queue&, const int32_t*, int, const uint8_t*, size_t, int32_t*, uint8_t*,
                                   uint32_t*, const uint32_t*);
template void ep_push_probe<true>(sycl::queue&, const int32_t*, int, const uint8_t*, size_t, int32_t*, uint8_t*,
                                  uint32_t*, const uint32_t*);

int ep_touch_pages(sycl::queue& writer, void* remote, sycl::queue& owner, void* local, size_t bytes) {
    if (remote == nullptr || local == nullptr || bytes < 4) return 0;
    const size_t pages = (bytes + 4095u) / 4096u;
    auto* dst = static_cast<uint32_t*>(remote);
    constexpr size_t WG = 256;
    writer
        .parallel_for<class b70_ep_touch_pages_k>(sycl::nd_range<1>(WG, WG),
                                                 [=](sycl::nd_item<1> it) {
                                                     for (size_t i = it.get_local_id(0); i < pages; i += WG) {
                                                         const size_t word = i * 1024u;
                                                         if ((word + 1u) * 4u <= bytes)
                                                             dst[word] = 0xA5A50000u + (uint32_t)i;
                                                     }
                                                 })
        .wait();
    const size_t last_word = (pages - 1u) * 1024u;
    if ((last_word + 1u) * 4u > bytes) return 0;
    uint32_t first = 0, last = 0;
    owner.memcpy(&first, local, 4).wait();
    owner.memcpy(&last, static_cast<uint32_t*>(local) + last_word, 4).wait();
    const uint32_t want_last = 0xA5A50000u + (uint32_t)(pages - 1u);
    return first == 0xA5A50000u && last == want_last;
}

void ep_store_flag_pages(sycl::queue& writer, uint32_t* remote, int pages, uint32_t value) {
    if (remote == nullptr || pages < 1) return;
    writer.single_task<class b70_ep_flag_pages_k>([=]() {
        for (int i = 0; i < pages; ++i) sys_store(remote + (size_t)i * 1024u, value);
    }).wait();
}

}  // namespace b70ep
