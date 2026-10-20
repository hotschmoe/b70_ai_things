// Expert-parallel transport for one process, two XPU queues.
// Ported from strata-ab/src-ep/sycl/src/kernels/cuda/ep_kernels.dp.cpp and
// include/strata/kernels/ep_kernels.hpp. Queue arguments replace Strata's
// void* stream. The publication fix is the system-scope group_barrier in
// ep_push (S28). ep_send_rows keeps Strata's default-scope barrier.
// Prefill reduce/finish and resident_plan_ep are not in this milestone.
#pragma once
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>

namespace b70ep {

void ep_copy_uncached(sycl::queue& q, void* dst, const void* src, size_t bytes);
void ep_bump(sycl::queue& q, uint32_t* ctr);
void ep_push(sycl::queue& q, const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes,
             int32_t* peer_ids, uint8_t* peer_xq, uint32_t* peer_flag, const uint32_t* ctr);
void ep_wait(sycl::queue& q, const uint32_t* flag, const uint32_t* ctr, uint32_t spin_max, uint32_t* err,
             uint32_t code);
void ep_wait_value(sycl::queue& q, const uint32_t* flag, uint32_t value, uint32_t spin_max, uint32_t* err,
                   uint32_t code);
void ep_send_rows(sycl::queue& q, float* peer_rows, const float* rows, const int32_t* ids, const int32_t* res0,
                  int n, int64_t row, uint32_t* done, uint32_t* peer_flag, const uint32_t* ctr);

// Posted float4 copy, then the last work-group stores *ctr into peer_flag.
// ep_push_signal in Strata takes a host value; a host value freezes inside a graph.
void ep_push_ctr(sycl::queue& q, void* peer_dst, const void* src, int64_t n4, uint32_t* done, uint32_t* peer_flag,
                 const uint32_t* ctr);

// y[i] = a[i] + uncached load of b[i]. n4 is the number of float4s.
void ep_add_uncached(sycl::queue& q, float* y, const float* a, const float* b, int64_t n4);
void ep_add_uncached_bf16(sycl::queue& q, uint16_t* y, const float* a, const float* b, int64_t n4);

// Remap expert ids and push weights plus bf16 activations. One kernel, system barrier, then flag.
// No maps: ids0[i] = id if 0 <= id < n0 else -1. peer_ids[i] = id - n0 if n0 <= id < n0+e1 else -1.
// With maps (Flash-Next snake): ids0[i] = map0[id] when that slot is in [0, n0), else -1.
// peer_ids[i] = map1[id] when that slot is in [0, e1), else -1. Out-of-range ids are -1.
// map0 and map1 are int32[nexp] on the writer device. Both null keeps the contiguous half split.
void ep_dispatch(sycl::queue& q, const int32_t* ids, const float* w, const uint16_t* x, int32_t* ids0,
                 int32_t* peer_ids, float* peer_w, uint16_t* peer_x, int n_ids, size_t x_bytes, int n0, int e1,
                 const int32_t* map0, const int32_t* map1, int nexp, uint32_t* done, uint32_t* peer_flag,
                 const uint32_t* ctr);

// Probe-local push. Scoped=false is the S28 mirror (default barrier). Not the production path.
template <bool Scoped>
void ep_push_probe(sycl::queue& q, const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes,
                   int32_t* peer_ids, uint8_t* peer_xq, uint32_t* peer_flag, const uint32_t* ctr);

// One scalar store per 4 KiB page through the peer pointer, then the owner
// reads the first and last words. Host-waits. Returns 1 on a match.
// A dense peer parallel_for of a tight allocation lost the device (H1b);
// this touches pages only, and the allocation is padded by one page.
int ep_touch_pages(sycl::queue& writer, void* remote, sycl::queue& owner, void* local, size_t bytes);

// Uncached store of one flag word per 4 KiB page. Host-waits.
void ep_store_flag_pages(sycl::queue& writer, uint32_t* remote, int pages, uint32_t value);

}  // namespace b70ep
