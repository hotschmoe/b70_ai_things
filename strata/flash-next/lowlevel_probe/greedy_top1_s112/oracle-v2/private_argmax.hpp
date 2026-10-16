// Isolated S112 private candidate; original production source/archive remain untouched.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include <algorithm>
namespace s112_private {
constexpr int kArgMaxBlocks = 128, kArgThreads = 256, kTopBlocks = 8;

// the larger value, on equality the lower index: sampler_greedy_kernel's order
__dpct_inline__ void arg_take(float ov, int oi, float &bv, int &bi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// a block's (value, index) pick, in thread 0; `none` is the index of no candidate
__dpct_inline__ void arg_block(float &bv, int &bi, int none) {
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sv = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    /*
    DPCT1115: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &si = *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int off = 16; off > 0; off >>= 1) {
        /*
        DPCT1108: '__shfl_down_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        /*
        DPCT1121: Make sure that the "bv" which is used in the SYCL group
        function/algorithm is initialized.
        */
        const float ov = dpct::experimental::shift_sub_group_left(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), bv,
            off);
        /*
        DPCT1108: '__shfl_down_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        /*
        DPCT1121: Make sure that the "bi" which is used in the SYCL group
        function/algorithm is initialized.
        */
        const int oi = dpct::experimental::shift_sub_group_left(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), bi,
            off);
        arg_take(ov, oi, bv, bi);
    }
    const int warp = (int)(item_ct1.get_local_id(2) >> 5),
              lane = (int)(item_ct1.get_local_id(2) & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0) {
        const int nw = (int)(item_ct1.get_local_range(2) >> 5);
        bv = lane < nw ? sv[lane] : sycl::bit_cast<float, int>(0xff800000);
        bi = lane < nw ? si[lane] : none;
        for (int off = 16; off > 0; off >>= 1) {
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            /*
            DPCT1121: Make sure that the "bv" which is used in the SYCL
            group function/algorithm is initialized.
            */
            const float ov = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bv, off);
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            /*
            DPCT1121: Make sure that the "bi" which is used in the SYCL
            group function/algorithm is initialized.
            */
            const int oi = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bi, off);
            arg_take(ov, oi, bv, bi);
        }
    }
}

// A row's scratch, whatever the launch's row count (one scratch serves windows of every size): its counter, then
// the blocks' values and indices.
struct ArgRow {
    unsigned counter;
    unsigned pad[3];
    float v[kArgMaxBlocks];
    int i[kArgMaxBlocks];
};

// grid (blocks, rows): block b scans [b * per_block, +per_block) of its row
__dpct_inline__ void argmax_rows_kernel(const float *__restrict__ logits, int n,
                                        int per_block,
                                        ArgRow *__restrict__ rows,
                                        int32_t *out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = (int)item_ct1.get_group(1), b = (int)item_ct1.get_group(2),
              nb = (int)item_ct1.get_group_range(2);
    unsigned* counter = &rows[row].counter;
    float* pv = rows[row].v;
    int* pi = rows[row].i;
    const float* l = logits + (size_t) row * n;
    const int lo = b * per_block, hi = sycl::min(n, lo + per_block);
    float bv = sycl::bit_cast<float, int>(0xff800000); // -inf
    int bi = n;
    for (int v = lo + (int)item_ct1.get_local_id(2); v < hi;
         v += (int)item_ct1.get_local_range(2)) {
        const float s = l[v];
        if (s > bv) { bv = s; bi = v; }
    }
    arg_block(bv, bi, n);
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<bool>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) {
        pv[b] = bv;
        pi[b] = bi;
        /*
        DPCT1078: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        const unsigned s112_prior =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(counter, 1u);
        // Sole hardening: acquire on the RMW leader before publishing last to its WG.
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
        last = s112_prior == (unsigned)(nb - 1);
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return;
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    bv = sycl::bit_cast<float, int>(0xff800000);
    bi = n;
    if ((int)item_ct1.get_local_id(2) < nb) {
        bv = ((volatile const float *)pv)[item_ct1.get_local_id(2)];
        bi = ((volatile const int *)pi)[item_ct1.get_local_id(2)];
    }
    arg_block(bv, bi, n);
    if (item_ct1.get_local_id(2) == 0) {
        out[row] = bi < n ? bi : 0;   // no value above -inf: 0, as the one-block kernel
        *counter = 0;
    }
}


inline void launch(const float* logits,int n,void* scratch,int32_t* out,sycl::queue& q) {
    const int nb=std::min(kArgMaxBlocks,std::max(1,(n+4095)/4096));
    const int per=(n+nb-1)/nb;
    const auto props=sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::use_root_sync};
    q.parallel_for<dpct_kernel_name<class GreedyTop1S112>>(
      sycl::nd_range<3>(sycl::range(1,1,nb)*sycl::range(1,1,kArgThreads),sycl::range(1,1,kArgThreads)),
      props,[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
        argmax_rows_kernel(logits,n,per,static_cast<ArgRow*>(scratch),out);
      });
}
}
