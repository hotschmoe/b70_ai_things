// S84 private prefill scorer. Original multi body copied byte-for-byte below.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/qsa_select.hpp"
namespace s84 {
using namespace strata::kernels;
constexpr int MQ=8,IDX_HEADS=4,IDX_DIM=128,R=4,SCORE_WARPS=8;
__dpct_inline__ void block_scores_multi_kernel(
    const float *__restrict__ pooled, const float *__restrict__ dead,
    const float *__restrict__ q_idx, const int32_t *__restrict__ steps, int nq,
    int64_t max_blocks, float *__restrict__ out, bool early_exit) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &qs = *sycl::ext::oneapi::group_local_memory_for_overwrite<
    float[MQ * IDX_HEADS * IDX_DIM]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nkv =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[MQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_nbid =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int64_t[MQ]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (!early_exit)
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < nq * IDX_HEADS * IDX_DIM;
             i += item_ct1.get_local_range(2)) qs[i] = q_idx[i];
    if (item_ct1.get_local_id(2) < nq) {
        s_nkv[item_ct1.get_local_id(2)] =
            steps[item_ct1.get_local_id(2) * kStepCount + kStepNKv];
        s_nbid[item_ct1.get_local_id(2)] =
            steps[item_ct1.get_local_id(2) * kStepCount + kStepNBid];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    int64_t top = 0;
#pragma unroll
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if (early_exit) {
        // (#783, stuchapin909) a CTA whose first block is past the batch's largest n_bid has no work: the grid is a
        // fixed 256 CTAs and a decode window reaches only a few of them, so those return before staging the queries.
        // Every thread of the CTA takes the same branch (top and blockIdx are CTA-uniform), so the barrier is safe.
        if ((int64_t)item_ct1.get_group(2) * SCORE_WARPS > top) return;
#pragma unroll
        for (int i = item_ct1.get_local_id(2); i < nq * IDX_HEADS * IDX_DIM;
             i += item_ct1.get_local_range(2)) qs[i] = q_idx[i];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    const int lane = item_ct1.get_local_id(2) & 31;
    const int64_t wstride = (int64_t)item_ct1.get_group_range(2) * SCORE_WARPS;
    for (int64_t b = (int64_t)item_ct1.get_group(2) * SCORE_WARPS +
                     (item_ct1.get_local_id(2) >> 5);
         b <= top && b < max_blocks; b += wstride) {
        const sycl::float4 kp = *reinterpret_cast<const sycl::float4 *>(
            pooled + b * IDX_DIM + lane * 4);
        const sycl::float4 kd =
            *reinterpret_cast<const sycl::float4 *>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const sycl::float4 k4 = (b == n_bid) ? kd : kp;
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sycl::float4 q4 =
                    *reinterpret_cast<const sycl::float4 *>(q + h * IDX_DIM);
                float d = k4.x() * q4.x() + k4.y() * q4.y() + k4.z() * q4.z() +
                          k4.w() * q4.w();
#pragma unroll
                /*
                DPCT1108: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                for (int o = 16; o > 0; o >>= 1) d +=
                    dpct::experimental::permute_sub_group_by_xor(
                        0xffffffffu,
                        sycl::ext::oneapi::this_work_item::get_sub_group(), d,
                        o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}
inline void launch(const float* pooled,const float* dead,const float* query,
                   const int32_t* steps,int nq,int64_t stride,float* scores,sycl::queue& q){
    // Caller validates exact host step/active bounds. Grid's Y is independent query tiles.
    q.parallel_for<class TiledScores>(sycl::nd_range<3>(
        sycl::range<3>(1,(nq+MQ-1)/MQ,256*SCORE_WARPS*32),
        sycl::range<3>(1,1,SCORE_WARPS*32)),
        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
            const int q0=int(it.get_group(1))*MQ;
            const int n=(nq-q0<MQ?nq-q0:MQ);
            block_scores_multi_kernel(pooled,dead,query+size_t(q0)*512,
                steps+size_t(q0)*kStepCount,n,stride,scores+size_t(q0)*stride,true);
        });
}
} // namespace s84
