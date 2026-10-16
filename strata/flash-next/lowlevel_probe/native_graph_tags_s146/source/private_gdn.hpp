// S145 SOURCE ONLY: exact current SYCL bodies with unique private names.
#pragma once
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <stdexcept>
namespace s145 {
constexpr int S=128,RG=4,RPG=S/RG;
__dpct_inline__ void
s145_conv_body(float *__restrict__ hist, const float *__restrict__ qkv,
                       int C, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

__dpct_inline__ void s145_state_body(
    float *__restrict__ state, const float *__restrict__ hbuf, int C,
    const float *__restrict__ gate, const float *__restrict__ beta, int h_k,
    int h_v, const int32_t *__restrict__ n_keep) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int n = *n_keep;
    if (n <= 0) return;
    auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2);
    const int col_local = item_ct1.get_local_id(2);
    const int col = item_ct1.get_group(1) * 32 + col_local;
    const int rg = item_ct1.get_local_id(1);
    const int tid = rg * 32 + col_local;    // 0 .. 127
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        sk[tid] = ht[qk + qh * S + tid];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float g = sycl::native::exp(gate[(size_t)t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

// Host adapter, fixed real geometry. One call/owner submits two kernel roles.
inline void commit_layers(sycl::queue&q,float*state,const float*qkv,const float*h,
                          const float*gate,const float*beta,const int32_t*keep) {
 constexpr int C=10240,HV=48,HK=16,L=36,MT=8;
 constexpr int64_t STRIDE=int64_t(S)*HV*S+int64_t(C)*3,CONV=int64_t(S)*HV*S;
 q.parallel_for<class s145_conv_layers>(
  sycl::nd_range<3>(sycl::range<3>(1,L,(C+255)/256)*sycl::range<3>(1,1,256),sycl::range<3>(1,1,256)),
  [=](sycl::nd_item<3>item){const int64_t l=item.get_group(1);
   s145_conv_body(state+l*STRIDE+CONV,qkv+l*MT*C,C,keep);
  });
 q.parallel_for<class s145_state_layers>(
  sycl::nd_range<3>(sycl::range<3>(L,4,HV)*sycl::range<3>(1,RG,32),sycl::range<3>(1,RG,32)),
  [=](sycl::nd_item<3>item){const int64_t l=item.get_group(0);
   s145_state_body(state+l*STRIDE,h+l*MT*C,C,gate+l*MT*HV,beta+l*MT*HV,HK,HV,keep);
  });
}
} // namespace s145
