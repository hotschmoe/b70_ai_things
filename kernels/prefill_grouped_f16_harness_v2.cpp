// Synthetic mixed-precision operator control, not model math or speed qualification.
#include <sycl/sycl.hpp>
#include "strata/prefill/gemm.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t LIMIT = size_t(128) << 20;
struct Buffers {
    sycl::queue& q;
    std::vector<void*> ptrs;
    size_t bytes = 0, host_bytes = 0;
    explicit Buffers(sycl::queue& queue): q(queue) { ptrs.reserve(96); }
    template<class T> T* take(size_t n) {
        if (n > (LIMIT - bytes) / sizeof(T)) throw std::runtime_error("128MiB device allocation cap");
        auto p = sycl::malloc_device<T>(n, q);
        if (!p) throw std::bad_alloc();
        ptrs.push_back(p); bytes += n * sizeof(T); return p;
    }
    template<class T> T* host(size_t n) {
        if (n > ((size_t(96)<<20)-host_bytes)/sizeof(T)) throw std::runtime_error("96MiB pinned host cap");
        auto p=sycl::malloc_host<T>(n,q);if(!p)throw std::bad_alloc();
        ptrs.push_back(p);host_bytes+=n*sizeof(T);return p;
    }
    void release() {
        q.wait_and_throw();
        while (!ptrs.empty()) { sycl::free(ptrs.back(), q); ptrs.pop_back(); }
    }
    ~Buffers() {
        try { q.wait_and_throw(); } catch (...) {}
        while (!ptrs.empty()) { try { sycl::free(ptrs.back(), q); } catch (...) {} ptrs.pop_back(); }
    }
};
uint16_t half(float x) {
    sycl::half h(x); uint16_t bits;
    static_assert(sizeof(h) == sizeof(bits)); std::memcpy(&bits, &h, 2); return bits;
}
float input(int expert, int row, int col) {
    return float((row * 3 + col * 5 + expert) % 17 - 8) * 0.125f;
}
float expected(int expert, int row, int col, int K) {
    int k = (col * 17 + expert * 13) % K;
    // Every product and sum is exactly representable in FP32, independent of BLAS reduction order.
    return input(expert, row, k) * 0.5f
         - input(expert, row, (k + 1) % K) * 0.25f
         + input(expert, row, (k + 11) % K) * 0.125f;
}
void run(sycl::queue& q, bool two_queue, int count, int N, int K) {
    sycl::queue copy(q.get_context(),q.get_device(),sycl::property::queue::in_order{});
    Buffers owned(q);
    struct Drain {
        sycl::queue &compute,&copy;
        ~Drain(){try{compute.wait_and_throw();}catch(...){}try{copy.wait_and_throw();}catch(...){}}
    } drain{q,copy}; // drains both queues before owned buffers are destroyed on any exception

    strata::prefill::Gemm gemm;
    std::string error;
    if (!gemm.init_external(&q, nullptr, 0, nullptr, 0, error)) throw std::runtime_error(error);
    constexpr int64_t row_cases[8] = {16, 32, 48, 64, 16, 32, 48, 64};
    std::array<uint16_t*,8> stage{}, hx_owned{}, hw_owned{};
    std::array<sycl::event,8> used{};
    std::array<const uint16_t*, 8> X{}, W{};
    std::array<int64_t, 8> rows{};
    std::array<std::array<float*, 8>, 4> Y{};
    for (int e = 0; e < count; ++e) {
        rows[e] = row_cases[e];
        auto x = owned.take<uint16_t>(rows[e] * K);
        auto w = owned.take<uint16_t>(size_t(N) * K);
        X[e] = x; W[e] = w;
        // Upload wait is fixture preparation only; it keeps temporary host buffers valid.
        std::vector<uint16_t> hx(rows[e] * K), hw(size_t(N) * K, half(0.0f));
        for (int r = 0; r < rows[e]; ++r)
            for (int k = 0; k < K; ++k) hx[r * K + k] = half(input(e, r, k));
        for (int n = 0; n < N; ++n) {
            int k = (n * 17 + e * 13) % K;
            hw[size_t(n) * K + k] = half(0.5f);
            hw[size_t(n) * K + (k + 1) % K] = half(-0.25f);
            hw[size_t(n) * K + (k + 11) % K] = half(0.125f);
        }
        if (two_queue) {
            stage[e]=owned.take<uint16_t>(size_t(N)*K);
            hx_owned[e]=owned.host<uint16_t>(hx.size()); hw_owned[e]=owned.host<uint16_t>(hw.size());
            std::memcpy(hx_owned[e],hx.data(),hx.size()*2);std::memcpy(hw_owned[e],hw.data(),hw.size()*2);
        } else {
            q.memcpy(x,hx.data(),hx.size()*2);q.memcpy(w,hw.data(),hw.size()*2);q.wait_and_throw();
        }
        for (int arm = 0; arm < 4; ++arm) {
            Y[arm][e] = owned.take<float>(rows[e] * N);
            q.memset(Y[arm][e], 0xff, rows[e] * N * 4); // NaN sentinel: beta=0 must overwrite.
        }
    }
    auto marker=[&](const char* phase,int round,int expert){
        std::fprintf(stderr,"GROUPED_V2_PROGRESS two_queue=%d count=%d N=%d K=%d phase=%s round=%d expert=%d\n",int(two_queue),count,N,K,phase,round,expert);std::fflush(stderr);
    };
    for(int arm=1;arm<4;++arm) {
        if(two_queue) {
            for(int e=0;e<count;++e) {
                marker("copy_submit_begin",arm,e);
                if(arm>1) copy.ext_oneapi_submit_barrier({used[e]});
                copy.memcpy(stage[e],hw_owned[e],size_t(N)*K*2);
                if(arm==1) copy.memcpy(const_cast<uint16_t*>(X[e]),hx_owned[e],rows[e]*K*2);
                auto copied=copy.ext_oneapi_submit_barrier();
                q.ext_oneapi_submit_barrier({copied});
                // Exact copy->compute dependency and compute->copy release ordering.
                // This memcpy stands in for dequant reading the source blob into a separate decoded slot.
                q.memcpy(const_cast<uint16_t*>(W[e]),stage[e],size_t(N)*K*2);
                used[e]=q.ext_oneapi_submit_barrier();
                marker("copy_submit_return",arm,e);
            }
        }
        if(arm==1) for(int e=0;e<count;++e) {
            marker("original_call_begin",arm,e);
            gemm.f16(X[e],W[e],Y[0][e],rows[e],N,K);
            marker("original_call_return",arm,e);
        }
        auto x=X;auto w=W;auto y=Y[arm];auto r=rows;
        marker("grouped_call_begin",arm,-1);
        gemm.f16_grouped(x.data(),w.data(),y.data(),r.data(),count,N,K);
        marker("grouped_call_return",arm,-1);
        x.fill(nullptr);w.fill(nullptr);y.fill(nullptr);r.fill(-1);
    }
    marker("compute_wait_begin",-1,-1);q.wait_and_throw();marker("compute_wait_return",-1,-1);
    marker("copy_wait_begin",-1,-1);copy.wait_and_throw();marker("copy_wait_return",-1,-1);
    size_t checked = 0;
    for (int arm = 0; arm < 4; ++arm) for (int e = 0; e < count; ++e) {
        std::vector<float> got(rows[e] * N);
        q.memcpy(got.data(), Y[arm][e], got.size() * 4).wait_and_throw();
        for (int r = 0; r < rows[e]; ++r) for (int n = 0; n < N; ++n) {
            float want = expected(e, r, n, K);
            // Numeric equality deliberately accepts either sign of exact zero.
            if (got[r * N + n] != want) {
                std::fprintf(stderr, "GROUPED_F16_MISMATCH arm=%d expert=%d row=%d col=%d got=%g expected=%g\n",
                             arm, e, r, n, got[r * N + n], want);
                throw std::runtime_error("known exact dyadic output differs");
            }
            ++checked;
        }
    }
    size_t bytes = owned.bytes;
    marker("release_begin",-1,-1);owned.release();marker("release_return",-1,-1);
    std::printf("GROUPED_F16_CASE two_queue=%d count=%d N=%d K=%d checked=%zu device_bytes=%zu original=1 grouped=3 equal=1 buffers_freed=1\n",
                int(two_queue), count, N, K, checked, bytes);
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--synthetic" || (std::string(argv[2])!="--single-queue" && std::string(argv[2])!="--two-queue"))
            throw std::invalid_argument("explicit --synthetic --single-queue or --two-queue required");
        bool two_queue=std::string(argv[2])=="--two-queue";
        const char* affinity = std::getenv("ZE_AFFINITY_MASK");
        const char* selector = std::getenv("ONEAPI_DEVICE_SELECTOR");
        if (!affinity || std::strcmp(affinity, "0") || !selector || std::strcmp(selector, "level_zero:gpu"))
            throw std::invalid_argument("card0 pin requires ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR=level_zero:gpu");
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        if (q.get_backend() != sycl::backend::ext_oneapi_level_zero)
            throw std::runtime_error("Level Zero GPU required");
        for (int count : {1, 3, 8}) {
            run(q, two_queue, count, 1280, 2560); // actual GU shape
            run(q, two_queue, count, 2560, 640);  // actual down shape
        }
        q.wait_and_throw();
        std::puts("GROUPED_F16_V2_SYNTHETIC_PASS cases=6 queue_retired=1 model_math_qualified=0 latency_qualified=0");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "GROUPED_F16_ERROR %s\n", e.what()); return 1;
    }
}
