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
    size_t bytes = 0;
    explicit Buffers(sycl::queue& queue): q(queue) { ptrs.reserve(48); }
    template<class T> T* take(size_t n) {
        if (n > (LIMIT - bytes) / sizeof(T)) throw std::runtime_error("128MiB device allocation cap");
        auto p = sycl::malloc_device<T>(n, q);
        if (!p) throw std::bad_alloc();
        ptrs.push_back(p); bytes += n * sizeof(T); return p;
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
void run(sycl::queue& q, int count, int N, int K) {
    Buffers owned(q);
    strata::prefill::Gemm gemm;
    std::string error;
    if (!gemm.init_external(&q, nullptr, 0, nullptr, 0, error)) throw std::runtime_error(error);
    constexpr int64_t row_cases[8] = {1, 2, 3, 5, 7, 9, 11, 13};
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
        q.memcpy(x, hx.data(), hx.size() * 2);
        q.memcpy(w, hw.data(), hw.size() * 2); q.wait_and_throw();
        for (int arm = 0; arm < 4; ++arm) {
            Y[arm][e] = owned.take<float>(rows[e] * N);
            q.memset(Y[arm][e], 0xff, rows[e] * N * 4); // NaN sentinel: beta=0 must overwrite.
        }
    }
    for (int e = 0; e < count; ++e) gemm.f16(X[e], W[e], Y[0][e], rows[e], N, K);
    for (int arm = 1; arm < 4; ++arm) {
        // The caller's pointer/row arrays die and are overwritten immediately after submission.
        auto x = X; auto w = W; auto y = Y[arm]; auto r = rows;
        gemm.f16_grouped(x.data(), w.data(), y.data(), r.data(), count, N, K);
        x.fill(nullptr); w.fill(nullptr); y.fill(nullptr); r.fill(-1);
    }
    q.wait_and_throw(); // Complete library work and metadata cleanup before result reads/teardown.
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
    owned.release();
    std::printf("GROUPED_F16_CASE count=%d N=%d K=%d checked=%zu device_bytes=%zu original=1 grouped=3 equal=1 buffers_freed=1\n",
                count, N, K, checked, bytes);
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2 || std::string(argv[1]) != "--synthetic")
            throw std::invalid_argument("explicit --synthetic required");
        const char* affinity = std::getenv("ZE_AFFINITY_MASK");
        const char* selector = std::getenv("ONEAPI_DEVICE_SELECTOR");
        if (!affinity || std::strcmp(affinity, "0") || !selector || std::strcmp(selector, "level_zero:gpu"))
            throw std::invalid_argument("card0 pin requires ZE_AFFINITY_MASK=0 ONEAPI_DEVICE_SELECTOR=level_zero:gpu");
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        if (q.get_backend() != sycl::backend::ext_oneapi_level_zero)
            throw std::runtime_error("Level Zero GPU required");
        for (int count : {1, 3, 8}) {
            run(q, count, 1280, 2560); // actual GU shape
            run(q, count, 2560, 640);  // actual down shape
        }
        q.wait_and_throw();
        std::puts("GROUPED_F16_SYNTHETIC_PASS cases=6 queue_retired=1 model_math_qualified=0 latency_qualified=0");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "GROUPED_F16_ERROR %s\n", e.what()); return 1;
    }
}
