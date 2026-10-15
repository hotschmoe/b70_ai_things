// S74 capability prerequisite only. No IQ decode or FP32 scaling/reduction.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

namespace jm = sycl::ext::oneapi::experimental::matrix;
class Int8K32Capability;
// Declared after allocations below, so unwinding drains before any USM free.
struct DrainQueue {
    sycl::queue* queue;
    ~DrainQueue() { try { queue->wait_and_throw(); } catch (...) {} }
};
struct UsmFree {
    sycl::queue* queue;
    template<class T> void operator()(T* p) const { if (p) sycl::free(p, *queue); }
};

int main(int argc, char** argv) {
    if (argc == 1) {
        std::puts("COMPILE_CAPABILITY_ONLY: runtime not requested; no queue constructed");
        return 0;
    }
    if (argc != 2 || std::strcmp(argv[1], "--run") != 0) {
        std::fputs("usage: int8_xmx_capability_s74 [--run]\n", stderr);
        return 2;
    }
    try {
        constexpr int CT = 8, ACTIVE = 6, RT = 16, KT = 32;
        alignas(64) std::array<std::int8_t, CT * KT> a{};
        // B is column-major K32 x output16: contiguous K for each output.
        alignas(64) std::array<std::int8_t, RT * KT> b{};
        alignas(64) std::array<std::int32_t, CT * RT> output{};
        constexpr std::int8_t values[] = {-128, 127, -127, -15, 15, -1, 0, 1};
        for (int c = 0; c < ACTIVE; ++c)
            for (int k = 0; k < KT; ++k) a[c * KT + k] = values[(k + 3 * c) % 8];
        for (int r = 0; r < RT; ++r)
            for (int k = 0; k < KT; ++k) b[r * KT + k] = values[(3 * k + r) % 8];
        output.fill(std::numeric_limits<std::int32_t>::min());
        sycl::queue q(sycl::gpu_selector_v,
            [](sycl::exception_list es) {
                for (const auto& e : es) std::rethrow_exception(e);
            }, sycl::property::queue::in_order{});
        // All allocations precede submission. Each row's worst absolute dot is
        // <=32*128*128=524288, including all partial sums, safely inside INT32.
        std::unique_ptr<std::int8_t, UsmFree> ad(
            sycl::aligned_alloc_device<std::int8_t>(64, a.size(), q), UsmFree{&q});
        std::unique_ptr<std::int8_t, UsmFree> bd(
            sycl::aligned_alloc_device<std::int8_t>(64, b.size(), q), UsmFree{&q});
        std::unique_ptr<std::int32_t, UsmFree> cd(
            sycl::aligned_alloc_device<std::int32_t>(64, output.size(), q), UsmFree{&q});
        if (!ad || !bd || !cd) throw std::bad_alloc();
        DrainQueue drain{&q};
        auto* ap = ad.get(); auto* bp = bd.get(); auto* cp = cd.get();
        q.memcpy(ap, a.data(), sizeof(a));
        q.memcpy(bp, b.data(), sizeof(b));
        q.memcpy(cp, output.data(), sizeof(output));
        q.submit([&](sycl::handler& h) {
            h.parallel_for<Int8K32Capability>(
                sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(16)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    const auto sg = it.get_sub_group();
                    jm::joint_matrix<sycl::sub_group, std::int8_t, jm::use::a,
                                     CT, KT, jm::layout::row_major> ma;
                    jm::joint_matrix<sycl::sub_group, std::int8_t, jm::use::b,
                                     KT, RT, jm::layout::col_major> mb;
                    jm::joint_matrix<sycl::sub_group, std::int32_t,
                                     jm::use::accumulator, CT, RT> mc;
                    jm::joint_matrix_fill(sg, mc, 0);
                    jm::joint_matrix_load(sg, ma,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(ap), KT);
                    jm::joint_matrix_load(sg, mb,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(bp), KT);
                    jm::joint_matrix_mad(sg, mc, ma, mb, mc);
                    jm::joint_matrix_store(sg, mc,
                        sycl::address_space_cast<sycl::access::address_space::global_space,
                                                 sycl::access::decorated::no>(cp),
                        RT, jm::layout::row_major);
                });
        });
        q.memcpy(output.data(), cp, sizeof(output));
        std::array<std::int8_t, CT * KT> a_after{};
        std::array<std::int8_t, RT * KT> b_after{};
        q.memcpy(a_after.data(), ap, sizeof(a_after));
        q.memcpy(b_after.data(), bp, sizeof(b_after));
        q.wait_and_throw();
        if (a_after != a || b_after != b) {
            std::fputs("FAIL_INPUT_CHANGED\n", stderr);
            return 5;
        }
        for (int c = 0; c < CT; ++c) for (int r = 0; r < RT; ++r) {
            std::int64_t expected = 0;
            for (int k = 0; k < KT; ++k)
                expected += std::int64_t(a[c * KT + k]) * std::int64_t(b[r * KT + k]);
            if (std::int64_t(output[c * RT + r]) != expected) {
                std::fprintf(stderr, "FAIL_DOT c=%d r=%d actual=%d expected=%lld\n",
                             c, r, int(output[c * RT + r]), (long long) expected);
                return 5;
            }
        }
        std::puts("PASS_INT8_K32: all128_INT32_cells; six_active_two_zero_rows; inputs_unchanged");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "RUNTIME_ERROR: %s\n", e.what());
        return 2;
    }
}
