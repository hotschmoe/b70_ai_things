// Source52 transport-only diagnostic. No model, GEMM, dequant or math changes.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/copy_queue.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

struct Owned {
    sycl::queue* compute;
    sycl::queue* copy;
    bool native;
    void* allocations[5] = {};
    ~Owned() {
        try {
            compute->wait_and_throw();
            if (copy) copy->wait_and_throw();
            if (copy) {
                if (native) strata::release_copy_engine_queue(copy);
                else dpct::get_current_device().destroy_queue(copy);
            }
            for (void* p : allocations) if (p) sycl::free(p, *compute);
        } catch (const std::exception& e) {
            // Never free an allocation still reachable by failed device work.
            std::fprintf(stderr, "FAIL source52 diagnostic drain; resources retained: %s\n", e.what());
        }
    }
};

int main(int argc, char** argv) try {
    if (argc != 2 || std::strcmp(argv[1], "--synthetic")) return 2;
    constexpr size_t bytes = 2u << 20;
    constexpr int ring = 2, jobs = 32;
    auto& compute = dpct::get_in_order_queue();
    const char* flag = std::getenv("STRATA_COPY_ENGINE");
    const bool on = flag && !std::strcmp(flag, "1");
    sycl::queue* copy = strata::make_copy_engine_queue("source52 fixture");
    sycl::queue* second = nullptr;
    if (on) {
        if (!copy || !copy->is_in_order() || !strata::is_copy_engine_queue(copy))
            throw std::runtime_error("dedicated queue unavailable");
        second = strata::make_copy_engine_queue("source52 second owner");
        if (second != copy) throw std::runtime_error("shared device queue differs");
        strata::release_copy_engine_queue(second);
        second = nullptr;
        if (!strata::copy_engine_active()) throw std::runtime_error("first owner lost queue");
    } else {
        if (copy || strata::copy_engine_active()) throw std::runtime_error("OFF created native queue");
        copy = dpct::get_current_device().create_queue(true);
    }
    Owned owned{&compute, copy, on};
    unsigned char* host = sycl::malloc_host<unsigned char>(bytes * ring, compute);
    unsigned char* dev = sycl::malloc_device<unsigned char>(bytes * ring, compute);
    uint64_t* done = sycl::malloc_host<uint64_t>(ring, compute);
    uint64_t* values = sycl::malloc_host<uint64_t>(4 * ring, compute);
    uint32_t* errors = sycl::malloc_device<uint32_t>(1, compute);
    owned.allocations[0] = host; owned.allocations[1] = dev;
    owned.allocations[2] = done; owned.allocations[3] = values; owned.allocations[4] = errors;
    if (!host || !dev || !done || !values || !errors) throw std::bad_alloc();
    std::memset(done, 0, ring * sizeof(uint64_t));
    std::memset(values, 0, 4 * ring * sizeof(uint64_t));
    compute.fill<uint32_t>(errors, 0, 1).wait_and_throw();
    sycl::event used[ring];
    uint64_t wanted[ring] = {};
    for (int j = 0; j < jobs; ++j) {
        const int slot = j % ring;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (*(volatile uint64_t*)(done + slot) < wanted[slot]) {
            if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("completion marker timed out");
            std::this_thread::yield();
        }
        const unsigned char expected = (unsigned char)(j + 7);
        std::memset(host + slot * bytes, expected, bytes);
        if (j >= ring) copy->ext_oneapi_submit_barrier({used[slot]});
        copy->memcpy(dev + slot * bytes, host + slot * bytes, bytes);
        const uint64_t sequence = (uint64_t)j + 1;
        wanted[slot] = sequence;
        if (on) {
            values[sequence % (4 * ring)] = sequence;
            copy->memcpy(done + slot, values + sequence % (4 * ring), sizeof(uint64_t));
        } else {
            copy->fill<uint64_t>(done + slot, sequence, 1);
        }
        auto copied = copy->ext_oneapi_submit_barrier();
        compute.ext_oneapi_submit_barrier({copied});
        auto* data = dev + slot * bytes;
        used[slot] = compute.parallel_for(sycl::range<1>(bytes), [=](sycl::id<1> i) {
            if (data[i] != expected) {
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                    sycl::memory_scope::device, sycl::access::address_space::global_space>(*errors).fetch_add(1);
            }
            data[i] = (unsigned char)(expected ^ 0x5a); // reuse must wait for this consumer
        });
    }
    compute.wait_and_throw();
    copy->wait_and_throw();
    uint32_t bad = 0;
    compute.memcpy(&bad, errors, sizeof bad).wait_and_throw();
    if (bad) throw std::runtime_error("cross-queue ring bytes changed");
    for (int i = 0; i < ring; ++i)
        if (done[i] != wanted[i]) throw std::runtime_error("final marker differs");
    if (on) {
        strata::release_copy_engine_queue(copy);
        owned.copy = nullptr;
        if (strata::copy_engine_active()) throw std::runtime_error("native queue not retired");
        // Recreate after last owner release; no stale registry/native list may survive.
        copy = strata::make_copy_engine_queue("source52 reopened owner");
        owned.copy = copy;
        if (!copy) throw std::runtime_error("queue recreate failed");
        copy->memcpy(dev, host, bytes).wait_and_throw();
        strata::release_copy_engine_queue(copy);
        owned.copy = nullptr;
        if (strata::copy_engine_active()) throw std::runtime_error("reopened queue not retired");
    } else {
        dpct::get_current_device().destroy_queue(copy);
        owned.copy = nullptr;
    }
    std::printf("PASS source52 transport copy_engine=%d bytes=%zu jobs=%d ring=%d errors=0 retired=1\n", int(on), bytes, jobs, ring);
    return 0;
} catch (const std::exception& e) {
    // The owner retains a failed process/lease until its actual queues retire.
    std::fprintf(stderr, "FAIL source52 transport: %s\n", e.what());
    return 1;
}
