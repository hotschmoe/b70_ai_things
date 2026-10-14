// Run under the pair lease; compare IPC reads with full-buffer freshness checks.
// Keep full-sized native allocations alive across sizes/modes. Do not recycle an
// IPC mapping while UR queues may still retain information about that pointer.
#include <chrono>
#include <cstdio>
#include <vector>
#include "strata/core/ep_mem.hpp"
#include "strata/copy_queue.hpp"
#include "strata/kernels/ep_kernels.hpp"
int main() {
    using namespace strata::core;
    if (dpct::device_count() != 2) return 2;
    auto& producer = dpct::dev_mgr::instance().get_device(1).in_order_queue();
    auto& consumer = dpct::dev_mgr::instance().get_device(0).in_order_queue();
    dpct::select_device(0);
    auto* dma = strata::make_copy_engine_queue("EP transfer probe");
    if (!dma) { std::puts("FAIL no copy engine"); return 3; }
    constexpr size_t capacity = size_t(25023)*2560*4;
    EpRegion region, destination;
    std::string error;
    if (!ep_region_alloc(1, 0, capacity, region, error) ||
        !ep_region_alloc(0, 1, capacity, destination, error)) {
        std::printf("FAIL %s\n", error.c_str()); return 4;
    }
    auto* dst = static_cast<uint32_t*>(destination.local);
    std::printf("ALLOC capacity=%zu source=%p imported=%p destination=%p\n",
                capacity, region.local, region.remote, dst);
    std::fflush(stdout);
    std::vector<uint32_t> host(capacity/4);
    for (size_t bytes : {size_t(16384), capacity}) {
        const size_t words = bytes/4;
        for (int mode=0; mode<3; ++mode) {
            for (uint32_t epoch=1; epoch<=4; ++epoch) {
                auto* src = static_cast<uint32_t*>(region.local);
                uint32_t salt = epoch + mode*100;
                std::printf("BEGIN bytes=%zu mode=%d epoch=%u\n",bytes,mode,epoch);
                std::fflush(stdout);
                producer.parallel_for(sycl::range<1>(words), [=](sycl::id<1> i) {
                    src[i] = uint32_t(i[0])*2654435761u ^ salt;
                }).wait();
                auto start=std::chrono::steady_clock::now();
                if (mode==0) {
                    strata::kernels::ep_copy_uncached(dst,region.remote,bytes,&consumer);
                    consumer.wait();
                } else if (mode==1) consumer.memcpy(dst,region.remote,bytes).wait();
                else dma->memcpy(dst,region.remote,bytes).wait();
                double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                consumer.memcpy(host.data(),dst,bytes).wait();
                size_t bad=0;
                for(size_t i=0;i<words;++i) bad += host[i] != (uint32_t(i)*2654435761u ^ salt);
                std::printf("TRANSFER bytes=%zu mode=%d epoch=%u ms=%.3f GBps=%.3f mismatches=%zu\n",bytes,mode,epoch,ms,bytes/ms/1e6,bad);
                std::fflush(stdout);
                if(bad) return 5;
            }
        }
    }
    producer.wait(); consumer.wait(); dma->wait();
    ep_region_free(destination);
    ep_region_free(region);
    std::puts("PASS all IPC transfer bytes fresh");
}
