// Run under the pair lease; compare IPC reads with full-buffer freshness checks.
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
    for (size_t bytes : {size_t(16384), size_t(25023)*2560*4}) {
        EpRegion region;
        std::string error;
        if (!ep_region_alloc(1, 0, bytes, region, error)) {
            std::printf("FAIL %s\n", error.c_str()); return 4;
        }
        const size_t words = bytes/4;
        auto* dst = sycl::malloc_device<uint32_t>(words, consumer);
        std::vector<uint32_t> host(words);
        for (int mode=0; mode<3; ++mode) {
            for (uint32_t epoch=1; epoch<=4; ++epoch) {
                auto* src = static_cast<uint32_t*>(region.local);
                uint32_t salt = epoch + mode*100;
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
        consumer.wait(); dma->wait();
        sycl::free(dst,consumer);
        ep_region_free(region);
    }
    std::puts("PASS all IPC transfer bytes fresh");
}
