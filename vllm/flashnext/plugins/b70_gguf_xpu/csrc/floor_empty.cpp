// Device-side launch floor. Card 0 only, via bin/gpu-run.
// Kernel events on this stack do not carry profiling unless the queue was
// created with enable_profiling, and that property did not stick on the
// default Level Zero queue. Timing uses submit_profiling_tag, which is the
// same clock torch.xpu.Event uses. A long burn kernel is queued first so the
// tags sit behind real work and the GPU does not drain between submits.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>

#include <cstdint>
#include <cstdio>
#include <vector>

static double us(uint64_t ns) { return ns > 0 ? (double)ns / 1e3 : 0.0; }

static uint64_t tag_end(sycl::event e) {
    return e.get_profiling_info<sycl::info::event_profiling::command_end>();
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order()}};
    std::printf("DEVICE %s profiling_prop=%d\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                (int)q.has_property<sycl::property::queue::enable_profiling>());

    // ~tens of ms of independent FMAs so later submits land before the GPU
    // reaches them. The result is stored so the loop cannot be deleted.
    float* sink = sycl::malloc_device<float>(1, q);
    q.memset(sink, 0, sizeof(float)).wait();
    constexpr int BURN_WG = 1024;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(BURN_WG * 256, 256), [=](sycl::nd_item<1> it) {
            float x = 1.f + 0.001f * (float)it.get_global_id(0);
#pragma unroll 1
            for (int i = 0; i < 200000; ++i) x = sycl::fma(x, 1.0000001f, 0.000001f);
            if (it.get_global_id(0) == 0) *sink = x;
        });
    });

    constexpr int N = 32;
    constexpr int WG = 64;
    std::vector<sycl::event> marks;
    marks.reserve(N + 1);
    marks.push_back(sycl::ext::oneapi::experimental::submit_profiling_tag(q));
    for (int i = 0; i < N; ++i) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1>) {});
        });
        marks.push_back(sycl::ext::oneapi::experimental::submit_profiling_tag(q));
    }

    // 8 MiB streaming read, same sandwich.
    constexpr int BYTES = 8 << 20;
    uint8_t* buf = sycl::malloc_device<uint8_t>(BYTES, q);
    q.memset(buf, 1, BYTES);
    int* out = sycl::malloc_device<int>(N, q);
    std::vector<sycl::event> rmarks;
    rmarks.reserve(N + 1);
    rmarks.push_back(sycl::ext::oneapi::experimental::submit_profiling_tag(q));
    for (int i = 0; i < N; ++i) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(256 * 256, 256), [=](sycl::nd_item<1> it) {
                const size_t n4 = BYTES / 4;
                const uint32_t* w = reinterpret_cast<const uint32_t*>(buf);
                uint32_t s = 0;
                for (size_t k = it.get_global_id(0); k < n4; k += it.get_global_range(0)) s += w[k];
                if (it.get_global_id(0) == 0) out[i] = (int)s;
            });
        });
        rmarks.push_back(sycl::ext::oneapi::experimental::submit_profiling_tag(q));
    }
    rmarks.back().wait();

    auto summarize = [](const char* name, const std::vector<sycl::event>& m) {
        double mn = 1e300, mx = 0, sum = 0;
        for (size_t i = 1; i < m.size(); ++i) {
            const double d = us(tag_end(m[i]) - tag_end(m[i - 1]));
            mn = d < mn ? d : mn;
            mx = d > mx ? d : mx;
            sum += d;
        }
        const double n = (double)(m.size() - 1);
        std::printf("%s n=%.0f span_us min/mean/max %.2f %.2f %.2f\n", name, n, mn, sum / n, mx);
    };
    summarize("empty_behind_burn", marks);
    summarize("read8mb_behind_burn", rmarks);

    // Idle-queue control: one empty kernel, tags submitted around it with a
    // wait only at the end. If the GPU drains the start tag before the kernel
    // is submitted, this span is the host gap plus the kernel.
    {
        auto a = sycl::ext::oneapi::experimental::submit_profiling_tag(q);
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1>) {});
        });
        auto b = sycl::ext::oneapi::experimental::submit_profiling_tag(q);
        b.wait();
        std::printf("empty_idle_one span_us %.2f\n", us(tag_end(b) - tag_end(a)));
    }
    {
        constexpr int M = 32;
        auto a = sycl::ext::oneapi::experimental::submit_profiling_tag(q);
        for (int i = 0; i < M; ++i) {
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1>) {});
            });
        }
        auto b = sycl::ext::oneapi::experimental::submit_profiling_tag(q);
        b.wait();
        std::printf("empty_idle_batch%d span_us %.2f per %.2f\n", M, us(tag_end(b) - tag_end(a)),
                    us(tag_end(b) - tag_end(a)) / M);
    }

    sycl::free(sink, q);
    sycl::free(buf, q);
    sycl::free(out, q);
    std::printf("FLOOR_EMPTY_OK\n");
    return 0;
}
