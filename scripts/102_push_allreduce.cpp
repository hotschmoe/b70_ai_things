// 102_push_allreduce.cpp -- hand-rolled 2-rank PUSH all-reduce for dual B70, vs oneCCL (H.12: 9.7 GB/s).
//
// Built on J.2/J.5: a posted peer WRITE streams at 11.3 GB/s; a peer READ or reduce-into-peer is
// 2.4-3.2. So an all-reduce must be all local reduces + posted peer writes. 2-GPU pairwise exchange:
//   step 1 (concurrent, opposite directions): d0 PUSHES bufA -> scratch_on_d1 ; d1 PUSHES bufB -> scratch_on_d0
//   step 2 (concurrent, local only):          d0: bufA += scratch_on_d0       ; d1: bufB += scratch_on_d1
// End state: bufA == bufB == A+B. Every cross-card byte is a posted write; reduces touch local mem only.
//
// Single process, ONE context spanning both devices (peer access enabled). The transport is identical
// to a 2-process oneCCL run, so the algbw is directly comparable to H.12's allreduce_bench numbers.
//
// Build: icpx -fsycl -O2 102_push_allreduce.cpp -o push_allreduce
// Run  : ZE_AFFINITY_MASK=0,1 ./push_allreduce
#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
#include <vector>
#include <cmath>
using namespace sycl;

static double sec() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main() {
    std::vector<device> gpus;
    for (auto &p : platform::get_platforms())
        for (auto &d : p.get_devices(info::device_type::gpu))
            if (d.get_backend() == backend::ext_oneapi_level_zero) gpus.push_back(d);
    if (gpus.size() < 2) { printf("need >=2 (ZE_AFFINITY_MASK=0,1)\n"); return 1; }
    device d0 = gpus[0], d1 = gpus[1];
    context ctx({d0, d1});
    queue q0(ctx, d0), q1(ctx, d1);
    if (d0.ext_oneapi_can_access_peer(d1)) d0.ext_oneapi_enable_peer_access(d1);
    if (d1.ext_oneapi_can_access_peer(d0)) d1.ext_oneapi_enable_peer_access(d0);
    printf("push-allreduce, 2 ranks (single ctx). peer d0<->d1 enabled.\n");

    size_t sizes[] = { 10240, 65536, 1u<<20, 16u<<20, 64u<<20, 256u<<20 };
    const char *labels[] = { "10KB(decode)", "64KB", "1MB", "16MB(prefill)", "64MB", "256MB" };
    int nsz = sizeof(sizes)/sizeof(sizes[0]);

    size_t maxn = (256u<<20) / sizeof(float);
    float *bufA = malloc_device<float>(maxn, q0);   // rank0 data + result (d0)
    float *bufB = malloc_device<float>(maxn, q1);   // rank1 data + result (d1)
    float *scrA = malloc_device<float>(maxn, q0);   // on d0: receives rank1's pushed data
    float *scrB = malloc_device<float>(maxn, q1);   // on d1: receives rank0's pushed data

    printf("%-15s %12s %12s %12s\n", "size", "algbw GB/s", "lat us", "verify");
    for (int s = 0; s < nsz; s++) {
        size_t bytes = sizes[s];
        size_t n = bytes / sizeof(float);
        // init: A=1, B=3 -> result must be 4 everywhere
        q0.fill(bufA, 1.0f, n); q1.fill(bufB, 3.0f, n);
        q0.wait(); q1.wait();

        auto step = [&] {
            // step 1: concurrent opposite-direction posted peer writes
            q0.parallel_for(range<1>(n), [=](id<1> i){ scrB[i] = bufA[i]; }); // d0 -> d1 (peer write)
            q1.parallel_for(range<1>(n), [=](id<1> i){ scrA[i] = bufB[i]; }); // d1 -> d0 (peer write)
            q0.wait(); q1.wait();
            // step 2: concurrent local-only reduces
            q0.parallel_for(range<1>(n), [=](id<1> i){ bufA[i] += scrA[i]; });
            q1.parallel_for(range<1>(n), [=](id<1> i){ bufB[i] += scrB[i]; });
            q0.wait(); q1.wait();
        };

        // correctness: one fresh all-reduce, check result == 4 everywhere
        step();
        std::vector<float> h(4);
        q0.memcpy(h.data(), bufA, 4*sizeof(float)).wait();
        bool ok = std::fabs(h[0]-4.0f) < 1e-3;
        // timing: pure step() (data accumulates across iters -- irrelevant to bytes-moved/BW)
        int iters = bytes <= (1u<<20) ? 300 : (bytes <= (16u<<20) ? 60 : 15);
        for (int i = 0; i < 5; i++) step();          // warmup
        double t0 = sec();
        for (int i = 0; i < iters; i++) step();
        double dt = (sec() - t0) / iters;
        // algbw = bytes / time (one buffer's worth moved per rank), comparable to H.12 busbw
        printf("%-15s %12.2f %12.2f %12s\n", labels[s], bytes/dt/1e9, dt*1e6, ok ? "OK(4.0)" : "BAD");
    }

    free(bufA,q0); free(bufB,q1); free(scrA,q0); free(scrB,q1);
    printf("DONE_PUSH_ALLREDUCE\n");
    return 0;
}
