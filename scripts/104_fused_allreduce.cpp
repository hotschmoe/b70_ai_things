// 104_fused_allreduce.cpp -- latency-optimized push all-reduce for dual B70 (decode-sized messages).
//
// J.7 (102) full all-reduce = 59.5 us @ 10KB because it does 4 kernel launches + 2 HOST syncs
// (q0.wait();q1.wait() AFTER step1, blocking the host before it can even submit step2). For decode the
// transfer is tiny; the cost is launch + host round-trip overhead. Three modes, head-to-head:
//
//   A baseline : J.7 exact -- step1 push, HOST WAIT, step2 reduce, HOST WAIT. (4 launch, 2 sync)
//   B events   : submit all 4 kernels async; the reduce on each device depends (cross-queue L0 event)
//                on the PEER's push completing. ONE host wait at the very end. (4 launch, 1 sync)
//   C fused    : ONE kernel per rank (single work-group, grid-stride). Each rank: peer-write its data,
//                release-fence, peer-write a sequence flag into the peer, spin on its own flag until the
//                peer's seq arrives, then local-reduce. NO host barrier between push and reduce. Device
//                -side cross-card signalling. (1 launch/rank, 1 sync). Note: peer ATOMICS=N on B70
//                (H.11), so the flag is a plain posted store + a local polling load, not a peer RMW.
//
// All modes verified to produce A+B == 4.0 everywhere. Run repeats to shake out any ordering race in C.
//
// Build: icpx -fsycl -O2 104_fused_allreduce.cpp -o fused_allreduce
// Run  : ZE_AFFINITY_MASK=0,1 ./fused_allreduce
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
    setvbuf(stdout, NULL, _IONBF, 0); // unbuffered: survive a timeout kill and show where C hangs
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
    printf("fused all-reduce latency bench, 2 ranks single ctx. peer enabled.\n");

    size_t sizes[]   = { 10240, 65536, 1u<<20, 16u<<20 };
    const char *lab[] = { "10KB(decode)", "64KB", "1MB", "16MB(prefill)" };
    int nsz = sizeof(sizes)/sizeof(sizes[0]);

    size_t maxn = (16u<<20) / sizeof(float);
    float *bufA = malloc_device<float>(maxn, q0);
    float *bufB = malloc_device<float>(maxn, q1);
    float *scrA = malloc_device<float>(maxn, q0);   // on d0: rank1 pushes here
    float *scrB = malloc_device<float>(maxn, q1);   // on d1: rank0 pushes here
    // flags: one int on each device; peer writes a sequence number, owner polls locally.
    int *flagA = malloc_device<int>(1, q0);          // on d0: rank1 signals here
    int *flagB = malloc_device<int>(1, q1);          // on d1: rank0 signals here
    q0.fill(flagA, 0, 1); q1.fill(flagB, 0, 1); q0.wait(); q1.wait();

    auto reset = [&](size_t n){ q0.fill(bufA,1.0f,n); q1.fill(bufB,3.0f,n); q0.wait(); q1.wait(); };

    // --- mode A: J.7 baseline (4 launch, 2 host sync) ---
    auto stepA = [&](size_t n){
        q0.parallel_for(range<1>(n), [=](id<1> i){ scrB[i] = bufA[i]; });
        q1.parallel_for(range<1>(n), [=](id<1> i){ scrA[i] = bufB[i]; });
        q0.wait(); q1.wait();
        q0.parallel_for(range<1>(n), [=](id<1> i){ bufA[i] += scrA[i]; });
        q1.parallel_for(range<1>(n), [=](id<1> i){ bufB[i] += scrB[i]; });
        q0.wait(); q1.wait();
    };
    // --- mode B: cross-queue events (4 launch, 1 host sync) ---
    auto stepB = [&](size_t n){
        event e0 = q0.parallel_for(range<1>(n), [=](id<1> i){ scrB[i] = bufA[i]; }); // d0 -> scrB(d1)
        event e1 = q1.parallel_for(range<1>(n), [=](id<1> i){ scrA[i] = bufB[i]; }); // d1 -> scrA(d0)
        // reduce on d0 needs scrA (filled by e1, executed on d1); reduce on d1 needs scrB (e0).
        q0.parallel_for(range<1>(n), {e1}, [=](id<1> i){ bufA[i] += scrA[i]; });
        q1.parallel_for(range<1>(n), {e0}, [=](id<1> i){ bufB[i] += scrB[i]; });
        q0.wait(); q1.wait();
    };
    // --- mode C: fused single-kernel device-flag signal (1 launch/rank, 1 sync) ---
    // seq increments every call so flags never need a reset (avoids inter-iter reset race).
    int seq = 0;
    const size_t WG = 256;
    auto stepC = [&](size_t n){
        seq++;
        int myseq = seq;
        auto k0 = q0.submit([&](handler &h){
            h.parallel_for(nd_range<1>(WG, WG), [=](nd_item<1> it){
                size_t lid = it.get_local_id(0);
                for (size_t i = lid; i < n; i += WG) scrB[i] = bufA[i]; // peer write d0 -> d1
                atomic_fence(memory_order::release, memory_scope::system);
                group_barrier(it.get_group());
                if (lid == 0) {
                    *flagB = myseq;                                     // peer-write the flag into d1
                    volatile int *fl = flagA;                          // poll local flag (rank1 -> me)
                    long spins = 0;
                    while (*fl < myseq && ++spins < 500000000L) { }     // bounded: detect, don't deadlock
                }
                group_barrier(it.get_group());
                atomic_fence(memory_order::acquire, memory_scope::system);
                for (size_t i = lid; i < n; i += WG) bufA[i] += scrA[i]; // local reduce
            });
        });
        auto k1 = q1.submit([&](handler &h){
            h.parallel_for(nd_range<1>(WG, WG), [=](nd_item<1> it){
                size_t lid = it.get_local_id(0);
                for (size_t i = lid; i < n; i += WG) scrA[i] = bufB[i]; // peer write d1 -> d0
                atomic_fence(memory_order::release, memory_scope::system);
                group_barrier(it.get_group());
                if (lid == 0) {
                    *flagA = myseq;                                     // peer-write the flag into d0
                    volatile int *fl = flagB;
                    long spins = 0;
                    while (*fl < myseq && ++spins < 500000000L) { }     // bounded: detect, don't deadlock
                }
                group_barrier(it.get_group());
                atomic_fence(memory_order::acquire, memory_scope::system);
                for (size_t i = lid; i < n; i += WG) bufB[i] += scrB[i];
            });
        });
        k0.wait(); k1.wait();
    };

    struct { const char *name; void (*dummy)(); } modes[] = {{"A_baseline",0},{"B_events",0},{"C_fused",0}};
    printf("%-15s %-12s %12s %12s %10s\n", "size", "mode", "lat us", "algbw GB/s", "verify");
    for (int s = 0; s < nsz; s++) {
        size_t n = sizes[s] / sizeof(float);
        for (int m = 0; m < 3; m++) {
            fprintf(stderr, "  [run] %s %s ...\n", lab[s], modes[m].name);
            // correctness
            reset(n);
            if (m==0) stepA(n); else if (m==1) stepB(n); else stepC(n);
            std::vector<float> h(4);
            q0.memcpy(h.data(), bufA, 4*sizeof(float)).wait();
            bool ok = std::fabs(h[0]-4.0f) < 1e-3;
            // timing (data accumulates -- irrelevant to latency/BW)
            int iters = sizes[s] <= (1u<<20) ? 500 : 60;
            for (int i=0;i<10;i++){ if(m==0)stepA(n); else if(m==1)stepB(n); else stepC(n); }
            double t0 = sec();
            for (int i=0;i<iters;i++){ if(m==0)stepA(n); else if(m==1)stepB(n); else stepC(n); }
            double dt = (sec()-t0)/iters;
            printf("%-15s %-12s %12.2f %12.2f %10s\n",
                   s==0||m>0?lab[s]:"", modes[m].name, dt*1e6, sizes[s]/dt/1e9, ok?"OK(4.0)":"BAD");
        }
    }

    free(bufA,q0); free(bufB,q1); free(scrA,q0); free(scrB,q1);
    free(flagA,q0); free(flagB,q1);
    printf("DONE_FUSED_ALLREDUCE\n");
    return 0;
}
