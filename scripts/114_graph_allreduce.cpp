// 114_graph_allreduce.cpp -- IS the push all-reduce SYCL-GRAPH-CAPTURABLE on B70? (the decode-capture question)
//
// CONTEXT (docs/P2P_GPU.md J.9, handoff_decode_push_ar.md): our push all-reduce is prefill-only because its
// rank-sync is a HOST barrier -> not graph-recordable -> decode all-reduces fall back to oneCCL inside the
// captured graph. J.9 found: (B) cross-queue SYCL EVENTS sync = 44us, correct, but it was never put in a
// graph; (C) an EU-spin device flag HANGS (mid-kernel peer write invisible on Xe; peer ATOMICS=N).
//
// THIS BENCH tests the missing piece from J.9 B: take the event-synced all-reduce and RECORD IT INTO A SYCL
// command_graph, then REPLAY it with no host involvement on the sync. The cross-device dependency (rank0's
// reduce must wait for rank1's push) is expressed as a graph EDGE (recorded SYCL event dep), which lowers to
// a command-streamer / L0-event wait -- a DIFFERENT mechanism from J.9-C's failed EU spin. If the graph
// replays correctly, the all-reduce is capturable and the decode path is unblocked (in principle).
//
// Single process / single L0 context / 2 queues (one per card). This proves CAPTURABILITY + REPLAY CORRECTNESS,
// the hard unknown. The 2-process IPC-event variant (vLLM worker topology) is the follow-up (115).
//
// Build: icpx -fsycl -O2 114_graph_allreduce.cpp -o graph_allreduce -lze_loader
// Run  : ZE_AFFINITY_MASK=0,1 ./graph_allreduce   (under gpu-run, timeout-wrapped)
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <cstdio>
#include <chrono>
#include <vector>
#include <cmath>
using namespace sycl;
namespace sgr = sycl::ext::oneapi::experimental;

static double sec() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::vector<device> gpus;
    for (auto &p : platform::get_platforms())
        for (auto &d : p.get_devices(info::device_type::gpu))
            if (d.get_backend() == backend::ext_oneapi_level_zero) gpus.push_back(d);
    if (gpus.size() < 2) { printf("need >=2 (ZE_AFFINITY_MASK=0,1)\n"); return 1; }
    device d0 = gpus[0], d1 = gpus[1];
    context ctx({d0, d1});
    // Graph recording needs OUT-OF-ORDER queues (in-order queues add implicit deps that recording rejects).
    queue q0(ctx, d0);
    queue q1(ctx, d1);
    if (d0.ext_oneapi_can_access_peer(d1)) d0.ext_oneapi_enable_peer_access(d1);
    if (d1.ext_oneapi_can_access_peer(d0)) d1.ext_oneapi_enable_peer_access(d0);
    printf("graph all-reduce: 2 ranks single ctx, peer enabled. SYCL command_graph capture+replay.\n");

    size_t sizes[]    = { 10240, 65536, 1u<<20, 16u<<20 };
    const char *lab[] = { "10KB(decode)", "64KB", "1MB", "16MB(prefill)" };
    int nsz = sizeof(sizes)/sizeof(sizes[0]);

    size_t maxn = (16u<<20) / sizeof(float);
    float *bufA = malloc_device<float>(maxn, q0);
    float *bufB = malloc_device<float>(maxn, q1);
    float *scrA = malloc_device<float>(maxn, q0);   // on d0: rank1 pushes here
    float *scrB = malloc_device<float>(maxn, q1);   // on d1: rank0 pushes here

    auto reset = [&](size_t n){ q0.fill(bufA,1.0f,n); q1.fill(bufB,3.0f,n); q0.wait(); q1.wait(); };

    printf("%-15s %12s %12s %12s %10s %10s\n",
           "size", "perLaunch us", "amort us", "algbw GB/s", "verifyA", "verifyB");
    for (int s = 0; s < nsz; s++) {
        size_t n = sizes[s] / sizeof(float);

        // --- record the all-reduce into a command_graph (multi-queue recording mode) ---
        sgr::command_graph graph(ctx, d0);
        graph.begin_recording(std::vector<queue>{q0, q1});
        event eA = q0.parallel_for(range<1>(n), [=](id<1> i){ scrB[i] = bufA[i]; }); // d0 -> scrB(d1) push
        event eB = q1.parallel_for(range<1>(n), [=](id<1> i){ scrA[i] = bufB[i]; }); // d1 -> scrA(d0) push
        // cross-device edges: reduceA(on d0) depends on eB(push done on d1); reduceB(on d1) depends on eA.
        q0.parallel_for(range<1>(n), {eB}, [=](id<1> i){ bufA[i] += scrA[i]; });
        q1.parallel_for(range<1>(n), {eA}, [=](id<1> i){ bufB[i] += scrB[i]; });
        graph.end_recording();
        sgr::command_graph<sgr::graph_state::executable> exec = graph.finalize();

        // --- correctness: reset, replay once, check bufA[0..3]==4 AND bufB[0..3]==4 ---
        reset(n);
        q0.ext_oneapi_graph(exec).wait();
        std::vector<float> hA(4), hB(4);
        q0.memcpy(hA.data(), bufA, 4*sizeof(float)).wait();
        q1.memcpy(hB.data(), bufB, 4*sizeof(float)).wait();
        bool okA = std::fabs(hA[0]-4.0f) < 1e-3 && std::fabs(hA[3]-4.0f) < 1e-3;
        bool okB = std::fabs(hB[0]-4.0f) < 1e-3 && std::fabs(hB[3]-4.0f) < 1e-3;

        // re-verify correctness across MANY replays (shake out a replay-only race)
        bool okRepeat = true;
        for (int r = 0; r < 50 && okRepeat; r++) {
            reset(n);
            q0.ext_oneapi_graph(exec).wait();
            q0.memcpy(hA.data(), bufA, 4*sizeof(float)).wait();
            if (std::fabs(hA[0]-4.0f) > 1e-3) okRepeat = false;
        }

        // --- timing (a) per-launch: submit+wait each iter (one decode token = one graph launch) ---
        int iters = sizes[s] <= (1u<<20) ? 300 : 60;
        for (int i=0;i<10;i++) q0.ext_oneapi_graph(exec).wait();
        double t0 = sec();
        for (int i=0;i<iters;i++) q0.ext_oneapi_graph(exec).wait();
        double perLaunch = (sec()-t0)/iters;

        // --- timing (b) amortized: submit N back-to-back, wait once (allreduce as 1 node in a big graph) ---
        double t1 = sec();
        event last;
        for (int i=0;i<iters;i++) last = q0.ext_oneapi_graph(exec);
        last.wait();
        double amort = (sec()-t1)/iters;

        printf("%-15s %12.2f %12.2f %12.2f %10s %10s\n",
               lab[s], perLaunch*1e6, amort*1e6, sizes[s]/amort/1e9,
               (okA&&okRepeat)?"OK(4.0)":"BAD", okB?"OK(4.0)":"BAD");
    }

    free(bufA,q0); free(bufB,q1); free(scrA,q0); free(scrB,q1);
    printf("DONE_GRAPH_ALLREDUCE\n");
    return 0;
}
