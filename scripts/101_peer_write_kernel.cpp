// 101_peer_write_kernel.cpp -- SYCL microkernel that PUSHES data GPU0 -> GPU1 via EU peer writes.
//
// J.2 showed a posted peer WRITE hits 11.3 GB/s vs 3.24 for a read. The copy-engine push is opaque;
// this is the EU (compute) version: a kernel running on dev0 whose work-items STORE into a USM pointer
// that lives in dev1's VRAM. That store compiles to Xe2 send instructions across the PCIe fabric -- the
// literal "assembly that transfers data from gpu0 to gpu1". Dump it with IGC_ShaderDumpEnable=1.
//
// Three kernels, all executed on dev0:
//   1. copy   : dst_peer[i] = src_local[i]            (pure push, EU)
//   2. addpush: dst_peer[i] = dst_peer[i] + src[i]    (push+reduce -- the allreduce inner step;
//               note this READS peer then WRITES peer -> exposes the read tax inside a reduce)
//   3. addlocal_thenpush: tmp_local = a+b; dst_peer = tmp  (reduce LOCAL, then push -- the fast shape)
//
// Build: icpx -fsycl -O2 101_peer_write_kernel.cpp -o peer_write
// Run  : ZE_AFFINITY_MASK=0,1 ./peer_write
#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
#include <vector>
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
    printf("L0 GPUs visible: %zu\n", gpus.size());
    if (gpus.size() < 2) { printf("need >=2 (ZE_AFFINITY_MASK=0,1)\n"); return 1; }
    device d0 = gpus[0], d1 = gpus[1];
    context ctx({d0, d1});
    queue q0(ctx, d0), q1(ctx, d1);

    bool can01 = d0.ext_oneapi_can_access_peer(d1);
    bool can10 = d1.ext_oneapi_can_access_peer(d0);
    printf("can_access_peer d0->d1=%d d1->d0=%d\n", can01, can10);
    if (can01) d0.ext_oneapi_enable_peer_access(d1);
    if (can10) d1.ext_oneapi_enable_peer_access(d0);

    const size_t N = 64ull << 20;            // 64 MB transfer
    const size_t n = N / sizeof(float);
    float *src = malloc_device<float>(n, q0);   // local to d0
    float *dst = malloc_device<float>(n, q1);   // peer (d1)
    float *acc = malloc_device<float>(n, q1);   // peer (d1), accumulator
    q0.fill(src, 1.0f, n).wait();
    q1.fill(dst, 0.0f, n).wait();
    q1.fill(acc, 2.0f, n).wait();

    auto bench = [&](const char *name, auto kern, int iters) {
        for (int i = 0; i < 5; i++) kern();        // warmup
        q0.wait(); q1.wait();
        double t0 = sec();
        for (int i = 0; i < iters; i++) kern();
        q0.wait(); q1.wait();
        double dt = (sec() - t0) / iters;
        printf("  %-26s %8.2f GB/s  (%.3f ms)\n", name, N / dt / 1e9, dt * 1e3);
    };

    printf("\n=== EU PUSH microkernels (kernel on dev0, target in dev1 VRAM), 64MB ===\n");
    // 1. pure push
    bench("copy: dstPeer=src", [&] {
        q0.parallel_for(range<1>(n), [=](id<1> i) { dst[i] = src[i]; }).wait();
    }, 30);
    // 2. push+reduce reading peer (exposes the read tax)
    bench("addpush: accPeer+=src", [&] {
        q0.parallel_for(range<1>(n), [=](id<1> i) { acc[i] = acc[i] + src[i]; }).wait();
    }, 30);
    // 3. reduce local then push (the fast shape: no peer read)
    float *loc = malloc_device<float>(n, q0);    // local scratch on d0
    q0.fill(loc, 3.0f, n).wait();
    bench("local-reduce then push", [&] {
        q0.parallel_for(range<1>(n), [=](id<1> i) { dst[i] = src[i] + loc[i]; }).wait();
    }, 30);

    // sanity: verify a push actually landed in peer memory
    q0.parallel_for(range<1>(n), [=](id<1> i) { dst[i] = src[i] * 7.0f; }).wait();
    std::vector<float> host(16);
    q1.memcpy(host.data(), dst, 16 * sizeof(float)).wait();
    printf("verify peer[0..3] = %.1f %.1f %.1f %.1f (expect 7.0)\n",
           host[0], host[1], host[2], host[3]);

    free(src, q0); free(dst, q1); free(acc, q1); free(loc, q0);
    printf("DONE_PEER_WRITE\n");
    return 0;
}
