// 100_ze_peer_copy.c -- authoritative DIRECT GPU0<->GPU1 transfer benchmark for dual Intel B70.
//
// Goes BELOW torch/oneCCL: raw Level Zero. Allocates device memory on dev0 and dev1 in ONE context
// (canAccessPeer=True on kernel 7.0, P2P_GPU H.11), then issues zeCommandListAppendMemoryCopy from a
// command list ON dev1 with the source pointer living in dev0's VRAM. That is a DIRECT peer DMA: the
// copy engine (BCS) or compute engine pulls bytes GPU0 -> GPU1 across the PCIe fabric without a host
// bounce. This is the hand-rolled equivalent of intel level-zero-tests `ze_peer`.
//
// Measures, on the CURRENT kernel+BIOS (reprofile 2026-06-24):
//   - copy-engine peer DMA bandwidth   d0->d1 and d1->d0, sizes 4KB..256MB
//   - compute-engine peer DMA bandwidth (same)
//   - small-message peer latency (8B ping)
//   - host-staged baseline (d0 -> host -> d1) for the direct-vs-bounce ratio
//
// Build (in vllm-xpu-env): gcc 100_ze_peer_copy.c -o ze_peer_copy -lze_loader
// Run under gpu-run (both cards). See 100_run_peer_copy.sh.
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(call) do { ze_result_t _r = (call); if (_r != ZE_RESULT_SUCCESS) { \
    fprintf(stderr, "FAIL %s -> 0x%x @ %s:%d\n", #call, _r, __FILE__, __LINE__); exit(2);} } while(0)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// find a command-queue-group ordinal on `dev` matching want_copy (COPY-only) or compute.
static uint32_t find_ordinal(ze_device_handle_t dev, int want_copy_only, uint32_t *out_numq) {
    uint32_t n = 0;
    CK(zeDeviceGetCommandQueueGroupProperties(dev, &n, NULL));
    ze_command_queue_group_properties_t *g = calloc(n, sizeof(*g));
    for (uint32_t i = 0; i < n; i++) g[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    CK(zeDeviceGetCommandQueueGroupProperties(dev, &n, g));
    uint32_t chosen = 0, numq = 1; int found = 0;
    for (uint32_t i = 0; i < n; i++) {
        int is_copy = (g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) != 0;
        int is_compute = (g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0;
        fprintf(stderr, "  group[%u] flags=0x%x copy=%d compute=%d numQueues=%u\n",
                i, g[i].flags, is_copy, is_compute, g[i].numQueues);
        if (want_copy_only && is_copy && !is_compute) { chosen = i; numq = g[i].numQueues; found = 1; }
        if (!want_copy_only && is_compute)            { chosen = i; numq = g[i].numQueues; found = 1; break; }
    }
    if (!found) { // fallback: ordinal 0
        chosen = 0; numq = g[0].numQueues;
        fprintf(stderr, "  (no %s group found; falling back to ordinal 0)\n",
                want_copy_only ? "copy-only" : "compute");
    }
    free(g);
    if (out_numq) *out_numq = numq;
    return chosen;
}

// Bench a peer copy: command list+queue on dst_dev, copy src_ptr(on src_dev) -> dst_ptr(on dst_dev).
static double bench_copy(ze_context_handle_t ctx, ze_device_handle_t dst_dev, uint32_t ordinal,
                         void *dst, void *src, size_t bytes, int iters) {
    ze_command_queue_desc_t qd = { ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, ordinal, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL };
    ze_command_list_desc_t ld = { ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, NULL, ordinal, 0 };
    ze_command_queue_handle_t q; ze_command_list_handle_t cl;
    CK(zeCommandQueueCreate(ctx, dst_dev, &qd, &q));
    CK(zeCommandListCreate(ctx, dst_dev, &ld, &cl));
    CK(zeCommandListAppendMemoryCopy(cl, dst, src, bytes, NULL, 0, NULL));
    CK(zeCommandListClose(cl));
    // warmup
    for (int i = 0; i < 3; i++) { CK(zeCommandQueueExecuteCommandLists(q, 1, &cl, NULL));
                                  CK(zeCommandQueueSynchronize(q, UINT64_MAX)); }
    double t0 = now_s();
    for (int i = 0; i < iters; i++) { CK(zeCommandQueueExecuteCommandLists(q, 1, &cl, NULL));
                                      CK(zeCommandQueueSynchronize(q, UINT64_MAX)); }
    double dt = now_s() - t0;
    zeCommandListDestroy(cl); zeCommandQueueDestroy(q);
    return dt / iters; // seconds per copy
}

int main(void) {
    CK(zeInit(0));
    uint32_t nd = 0; CK(zeDriverGet(&nd, NULL));
    if (nd < 1) { fprintf(stderr, "no L0 drivers\n"); return 1; }
    ze_driver_handle_t *drv = calloc(nd, sizeof(*drv)); CK(zeDriverGet(&nd, drv));
    ze_driver_handle_t driver = drv[0];
    uint32_t ndev = 0; CK(zeDeviceGet(driver, &ndev, NULL));
    printf("L0 driver[0] devices: %u\n", ndev);
    if (ndev < 2) { fprintf(stderr, "need >=2 devices (ZE_AFFINITY_MASK?)\n"); return 1; }
    ze_device_handle_t *dev = calloc(ndev, sizeof(*dev)); CK(zeDeviceGet(driver, &ndev, dev));
    ze_device_handle_t d0 = dev[0], d1 = dev[1];

    ze_bool_t can01 = 0, can10 = 0;
    CK(zeDeviceCanAccessPeer(d0, d1, &can01));
    CK(zeDeviceCanAccessPeer(d1, d0, &can10));
    printf("canAccessPeer d0->d1=%d d1->d0=%d\n", can01, can10);

    ze_context_desc_t cd = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0 };
    ze_context_handle_t ctx; CK(zeContextCreate(driver, &cd, &ctx));

    fprintf(stderr, "[dev1 queue groups]\n");
    uint32_t nq; uint32_t copy_ord = find_ordinal(d1, 1, &nq);
    uint32_t comp_ord = find_ordinal(d1, 0, &nq);
    fprintf(stderr, "[dev0 queue groups]\n");
    uint32_t copy_ord0 = find_ordinal(d0, 1, &nq);
    printf("dev1 copy_ordinal=%u compute_ordinal=%u ; dev0 copy_ordinal=%u\n",
           copy_ord, comp_ord, copy_ord0);

    size_t maxb = 256ull << 20;
    ze_device_mem_alloc_desc_t md = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, NULL, 0, 0 };
    void *p0, *p1; // device buffers on d0 and d1
    CK(zeMemAllocDevice(ctx, &md, maxb, 4096, d0, &p0));
    CK(zeMemAllocDevice(ctx, &md, maxb, 4096, d1, &p1));
    ze_host_mem_alloc_desc_t hd = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, NULL, 0 };
    void *ph; CK(zeMemAllocHost(ctx, &hd, maxb, 4096, &ph)); // pinned host staging buffer

    size_t sizes[] = { 4096, 65536, 1u<<20, 4u<<20, 16u<<20, 64u<<20, 256u<<20 };
    int nsz = sizeof(sizes)/sizeof(sizes[0]);

    // PULL = copy executes on the DESTINATION device, reading peer src  (non-posted PCIe reads).
    // PUSH = copy executes on the SOURCE device, writing peer dst        (posted PCIe writes).
    // Hypothesis: PUSH >> PULL on PCIe because reads are round-trip/credit-limited, writes are posted.
    printf("\n=== DIRECT PEER COPY -- PULL vs PUSH (copy engine), data moving d0->d1 ===\n");
    printf("%-10s %14s %14s %12s\n", "size", "PULL GB/s", "PUSH GB/s", "lat_us(pull)");
    for (int i = 0; i < nsz; i++) {
        size_t b = sizes[i];
        int iters = b <= (1u<<20) ? 200 : (b <= (16u<<20) ? 50 : 15);
        // PULL: exec on d1 (dst), src on d0 -> d1 reads peer
        double tpull = bench_copy(ctx, d1, copy_ord, p1, p0, b, iters);
        // PUSH: exec on d0 (src), dst on d1 -> d0 writes peer
        double tpush = bench_copy(ctx, d0, copy_ord0, p1, p0, b, iters);
        printf("%-10zu %14.2f %14.2f %12.1f\n", b, b/tpull/1e9, b/tpush/1e9, tpull*1e6);
    }

    printf("\n=== DIRECT PEER COPY -- PULL vs PUSH (compute engine), data moving d0->d1 ===\n");
    printf("%-10s %14s %14s\n", "size", "PULL GB/s", "PUSH GB/s");
    for (int i = 0; i < nsz; i++) {
        size_t b = sizes[i];
        int iters = b <= (1u<<20) ? 200 : (b <= (16u<<20) ? 50 : 15);
        double tpull = bench_copy(ctx, d1, comp_ord, p1, p0, b, iters);
        uint32_t comp_ord0 = 0; // compute group on d0 is ordinal 0 (flags 0x7)
        double tpush = bench_copy(ctx, d0, comp_ord0, p1, p0, b, iters);
        printf("%-10zu %14.2f %14.2f\n", b, b/tpull/1e9, b/tpush/1e9);
    }

    printf("\n=== SMALL-MESSAGE PEER LATENCY (copy engine, 8B) ===\n");
    {
        double t = bench_copy(ctx, d1, copy_ord, p1, p0, 8, 2000);
        printf("8B d0->d1 ping: %.2f us/copy\n", t*1e6);
    }

    printf("\n=== HOST-STAGED BASELINE (d0 -> host -> d1, 2 copies) ===\n");
    printf("%-10s %12s\n", "size", "GB/s(eff)");
    for (int i = 0; i < nsz; i++) {
        size_t b = sizes[i];
        int iters = b <= (1u<<20) ? 100 : (b <= (16u<<20) ? 30 : 10);
        // d0 -> host (queue on d0), host -> d1 (queue on d1); time the pair
        double acc = 0; int warm = 3;
        for (int it = 0; it < iters + warm; it++) {
            double s = now_s();
            // reuse bench_copy's pattern inline via single-shot lists would be heavy; approximate with
            // two synchronous immediate copies:
            ze_command_queue_desc_t qd0 = { ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, copy_ord0, 0, 0,
                ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL };
            ze_command_queue_desc_t qd1 = { ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, copy_ord, 0, 0,
                ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL };
            ze_command_list_handle_t cl0, cl1;
            CK(zeCommandListCreateImmediate(ctx, d0, &qd0, &cl0));
            CK(zeCommandListCreateImmediate(ctx, d1, &qd1, &cl1));
            CK(zeCommandListAppendMemoryCopy(cl0, ph, p0, b, NULL, 0, NULL)); // d0 -> host
            CK(zeCommandListAppendMemoryCopy(cl1, p1, ph, b, NULL, 0, NULL)); // host -> d1
            zeCommandListDestroy(cl0); zeCommandListDestroy(cl1);
            if (it >= warm) acc += now_s() - s;
        }
        double t = acc / iters;
        printf("%-10zu %12.2f\n", b, b/t/1e9);
    }

    printf("\nDONE_PEER_COPY\n");
    zeMemFree(ctx, p0); zeMemFree(ctx, p1); zeMemFree(ctx, ph);
    zeContextDestroy(ctx);
    return 0;
}
