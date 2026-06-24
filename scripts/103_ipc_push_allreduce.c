// 103_ipc_push_allreduce.c -- 2-PROCESS IPC push exchange for dual B70 (the vLLM-worker path).
//
// J.7 (102) proved a hand-rolled push all-reduce beats oneCCL -- but in ONE process / ONE context.
// The real vLLM TP path is TWO processes (one worker per card), so each worker must map the PEER
// worker's scratch buffer via a Level-Zero IPC handle (zeMemGetIpcHandle / zeMemOpenIpcHandle, proven
// OK on kernel 7.0 in P2P_GPU H.11) before it can push into it. H.13 showed oneCCL's own P2P path
// DEVICE_LOSTs inside the vLLM multiproc worker; this measures whether OUR posted-write transport
// survives the 2-process boundary at full speed -- the precondition for a custom XPU all-reduce op.
//
// Design (rank r in {0,1}, peer pr = 1-r, data moves both ways = an all-reduce step-1 exchange):
//   - separate fork()'d process per rank; each zeInit, sees both devices, owns its own context.
//   - rank r allocates localBuf + localScratch on device[r]; exports an IPC handle for localScratch.
//   - exchange handles over a Unix socketpair, passing the embedded dma-buf fd via SCM_RIGHTS.
//   - rank r zeMemOpenIpcHandle(peer's handle) -> peerScratch ptr (lives in device[pr] VRAM).
//   - PUSH: a copy EXECUTED ON device[r] writes localBuf -> peerScratch (posted peer write = fast).
//   - socket barrier between iters keeps both ranks in lockstep (= the real collective's step boundary).
// Verify: after the exchange rank r's localScratch holds the PEER's fill value -> cross-process peer
// write landed. The local reduce (buf += localScratch) is local-VRAM compute (~100s GB/s), as in J.7.
//
// Build: gcc 103_ipc_push_allreduce.c -o ipc_push_allreduce -lze_loader
// Run under gpu-run (both cards). See 103_run_ipc_push_allreduce.sh.
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>

#define CK(call) do { ze_result_t _r = (call); if (_r != ZE_RESULT_SUCCESS) { \
    fprintf(stderr, "[rank?] FAIL %s -> 0x%x @ %s:%d\n", #call, _r, __FILE__, __LINE__); _exit(2);} } while(0)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// --- SCM_RIGHTS fd passing: send a 64-byte IPC handle blob WITH its embedded dma-buf fd ---
static void send_handle(int sock, const ze_ipc_mem_handle_t *h) {
    int fd = *(const int *)h->data;           // xe L0 embeds the dma-buf fd at the start of the blob
    struct iovec io = { (void *)h->data, sizeof(h->data) };
    char cbuf[CMSG_SPACE(sizeof(int))]; memset(cbuf, 0, sizeof(cbuf));
    struct msghdr m = {0};
    m.msg_iov = &io; m.msg_iovlen = 1; m.msg_control = cbuf; m.msg_controllen = sizeof(cbuf);
    struct cmsghdr *c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), &fd, sizeof(int));
    if (sendmsg(sock, &m, 0) < 0) { perror("sendmsg"); _exit(3); }
}
static void recv_handle(int sock, ze_ipc_mem_handle_t *h) {
    struct iovec io = { h->data, sizeof(h->data) };
    char cbuf[CMSG_SPACE(sizeof(int))]; memset(cbuf, 0, sizeof(cbuf));
    struct msghdr m = {0};
    m.msg_iov = &io; m.msg_iovlen = 1; m.msg_control = cbuf; m.msg_controllen = sizeof(cbuf);
    if (recvmsg(sock, &m, 0) < 0) { perror("recvmsg"); _exit(3); }
    struct cmsghdr *c = CMSG_FIRSTHDR(&m);
    int fd; memcpy(&fd, CMSG_DATA(c), sizeof(int));
    *(int *)h->data = fd;                     // patch in the fd valid in OUR process's table
}
// 1-byte rendezvous barrier over the bidirectional socketpair.
static void barrier(int sock) {
    char x = 1; if (write(sock, &x, 1) != 1) _exit(4);
    char y;     if (read(sock, &y, 1) != 1)  _exit(4);
}

static void run_rank(int rank, int sock) {
    int pr = 1 - rank;
    setvbuf(stdout, NULL, _IONBF, 0); // _exit() skips stdio flush; go unbuffered so prints survive
    CK(zeInit(0));
    uint32_t nd = 0; CK(zeDriverGet(&nd, NULL));
    ze_driver_handle_t *drv = calloc(nd, sizeof(*drv)); CK(zeDriverGet(&nd, drv));
    ze_driver_handle_t driver = drv[0];
    uint32_t ndev = 0; CK(zeDeviceGet(driver, &ndev, NULL));
    if (ndev < 2) { fprintf(stderr, "[rank %d] need >=2 devices\n", rank); _exit(1); }
    ze_device_handle_t *dev = calloc(ndev, sizeof(*dev)); CK(zeDeviceGet(driver, &ndev, dev));
    ze_device_handle_t mydev = dev[rank], peerdev = dev[pr];

    ze_context_desc_t cd = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0 };
    ze_context_handle_t ctx; CK(zeContextCreate(driver, &cd, &ctx));

    // compute-engine ordinal on my device is 0 (flags 0x7 on B70, per 100_ze_peer_copy).
    uint32_t ord = 0;

    size_t maxb = 256ull << 20;
    ze_device_mem_alloc_desc_t md = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, NULL, 0, 0 };
    void *localBuf, *localScratch;
    CK(zeMemAllocDevice(ctx, &md, maxb, 4096, mydev, &localBuf));
    CK(zeMemAllocDevice(ctx, &md, maxb, 4096, mydev, &localScratch));

    // export localScratch (the buffer the PEER will push into), import peer's.
    ze_ipc_mem_handle_t myH, peerH;
    CK(zeMemGetIpcHandle(ctx, localScratch, &myH));
    if (rank == 0) { send_handle(sock, &myH); recv_handle(sock, &peerH); }
    else           { recv_handle(sock, &peerH); send_handle(sock, &myH); }
    void *peerScratch; // lives in peer device VRAM, mapped into MY context
    CK(zeMemOpenIpcHandle(ctx, peerdev, peerH, 0, &peerScratch));
    if (rank == 0) printf("[rank0] IPC peer scratch mapped OK (cross-process peer ptr acquired)\n");

    // a synchronous immediate command list on MY device executes the PUSH (exec-on-source = posted write).
    ze_command_queue_desc_t qd = { ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, ord, 0, 0,
                                   ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL };
    ze_command_list_handle_t cl; CK(zeCommandListCreateImmediate(ctx, mydev, &qd, &cl));

    // fill localBuf with rank-distinct value so the peer can verify the write landed.
    float fillv = (rank == 0) ? 1.0f : 3.0f;
    // init via host buffer copy (small) -- fill first 4 floats is enough for verify; fill whole buf via pattern.
    ze_host_mem_alloc_desc_t hd = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, NULL, 0 };
    void *ph; CK(zeMemAllocHost(ctx, &hd, 4096, 4096, &ph));
    for (int i = 0; i < 1024; i++) ((float *)ph)[i] = fillv;
    // splat the pattern across localBuf
    for (size_t off = 0; off < maxb; off += 4096) {
        size_t chunk = (maxb - off < 4096) ? (maxb - off) : 4096;
        CK(zeCommandListAppendMemoryCopy(cl, (char *)localBuf + off, ph, chunk, NULL, 0, NULL));
    }

    size_t sizes[] = { 10240, 65536, 1u<<20, 16u<<20, 64u<<20, 256u<<20 };
    const char *labels[] = { "10KB(decode)", "64KB", "1MB", "16MB(prefill)", "64MB", "256MB" };
    int nsz = sizeof(sizes)/sizeof(sizes[0]);

    if (rank == 0)
        printf("%-15s %14s %12s %14s\n", "size", "push GB/s", "lat us", "verify(peer)");
    barrier(sock);

    for (int s = 0; s < nsz; s++) {
        size_t bytes = sizes[s];
        int iters = bytes <= (1u<<20) ? 300 : (bytes <= (16u<<20) ? 60 : 15);

        // one correctness exchange: PUSH localBuf -> peerScratch, barrier, check localScratch holds peer fill.
        CK(zeCommandListAppendMemoryCopy(cl, peerScratch, localBuf, bytes, NULL, 0, NULL));
        barrier(sock); // both pushes landed
        float chk = -1.0f;
        CK(zeCommandListAppendMemoryCopy(cl, ph, localScratch, sizeof(float), NULL, 0, NULL));
        chk = ((float *)ph)[0];
        float expect = (rank == 0) ? 3.0f : 1.0f; // peer's fill
        int ok = (chk > expect - 1e-3f && chk < expect + 1e-3f);

        for (int i = 0; i < 5; i++) { // warmup
            CK(zeCommandListAppendMemoryCopy(cl, peerScratch, localBuf, bytes, NULL, 0, NULL));
        }
        barrier(sock);
        double t0 = now_s();
        for (int i = 0; i < iters; i++) {
            CK(zeCommandListAppendMemoryCopy(cl, peerScratch, localBuf, bytes, NULL, 0, NULL));
        }
        double dt = (now_s() - t0) / iters;
        barrier(sock);
        if (rank == 0)
            printf("%-15s %14.2f %12.2f %14s\n", labels[s], bytes/dt/1e9, dt*1e6, ok ? "OK" : "BAD");
    }

    barrier(sock);
    if (rank == 0) printf("DONE_IPC_PUSH\n");
    zeMemCloseIpcHandle(ctx, peerScratch);
    zeMemFree(ctx, localBuf); zeMemFree(ctx, localScratch); zeMemFree(ctx, ph);
    zeContextDestroy(ctx);
    _exit(0);
}

int main(void) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) < 0) { perror("socketpair"); return 1; }
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) { close(sv[0]); run_rank(1, sv[1]); }
    else          { close(sv[1]); run_rank(0, sv[0]); }
    int st; waitpid(pid, &st, 0);
    return 0;
}
