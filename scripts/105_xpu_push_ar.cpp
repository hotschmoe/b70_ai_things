// 105_xpu_push_ar.cpp -- C-ABI shared library: 2-PROCESS push all-reduce for dual B70.
//
// This is the deployable core of the vLLM custom op (J.10). Unlike J.8 (one fork()'d binary), the two
// ranks here are INDEPENDENT processes (exactly how vLLM spawns TP workers -- no shared fds), so the
// Level-Zero IPC handle's embedded dma-buf fd is passed over a NAMED Unix socket via SCM_RIGHTS
// (oneCCL's "sockets" mode; robust vs pidfd_getfd which Ubuntu's ptrace_scope=1 blocks between siblings).
//
// Built with icpx -fsycl so we get SYCL kernels (push + local reduce) AND raw Level-Zero (IPC handles),
// bridged by sycl::get_native<level_zero>(context) -> ze_context_handle_t for zeMemGetIpcHandle.
//
// Synchronisation respects the J.9 finding: a peer write is only visible at KERNEL-COMPLETION, never
// mid-kernel. So each step's kernel is waited (q.wait) and the push->reduce boundary is host-driven
// (the caller does a process-group barrier between ar_push and ar_reduce).
//
// Algorithm per rank r (peer pr), data moving both ways = a 2-rank all-reduce:
//   ar_push  : SYCL kernel on my queue writes my data -> peerScratch (posted peer write); q.wait().
//   <caller barrier over the cpu group: both pushes have landed>
//   ar_reduce: SYCL kernel my data += myScratch (the value the peer pushed into me); q.wait().
//   end state: both ranks hold data == sum.  (extends trivially to ring for world>2.)
//
// Build: icpx -fsycl -O2 -fPIC -shared 105_xpu_push_ar.cpp -o libxpu_push_ar.so -lze_loader
// Driven by 105_ar_harness.py (torch.distributed gloo for handle exchange + barrier).
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
using namespace sycl;

// 2-process sense-reversing spin barrier in shared memory: ~1-2 us vs gloo TCP barrier's ~150 us.
struct ShmBar { int count; int sense; };

namespace {
device g_mydev, g_peerdev;
context *g_ctx = nullptr;
queue *g_q = nullptr;
ze_context_handle_t g_zectx = nullptr;
ze_device_handle_t g_ze_peerdev = nullptr;
void *g_data = nullptr;        // my local data buffer (the "tensor")
void *g_scratch = nullptr;     // my scratch: the peer pushes here (IPC-exported)
void *g_peerScratch = nullptr; // peer's scratch mapped into my context (I push here)
ze_ipc_mem_handle_t g_myH;
int g_rank = -1;
size_t g_max = 0;
ShmBar *g_bar = nullptr;   // shared-memory barrier (mmap'd, both processes)
int g_local_sense = 0;

void log(const char *m) { fprintf(stderr, "[ar rank%d] %s\n", g_rank, m); fflush(stderr); }
}

extern "C" int ar_setup(int rank, long max_bytes) {
    g_rank = rank; g_max = (size_t)max_bytes;
    std::vector<device> gpus;
    for (auto &p : platform::get_platforms())
        for (auto &d : p.get_devices(info::device_type::gpu))
            if (d.get_backend() == backend::ext_oneapi_level_zero) gpus.push_back(d);
    if (gpus.size() < 2) { log("need >=2 devices (ZE_AFFINITY_MASK=0,1)"); return 1; }
    g_mydev = gpus[rank]; g_peerdev = gpus[1 - rank];
    g_ctx = new context({g_mydev, g_peerdev});
    g_q = new queue(*g_ctx, g_mydev);
    if (g_mydev.ext_oneapi_can_access_peer(g_peerdev)) g_mydev.ext_oneapi_enable_peer_access(g_peerdev);
    g_zectx = get_native<backend::ext_oneapi_level_zero>(*g_ctx);
    g_ze_peerdev = get_native<backend::ext_oneapi_level_zero>(g_peerdev);
    g_data    = malloc_device(g_max, *g_q);
    g_scratch = malloc_device(g_max, *g_q);
    if (!g_data || !g_scratch) { log("malloc_device failed"); return 2; }
    // export an IPC handle for my scratch (the buffer the PEER will push into).
    ze_result_t r = zeMemGetIpcHandle(g_zectx, g_scratch, &g_myH);
    if (r != ZE_RESULT_SUCCESS) { fprintf(stderr,"[ar rank%d] zeMemGetIpcHandle 0x%x\n",rank,r); return 3; }
    log("setup OK (ctx, queue, scratch, ipc handle exported)");
    return 0;
}

// SCM_RIGHTS send/recv of {64-byte handle blob + embedded fd}.
static int send_h(int s, const ze_ipc_mem_handle_t *h) {
    int fd = *(const int *)h->data;
    struct iovec io = { (void*)h->data, sizeof(h->data) };
    char cb[CMSG_SPACE(sizeof(int))]; memset(cb,0,sizeof(cb));
    struct msghdr m = {}; m.msg_iov=&io; m.msg_iovlen=1; m.msg_control=cb; m.msg_controllen=sizeof(cb);
    struct cmsghdr *c = CMSG_FIRSTHDR(&m);
    c->cmsg_level=SOL_SOCKET; c->cmsg_type=SCM_RIGHTS; c->cmsg_len=CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c),&fd,sizeof(int));
    return sendmsg(s,&m,0) < 0 ? -1 : 0;
}
static int recv_h(int s, ze_ipc_mem_handle_t *h) {
    struct iovec io = { h->data, sizeof(h->data) };
    char cb[CMSG_SPACE(sizeof(int))]; memset(cb,0,sizeof(cb));
    struct msghdr m = {}; m.msg_iov=&io; m.msg_iovlen=1; m.msg_control=cb; m.msg_controllen=sizeof(cb);
    if (recvmsg(s,&m,0) < 0) return -1;
    struct cmsghdr *c = CMSG_FIRSTHDR(&m);
    int fd; memcpy(&fd,CMSG_DATA(c),sizeof(int));
    *(int*)h->data = fd;  // fd valid in OUR table now
    return 0;
}

// Exchange scratch IPC handles over a named Unix socket; rank0 listens, rank1 connects. Then open peer.
extern "C" int ar_exchange(int rank, const char *sockpath) {
    int sock = -1;
    if (rank == 0) {
        unlink(sockpath);
        int ls = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un a = {}; a.sun_family=AF_UNIX; strncpy(a.sun_path, sockpath, sizeof(a.sun_path)-1);
        if (bind(ls,(struct sockaddr*)&a,sizeof(a))<0){log("bind fail");return 10;}
        listen(ls,1);
        sock = accept(ls,nullptr,nullptr);
        close(ls);
    } else {
        struct sockaddr_un a = {}; a.sun_family=AF_UNIX; strncpy(a.sun_path, sockpath, sizeof(a.sun_path)-1);
        // retry connect until rank0 has bound
        for (int i=0;i<2000;i++){ sock=socket(AF_UNIX,SOCK_STREAM,0);
            if (connect(sock,(struct sockaddr*)&a,sizeof(a))==0) break; close(sock); sock=-1; usleep(2000); }
        if (sock<0){log("connect fail");return 11;}
    }
    ze_ipc_mem_handle_t peerH;
    // rank0 sends then recvs; rank1 recvs then sends (deadlock-free ordering).
    if (rank==0){ if(send_h(sock,&g_myH))return 12; if(recv_h(sock,&peerH))return 13; }
    else        { if(recv_h(sock,&peerH))return 13; if(send_h(sock,&g_myH))return 12; }
    close(sock);
    ze_result_t r = zeMemOpenIpcHandle(g_zectx, g_ze_peerdev, peerH, 0, &g_peerScratch);
    if (r != ZE_RESULT_SUCCESS){ fprintf(stderr,"[ar rank%d] zeMemOpenIpcHandle 0x%x\n",rank,r); return 14; }

    // shared-memory barrier setup (socket handshake above guarantees rank0 runs first).
    const char *bname = "/ar_shmbar";
    int fd = shm_open(bname, O_CREAT | O_RDWR, 0600);
    if (fd < 0) { log("shm_open fail"); return 15; }
    if (rank == 0) { if (ftruncate(fd, sizeof(ShmBar)) < 0) { log("ftruncate fail"); return 16; } }
    g_bar = (ShmBar*)mmap(nullptr, sizeof(ShmBar), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (g_bar == MAP_FAILED) { g_bar=nullptr; log("mmap fail"); return 17; }
    if (rank == 0) { g_bar->count = 0; g_bar->sense = 0; }
    log("exchange OK (peer scratch mapped cross-process; shm barrier ready)");
    return 0;
}

// 2-process sense-reversing spin barrier (world=2). ~1-2 us.
extern "C" void ar_barrier(void) {
    int my = !g_local_sense;
    if (__atomic_add_fetch(&g_bar->count, 1, __ATOMIC_ACQ_REL) == 2) {
        g_bar->count = 0;
        __atomic_store_n(&g_bar->sense, my, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&g_bar->sense, __ATOMIC_ACQUIRE) != my) { /* spin */ }
    }
    g_local_sense = my;
}

extern "C" void ar_fill(float v, long n)  { g_q->fill((float*)g_data, v, (size_t)n).wait(); }
extern "C" float ar_peek(void)            { float h; g_q->memcpy(&h,g_data,sizeof(float)).wait(); return h; }

// push my data -> peer's scratch (posted peer write); wait so the write is flushed at kernel boundary.
extern "C" void ar_push(long nbytes) {
    size_t n = (size_t)nbytes/sizeof(float);
    float *src=(float*)g_data, *dst=(float*)g_peerScratch;
    g_q->parallel_for(range<1>(n), [=](id<1> i){ dst[i]=src[i]; }).wait();
}
// local reduce: my data += my scratch (what the peer pushed into me).
extern "C" void ar_reduce(long nbytes) {
    size_t n = (size_t)nbytes/sizeof(float);
    float *d=(float*)g_data, *s=(float*)g_scratch;
    g_q->parallel_for(range<1>(n), [=](id<1> i){ d[i]+=s[i]; }).wait();
}

// full all-reduce in ONE call (what the vLLM op invokes): push -> shm barrier -> local reduce.
extern "C" void ar_allreduce(long nbytes) {
    size_t n = (size_t)nbytes/sizeof(float);
    float *src=(float*)g_data, *dst=(float*)g_peerScratch, *scr=(float*)g_scratch;
    g_q->parallel_for(range<1>(n), [=](id<1> i){ dst[i]=src[i]; }).wait(); // push (flush at kernel boundary)
    ar_barrier();                                                          // both pushes landed
    g_q->parallel_for(range<1>(n), [=](id<1> i){ src[i]+=scr[i]; }).wait();// local reduce
}

extern "C" void ar_teardown(void) {
    if (g_bar) { munmap(g_bar, sizeof(ShmBar)); if (g_rank==0) shm_unlink("/ar_shmbar"); }
    if (g_peerScratch) zeMemCloseIpcHandle(g_zectx, g_peerScratch);
    if (g_data) free(g_data,*g_q);
    if (g_scratch) free(g_scratch,*g_q);
}
