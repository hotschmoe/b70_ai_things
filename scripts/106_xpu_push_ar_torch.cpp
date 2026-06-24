// 106_xpu_push_ar_torch.cpp -- custom all-reduce that runs IN TORCH'S L0 context (the live-serve bind).
//
// J.10 (105) proved the 2-process push all-reduce beats oneCCL, but in OUR own SYCL context. A torch
// tensor's USM pointer is only valid in TORCH's context, so to all-reduce a real vLLM activation we must
// run on torch's queue/context. torch-xpu exposes the address of its `sycl::queue` as an int via
// `torch.xpu.current_stream().sycl_queue`; we reinterpret that to a sycl::queue* and take its context.
// Then: input tensor ptr (torch ctx), my scratch (raw zeMemAllocDevice in torch ctx -> base-aligned,
// cleanly IPC-exportable), and the peer's mapped scratch all live in ONE context -> no cross-context use.
//
// Differences from 105: ar_setup_torch(rank, torch_queue_addr, max) instead of ar_setup; ar_allreduce_ptr
// (input pointer passed in, in-place) instead of the self-allocated test buffer. ar_exchange / ar_barrier
// are identical (named-socket SCM_RIGHTS fd-pass + shm spin barrier).
//
// Build: icpx -fsycl -O2 -fPIC -shared 106_xpu_push_ar_torch.cpp -o libxpu_push_ar_torch.so -lze_loader -lrt
// Driven by 106_ar_torch_harness.py (real torch.xpu tensors). If it verifies, the vLLM monkeypatch is trivial.
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
using namespace sycl;

struct ShmBar { int count; int sense; };

namespace {
queue *g_q = nullptr;                 // TORCH's queue (not owned)
context g_ctx;
device g_mydev, g_peerdev;
ze_context_handle_t g_zectx = nullptr;
ze_device_handle_t g_ze_peerdev = nullptr;
void *g_scratch = nullptr;            // raw-L0 scratch in torch's ctx (peer pushes here)
void *g_peerScratch = nullptr;        // peer's scratch mapped into torch's ctx (I push here)
ze_ipc_mem_handle_t g_myH;
ShmBar *g_bar = nullptr;
int g_local_sense = 0;
int g_rank = -1;
void log(const char *m){ fprintf(stderr,"[artorch r%d] %s\n",g_rank,m); fflush(stderr); }
}

extern "C" int ar_setup_torch(int rank, unsigned long long torch_q_addr, long max_bytes) {
    g_rank = rank;
    g_q = reinterpret_cast<queue*>(torch_q_addr);  // torch's real sycl::queue
    g_ctx = g_q->get_context();
    g_mydev = g_q->get_device();
    // find the peer device in the same context.
    auto devs = g_ctx.get_devices();
    g_peerdev = g_mydev;
    for (auto &d : devs) if (d != g_mydev) { g_peerdev = d; break; }
    if (g_peerdev == g_mydev) {
        // torch ctx may hold only this rank's device; pull both from the platform and enable peer.
        std::vector<device> gpus;
        for (auto &p : platform::get_platforms())
            for (auto &d : p.get_devices(info::device_type::gpu))
                if (d.get_backend()==backend::ext_oneapi_level_zero) gpus.push_back(d);
        if (gpus.size()>=2) g_peerdev = gpus[1-rank];
    }
    if (g_mydev.ext_oneapi_can_access_peer(g_peerdev)) g_mydev.ext_oneapi_enable_peer_access(g_peerdev);
    g_zectx = get_native<backend::ext_oneapi_level_zero>(g_ctx);
    g_ze_peerdev = get_native<backend::ext_oneapi_level_zero>(g_peerdev);
    // raw-L0 scratch alloc in torch's ctx (base-aligned -> IPC offset 0).
    ze_device_mem_alloc_desc_t md = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, NULL, 0, 0 };
    ze_device_handle_t myze = get_native<backend::ext_oneapi_level_zero>(g_mydev);
    ze_result_t r = zeMemAllocDevice(g_zectx, &md, (size_t)max_bytes, 4096, myze, &g_scratch);
    if (r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[artorch r%d] zeMemAllocDevice 0x%x\n",rank,r); return 2; }
    r = zeMemGetIpcHandle(g_zectx, g_scratch, &g_myH);
    if (r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[artorch r%d] zeMemGetIpcHandle 0x%x\n",rank,r); return 3; }
    log("setup_torch OK (running in torch's L0 context)");
    return 0;
}

static int send_h(int s, const ze_ipc_mem_handle_t *h){
    int fd=*(const int*)h->data; struct iovec io={(void*)h->data,sizeof(h->data)};
    char cb[CMSG_SPACE(sizeof(int))]; memset(cb,0,sizeof(cb));
    struct msghdr m={}; m.msg_iov=&io; m.msg_iovlen=1; m.msg_control=cb; m.msg_controllen=sizeof(cb);
    struct cmsghdr *c=CMSG_FIRSTHDR(&m); c->cmsg_level=SOL_SOCKET; c->cmsg_type=SCM_RIGHTS;
    c->cmsg_len=CMSG_LEN(sizeof(int)); memcpy(CMSG_DATA(c),&fd,sizeof(int));
    return sendmsg(s,&m,0)<0?-1:0;
}
static int recv_h(int s, ze_ipc_mem_handle_t *h){
    struct iovec io={h->data,sizeof(h->data)}; char cb[CMSG_SPACE(sizeof(int))]; memset(cb,0,sizeof(cb));
    struct msghdr m={}; m.msg_iov=&io; m.msg_iovlen=1; m.msg_control=cb; m.msg_controllen=sizeof(cb);
    if(recvmsg(s,&m,0)<0)return -1;
    struct cmsghdr *c=CMSG_FIRSTHDR(&m); int fd; memcpy(&fd,CMSG_DATA(c),sizeof(int)); *(int*)h->data=fd; return 0;
}

extern "C" int ar_exchange(int rank, const char *sockpath) {
    int sock=-1;
    if (rank==0){ unlink(sockpath); int ls=socket(AF_UNIX,SOCK_STREAM,0);
        struct sockaddr_un a={}; a.sun_family=AF_UNIX; strncpy(a.sun_path,sockpath,sizeof(a.sun_path)-1);
        if(bind(ls,(struct sockaddr*)&a,sizeof(a))<0){log("bind fail");return 10;}
        listen(ls,1); sock=accept(ls,nullptr,nullptr); close(ls);
    } else {
        struct sockaddr_un a={}; a.sun_family=AF_UNIX; strncpy(a.sun_path,sockpath,sizeof(a.sun_path)-1);
        for(int i=0;i<2000;i++){ sock=socket(AF_UNIX,SOCK_STREAM,0);
            if(connect(sock,(struct sockaddr*)&a,sizeof(a))==0)break; close(sock); sock=-1; usleep(2000);}
        if(sock<0){log("connect fail");return 11;}
    }
    ze_ipc_mem_handle_t peerH;
    if(rank==0){ if(send_h(sock,&g_myH))return 12; if(recv_h(sock,&peerH))return 13; }
    else       { if(recv_h(sock,&peerH))return 13; if(send_h(sock,&g_myH))return 12; }
    close(sock);
    ze_result_t r=zeMemOpenIpcHandle(g_zectx,g_ze_peerdev,peerH,0,&g_peerScratch);
    if(r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[artorch r%d] zeMemOpenIpcHandle 0x%x\n",rank,r); return 14; }
    const char *bn="/ar_shmbar_torch"; int fd=shm_open(bn,O_CREAT|O_RDWR,0600);
    if(fd<0){log("shm_open fail");return 15;}
    if(rank==0){ if(ftruncate(fd,sizeof(ShmBar))<0){log("ftruncate fail");return 16;} }
    g_bar=(ShmBar*)mmap(nullptr,sizeof(ShmBar),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);
    if(g_bar==MAP_FAILED){g_bar=nullptr;log("mmap fail");return 17;}
    if(rank==0){ g_bar->count=0; g_bar->sense=0; }
    log("exchange OK (peer scratch mapped in torch ctx; shm barrier ready)");
    return 0;
}

extern "C" void ar_barrier(void){
    int my=!g_local_sense;
    if(__atomic_add_fetch(&g_bar->count,1,__ATOMIC_ACQ_REL)==2){ g_bar->count=0;
        __atomic_store_n(&g_bar->sense,my,__ATOMIC_RELEASE);
    } else { while(__atomic_load_n(&g_bar->sense,__ATOMIC_ACQUIRE)!=my){} }
    g_local_sense=my;
}

// all-reduce a buffer at `inout` (torch tensor data_ptr), in place, fp32. nbytes total.
extern "C" void ar_allreduce_ptr(unsigned long long inout, long nbytes) {
    size_t n=(size_t)nbytes/sizeof(float);
    float *src=reinterpret_cast<float*>(inout), *dst=(float*)g_peerScratch, *scr=(float*)g_scratch;
    g_q->parallel_for(range<1>(n),[=](id<1> i){ dst[i]=src[i]; }).wait(); // push to peer
    ar_barrier();
    g_q->parallel_for(range<1>(n),[=](id<1> i){ src[i]+=scr[i]; }).wait();// local reduce
}

extern "C" void ar_teardown(void){
    if(g_bar){ munmap(g_bar,sizeof(ShmBar)); if(g_rank==0) shm_unlink("/ar_shmbar_torch"); }
    if(g_peerScratch) zeMemCloseIpcHandle(g_zectx,g_peerScratch);
    if(g_scratch) zeMemFree(g_zectx,g_scratch);
}
