// xpu_push_ar_fused_rmsnorm.cpp -- experiment-118 push all-reduce plus an
// eager BF16 all-reduce + residual + Gemma RMSNorm boundary kernel.
//
// Existing experiment-118 ABI is preserved. ar_setup_torch allocates its
// original max_bytes scratch followed by two aligned fused-operation slots.
//
// Build:
//   icpx -fsycl -O2 -fPIC -shared xpu_push_ar_fused_rmsnorm.cpp \
//     -o libxpu_push_ar_fused_rmsnorm.so -lze_loader -lrt
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <level_zero/ze_api.h>
#include <cstdint>
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
queue *g_q = nullptr;
context g_ctx;
device g_mydev, g_peerdev;
ze_context_handle_t g_zectx = nullptr;
ze_device_handle_t g_myze = nullptr, g_ze_peerdev = nullptr;
void *g_scratch = nullptr;           // local: peer pushes here
void *g_peerScratch = nullptr;       // peer's scratch mapped here: I push here
ze_ipc_mem_handle_t g_myH;
ShmBar *g_bar = nullptr;
int g_local_sense = 0;
int g_rank = -1;
// event sync (experiment 118 K.3-K.5)
ze_event_pool_handle_t g_pool = nullptr;
ze_event_handle_t g_S_A = nullptr, g_S_B = nullptr;
ze_event_handle_t g_sigEv = nullptr, g_waitEv = nullptr;

constexpr size_t kAllocAlignment = 4096;
constexpr int kFusedMaxRows = 128;
#ifndef B70_FUSED_FAST_MAX_ROWS
#define B70_FUSED_FAST_MAX_ROWS 2
#endif
#ifndef B70_FUSED_WORKGROUP_SIZE
#define B70_FUSED_WORKGROUP_SIZE 512
#endif
constexpr int kFusedFastMaxRows = B70_FUSED_FAST_MAX_ROWS;
static_assert(kFusedFastMaxRows >= 1 && kFusedFastMaxRows <= kFusedMaxRows);
constexpr size_t kFusedWorkGroupSize = B70_FUSED_WORKGROUP_SIZE;
static_assert(kFusedWorkGroupSize >= 16 && kFusedWorkGroupSize <= 512);
static_assert(kFusedWorkGroupSize % 16 == 0);
constexpr int kFusedHidden = 5120;
constexpr int kFusedRingSlots = 2;
constexpr size_t kBf16Bytes = 2;
constexpr size_t kFusedPayloadBytes =
    (size_t)kFusedMaxRows * kFusedHidden * kBf16Bytes;
constexpr size_t align_up_const(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}
constexpr size_t kFusedSlotBytes =
    align_up_const(kFusedPayloadBytes, kAllocAlignment);
size_t g_fused_base_offset = 0;
size_t g_total_alloc_bytes = 0;
uint64_t g_fused_sequence = 0;
queue *g_fused_q = nullptr;

void log(const char *m){ fprintf(stderr,"[argraph r%d] %s\n",g_rank,m); fflush(stderr); }
}

extern "C" int ar_setup_torch(int rank, unsigned long long torch_q_addr, long max_bytes) {
    if (max_bytes <= 0) return 1;
    g_rank = rank;
    g_q = reinterpret_cast<queue*>(torch_q_addr);
    g_ctx = g_q->get_context();
    g_mydev = g_q->get_device();
    auto devs = g_ctx.get_devices();
    g_peerdev = g_mydev;
    for (auto &d : devs) if (d != g_mydev) { g_peerdev = d; break; }
    if (g_peerdev == g_mydev) {
        std::vector<device> gpus;
        for (auto &p : platform::get_platforms())
            for (auto &d : p.get_devices(info::device_type::gpu))
                if (d.get_backend()==backend::ext_oneapi_level_zero) gpus.push_back(d);
        if (gpus.size()>=2) g_peerdev = gpus[1-rank];
    }
    if (g_mydev.ext_oneapi_can_access_peer(g_peerdev)) g_mydev.ext_oneapi_enable_peer_access(g_peerdev);
    g_zectx = get_native<backend::ext_oneapi_level_zero>(g_ctx);
    g_ze_peerdev = get_native<backend::ext_oneapi_level_zero>(g_peerdev);
    g_myze = get_native<backend::ext_oneapi_level_zero>(g_mydev);
    g_fused_base_offset = align_up_const((size_t)max_bytes, kAllocAlignment);
    g_total_alloc_bytes =
        g_fused_base_offset + kFusedRingSlots * kFusedSlotBytes;
    g_fused_sequence = 0;
    g_fused_q = nullptr;
    ze_device_mem_alloc_desc_t md = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, NULL, 0, 0 };
    ze_result_t r = zeMemAllocDevice(
        g_zectx, &md, g_total_alloc_bytes, kAllocAlignment, g_myze, &g_scratch);
    if (r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[argraph r%d] zeMemAllocDevice 0x%x\n",rank,r); return 2; }
    r = zeMemGetIpcHandle(g_zectx, g_scratch, &g_myH);
    if (r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[argraph r%d] zeMemGetIpcHandle 0x%x\n",rank,r); return 3; }
    log("setup_torch OK");
    return 0;
}

static int send_blob(int s, const char *data, size_t n){
    int fd=*(const int*)data; struct iovec io={(void*)data,n};
    char cb[CMSG_SPACE(sizeof(int))]; memset(cb,0,sizeof(cb));
    struct msghdr m={}; m.msg_iov=&io; m.msg_iovlen=1; m.msg_control=cb; m.msg_controllen=sizeof(cb);
    struct cmsghdr *c=CMSG_FIRSTHDR(&m); c->cmsg_level=SOL_SOCKET; c->cmsg_type=SCM_RIGHTS;
    c->cmsg_len=CMSG_LEN(sizeof(int)); memcpy(CMSG_DATA(c),&fd,sizeof(int));
    return sendmsg(s,&m,0)<0?-1:0;
}
static int recv_blob(int s, char *data, size_t n){
    struct iovec io={data,n}; char cb[CMSG_SPACE(sizeof(int))]; memset(cb,0,sizeof(cb));
    struct msghdr m={}; m.msg_iov=&io; m.msg_iovlen=1; m.msg_control=cb; m.msg_controllen=sizeof(cb);
    if(recvmsg(s,&m,0)<0)return -1;
    struct cmsghdr *c=CMSG_FIRSTHDR(&m); int fd; memcpy(&fd,CMSG_DATA(c),sizeof(int)); *(int*)data=fd; return 0;
}

// ar_exchange: scratch IPC (peer push target) + shm host barrier (eager) +
// IPC event pool (graph sync).
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
    // --- scratch handle exchange ---
    ze_ipc_mem_handle_t peerH;
    if(rank==0){ if(send_blob(sock,g_myH.data,sizeof(g_myH.data)))return 12; if(recv_blob(sock,peerH.data,sizeof(peerH.data)))return 13; }
    else       { if(recv_blob(sock,peerH.data,sizeof(peerH.data)))return 13; if(send_blob(sock,g_myH.data,sizeof(g_myH.data)))return 12; }
    ze_result_t r=zeMemOpenIpcHandle(g_zectx,g_ze_peerdev,peerH,0,&g_peerScratch);
    if(r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[argraph r%d] zeMemOpenIpcHandle 0x%x\n",rank,r); return 14; }
    // --- shm host barrier (kept for the eager paths) ---
    const char *bn="/ar_shmbar_graph"; int fd=shm_open(bn,O_CREAT|O_RDWR,0600);
    if(fd<0){log("shm_open fail");return 15;}
    if(rank==0){ if(ftruncate(fd,sizeof(ShmBar))<0){log("ftruncate fail");return 16;} }
    g_bar=(ShmBar*)mmap(nullptr,sizeof(ShmBar),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);
    if(g_bar==MAP_FAILED){g_bar=nullptr;log("mmap fail");return 17;}
    if(rank==0){ g_bar->count=0; g_bar->sense=0; }
    // --- IPC EVENT POOL (must span both devices, experiment 118 K.5) ---
    ze_device_handle_t both[2]={g_myze,g_ze_peerdev};
    if(rank==0){
        ze_event_pool_desc_t epd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,NULL,
            ZE_EVENT_POOL_FLAG_IPC|ZE_EVENT_POOL_FLAG_HOST_VISIBLE,2};
        r=zeEventPoolCreate(g_zectx,&epd,2,both,&g_pool);
        if(r!=ZE_RESULT_SUCCESS){fprintf(stderr,"[argraph r%d] EventPoolCreate 0x%x\n",rank,r);return 20;}
        ze_ipc_event_pool_handle_t iph; if(zeEventPoolGetIpcHandle(g_pool,&iph)!=ZE_RESULT_SUCCESS)return 21;
        if(send_blob(sock,iph.data,sizeof(iph.data)))return 22;
    } else {
        ze_ipc_event_pool_handle_t iph; memset(&iph,0,sizeof(iph));
        if(recv_blob(sock,iph.data,sizeof(iph.data)))return 22;
        r=zeEventPoolOpenIpcHandle(g_zectx,iph,&g_pool);
        if(r!=ZE_RESULT_SUCCESS){fprintf(stderr,"[argraph r%d] EventPoolOpenIpc 0x%x\n",rank,r);return 23;}
    }
    ze_event_desc_t dA={ZE_STRUCTURE_TYPE_EVENT_DESC,NULL,0,ZE_EVENT_SCOPE_FLAG_HOST,ZE_EVENT_SCOPE_FLAG_HOST};
    ze_event_desc_t dB={ZE_STRUCTURE_TYPE_EVENT_DESC,NULL,1,ZE_EVENT_SCOPE_FLAG_HOST,ZE_EVENT_SCOPE_FLAG_HOST};
    if(zeEventCreate(g_pool,&dA,&g_S_A)!=ZE_RESULT_SUCCESS)return 24;
    if(zeEventCreate(g_pool,&dB,&g_S_B)!=ZE_RESULT_SUCCESS)return 25;
    zeEventHostReset(g_S_A); zeEventHostReset(g_S_B);
    g_sigEv  = (rank==0)?g_S_A:g_S_B;
    g_waitEv = (rank==0)?g_S_B:g_S_A;
    close(sock);
    log("exchange OK (scratch + shm barrier + IPC event pool)");
    return 0;
}

extern "C" void ar_barrier(void){
    int my=!g_local_sense;
    if(__atomic_add_fetch(&g_bar->count,1,__ATOMIC_ACQ_REL)==2){ g_bar->count=0;
        __atomic_store_n(&g_bar->sense,my,__ATOMIC_RELEASE);
    } else { while(__atomic_load_n(&g_bar->sense,__ATOMIC_ACQUIRE)!=my){} }
    g_local_sense=my;
}

// ---- EAGER experiment-118 path, unchanged apart from the larger allocation ----
template<typename T>
static void do_ar(unsigned long long inout, size_t nbytes) {
    size_t nw = nbytes/4;
    uint32_t *ws=reinterpret_cast<uint32_t*>(inout), *wd=(uint32_t*)g_peerScratch;
    g_q->parallel_for(range<1>(nw),[=](id<1> i){ wd[i]=ws[i]; }).wait();
    ar_barrier();
    size_t n=nbytes/sizeof(T);
    T *src=reinterpret_cast<T*>(inout), *scr=(T*)g_scratch;
    g_q->parallel_for(range<1>(n),[=](id<1> i){ src[i]=(T)(float(src[i])+float(scr[i])); }).wait();
}
extern "C" void ar_allreduce_ptr_dt(unsigned long long inout, long nbytes, int dtype) {
    if (dtype==1)      do_ar<sycl::ext::oneapi::bfloat16>(inout,(size_t)nbytes);
    else if (dtype==2) do_ar<sycl::half>(inout,(size_t)nbytes);
    else               do_ar<float>(inout,(size_t)nbytes);
}

// Fused eager boundary. The push wait and host barrier match experiment 118.
// The final kernel is submitted without a host wait. Requiring one stable
// in-order queue, plus alternating scratch slots, prevents a peer from
// overwriting a slot while its prior asynchronous normalization still reads it.
extern "C" int ar_allreduce_residual_gemma_rmsnorm_bf16(
    uint64_t q_addr,
    uint64_t x_addr,
    uint64_t residual_addr,
    uint64_t raw_weight_addr,
    int rows,
    int hidden,
    float eps) {
    using bf16 = sycl::ext::oneapi::bfloat16;
    constexpr size_t kWorkGroupSize = kFusedWorkGroupSize;
    constexpr size_t kVecSize = 8;
    constexpr size_t kIters =
        (kFusedHidden + kWorkGroupSize * kVecSize - 1) /
        (kWorkGroupSize * kVecSize);
    struct alignas(16) Bf16Vec {
        bf16 value[kVecSize];
    };

    if (!g_q || !g_scratch || !g_peerScratch || !g_bar) return 30;
    if (rows < 1 || rows > kFusedMaxRows || hidden != kFusedHidden) return 31;
    // This ABI is a decode fast path. Return a distinct status before any
    // communication or mutation so the caller can run the stock collective
    // and Gemma RMSNorm for larger batches.
    if (rows > kFusedFastMaxRows) return 38;
    if (!q_addr || !x_addr || !residual_addr || !raw_weight_addr) return 32;
    if ((x_addr | residual_addr | raw_weight_addr) & 0xFULL) return 33;

    queue *q = reinterpret_cast<queue*>(q_addr);
    if (!q->has_property<property::queue::in_order>()) return 34;
    if (q->get_context() != g_ctx || q->get_device() != g_mydev) return 35;
    if (g_fused_q && q != g_fused_q) return 36;
    g_fused_q = q;

    const size_t elements = (size_t)rows * hidden;
    const size_t nbytes = elements * sizeof(bf16);
    if (nbytes > kFusedSlotBytes) return 37;
    const size_t slot = (size_t)(g_fused_sequence++ & 1ULL);
    const size_t slot_offset = g_fused_base_offset + slot * kFusedSlotBytes;
    auto *x = reinterpret_cast<bf16*>(x_addr);
    auto *residual = reinterpret_cast<bf16*>(residual_addr);
    auto *raw_weight = reinterpret_cast<const bf16*>(raw_weight_addr);
    auto *local_slot = reinterpret_cast<bf16*>(
        static_cast<char*>(g_scratch) + slot_offset);
    auto *peer_slot = reinterpret_cast<bf16*>(
        static_cast<char*>(g_peerScratch) + slot_offset);

    // Push two BF16 values per work-item, matching the proven experiment-118
    // copy geometry instead of launching one work-item per BF16 element.
    const size_t copy_words = nbytes / sizeof(uint32_t);
    auto *copy_src = reinterpret_cast<const uint32_t*>(x);
    auto *copy_dst = reinterpret_cast<uint32_t*>(peer_slot);
    q->parallel_for(range<1>(copy_words), [=](id<1> i) {
        copy_dst[i] = copy_src[i];
    }).wait();
    ar_barrier();

    // One asynchronous kernel: rounded AR, rounded residual, FP32 sumsq, then
    // Gemma RMSNorm with raw BF16 weight interpreted as (1 + weight). Match
    // sgl-kernel-xpu's BF16 no-rstd path: vec8 inputs, a configurable
    // subgroup-16 workgroup, cached iterations, and reduce_over_group. The
    // production default is WG512; the macro exists for bounded lab sweeps.
    q->submit([&](handler &h) {
        h.parallel_for(
            nd_range<1>(range<1>((size_t)rows * kWorkGroupSize),
                        range<1>(kWorkGroupSize)),
            [=](nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const size_t row = item.get_group(0);
                const size_t lane = item.get_local_id(0);
                const size_t base = row * (size_t)hidden;
                float sumsq = 0.0f;
                Bf16Vec cached[kIters];

                for (size_t iter = 0; iter < kIters; ++iter) {
                    const size_t col =
                        (iter * kWorkGroupSize + lane) * kVecSize;
                    if (col < (size_t)hidden) {
                        const size_t index = base + col;
                        const Bf16Vec x_vec =
                            *reinterpret_cast<const Bf16Vec*>(x + index);
                        const Bf16Vec peer_vec =
                            *reinterpret_cast<const Bf16Vec*>(local_slot + index);
                        const Bf16Vec old_residual_vec =
                            *reinterpret_cast<const Bf16Vec*>(residual + index);
                        Bf16Vec residual_vec;
#pragma unroll
                        for (size_t v = 0; v < kVecSize; ++v) {
                            const bf16 ar_bf16 = bf16(
                                float(x_vec.value[v]) +
                                float(peer_vec.value[v]));
                            const bf16 residual_bf16 = bf16(
                                float(ar_bf16) +
                                float(old_residual_vec.value[v]));
                            residual_vec.value[v] = residual_bf16;
                            const float rounded_residual =
                                float(residual_bf16);
                            sumsq += rounded_residual * rounded_residual;
                        }
                        *reinterpret_cast<Bf16Vec*>(residual + index) =
                            residual_vec;
                        cached[iter] = residual_vec;
                    }
                }

                sumsq = sycl::reduce_over_group(
                    item.get_group(), sumsq, sycl::plus<float>());
                sumsq = sumsq < 0.0f ? 0.0f : sumsq;
                const float rstd =
                    sycl::rsqrt(sumsq / float(hidden) + eps);

                for (size_t iter = 0; iter < kIters; ++iter) {
                    const size_t col =
                        (iter * kWorkGroupSize + lane) * kVecSize;
                    if (col < (size_t)hidden) {
                        const size_t index = base + col;
                        const Bf16Vec weight_vec =
                            *reinterpret_cast<const Bf16Vec*>(
                                raw_weight + col);
                        Bf16Vec output_vec;
#pragma unroll
                        for (size_t v = 0; v < kVecSize; ++v) {
                            output_vec.value[v] = bf16(
                                float(cached[iter].value[v]) * rstd *
                                (1.0f + float(weight_vec.value[v])));
                        }
                        *reinterpret_cast<Bf16Vec*>(x + index) = output_vec;
                    }
                }
            });
    });
    return 0;
}

// ---- CAPTURABLE experiment-118 path, unchanged ----
template<typename T>
static void do_ar_graph(queue *q, unsigned long long inout, size_t nbytes) {
    size_t nw = nbytes/4;
    uint32_t *ws=reinterpret_cast<uint32_t*>(inout), *wd=(uint32_t*)g_peerScratch;
    event ep = q->parallel_for(range<1>(nw),[=](id<1> i){ wd[i]=ws[i]; });
    [[maybe_unused]] ze_event_handle_t sigEv=g_sigEv, waitEv=g_waitEv;
    event es = q->submit([&](handler &h){ h.depends_on(ep);
        h.ext_codeplay_enqueue_native_command([=]([[maybe_unused]] interop_handle ih){
#ifndef __SYCL_DEVICE_ONLY__
            ze_command_list_handle_t cl=ih.ext_codeplay_get_native_graph<backend::ext_oneapi_level_zero>();
            zeCommandListAppendSignalEvent(cl,sigEv);
            ze_event_handle_t w=waitEv; zeCommandListAppendWaitOnEvents(cl,1,&w);
            zeCommandListAppendEventReset(cl,waitEv);
#endif
        }); });
    size_t n=nbytes/sizeof(T);
    T *src=reinterpret_cast<T*>(inout), *scr=(T*)g_scratch;
    q->submit([&](handler &h){ h.depends_on(es);
        h.parallel_for(range<1>(n),[=](id<1> i){ src[i]=(T)(float(src[i])+float(scr[i])); }); });
}
extern "C" void ar_allreduce_graph(unsigned long long q_addr, unsigned long long inout, long nbytes, int dtype) {
    queue *q = q_addr ? reinterpret_cast<queue*>(q_addr) : g_q;
    if (dtype==1)      do_ar_graph<sycl::ext::oneapi::bfloat16>(q,inout,(size_t)nbytes);
    else if (dtype==2) do_ar_graph<sycl::half>(q,inout,(size_t)nbytes);
    else               do_ar_graph<float>(q,inout,(size_t)nbytes);
}

extern "C" void ar_teardown(void){
    if(g_S_A) zeEventDestroy(g_S_A); if(g_S_B) zeEventDestroy(g_S_B);
    if(g_pool) zeEventPoolDestroy(g_pool);
    if(g_bar){ munmap(g_bar,sizeof(ShmBar)); if(g_rank==0) shm_unlink("/ar_shmbar_graph"); }
    if(g_peerScratch) zeMemCloseIpcHandle(g_zectx,g_peerScratch);
    if(g_scratch) zeMemFree(g_zectx,g_scratch);
    g_q=nullptr; g_scratch=nullptr; g_peerScratch=nullptr; g_bar=nullptr;
    g_pool=nullptr; g_S_A=nullptr; g_S_B=nullptr; g_sigEv=nullptr; g_waitEv=nullptr;
    g_fused_base_offset=0; g_total_alloc_bytes=0; g_fused_sequence=0; g_fused_q=nullptr;
}
