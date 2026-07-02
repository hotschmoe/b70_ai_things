// 118_xpu_push_ar_graph.cpp -- CAPTURABLE push all-reduce for vLLM decode (extends 106 with a graph path).
//
// 106 made the push all-reduce run on torch's L0 context, but its rank sync is a HOST barrier (shm spin +
// host .wait()), so it is NOT graph-capturable -> decode (captured) falls back to oneCCL (P2P_GPU J.14/J.17).
// K.3-K.5 proved the fix: a cross-device L0-EVENT wait (command-streamer / HW semaphore, NOT the J.9-C EU spin)
// is correct + replayable, records into a SYCL command_graph via ext_codeplay_enqueue_native_command, and works
// cross-process via an IPC event pool (which MUST span both devices). torch-xpu's XPUGraph IS sycl command_graph
// so anything we submit on torch's stream during capture becomes a graph node.
//
// This file keeps the proven EAGER path (ar_allreduce_ptr_dt, host barrier -- for prefill) and ADDS:
//   ar_setup_events / ar_exchange_events  : create+share an IPC event pool (both devices), make S_A/S_B.
//   ar_allreduce_graph(ptr,nbytes,dtype)  : push kernel -> native cmd[signal mine; wait peer; reset peer] ->
//                                           reduce kernel, all on torch's queue. NO host barrier. During
//                                           torch XPUGraph capture this records; on replay it runs device-only.
// The monkeypatch routes: torch.xpu.is_current_stream_capturing() -> ar_allreduce_graph (decode, captured)
//   else ar_allreduce_ptr_dt (prefill, eager). NOTE: ar_allreduce_graph is CAPTURE-ONLY (it uses
//   ext_codeplay_get_native_graph, valid only while recording; get_native_queue<level_zero> is broken in DPC++).
//
// Build: icpx -fsycl -O2 -fPIC -shared 118_xpu_push_ar_graph.cpp -o libxpu_push_ar_graph.so -lze_loader -lrt
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
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
// event sync (K.3-K.5)
ze_event_pool_handle_t g_pool = nullptr;
ze_event_handle_t g_S_A = nullptr, g_S_B = nullptr;  // S_A signaled by rank0, S_B by rank1
ze_event_handle_t g_sigEv = nullptr, g_waitEv = nullptr;
void log(const char *m){ fprintf(stderr,"[argraph r%d] %s\n",g_rank,m); fflush(stderr); }
}

extern "C" int ar_setup_torch(int rank, unsigned long long torch_q_addr, long max_bytes) {
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
    ze_device_mem_alloc_desc_t md = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, NULL, 0, 0 };
    ze_result_t r = zeMemAllocDevice(g_zectx, &md, (size_t)max_bytes, 4096, g_myze, &g_scratch);
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

// ar_exchange: scratch IPC (peer push target) + shm host barrier (eager) + IPC EVENT POOL (graph sync).
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
    // --- shm host barrier (kept for the eager path) ---
    const char *bn="/ar_shmbar_graph"; int fd=shm_open(bn,O_CREAT|O_RDWR,0600);
    if(fd<0){log("shm_open fail");return 15;}
    if(rank==0){ if(ftruncate(fd,sizeof(ShmBar))<0){log("ftruncate fail");return 16;} }
    g_bar=(ShmBar*)mmap(nullptr,sizeof(ShmBar),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);
    if(g_bar==MAP_FAILED){g_bar=nullptr;log("mmap fail");return 17;}
    if(rank==0){ g_bar->count=0; g_bar->sense=0; }
    // --- IPC EVENT POOL (must span BOTH devices, K.5) ---
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

// ---- EAGER path (host barrier), unchanged from 106 ----
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

// ---- CAPTURABLE path (device event sync, no host barrier). CAPTURE-ONLY. ----
// IMPORTANT: torch captures on a DEDICATED capture stream, not the default stream cached at setup. The
// caller MUST pass the CURRENT stream's sycl_queue (torch.xpu.current_stream().sycl_queue) so our ops land
// on the stream torch is recording -> ext_codeplay_get_native_graph then returns the graph command list.
template<typename T>
static void do_ar_graph(queue *q, unsigned long long inout, size_t nbytes) {
    size_t nw = nbytes/4;
    uint32_t *ws=reinterpret_cast<uint32_t*>(inout), *wd=(uint32_t*)g_peerScratch;
    // push (4-byte words) -> peer scratch
    event ep = q->parallel_for(range<1>(nw),[=](id<1> i){ wd[i]=ws[i]; });
    // cross-device sync injected as native L0 commands recorded into torch's graph
    ze_event_handle_t sigEv=g_sigEv, waitEv=g_waitEv;
    event es = q->submit([&](handler &h){ h.depends_on(ep);
        h.ext_codeplay_enqueue_native_command([=](interop_handle ih){
#ifndef __SYCL_DEVICE_ONLY__
            ze_command_list_handle_t cl=ih.ext_codeplay_get_native_graph<backend::ext_oneapi_level_zero>();
            zeCommandListAppendSignalEvent(cl,sigEv);          // my push done
            ze_event_handle_t w=waitEv; zeCommandListAppendWaitOnEvents(cl,1,&w); // wait peer push
            zeCommandListAppendEventReset(cl,waitEv);          // consumer-reset for next replay
#endif
        }); });
    // local reduce (dtype-aware), depends on the sync
    size_t n=nbytes/sizeof(T);
    T *src=reinterpret_cast<T*>(inout), *scr=(T*)g_scratch;
    q->submit([&](handler &h){ h.depends_on(es);
        h.parallel_for(range<1>(n),[=](id<1> i){ src[i]=(T)(float(src[i])+float(scr[i])); }); });
}
// ---- MODE 4: pure-SYCL spin-kernel sync (no native commands -- they break sycl-graph finalize
// on DPC++ 2025.3/torch 2.12, bisect MODE 2). Design: the push kernel writes payload to peer
// scratch, then (after all payload words) writes a SEQ flag word. The reduce kernel first spins
// on the LOCAL flag reaching the expected seq (peer's push done), then reduces. Seq state lives
// in device memory (g_seq buffers at the END of scratch): replay-safe -- each executed push
// increments the seq it writes, each reduce increments its expectation, all device-side.
// Xe caveat (J.9): peer writes from a STILL-RUNNING kernel are not visible to a concurrently
// spinning kernel -- here the push KERNEL COMPLETES (kernel-boundary flush) before the peer's
// reduce spin can require it, and the spin polls with atomic_ref acquire on LOCAL device memory.
// ---- BISECT variants (2026-07-02 capture_end-hang hunt) ----
// mode 0=full (signal+wait+reset), 1=NO native cmd (push+reduce only, no cross sync -- WRONG result,
// capture-behavior probe only), 2=EMPTY native cmd (get_native_graph, append nothing), 3=signal only.
template<typename T>
static void do_ar_graph_mode(queue *q, unsigned long long inout, size_t nbytes, int mode) {
    size_t nw = nbytes/4;
    uint32_t *ws=reinterpret_cast<uint32_t*>(inout), *wd=(uint32_t*)g_peerScratch;
    event ep = q->parallel_for(range<1>(nw),[=](id<1> i){ wd[i]=ws[i]; });
    event es = ep;
    if (mode != 1) {
        ze_event_handle_t sigEv=g_sigEv, waitEv=g_waitEv;
        es = q->submit([&](handler &h){ h.depends_on(ep);
            h.ext_codeplay_enqueue_native_command([=](interop_handle ih){
#ifndef __SYCL_DEVICE_ONLY__
                ze_command_list_handle_t cl=ih.ext_codeplay_get_native_graph<backend::ext_oneapi_level_zero>();
                if (mode == 2) { (void)cl; return; }
                zeCommandListAppendSignalEvent(cl,sigEv);
                if (mode == 0) {
                    ze_event_handle_t w=waitEv; zeCommandListAppendWaitOnEvents(cl,1,&w);
                    zeCommandListAppendEventReset(cl,waitEv);
                }
#endif
            }); });
    }
    size_t n=nbytes/sizeof(T);
    T *src=reinterpret_cast<T*>(inout), *scr=(T*)g_scratch;
    q->submit([&](handler &h){ h.depends_on(es);
        h.parallel_for(range<1>(n),[=](id<1> i){ src[i]=(T)(float(src[i])+float(scr[i])); }); });
}
// device-side seq state: carve the LAST 256KB of the local scratch.
// PER-NODE layout (v2, run-24 hang fix): each RECORDED AR call gets a unique node index n (host-side
// counter at record time; identical on both ranks -- same model, same record order). Page A (offset 0):
// flags[n] -- peer's push for node n stores its per-node replay seq here. Page B (offset 128KB):
// counts[2n]=my push count for node n, counts[2n+1]=my expect count for node n. A spin only waits on
// ITS OWN node's flag, so any cross-node execution reordering (captured side streams -> parallel graph
// branches) cannot circular-wait. 16K nodes max.
// device-side seq state: carve the LAST 512KB of the local scratch = 4 pages of 32768 words each:
//   page 0 flags[n]        -- peer's PUSH signals its push-count here (my reduce waits on it)
//   page 1 counts[2n/2n+1] -- my push-count / my expect-count for node n
//   page 2 consumed[n]     -- peer's REDUCE signals its consume-count here (my push waits on it)
//   page 3 consume_ct[n]   -- my consume-count for node n
// The consumed/consume_ct pages (v3, run-26 fix) add a per-node CONSUME-ACK so a peer cannot overwrite
// my slot for generation g+1 until I have reduced generation g -- prevents the cross-replay slot clobber
// that corrupts logits when replay cadence is not lockstep (eager draft steps between verify replays).
static const long SEQ_BYTES = 524288;   // 512KB
static inline uint32_t *seq_base(void *scratch, long max_bytes) {
    return reinterpret_cast<uint32_t *>((char *)scratch + max_bytes - SEQ_BYTES);
}
static int g_node_counter = 0;
// PAYLOAD bump-pointer (run-25 garbage-logits fix): the old (node%64)*1MB scheme aliased within a
// single graph (~129 nodes > 64 slots) AND the ~11MB logits all_gather overran its 1MB slot. Instead
// each recorded node gets a distinct, exactly-sized slot via a bump cursor RESET AT capture_begin
// (ar_graph_new_capture). Payload memory is thus bounded by the LARGEST single captured graph, not the
// sum of all buckets, so MAXB stays small. node_counter (flag index) stays GLOBAL -- flag words are
// never reused across graphs (decouples sync correctness from payload memory); only the payload cursor
// resets per graph. Both TP ranks capture identical bucket order -> identical cursor progression ->
// identical slot_off per (graph, node). AR_SLOT_BASE(32MB) sits above the eager prefill push region.
static long g_slot_cursor = 0;
static const long AR_SLOT_BASE = (32L << 20);
extern "C" int ar_graph_spin_init(long max_bytes) {
    // zero the flag+count pages (called once, eager, before any capture)
    uint32_t *loc = seq_base(g_scratch, max_bytes);
    g_q->memset(loc, 0, SEQ_BYTES).wait();
    g_node_counter = 0;
    g_slot_cursor = 0;
    return 0;
}
// Called once at the START of each captured graph (hooked from _B70XPUGraph.capture_begin). Resets the
// payload bump-pointer so each graph reuses the same scratch region. RECORD-time / host-side only.
extern "C" void ar_graph_new_capture(void) { g_slot_cursor = 0; }
template<typename T>
static void do_ar_graph_spin(queue *q, unsigned long long inout, size_t nbytes, long max_bytes) {
    size_t nw = nbytes/4;
    int node = g_node_counter++;              // record-order node id, identical across ranks (flag index)
    if (node >= 16384) node = node % 16384;   // flag-slot wrap (should never happen: ~400 nodes/serve)
    // PER-NODE payload slot via bump-pointer (reset per graph at capture_begin): exactly-sized, distinct
    // within a graph, so ~129 nodes never alias and an 11MB all_gather gets its full 11MB. 4KB-aligned.
    long slot_off = AR_SLOT_BASE + g_slot_cursor;
    long slot_sz  = ((long)nbytes + 4095L) & ~4095L;
    g_slot_cursor += slot_sz;
    long flags_off = max_bytes - SEQ_BYTES;   // seq/flag region at scratch tail (must not be overrun)
    if (slot_off + (long)nbytes > flags_off) {
        fprintf(stderr, "[argraph r%d] PAYLOAD SLOT OVERFLOW node=%d off=%ld nbytes=%zu flags_off=%ld "
                "-- bump PUSH_AR_MAXB (wrapping to base; will alias)\n", g_rank, node, slot_off, nbytes, flags_off);
        slot_off = AR_SLOT_BASE; g_slot_cursor = slot_sz;
    }
    uint32_t *ws=reinterpret_cast<uint32_t*>(inout);
    uint32_t *wd=(uint32_t*)((char*)g_peerScratch + slot_off);
    uint32_t *peer_flags    = seq_base(g_peerScratch, max_bytes);
    uint32_t *my_flags      = seq_base(g_scratch,     max_bytes);
    uint32_t *my_counts     = my_flags + 32768;      // page 1: [2n]=push count, [2n+1]=expect count
    uint32_t *my_consumed   = my_flags + 65536;      // page 2: peer's reduce signals its consume-count here
    uint32_t *peer_consumed = peer_flags + 65536;    // page 2 in peer's region: my reduce signals here
    uint32_t *my_consume_ct = my_flags + 98304;      // page 3: my consume-count for node n
    // CONSUME-ACK PROLOGUE: do not overwrite peer's slot for this (g-th) push until peer has REDUCED my
    // previous (g-1)-th push. my_counts[2n] currently holds g-1 (pushes done so far); wait my_consumed>=g-1.
    event epre = q->single_task([=](){
        uint32_t need = my_counts[2*node];
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
            sycl::access::address_space::global_space> af(my_consumed[node]);
        while (af.load(sycl::memory_order::acquire) < need) { }
    });
    event ep = q->submit([&](handler &h){ h.depends_on(epre);
        h.parallel_for(range<1>(nw),[=](id<1> i){ wd[i]=ws[i]; }); });
    event ef = q->submit([&](handler &h){ h.depends_on(ep);
        h.single_task([=](){
            uint32_t s = my_counts[2*node] + 1; my_counts[2*node] = s;
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                sycl::access::address_space::global_space> af(peer_flags[node]);
            af.store(s, sycl::memory_order::release);
        }); });
    event es = q->submit([&](handler &h){ h.depends_on(ef);
        h.single_task([=](){
            uint32_t want = my_counts[2*node+1] + 1; my_counts[2*node+1] = want;
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                sycl::access::address_space::global_space> af(my_flags[node]);
            while (af.load(sycl::memory_order::acquire) < want) { }
        }); });
    size_t n=nbytes/sizeof(T);
    T *src=reinterpret_cast<T*>(inout);
    T *scr=(T*)((char*)g_scratch + slot_off);
    event ered = q->submit([&](handler &h){ h.depends_on(es);
        h.parallel_for(range<1>(n),[=](id<1> i){ src[i]=(T)(float(src[i])+float(scr[i])); }); });
    // CONSUME-ACK EPILOGUE: I have read my slot[node] for this generation -> tell peer it may reuse it.
    q->submit([&](handler &h){ h.depends_on(ered);
        h.single_task([=](){
            uint32_t c = my_consume_ct[node] + 1; my_consume_ct[node] = c;
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system,
                sycl::access::address_space::global_space> af(peer_consumed[node]);
            af.store(c, sycl::memory_order::release);
        }); });
}
extern "C" void ar_allreduce_graph_spin(unsigned long long q_addr, unsigned long long inout, long nbytes, int dtype, long max_bytes) {
    queue *q = q_addr ? reinterpret_cast<queue*>(q_addr) : g_q;
    if (dtype==1)      do_ar_graph_spin<sycl::ext::oneapi::bfloat16>(q,inout,(size_t)nbytes,max_bytes);
    else if (dtype==2) do_ar_graph_spin<sycl::half>(q,inout,(size_t)nbytes,max_bytes);
    else               do_ar_graph_spin<float>(q,inout,(size_t)nbytes,max_bytes);
}

extern "C" void ar_allreduce_graph_mode(unsigned long long q_addr, unsigned long long inout, long nbytes, int dtype, int mode) {
    queue *q = q_addr ? reinterpret_cast<queue*>(q_addr) : g_q;
    if (dtype==1)      do_ar_graph_mode<sycl::ext::oneapi::bfloat16>(q,inout,(size_t)nbytes,mode);
    else if (dtype==2) do_ar_graph_mode<sycl::half>(q,inout,(size_t)nbytes,mode);
    else               do_ar_graph_mode<float>(q,inout,(size_t)nbytes,mode);
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
}
