// 119_xpu_push_ar_eager_async.cpp -- EAGER, low-host-cost push all-reduce for the DECODE all_gather path.
//
// CONTEXT (research/profiling/pushar_decode_gap.md + eager_async_ar_plan.md):
//   In captured decode ~39% of device time is a slow oneCCL SUM all-reduce emitted INSIDE vLLM's
//   `all_gather` (631 calls/step, ~0.75 ms each). Those gathers run EAGER, between the piecewise
//   captured subgraphs, so the capturable device-event push-AR (118 `ar_allreduce_graph`, ~0.034 ms)
//   CANNOT reach them. Routing them to the host-BARRIER eager push-AR (118 `ar_allreduce_ptr_dt`:
//   push.wait() + shm-spin barrier + reduce.wait() == ~2 host waits + a busy spin per call) was
//   COHERENT but 2.4x SLOWER: 631 host round-trips/step dwarf the device saving.
//
// GOAL of this file: an eager path that does the LEAST host synchronization that is still CORRECT,
// by moving the push + the cross-card rendezvous onto our OWN Level-Zero IMMEDIATE command list
// (zeCommandListCreateImmediate on g_myze) -- NOT torch's SYCL queue -- and using the co-located
// consumer-reset (same in-order list) for the cross-card events, exactly like the proven graph path.
//
// ============================ THE HARD PART: cross-queue ordering ============================
// The all-reduce touches memory that torch produces/consumes on torch's in-order SYCL queue Q_t, but
// our push + cross-card sync live on a SEPARATE stream (our immediate list `g_imm`). So per call we
// have two cross-stream dependencies:
//   (a) g_imm's push must not read `inout` until Q_t has finished WRITING it.
//   (b) Q_t's next op must not read the reduced result until our all-reduce has finished.
//
// (a) INPUT-READY  -- solved WITHOUT a host wait: submit an in-order barrier on Q_t (fires after the
//     write, since Q_t is in-order), take its native L0 event via get_native<level_zero>(event), and
//     make g_imm's first command `zeCommandListAppendWaitOnEvents` on it. (Env ASYNC_HOSTWAIT_INPUT=1
//     falls back to a host ew.wait() -- one extra host wait, no interop -- for first-bringup safety.)
//
// (b) RESULT-READY -- this is the wall. A device-async (zero-host) handoff from g_imm back to Q_t needs
//     an event that g_imm SIGNALS and Q_t WAITS on, RECYCLED every call. Recycling it race-free requires
//     resetting it AFTER the Q_t consumer, co-located in Q_t's own in-order list -- which eager SYCL
//     interop cannot do (get_native_queue<level_zero> is broken; only capture can inject L0 onto Q_t).
//     Both the g_imm reset and the Q_t wait are released by the SAME upstream (the input barrier), with
//     NO happens-before between them, so Q_t's wait can observe the STALE prior signal (a latch that
//     waiting does not clear) and proceed early -> reads unpopulated scratch -> INCOHERENT. An event
//     RING does not fix it (the racing pair reset(c) vs reduce(c) is on the SAME index at the SAME call).
//     The monotonic-counter escape (waiter waits mem>=c, never resets) needs either L0-on-Q_t (broken)
//     or an EU spin-wait in the reduce kernel (J.9-C: HANGS on B70). So the zero-host result-ready
//     handoff is a NO-GO on this platform -- see eager_async_ar_plan.md for the full argument.
//
//     Therefore this op keeps EXACTLY ONE host synchronization per call, on the result side only:
//     `zeEventHostSynchronize(g_done)` after g_imm has done push + cross-card rendezvous. That host
//     point (i) is the correct, race-free g_imm->Q_t handoff (host owns g_done: sync then reset, no
//     concurrent waiter), and (ii) lets us then submit the REDUCE on Q_t (scratch now populated), whose
//     in-order successor covers (b) for free. The reduce itself is NOT host-waited -- it overlaps torch's
//     next work -- so the host blocks only for push + cross-card latency (~one AR device latency), i.e.
//     ~half the failed host-barrier path (which also waited the reduce and busy-spun the barrier).
//
//     Net: input-ready host wait ELIMINATED (device event), reduce host wait ELIMINATED (async on Q_t),
//     shm busy-spin ELIMINATED (cross-card via co-located L0 events). One host wait remains and is the
//     proven floor for a correct eager path. Whether that floor beats oneCCL is a MEASUREMENT question
//     (plan doc: predicted marginal); this file exists so the coordinator can measure it, wedge-guarded.
//
// WEDGE SAFETY: the dangerous state is the cross-card sig/wait latch. It is handled EXACTLY as the
// proven graph path -- consumer-reset co-located in an in-order list -- PLUS a small event RING and a
// host sync per call, which keep the two ranks in lockstep within <1 iteration so no signal is ever
// reset away (lost-signal deadlock). The only remaining hang mode is a peer that dies mid-rendezvous
// (device imm waits a never-coming signal); that is a HANG recoverable by bin/xpu-health + bin/xe-reset
// (or reboot), not a silent corruption, and identical in class to the existing push-AR paths.
//
// Build (CPU compile-check only; no GPU needed to compile):
//   icpx -fsycl -O2 -fPIC -shared 119_xpu_push_ar_eager_async.cpp -o libxpu_push_ar_eager.so -lze_loader -lrt
// C-ABI: ar_ea_setup / ar_ea_exchange / ar_allreduce_eager_async / ar_ea_teardown.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <unistd.h>
using namespace sycl;

namespace {
queue *g_q = nullptr;
context g_ctx;
device g_mydev, g_peerdev;
ze_context_handle_t g_zectx = nullptr;
ze_device_handle_t  g_myze = nullptr, g_ze_peerdev = nullptr;
int g_rank = -1;
int g_ring = 0;
size_t g_chunk = 0;
unsigned long long g_call = 0;

// scratch ring: g_scratch = ring*chunk (peer pushes into MY buffer here); g_peerScratch = peer's base.
void *g_scratch = nullptr;
void *g_peerScratch = nullptr;
ze_ipc_mem_handle_t g_myH;

// cross-card event ring (IPC pool spanning BOTH devices -- K.5). g_sig[i] signaled by me, g_wait[i] by peer.
ze_event_pool_handle_t g_xpool = nullptr;
std::vector<ze_event_handle_t> g_sig, g_wait;

// result-ready event (local, host-visible): g_imm signals it, host synchronizes+resets it each call.
ze_event_pool_handle_t g_dpool = nullptr;
ze_event_handle_t g_done = nullptr;

// our immediate command list (async) on MY device -- the separate stream that does push + rendezvous.
ze_command_list_handle_t g_imm = nullptr;

// keep-alive ring of the Q_t input barriers so SYCL does not recycle their L0 events under our g_imm wait.
std::vector<event> g_ew;

bool g_hostwait_input = false;  // ASYNC_HOSTWAIT_INPUT=1: host ew.wait() for (a) instead of get_native.

void log(const char *m){ fprintf(stderr,"[areager r%d] %s\n",g_rank,m); fflush(stderr); }

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
} // namespace

// ar_ea_setup: bind to torch's L0 context/queue, alloc the scratch RING, create the immediate list,
// the local result-ready event, and the keep-alive rings. chunk_bytes = max tensor bytes accepted.
extern "C" int ar_ea_setup(int rank, unsigned long long torch_q_addr, int ring, long chunk_bytes) {
    g_rank = rank;
    g_ring = ring < 1 ? 1 : ring;
    g_chunk = (size_t)chunk_bytes;
    g_hostwait_input = (getenv("ASYNC_HOSTWAIT_INPUT") && !strcmp(getenv("ASYNC_HOSTWAIT_INPUT"),"1"));
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

    // scratch ring (contiguous), one IPC handle
    ze_device_mem_alloc_desc_t md = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, NULL, 0, 0 };
    ze_result_t r = zeMemAllocDevice(g_zectx, &md, g_chunk*(size_t)g_ring, 4096, g_myze, &g_scratch);
    if (r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[areager r%d] zeMemAllocDevice 0x%x\n",rank,r); return 2; }
    r = zeMemGetIpcHandle(g_zectx, g_scratch, &g_myH);
    if (r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[areager r%d] zeMemGetIpcHandle 0x%x\n",rank,r); return 3; }

    // our async immediate command list on the default compute engine (ordinal 0)
    ze_command_queue_desc_t cqd = { ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, 0, 0, 0,
                                    ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL };
    r = zeCommandListCreateImmediate(g_zectx, g_myze, &cqd, &g_imm);
    if (r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[areager r%d] CreateImmediate 0x%x\n",rank,r); return 4; }

    // local host-visible result-ready event (host synchronizes + resets it each call)
    ze_event_pool_desc_t dpd = { ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, NULL, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1 };
    r = zeEventPoolCreate(g_zectx, &dpd, 1, &g_myze, &g_dpool);
    if (r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[areager r%d] done pool 0x%x\n",rank,r); return 5; }
    ze_event_desc_t ded = { ZE_STRUCTURE_TYPE_EVENT_DESC, NULL, 0, ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST };
    if (zeEventCreate(g_dpool,&ded,&g_done)!=ZE_RESULT_SUCCESS) return 6;
    zeEventHostReset(g_done);

    g_ew.assign(g_ring, event{});
    g_sig.assign(g_ring, nullptr);
    g_wait.assign(g_ring, nullptr);
    log("ar_ea_setup OK");
    return 0;
}

// ar_ea_exchange: socket-exchange the scratch IPC handle and an IPC EVENT POOL (2*ring events, spanning
// BOTH devices -- K.5), then create the ringed cross-card sig/wait events. Uses a distinct socket path
// so it does not collide with 118's ar_exchange.
extern "C" int ar_ea_exchange(int rank, const char *sockpath) {
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
    // scratch handle exchange
    ze_ipc_mem_handle_t peerH;
    if(rank==0){ if(send_blob(sock,g_myH.data,sizeof(g_myH.data)))return 12; if(recv_blob(sock,peerH.data,sizeof(peerH.data)))return 13; }
    else       { if(recv_blob(sock,peerH.data,sizeof(peerH.data)))return 13; if(send_blob(sock,g_myH.data,sizeof(g_myH.data)))return 12; }
    ze_result_t r=zeMemOpenIpcHandle(g_zectx,g_ze_peerdev,peerH,0,&g_peerScratch);
    if(r!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[areager r%d] OpenIpcHandle 0x%x\n",rank,r); return 14; }

    // IPC EVENT POOL: 2*ring events, MUST span both devices (K.5). rank0 creates + sends; rank1 opens.
    ze_device_handle_t both[2]={g_myze,g_ze_peerdev};
    uint32_t cap = (uint32_t)(2*g_ring);
    if(rank==0){
        ze_event_pool_desc_t epd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,NULL,
            ZE_EVENT_POOL_FLAG_IPC|ZE_EVENT_POOL_FLAG_HOST_VISIBLE,cap};
        r=zeEventPoolCreate(g_zectx,&epd,2,both,&g_xpool);
        if(r!=ZE_RESULT_SUCCESS){fprintf(stderr,"[areager r%d] xpool create 0x%x\n",rank,r);return 20;}
        ze_ipc_event_pool_handle_t iph; if(zeEventPoolGetIpcHandle(g_xpool,&iph)!=ZE_RESULT_SUCCESS)return 21;
        if(send_blob(sock,iph.data,sizeof(iph.data)))return 22;
    } else {
        ze_ipc_event_pool_handle_t iph; memset(&iph,0,sizeof(iph));
        if(recv_blob(sock,iph.data,sizeof(iph.data)))return 22;
        r=zeEventPoolOpenIpcHandle(g_zectx,iph,&g_xpool);
        if(r!=ZE_RESULT_SUCCESS){fprintf(stderr,"[areager r%d] xpool open 0x%x\n",rank,r);return 23;}
    }
    // Per ring slot i, index 2*i is "rank0's signal / rank1's wait", 2*i+1 the reverse. Both ranks
    // create handles for BOTH indices on the shared pool -> same physical events across processes.
    for(int i=0;i<g_ring;i++){
        ze_event_desc_t d0={ZE_STRUCTURE_TYPE_EVENT_DESC,NULL,(uint32_t)(2*i),  ZE_EVENT_SCOPE_FLAG_HOST,ZE_EVENT_SCOPE_FLAG_HOST};
        ze_event_desc_t d1={ZE_STRUCTURE_TYPE_EVENT_DESC,NULL,(uint32_t)(2*i+1),ZE_EVENT_SCOPE_FLAG_HOST,ZE_EVENT_SCOPE_FLAG_HOST};
        ze_event_handle_t e0=nullptr,e1=nullptr;
        if(zeEventCreate(g_xpool,&d0,&e0)!=ZE_RESULT_SUCCESS)return 24;
        if(zeEventCreate(g_xpool,&d1,&e1)!=ZE_RESULT_SUCCESS)return 25;
        zeEventHostReset(e0); zeEventHostReset(e1);
        g_sig[i]  = (rank==0)?e0:e1;   // I signal this
        g_wait[i] = (rank==0)?e1:e0;   // peer signals this; I wait + reset it (consumer-reset)
    }
    close(sock);
    log("ar_ea_exchange OK (scratch ring + IPC event ring)");
    return 0;
}

// One eager all-reduce. push + cross-card rendezvous on g_imm (in-order, consumer-reset events); ONE
// host sync for the result-ready handoff; reduce async on torch's queue (in-order covers the consumer).
template<typename T>
static void do_ea(queue *q, unsigned long long inout, size_t nbytes) {
    unsigned long long c = g_call++;
    int i = (int)(c % (unsigned long long)g_ring);
    void *myScr   = (char*)g_scratch     + (size_t)i*g_chunk;   // peer pushed peer-data here (this call)
    void *peerScr = (char*)g_peerScratch + (size_t)i*g_chunk;   // I push my snapshot here

    // (a) INPUT-READY: an in-order barrier on Q_t fires after torch finished writing `inout`.
    event ew = q->ext_oneapi_submit_barrier();
    g_ew[i] = ew;  // keep alive (evicts slot i's barrier from `ring` calls ago, long consumed)
    if (g_hostwait_input) {
        ew.wait();  // safe-mode: one extra host wait, no interop
    }

    // g_imm: [wait torch-write] push(inout->peerScr) [signal mine] [wait peer] [reset peer] [signal done]
    if (!g_hostwait_input) {
        ze_event_handle_t zew = get_native<backend::ext_oneapi_level_zero>(ew);
        zeCommandListAppendWaitOnEvents(g_imm, 1, &zew);
    }
    zeCommandListAppendMemoryCopy(g_imm, peerScr, (void*)inout, nbytes, nullptr, 0, nullptr);
    zeCommandListAppendSignalEvent(g_imm, g_sig[i]);
    ze_event_handle_t w = g_wait[i];
    zeCommandListAppendWaitOnEvents(g_imm, 1, &w);
    zeCommandListAppendEventReset(g_imm, g_wait[i]);   // consumer-reset, co-located in this in-order list
    zeCommandListAppendSignalEvent(g_imm, g_done);

    // (b) RESULT-READY: the single, correct g_imm->Q_t handoff. Host owns g_done: sync then reset, no
    // concurrent waiter -> no stale-signal race. Blocks host only for push + cross-card latency.
    zeEventHostSynchronize(g_done, UINT64_MAX);
    zeEventHostReset(g_done);

    // reduce on torch's queue: scratch is now populated; this is ASYNC (not host-waited) and torch's
    // in-order successor cannot read `inout` until it completes -> covers (b).
    size_t n = nbytes/sizeof(T);
    T *src=reinterpret_cast<T*>(inout), *scr=reinterpret_cast<T*>(myScr);
    q->submit([&](handler &h){
        h.parallel_for(range<1>(n),[=](id<1> k){ src[k]=(T)(float(src[k])+float(scr[k])); });
    });
}

extern "C" void ar_allreduce_eager_async(unsigned long long q_addr, unsigned long long inout,
                                          long nbytes, int dtype) {
    queue *q = q_addr ? reinterpret_cast<queue*>(q_addr) : g_q;
    if (dtype==1)      do_ea<sycl::ext::oneapi::bfloat16>(q, inout, (size_t)nbytes);
    else if (dtype==2) do_ea<sycl::half>(q, inout, (size_t)nbytes);
    else               do_ea<float>(q, inout, (size_t)nbytes);
}

extern "C" void ar_ea_teardown(void){
    for (auto &e : g_sig)  if (e) zeEventDestroy(e);
    for (auto &e : g_wait) if (e) zeEventDestroy(e);
    if (g_done)  zeEventDestroy(g_done);
    if (g_dpool) zeEventPoolDestroy(g_dpool);
    if (g_xpool) zeEventPoolDestroy(g_xpool);
    if (g_imm)   zeCommandListDestroy(g_imm);
    if (g_peerScratch) zeMemCloseIpcHandle(g_zectx, g_peerScratch);
    if (g_scratch)     zeMemFree(g_zectx, g_scratch);
}
