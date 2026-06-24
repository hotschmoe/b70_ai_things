// 116_graph_native_ar.cpp -- the INTEGRATION-SHAPED proof: a per-rank SYCL command_graph push all-reduce whose
// cross-device sync is an L0 event signal/wait/reset injected via ext_codeplay_enqueue_native_command.
//
// WHY THIS SHAPE (P2P_GPU.md K.2/K.3): torch-xpu's XPUGraph IS sycl::...::command_graph (confirmed in
// ATen/xpu/XPUGraph.h: xpuGraph_t = command_graph<modifiable>). So vLLM GRAPH=1 capture = SYCL command-graph
// queue-recording on the capture stream, and ANYTHING we submit on torch's stream during capture becomes a
// graph node. K.3 proved the cross-device command-streamer L0-event wait is correct + replayable. K.2 proved
// a graph is single-device, so each rank captures its OWN graph and the cross-rank sync must be EXTERNAL.
// This bench unifies them: each rank records a command_graph [push kernel] -> [native cmd: signal my event,
// wait peer event, RESET the consumed event] -> [reduce kernel], finalizes, and REPLAYS it. If correct across
// replays, the SAME construction injected into torch's captured graph makes the DECODE all-reduce capturable.
//
// Single process / one L0 context / 2 queues (proxy for the 2 TP workers; cross-process IPC events = 117).
// Consumer-reset: the rank that WAITS on an event RESETS it right after (so next token's signal is clean).
// In the real serve the ~64 per-token all-reduces couple the ranks to <1-allreduce drift, so the reset is
// race-free; here a per-iteration sync keeps the same lockstep (free-running mode also reported as a stress).
//
// Build: icpx -fsycl -O2 116_graph_native_ar.cpp -o graph_native_ar -lze_loader
// Run  : ZE_AFFINITY_MASK=0,1 ./graph_native_ar   (under gpu-run, timeout-wrapped)
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <level_zero/ze_api.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <cmath>
using namespace sycl;
namespace sgr = sycl::ext::oneapi::experimental;

#define ZK(call) do { ze_result_t _r=(call); if(_r!=ZE_RESULT_SUCCESS){ \
    fprintf(stderr,"ZE FAIL %s -> 0x%x @ %d\n",#call,_r,__LINE__); exit(2);} } while(0)

static double sec(){ return std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }

int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    std::vector<device> gpus;
    for(auto&p:platform::get_platforms())
        for(auto&d:p.get_devices(info::device_type::gpu))
            if(d.get_backend()==backend::ext_oneapi_level_zero) gpus.push_back(d);
    if(gpus.size()<2){ printf("need >=2 (ZE_AFFINITY_MASK=0,1)\n"); return 1; }
    device d0=gpus[0], d1=gpus[1];
    context ctx({d0,d1});
    queue q0(ctx,d0), q1(ctx,d1);
    if(d0.ext_oneapi_can_access_peer(d1)) d0.ext_oneapi_enable_peer_access(d1);
    if(d1.ext_oneapi_can_access_peer(d0)) d1.ext_oneapi_enable_peer_access(d0);
    printf("graph+native-cmd all-reduce: per-rank command_graph, L0-event cross-device sync.\n");

    // L0 event pool (HOST_VISIBLE) + 2 events: S_A signaled by rank0's push, S_B by rank1's.
    auto zectx = get_native<backend::ext_oneapi_level_zero>(ctx);
    ze_device_handle_t zd0 = get_native<backend::ext_oneapi_level_zero>(d0);
    ze_device_handle_t zd1 = get_native<backend::ext_oneapi_level_zero>(d1);
    ze_device_handle_t devs[2]={zd0,zd1};
    ze_event_pool_desc_t epd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,NULL,ZE_EVENT_POOL_FLAG_HOST_VISIBLE,2};
    ze_event_pool_handle_t pool; ZK(zeEventPoolCreate(zectx,&epd,2,devs,&pool));
    ze_event_desc_t eA={ZE_STRUCTURE_TYPE_EVENT_DESC,NULL,0,ZE_EVENT_SCOPE_FLAG_HOST,ZE_EVENT_SCOPE_FLAG_HOST};
    ze_event_desc_t eB={ZE_STRUCTURE_TYPE_EVENT_DESC,NULL,1,ZE_EVENT_SCOPE_FLAG_HOST,ZE_EVENT_SCOPE_FLAG_HOST};
    ze_event_handle_t S_A,S_B; ZK(zeEventCreate(pool,&eA,&S_A)); ZK(zeEventCreate(pool,&eB,&S_B));

    size_t sizes[]   ={ 10240, 65536, 1u<<20 };
    const char* lab[]={"10KB(decode)","64KB","1MB"};
    int nsz=sizeof(sizes)/sizeof(sizes[0]);

    size_t maxn=(1u<<20)/sizeof(float);
    float *bufA=malloc_device<float>(maxn,q0), *scrA=malloc_device<float>(maxn,q0);
    float *bufB=malloc_device<float>(maxn,q1), *scrB=malloc_device<float>(maxn,q1);
    auto reset=[&](size_t n,float va,float vb){ q0.fill(bufA,va,n); q1.fill(bufB,vb,n);
                                                q0.fill(scrA,-1.f,n); q1.fill(scrB,-1.f,n);
                                                q0.wait(); q1.wait(); };

    printf("%-15s %12s %12s %12s %10s %10s\n",
           "size","perLaunch us","amort us","algbw GB/s","verifyA","verifyB");
    for(int s=0;s<nsz;s++){
        size_t n=sizes[s]/sizeof(float);

        // --- record rank0's graph on d0 ---
        sgr::command_graph g0(ctx,d0);
        g0.begin_recording(q0);
        event p0 = q0.parallel_for(range<1>(n),[=](id<1> i){ scrB[i]=bufA[i]; });     // push d0->scrB(d1)
        event s0 = q0.submit([&](handler&h){ h.depends_on(p0);
            h.ext_codeplay_enqueue_native_command([=](interop_handle ih){
#ifndef __SYCL_DEVICE_ONLY__
                ze_command_list_handle_t cl=ih.ext_codeplay_get_native_graph<backend::ext_oneapi_level_zero>();
                zeCommandListAppendSignalEvent(cl,S_A);          // my push done
                ze_event_handle_t w=S_B; zeCommandListAppendWaitOnEvents(cl,1,&w);      // wait peer push
                zeCommandListAppendEventReset(cl,S_B);           // consumer-reset what I waited on
#endif
            }); });
        q0.submit([&](handler&h){ h.depends_on(s0);
            h.parallel_for(range<1>(n),[=](id<1> i){ bufA[i]+=scrA[i]; }); });        // local reduce
        g0.end_recording();
        auto x0=g0.finalize();

        // --- record rank1's graph on d1 (symmetric) ---
        sgr::command_graph g1(ctx,d1);
        g1.begin_recording(q1);
        event p1 = q1.parallel_for(range<1>(n),[=](id<1> i){ scrA[i]=bufB[i]; });     // push d1->scrA(d0)
        event s1 = q1.submit([&](handler&h){ h.depends_on(p1);
            h.ext_codeplay_enqueue_native_command([=](interop_handle ih){
#ifndef __SYCL_DEVICE_ONLY__
                ze_command_list_handle_t cl=ih.ext_codeplay_get_native_graph<backend::ext_oneapi_level_zero>();
                zeCommandListAppendSignalEvent(cl,S_B);
                ze_event_handle_t w=S_A; zeCommandListAppendWaitOnEvents(cl,1,&w);
                zeCommandListAppendEventReset(cl,S_A);
#endif
            }); });
        q1.submit([&](handler&h){ h.depends_on(s1);
            h.parallel_for(range<1>(n),[=](id<1> i){ bufB[i]+=scrB[i]; }); });
        g1.end_recording();
        auto x1=g1.finalize();

        // start events cleared
        ZK(zeEventHostReset(S_A)); ZK(zeEventHostReset(S_B));

        // --- correctness across 200 replays, per-iter lockstep sync, fresh seq value each iter ---
        int badA=0,badB=0;
        for(int it=0; it<200; it++){
            float va=(float)(it%97), vb=(float)(it%89+1000);
            reset(n,va,vb);
            q0.ext_oneapi_graph(x0);
            q1.ext_oneapi_graph(x1);
            q0.wait(); q1.wait();
            float hA,hB;
            q0.memcpy(&hA,bufA,sizeof(float)).wait();   // expect va+vb
            q1.memcpy(&hB,bufB,sizeof(float)).wait();
            if(std::fabs(hA-(va+vb))>1e-3) badA++;
            if(std::fabs(hB-(va+vb))>1e-3) badB++;
        }

        // --- timing (a) per-launch lockstep (one decode token = one graph launch each rank) ---
        int iters = sizes[s]<=(1u<<20)?300:60;
        for(int i=0;i<10;i++){ q0.ext_oneapi_graph(x0); q1.ext_oneapi_graph(x1); q0.wait(); q1.wait(); }
        double t0=sec();
        for(int i=0;i<iters;i++){ q0.ext_oneapi_graph(x0); q1.ext_oneapi_graph(x1); q0.wait(); q1.wait(); }
        double perLaunch=(sec()-t0)/iters;

        // --- timing (b) amortized: back-to-back replays, sync once (allreduce as 1 node in a big graph) ---
        double t1=sec(); event la,lb;
        for(int i=0;i<iters;i++){ la=q0.ext_oneapi_graph(x0); lb=q1.ext_oneapi_graph(x1); }
        la.wait(); lb.wait();
        double amort=(sec()-t1)/iters;

        printf("%-15s %12.2f %12.2f %12.2f %10s %10s\n", lab[s], perLaunch*1e6, amort*1e6,
               sizes[s]/amort/1e9, badA?"BAD":"OK(sum)", badB?"BAD":"OK(sum)");
        if(badA||badB) printf("   [!] badA=%d badB=%d / 200\n",badA,badB);
    }

    free(bufA,q0); free(scrA,q0); free(bufB,q1); free(scrB,q1);
    printf("DONE_GRAPH_NATIVE_AR\n");
    return 0;
}
