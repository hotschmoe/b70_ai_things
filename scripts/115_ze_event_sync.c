// 115_ze_event_sync.c -- KEYSTONE: is a cross-device command-streamer event wait correct + REPLAYABLE on B70?
//
// The decode-capture blocker (P2P_GPU.md J.9, K.2): push-ar's rank sync is a HOST barrier -> not graph
// recordable. J.9-C proved an EU-spin device flag HANGS (mid-kernel peer write invisible on Xe, peer
// ATOMICS=N). BUT there is a DIFFERENT cross-device sync path that J.9 never tried and that oneCCL's own
// capturable sycl_algorithms allreduce relies on: a Level-Zero EVENT signaled by one device's command and
// waited on by the OTHER device's COMMAND STREAMER (zeCommandListAppendWaitOnEvents) -- a hardware semaphore
// wait, not an EU spin. And a CLOSED L0 command list is recorded once and re-executed = exactly "replay".
//
// THIS BENCH: two closed command lists (one per card), each does push(peer memcpy)+signal, then
// waitOnEvents(peer), then a proxy "reduce" read of what the peer pushed. Re-executed N times with a fresh
// per-iteration SEQUENCE value so a MISSED sync (reduce ran before peer push landed) is caught as a stale
// read. If all N iters verify, the cross-device command-streamer wait is correct AND replayable -> the
// decode all-reduce can be made graph-capturable (the J.9-C dead end is bypassed). Pure L0, no SPIR-V.
//
// Build: gcc 115_ze_event_sync.c -o ze_event_sync -lze_loader
// Run  : ZE_AFFINITY_MASK=0,1 ./ze_event_sync   (under gpu-run, timeout-wrapped)
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(call) do { ze_result_t _r = (call); if (_r != ZE_RESULT_SUCCESS) { \
    fprintf(stderr, "FAIL %s -> 0x%x @ %s:%d\n", #call, _r, __FILE__, __LINE__); exit(2);} } while(0)

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; }

// compute-capable queue-group ordinal (memory copy can run on it); fall back to ordinal 0.
static uint32_t compute_ordinal(ze_device_handle_t dev){
    uint32_t n=0; CK(zeDeviceGetCommandQueueGroupProperties(dev,&n,NULL));
    ze_command_queue_group_properties_t *g=calloc(n,sizeof(*g));
    for(uint32_t i=0;i<n;i++) g[i].stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    CK(zeDeviceGetCommandQueueGroupProperties(dev,&n,g));
    uint32_t ord=0;
    for(uint32_t i=0;i<n;i++) if(g[i].flags&ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE){ord=i;break;}
    free(g); return ord;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    CK(zeInit(0));
    uint32_t nd=0; CK(zeDriverGet(&nd,NULL));
    ze_driver_handle_t *drv=calloc(nd,sizeof(*drv)); CK(zeDriverGet(&nd,drv));
    ze_driver_handle_t driver=drv[0];
    uint32_t ndev=0; CK(zeDeviceGet(driver,&ndev,NULL));
    if(ndev<2){fprintf(stderr,"need >=2 devices (ZE_AFFINITY_MASK=0,1)\n");return 1;}
    ze_device_handle_t *dev=calloc(ndev,sizeof(*dev)); CK(zeDeviceGet(driver,&ndev,dev));
    ze_device_handle_t d0=dev[0], d1=dev[1];
    ze_bool_t c01=0,c10=0; CK(zeDeviceCanAccessPeer(d0,d1,&c01)); CK(zeDeviceCanAccessPeer(d1,d0,&c10));
    printf("devices=%u canAccessPeer d0->d1=%d d1->d0=%d\n",ndev,c01,c10);

    ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC,NULL,0};
    ze_context_handle_t ctx; CK(zeContextCreate(driver,&cd,&ctx));
    uint32_t ord0=compute_ordinal(d0), ord1=compute_ordinal(d1);
    printf("compute ordinals d0=%u d1=%u\n",ord0,ord1);

    // event pool: HOST_VISIBLE so the event state lives where both devices' command streamers can poll it
    // across PCIe. 2 events: E0 signaled by d0's push, E1 by d1's push.
    ze_event_pool_desc_t epd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,NULL,ZE_EVENT_POOL_FLAG_HOST_VISIBLE,2};
    ze_event_pool_handle_t pool;
    CK(zeEventPoolCreate(ctx,&epd,ndev,dev,&pool)); // both devices may signal/wait
    ze_event_desc_t ed0={ZE_STRUCTURE_TYPE_EVENT_DESC,NULL,0,
        ZE_EVENT_SCOPE_FLAG_HOST,ZE_EVENT_SCOPE_FLAG_HOST};   // index 0 = E0
    ze_event_desc_t ed1={ZE_STRUCTURE_TYPE_EVENT_DESC,NULL,1,
        ZE_EVENT_SCOPE_FLAG_HOST,ZE_EVENT_SCOPE_FLAG_HOST};   // index 1 = E1
    ze_event_handle_t E0,E1; CK(zeEventCreate(pool,&ed0,&E0)); CK(zeEventCreate(pool,&ed1,&E1));

    size_t sizes[]={ 10240, 65536, 1u<<20 };
    const char *lab[]={"10KB(decode)","64KB","1MB"};
    int nsz=sizeof(sizes)/sizeof(sizes[0]);
    int ITER=200;

    ze_device_mem_alloc_desc_t md={ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,NULL,0,0};
    ze_host_mem_alloc_desc_t hd={ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,NULL,0};
    size_t maxb=1u<<20;
    void *bufA,*scrA,*outA,*bufB,*scrB,*outB,*hChk;
    CK(zeMemAllocDevice(ctx,&md,maxb,4096,d0,&bufA));
    CK(zeMemAllocDevice(ctx,&md,maxb,4096,d0,&scrA)); // d1 pushes here
    CK(zeMemAllocDevice(ctx,&md,maxb,4096,d0,&outA));
    CK(zeMemAllocDevice(ctx,&md,maxb,4096,d1,&bufB));
    CK(zeMemAllocDevice(ctx,&md,maxb,4096,d1,&scrB)); // d0 pushes here
    CK(zeMemAllocDevice(ctx,&md,maxb,4096,d1,&outB));
    CK(zeMemAllocHost(ctx,&hd,maxb,4096,&hChk));

    // immediate (synchronous) lists used only to set up per-iteration input values
    ze_command_queue_desc_t iq0={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,NULL,ord0,0,0,
        ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_queue_desc_t iq1={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,NULL,ord1,0,0,
        ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t im0,im1;
    CK(zeCommandListCreateImmediate(ctx,d0,&iq0,&im0));
    CK(zeCommandListCreateImmediate(ctx,d1,&iq1,&im1));

    printf("%-15s %8s %12s %12s %12s\n","size","iters","sync_us","verifyA","verifyB");
    for(int s=0;s<nsz;s++){
        size_t bytes=sizes[s]; size_t n=bytes/sizeof(float);

        // async queues + CLOSED (replayable) command lists, one per card
        ze_command_queue_desc_t qd0={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,NULL,ord0,0,0,
            ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
        ze_command_queue_desc_t qd1={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,NULL,ord1,0,0,
            ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
        ze_command_list_desc_t ld0={ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,NULL,ord0,0};
        ze_command_list_desc_t ld1={ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,NULL,ord1,0};
        ze_command_queue_handle_t q0,q1; ze_command_list_handle_t cl0,cl1;
        CK(zeCommandQueueCreate(ctx,d0,&qd0,&q0)); CK(zeCommandQueueCreate(ctx,d1,&qd1,&q1));
        CK(zeCommandListCreate(ctx,d0,&ld0,&cl0)); CK(zeCommandListCreate(ctx,d1,&ld1,&cl1));

        // CL0 (d0): push bufA -> scrB(on d1), SIGNAL E0; WAIT E1 (d1's push); read scrA -> outA.
        CK(zeCommandListAppendMemoryCopy(cl0,scrB,bufA,bytes,E0,0,NULL));
        CK(zeCommandListAppendWaitOnEvents(cl0,1,&E1));
        CK(zeCommandListAppendMemoryCopy(cl0,outA,scrA,bytes,NULL,0,NULL));
        CK(zeCommandListClose(cl0));
        // CL1 (d1): push bufB -> scrA(on d0), SIGNAL E1; WAIT E0; read scrB -> outB.
        CK(zeCommandListAppendMemoryCopy(cl1,scrA,bufB,bytes,E1,0,NULL));
        CK(zeCommandListAppendWaitOnEvents(cl1,1,&E0));
        CK(zeCommandListAppendMemoryCopy(cl1,outB,scrB,bytes,NULL,0,NULL));
        CK(zeCommandListClose(cl1));

        int badA=0,badB=0; double acc=0;
        for(int it=0; it<ITER; it++){
            // fresh per-iteration sentinel: bufA = it, bufB = it+1000 (every element)
            float va=(float)it, vb=(float)(it+1000);
            // fill via host-visible staging + copy (sync immediate lists)
            for(size_t k=0;k<n;k++) ((float*)hChk)[k]=va;
            CK(zeCommandListAppendMemoryCopy(im0,bufA,hChk,bytes,NULL,0,NULL));
            for(size_t k=0;k<n;k++) ((float*)hChk)[k]=vb;
            CK(zeCommandListAppendMemoryCopy(im1,bufB,hChk,bytes,NULL,0,NULL));
            // also poison the scratch + outputs so a stale/missed read is visible as the poison or old seq
            float poison=-1.0f;
            for(size_t k=0;k<n;k++) ((float*)hChk)[k]=poison;
            CK(zeCommandListAppendMemoryCopy(im0,scrA,hChk,bytes,NULL,0,NULL));
            CK(zeCommandListAppendMemoryCopy(im0,outA,hChk,bytes,NULL,0,NULL));
            CK(zeCommandListAppendMemoryCopy(im1,scrB,hChk,bytes,NULL,0,NULL));
            CK(zeCommandListAppendMemoryCopy(im1,outB,hChk,bytes,NULL,0,NULL));

            CK(zeEventHostReset(E0)); CK(zeEventHostReset(E1));

            double t0=now_s();
            // launch both cards' closed lists concurrently, then wait both
            CK(zeCommandQueueExecuteCommandLists(q0,1,&cl0,NULL));
            CK(zeCommandQueueExecuteCommandLists(q1,1,&cl1,NULL));
            CK(zeCommandQueueSynchronize(q0,UINT64_MAX));
            CK(zeCommandQueueSynchronize(q1,UINT64_MAX));
            if(it>=5) acc += now_s()-t0;

            // verify: outA must == vb (what d1 pushed into scrA); outB == va.
            CK(zeCommandListAppendMemoryCopy(im0,hChk,outA,sizeof(float),NULL,0,NULL));
            if(((float*)hChk)[0]!=vb) badA++;
            CK(zeCommandListAppendMemoryCopy(im1,hChk,outB,sizeof(float),NULL,0,NULL));
            if(((float*)hChk)[0]!=va) badB++;
        }
        double per=acc/(ITER-5);
        char vA[24],vB[24];
        snprintf(vA,sizeof vA, badA?"BAD(%d/%d)":"OK", badA, ITER);
        snprintf(vB,sizeof vB, badB?"BAD(%d/%d)":"OK", badB, ITER);
        printf("%-15s %8d %12.2f %12s %12s\n",lab[s],ITER,per*1e6,vA,vB);

        zeCommandListDestroy(cl0); zeCommandListDestroy(cl1);
        zeCommandQueueDestroy(q0); zeCommandQueueDestroy(q1);
    }

    printf("DONE_EVENT_SYNC\n");
    return 0;
}
