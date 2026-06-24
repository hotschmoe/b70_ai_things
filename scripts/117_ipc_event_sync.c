// 117_ipc_event_sync.c -- CROSS-PROCESS keystone: IPC event pool + cross-device command-streamer wait,
// replayed, in the real vLLM 2-worker topology. K.3 (115) proved the cross-device L0-event wait single-ctx;
// K.4 (116) proved it records into a SYCL graph + replays. The remaining plumbing for the 2 TP workers is
// sharing the EVENT POOL across processes (zeEventPoolGetIpcHandle / zeEventPoolOpenIpcHandle), since each
// worker is a separate process with its own L0 context (scratch IPC already proven J.8/J.10).
//
// Design (mirrors 103 for scratch IPC + adds IPC event pool): fork 2 ranks, rank r on device[r], own context.
//   - rank0 creates an IPC|HOST_VISIBLE event pool (2 events), exports its IPC handle; rank1 opens it.
//   - both create S_A(idx0), S_B(idx1) from the shared pool -> same underlying event slots cross-process.
//   - scratch IPC exchange (SCM_RIGHTS) as in 103.
//   - per rank a CLOSED (replayable) command list: push(bufmine -> peerScratch) SIGNAL S_mine;
//     WaitOnEvents(S_peer); read my scratch -> out; EventReset(S_peer) [consumer-reset].
//   - replay 200x with a per-iter socket barrier + fresh seq value; verify the peer's pushed value landed
//     AFTER the wait (a missed cross-process sync -> stale/poison -> caught).
// If correct, the cross-device sync works cross-process and is replayable -> ready to wire into the .so/serve.
//
// Build: gcc 117_ipc_event_sync.c -o ipc_event_sync -lze_loader
// Run under gpu-run (both cards). See 117_run_ipc_event_sync.sh.
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>

#define CK(call) do { ze_result_t _r=(call); if(_r!=ZE_RESULT_SUCCESS){ \
    fprintf(stderr,"[r%d] FAIL %s -> 0x%x @ %d\n",g_rank,#call,_r,__LINE__); _exit(2);} } while(0)
static int g_rank=-1;
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9; }

// SCM_RIGHTS fd passing for a generic IPC blob (mem handle OR event-pool handle: xe embeds the fd at data[0]).
static void send_blob(int sock, const char *data, size_t n){
    int fd=*(const int*)data; struct iovec io={(void*)data,n};
    char cb[CMSG_SPACE(sizeof(int))]; memset(cb,0,sizeof(cb));
    struct msghdr m={0}; m.msg_iov=&io; m.msg_iovlen=1; m.msg_control=cb; m.msg_controllen=sizeof(cb);
    struct cmsghdr *c=CMSG_FIRSTHDR(&m); c->cmsg_level=SOL_SOCKET; c->cmsg_type=SCM_RIGHTS;
    c->cmsg_len=CMSG_LEN(sizeof(int)); memcpy(CMSG_DATA(c),&fd,sizeof(int));
    if(sendmsg(sock,&m,0)<0){perror("sendmsg");_exit(3);}
}
static void recv_blob(int sock, char *data, size_t n){
    struct iovec io={data,n}; char cb[CMSG_SPACE(sizeof(int))]; memset(cb,0,sizeof(cb));
    struct msghdr m={0}; m.msg_iov=&io; m.msg_iovlen=1; m.msg_control=cb; m.msg_controllen=sizeof(cb);
    if(recvmsg(sock,&m,0)<0){perror("recvmsg");_exit(3);}
    struct cmsghdr *c=CMSG_FIRSTHDR(&m); int fd; memcpy(&fd,CMSG_DATA(c),sizeof(int));
    *(int*)data=fd;
}
static void barrier(int sock){ char x=1; if(write(sock,&x,1)!=1)_exit(4); char y; if(read(sock,&y,1)!=1)_exit(4); }

static void run_rank(int rank,int sock){
    g_rank=rank; int pr=1-rank; setvbuf(stdout,NULL,_IONBF,0);
    CK(zeInit(0));
    uint32_t nd=0; CK(zeDriverGet(&nd,NULL));
    ze_driver_handle_t *drv=calloc(nd,sizeof(*drv)); CK(zeDriverGet(&nd,drv));
    ze_driver_handle_t driver=drv[0];
    uint32_t ndev=0; CK(zeDeviceGet(driver,&ndev,NULL));
    if(ndev<2){fprintf(stderr,"[r%d] need >=2 devices\n",rank);_exit(1);}
    ze_device_handle_t *dev=calloc(ndev,sizeof(*dev)); CK(zeDeviceGet(driver,&ndev,dev));
    ze_device_handle_t mydev=dev[rank], peerdev=dev[pr];
    ze_context_desc_t cd={ZE_STRUCTURE_TYPE_CONTEXT_DESC,NULL,0};
    ze_context_handle_t ctx; CK(zeContextCreate(driver,&cd,&ctx));
    uint32_t ord=0;

    // ---- shared IPC event pool: rank0 creates+exports, rank1 opens ----
    ze_event_pool_handle_t pool;
    ze_device_handle_t both[2]={dev[0],dev[1]}; // pool must span BOTH devices so either card's
                                                // command streamer can wait on its events (else crash)
    if(rank==0){
        ze_event_pool_desc_t epd={ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,NULL,
            ZE_EVENT_POOL_FLAG_IPC|ZE_EVENT_POOL_FLAG_HOST_VISIBLE,2};
        CK(zeEventPoolCreate(ctx,&epd,2,both,&pool));
        ze_ipc_event_pool_handle_t iph; CK(zeEventPoolGetIpcHandle(pool,&iph));
        send_blob(sock,iph.data,sizeof(iph.data));
        printf("[r0] IPC event pool created + exported\n");
    } else {
        ze_ipc_event_pool_handle_t iph; memset(&iph,0,sizeof(iph));
        recv_blob(sock,iph.data,sizeof(iph.data));
        CK(zeEventPoolOpenIpcHandle(ctx,iph,&pool));
        printf("[r1] IPC event pool opened OK\n");
    }
    // both ranks create the same 2 events from the shared pool
    ze_event_desc_t dA={ZE_STRUCTURE_TYPE_EVENT_DESC,NULL,0,ZE_EVENT_SCOPE_FLAG_HOST,ZE_EVENT_SCOPE_FLAG_HOST};
    ze_event_desc_t dB={ZE_STRUCTURE_TYPE_EVENT_DESC,NULL,1,ZE_EVENT_SCOPE_FLAG_HOST,ZE_EVENT_SCOPE_FLAG_HOST};
    ze_event_handle_t S_A,S_B; CK(zeEventCreate(pool,&dA,&S_A)); CK(zeEventCreate(pool,&dB,&S_B));
    ze_event_handle_t S_mine = (rank==0)?S_A:S_B;   // I signal this
    ze_event_handle_t S_peer = (rank==0)?S_B:S_A;   // I wait on this

    // ---- scratch IPC exchange (peer pushes into my scratch) ----
    size_t maxb=1u<<20;
    ze_device_mem_alloc_desc_t md={ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,NULL,0,0};
    ze_host_mem_alloc_desc_t hd={ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,NULL,0};
    void *myBuf,*myScratch,*myOut,*ph;
    CK(zeMemAllocDevice(ctx,&md,maxb,4096,mydev,&myBuf));
    CK(zeMemAllocDevice(ctx,&md,maxb,4096,mydev,&myScratch)); // peer pushes here
    CK(zeMemAllocDevice(ctx,&md,maxb,4096,mydev,&myOut));
    CK(zeMemAllocHost(ctx,&hd,maxb,4096,&ph));
    ze_ipc_mem_handle_t myH,peerH;
    CK(zeMemGetIpcHandle(ctx,myScratch,&myH));
    if(rank==0){ send_blob(sock,myH.data,sizeof(myH.data)); recv_blob(sock,peerH.data,sizeof(peerH.data)); }
    else       { recv_blob(sock,peerH.data,sizeof(peerH.data)); send_blob(sock,myH.data,sizeof(myH.data)); }
    void *peerScratch; CK(zeMemOpenIpcHandle(ctx,peerdev,peerH,0,&peerScratch));

    // immediate (sync) list for per-iter setup + readback
    ze_command_queue_desc_t iqd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,NULL,ord,0,0,
        ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t im; CK(zeCommandListCreateImmediate(ctx,mydev,&iqd,&im));

    // ---- PROBE: does rank0's signal become visible to rank1 via the IPC-shared event? ----
    if(rank==0){
        ze_command_list_handle_t pl; CK(zeCommandListCreateImmediate(ctx,mydev,&iqd,&pl));
        CK(zeCommandListAppendSignalEvent(pl,S_A)); // immediate list -> executes now
        printf("[r0] signaled S_A (IPC probe)\n");
        barrier(sock);
        barrier(sock); // wait for r1 to finish its host-sync
    } else {
        barrier(sock);
        ze_result_t r=zeEventHostSynchronize(S_A, 3000000000ULL); // 3s
        printf("[r1] host-sync S_A via IPC -> %s (0x%x)\n",
               r==ZE_RESULT_SUCCESS?"SEEN":"NOT-SEEN/timeout", r);
        barrier(sock);
    }
    CK(zeEventHostReset(S_A)); barrier(sock);

    size_t sizes[]={10240,65536,1u<<20}; const char*lab[]={"10KB(decode)","64KB","1MB"};
    int nsz=3, ITER=(getenv("ITER")?atoi(getenv("ITER")):200);
    int DBG=getenv("DBG")?1:0;
    if(rank==0) printf("%-15s %8s %12s %12s\n","size","iters","sync_us","verify(peer landed after wait)");
    barrier(sock);

    for(int s=0;s<nsz;s++){
        size_t bytes=sizes[s]; size_t n=bytes/sizeof(float);
        // async queue + CLOSED command list (replayable)
        ze_command_queue_desc_t qd={ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,NULL,ord,0,0,
            ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
        ze_command_list_desc_t ld={ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,NULL,ord,0};
        ze_command_queue_handle_t q; ze_command_list_handle_t cl;
        CK(zeCommandQueueCreate(ctx,mydev,&qd,&q)); CK(zeCommandListCreate(ctx,mydev,&ld,&cl));
        if(DBG)fprintf(stderr,"[r%d] s=%d A: q/cl created\n",rank,s);
        // push myBuf -> peerScratch, signal S_mine; wait S_peer; read myScratch->myOut; reset S_peer
        CK(zeCommandListAppendMemoryCopy(cl,peerScratch,myBuf,bytes,S_mine,0,NULL));
        if(DBG)fprintf(stderr,"[r%d] s=%d B: push+signal appended\n",rank,s);
        CK(zeCommandListAppendWaitOnEvents(cl,1,&S_peer));
        if(DBG)fprintf(stderr,"[r%d] s=%d C: wait appended\n",rank,s);
        CK(zeCommandListAppendMemoryCopy(cl,myOut,myScratch,bytes,NULL,0,NULL));
        if(DBG)fprintf(stderr,"[r%d] s=%d D: read appended\n",rank,s);
        CK(zeCommandListAppendEventReset(cl,S_peer));
        if(DBG)fprintf(stderr,"[r%d] s=%d E: reset appended\n",rank,s);
        CK(zeCommandListClose(cl));
        if(DBG)fprintf(stderr,"[r%d] s=%d cl built+closed\n",rank,s);

        int bad=0; double acc=0;
        CK(zeEventHostReset(S_A)); CK(zeEventHostReset(S_B)); barrier(sock);
        if(DBG)fprintf(stderr,"[r%d] s=%d past B5\n",rank,s);
        for(int it=0; it<ITER; it++){
            float mine=(float)(rank==0? it%97 : it%89+1000);
            float expect=(float)(rank==0? it%89+1000 : it%97); // peer's value
            for(size_t k=0;k<n;k++) ((float*)ph)[k]=mine;
            CK(zeCommandListAppendMemoryCopy(im,myBuf,ph,bytes,NULL,0,NULL));
            for(size_t k=0;k<n;k++) ((float*)ph)[k]=-1.0f;      // poison scratch + out
            CK(zeCommandListAppendMemoryCopy(im,myScratch,ph,bytes,NULL,0,NULL));
            CK(zeCommandListAppendMemoryCopy(im,myOut,ph,bytes,NULL,0,NULL));
            if(DBG)fprintf(stderr,"[r%d] it=%d pre-barrier\n",rank,it);
            barrier(sock);                                       // both ready
            if(DBG)fprintf(stderr,"[r%d] it=%d post-barrier, executing\n",rank,it);
            double t0=now_s();
            CK(zeCommandQueueExecuteCommandLists(q,1,&cl,NULL));
            ze_result_t sr=zeCommandQueueSynchronize(q,3000000000ULL); // 3s -> detect deadlock
            if(sr==ZE_RESULT_NOT_READY){ fprintf(stderr,"[r%d] WAIT TIMEOUT iter=%d size=%s -- cross-process event wait deadlocked\n",rank,it,lab[s]); _exit(7); }
            if(sr!=ZE_RESULT_SUCCESS){ fprintf(stderr,"[r%d] sync 0x%x\n",rank,sr); _exit(7); }
            if(it>=5) acc+=now_s()-t0;
            if(DBG)fprintf(stderr,"[r%d] it=%d synced ok\n",rank,it);
            CK(zeCommandListAppendMemoryCopy(im,ph,myOut,sizeof(float),NULL,0,NULL));
            if(((float*)ph)[0]!=expect) bad++;
            barrier(sock);                                       // keep lockstep before next reset
        }
        double per=acc/(ITER-5);
        if(rank==0){ char v[24]; snprintf(v,sizeof v, bad?"BAD(%d)":"OK", bad);
            printf("%-15s %8d %12.2f %12s\n",lab[s],ITER,per*1e6,v); }
        // rank1 reports its own bad count via stderr for completeness
        if(rank==1 && bad) fprintf(stderr,"[r1] bad=%d/%d @ %s\n",bad,ITER,lab[s]);
        zeCommandListDestroy(cl); zeCommandQueueDestroy(q);
        barrier(sock);
    }
    barrier(sock);
    if(rank==0) printf("DONE_IPC_EVENT_SYNC\n");
    _exit(0);
}

int main(void){
    int sv[2]; if(socketpair(AF_UNIX,SOCK_STREAM,0,sv)<0){perror("socketpair");return 1;}
    pid_t pid=fork(); if(pid<0){perror("fork");return 1;}
    if(pid==0){ close(sv[0]); run_rank(1,sv[1]); }
    else      { close(sv[1]); run_rank(0,sv[0]); }
    int st; waitpid(pid,&st,0);
    if(WIFSIGNALED(st)) fprintf(stderr,"[main] child rank1 KILLED by signal %d\n",WTERMSIG(st));
    else if(WIFEXITED(st)&&WEXITSTATUS(st)) fprintf(stderr,"[main] child rank1 exited %d\n",WEXITSTATUS(st));
    return 0;
}
