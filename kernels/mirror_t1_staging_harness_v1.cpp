// Synthetic byte/routing control; not native expert arithmetic or model qualification.
#include <sycl/sycl.hpp>
#include "strata/kernels/resident_plan_mirror.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
constexpr int NE=16,K=10,BLOB=64,CAP=K;
constexpr int START=4,DST=START+CAP+1,TOK=DST+CAP,PO=(TOK+CAP+1)&~1;
constexpr int START2=PO+4*CAP,WORDS=START2+CAP+2;
struct Owned {
    sycl::queue& q;std::vector<void*> ptrs;size_t bytes=0;
    explicit Owned(sycl::queue& v):q(v){ptrs.reserve(20);}
    template<class T>T* take(size_t n,bool host=false){
        if(n>(65536-bytes)/sizeof(T))throw std::runtime_error("64KiB fixture allocation cap");
        T* p=host?sycl::malloc_host<T>(n,q):sycl::malloc_device<T>(n,q);
        if(!p)throw std::bad_alloc();ptrs.push_back(p);bytes+=n*sizeof(T);return p;
    }
    void release(){q.wait_and_throw();while(!ptrs.empty()){sycl::free(ptrs.back(),q);ptrs.pop_back();}}
    ~Owned(){try{q.wait_and_throw();}catch(...){}while(!ptrs.empty()){try{sycl::free(ptrs.back(),q);}catch(...){}ptrs.pop_back();}}
};
unsigned char byte(int expert,int offset){return unsigned((expert*31+offset*7)%251);}
std::vector<int32_t> cpu_layout(const std::array<int32_t,K>& ids,const std::array<int32_t,NE>& res,
                              const std::array<unsigned long long,NE>& mirrors,
                              const std::array<unsigned long long,NE>& offsets,const uint8_t* cache){
    std::vector<int32_t> p(WORDS,-1);int groups=0,fetches=0,entries=0;
    // Existing CPU-published layout contract (expert_source.cpp:3203-3234).
    for(int tier=0;tier<2;++tier){
        for(int i=0;i<K;++i){
            int e=ids[i];if((tier==0)!=(res[e]>=0))continue;
            bool first=true;for(int j=0;j<i;++j)if(ids[j]==e)first=false;if(!first)continue;
            unsigned long long ptr;
            if(tier==0){ptr=(unsigned long long)(cache+offsets[res[e]]);p[START+groups]=entries;std::memcpy(&p[PO+2*groups],&ptr,8);++groups;}
            else{ptr=mirrors[e];if(!ptr)throw std::runtime_error("CPU reference missing mirror");p[START2+fetches]=entries;std::memcpy(&p[PO+2*CAP+2*fetches],&ptr,8);++fetches;}
            for(int j=i;j<K;++j)if(ids[j]==e){p[DST+entries]=j;p[TOK+entries]=j/K;++entries;}
        }
        if(tier==0)p[START+groups]=entries;
    }
    p[START2+fetches]=entries;p[0]=groups;p[1]=entries;p[2]=fetches;return p;
}
void same_layout(const std::vector<int32_t>& actual,const std::vector<int32_t>& ref){
    for(int i=0;i<3;++i)if(actual[i]!=ref[i])throw std::runtime_error("plan count differs");
    for(int i=0;i<=ref[0];++i)if(actual[START+i]!=ref[START+i])throw std::runtime_error("VRAM starts differ");
    for(int i=0;i<=ref[2];++i)if(actual[START2+i]!=ref[START2+i])throw std::runtime_error("mirror starts differ");
    for(int i=0;i<K;++i)if(actual[DST+i]!=ref[DST+i]||actual[TOK+i]!=ref[TOK+i])throw std::runtime_error("entry route differs");
    for(int i=0;i<ref[0]*2;++i)if(actual[PO+i]!=ref[PO+i])throw std::runtime_error("VRAM pointer differs");
    for(int i=0;i<ref[2]*2;++i)if(actual[PO+2*CAP+i]!=ref[PO+2*CAP+i])throw std::runtime_error("mirror pointer differs");
}
class MirrorStageEchoV1;
void echo(sycl::queue& q,int32_t* p,uint8_t* out){
    q.parallel_for<MirrorStageEchoV1>(sycl::range<1>(K*BLOB),[=](sycl::id<1> i){
        int pos=int(i[0])/BLOB,off=int(i[0])%BLOB;const auto* ptr=(const unsigned long long*)(p+PO);
        for(int g=0;g<p[0];++g)for(int j=p[START+g];j<p[START+g+1];++j)
            if(j==pos)out[p[DST+j]*BLOB+off]=((const uint8_t*)ptr[g])[off];
        for(int g=0;g<p[2];++g)for(int j=p[START2+g];j<p[START2+g+1];++j)
            if(j==pos)out[p[DST+j]*BLOB+off]=((const uint8_t*)ptr[CAP+g])[off];
    });
}
void run(sycl::queue& q,int kind){
    Owned owned(q);std::array<int32_t,K> ids{0,1,2,3,4,5,6,7,8,9};
    if(kind==2)ids={2,5,2,7,4,5,9,1,7,3};
    std::array<int32_t,NE> res;res.fill(-1);std::array<unsigned long long,NE> offsets{},mirrors{};
    auto cache=owned.take<uint8_t>(NE*128);auto host=owned.take<uint8_t>(NE*BLOB,true);
    std::vector<uint8_t> hc(NE*128,0);
    for(int e=0;e<NE;++e){offsets[e]=e*128;for(int d=0;d<BLOB;++d){hc[e*128+d]=byte(e,d);host[e*BLOB+d]=byte(e,d);}mirrors[e]=(unsigned long long)(host+e*BLOB);if(kind==0||(kind==2&&e%2==0))res[e]=e;}
    if(kind==3)mirrors[3]=0;
    auto di=owned.take<int32_t>(K);auto dr=owned.take<int32_t>(NE);auto dm=owned.take<unsigned long long>(NE);auto ds=owned.take<unsigned long long>(NE);
    auto plan=owned.take<int32_t>(WORDS);auto flag=owned.take<uint32_t>(1);auto err=owned.take<uint32_t>(1);
    q.memcpy(cache,hc.data(),hc.size());q.memcpy(di,ids.data(),K*4);q.memcpy(dr,res.data(),NE*4);q.memcpy(dm,mirrors.data(),NE*8);q.memcpy(ds,offsets.data(),NE*8);
    q.memset(plan,0xff,WORDS*4);q.memset(flag,0,4);q.memset(err,0,4);
    strata::kernels::resident_plan_bound(di,K,K,dr,NE,cache,ds,BLOB,plan,CAP,flag,17,&q,dm,err,true);
    std::vector<int32_t> hp(WORDS);uint32_t hf=0,he=0;
    q.memcpy(hp.data(),plan,WORDS*4);q.memcpy(&hf,flag,4);q.memcpy(&he,err,4);q.wait_and_throw();
    if(kind==3){if(hf!=0)throw std::runtime_error("missing mirror published a device plan");owned.release();std::puts("MIRROR_T1_CASE missing_mirror original_fallback=1 no_fetch=1 freed=1");return;}
    if(hf!=17||he!=0)throw std::runtime_error("valid plan failed publication");
    auto ref=cpu_layout(ids,res,mirrors,offsets,cache);same_layout(hp,ref);
    if(hp[2]>16)throw std::runtime_error("mirror groups exceed stage cap");
    auto baseline=owned.take<int32_t>(WORDS);q.memcpy(baseline,ref.data(),WORDS*4);
    std::array<uint8_t*,2> stage{owned.take<uint8_t>(16*BLOB),owned.take<uint8_t>(16*BLOB)};
    std::array<uint8_t*,2> out{owned.take<uint8_t>(K*BLOB),owned.take<uint8_t>(K*BLOB)};
    for(int arm=0;arm<2;++arm){
        int32_t* p=arm?baseline:plan;auto ptr2=(unsigned long long*)(p+PO)+CAP;
        q.memset(out[arm],0xff,K*BLOB);
        strata::kernels::fetch_blobs(ptr2,p+2,stage[arm],BLOB,16,&q);
        strata::kernels::rebase_ptrs(ptr2,p+2,stage[arm],BLOB,&q);echo(q,p,out[arm]);
    }
    std::vector<uint8_t>a(K*BLOB),b(K*BLOB);q.memcpy(a.data(),out[0],a.size());q.memcpy(b.data(),out[1],b.size());
    q.memcpy(hp.data(),plan,WORDS*4);q.wait_and_throw();
    if(a!=b)throw std::runtime_error("device/CPU-layout byte output differs");
    for(int i=0;i<K;++i)for(int d=0;d<BLOB;++d)if(a[i*BLOB+d]!=byte(ids[i],d))throw std::runtime_error("original destination blob differs");
    for(int g=0;g<hp[2];++g){unsigned long long p;std::memcpy(&p,&hp[PO+2*CAP+2*g],8);if(p!=(unsigned long long)(stage[0]+g*BLOB))throw std::runtime_error("rebase pointer differs");}
    size_t bytes=owned.bytes;owned.release();
    std::printf("MIRROR_T1_CASE kind=%d VRAM_groups=%d mirror_groups=%d entries=%d exact_layout=1 exact_bytes=1 allocation_bytes=%zu freed=1\n",kind,ref[0],ref[2],ref[1],bytes);
}
}
int main(int argc,char**argv){try{
    if(argc!=2||std::string(argv[1])!="--synthetic")throw std::invalid_argument("--synthetic required");
    const char*mask=std::getenv("ZE_AFFINITY_MASK"),*sel=std::getenv("ONEAPI_DEVICE_SELECTOR");
    if(!mask||std::strcmp(mask,"0")||!sel||std::strcmp(sel,"level_zero:gpu"))throw std::invalid_argument("card0 LevelZero pins required");
    {sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});for(int k=0;k<4;++k)run(q,k);q.wait_and_throw();}
    std::puts("MIRROR_T1_SYNTHETIC_PASS cases=4 context_retired=1 native_math_qualified=0 model_qualified=0 traffic_or_speed_qualified=0");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"MIRROR_T1_ERROR %s\n",e.what());return 1;}}
