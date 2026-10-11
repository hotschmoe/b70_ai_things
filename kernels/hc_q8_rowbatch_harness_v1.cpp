// Explicit synthetic control. Compile unchanged parent projection as renamed reference.
#include <sycl/sycl.hpp>
#include "strata/kernels/hc_native_projection.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace strata::kernels {
bool b70_original_hc_q8_0_project_f32(const uint8_t*,size_t,const float*,size_t,float*,size_t,int,int,int,void*);
}
namespace {
struct Owned {
    sycl::queue& q; std::vector<void*> p;
    explicit Owned(sycl::queue& queue):q(queue) {p.reserve(4);}
    template<class T>T* take(size_t n){auto x=sycl::malloc_device<T>(n,q);if(!x)throw std::bad_alloc();p.push_back(x);return x;}
    void release(){q.wait_and_throw();while(!p.empty()){sycl::free(p.back(),q);p.pop_back();}}
    ~Owned(){try{q.wait_and_throw();}catch(...){}while(!p.empty()){try{sycl::free(p.back(),q);}catch(...){}p.pop_back();}}
};
void run(sycl::queue& q,int T,int K,int M){
    const size_t wb=size_t(M)*(K/32)*34, nx=size_t(T)*K, ny=size_t(T)*M;
    std::vector<uint8_t> w(wb);std::vector<float>x(nx),a(ny),b(ny);
    for(size_t block=0;block<wb/34;++block){
        w[block*34]=0;w[block*34+1]=0x20; // exactly represented FP16 scale 2^-7
        for(int lane=0;lane<32;++lane){int v=int((block*11+lane*7)%255)-127;w[block*34+2+lane]=uint8_t(v<0?v+256:v);}
    }
    for(size_t i=0;i<nx;++i){uint32_t bits=0x3f000000u|uint32_t((i*7919+37)&0x7fffff);if(i%3==0)bits|=0x80000000u;std::memcpy(&x[i],&bits,4);}
    Owned owned(q);auto W=owned.take<uint8_t>(wb);auto X=owned.take<float>(nx);auto A=owned.take<float>(ny);auto B=owned.take<float>(ny);
    q.memcpy(W,w.data(),wb);q.memcpy(X,x.data(),nx*4);q.memset(A,0xff,ny*4);q.memset(B,0xff,ny*4);
    if(!strata::kernels::b70_original_hc_q8_0_project_f32(W,wb,X,nx,A,ny,K,M,T,&q))throw std::runtime_error("original rejected");
    if(!strata::kernels::hc_q8_0_project_f32(W,wb,X,nx,B,ny,K,M,T,&q))throw std::runtime_error("new rejected");
    q.memcpy(a.data(),A,ny*4);q.memcpy(b.data(),B,ny*4);q.wait_and_throw();
    for(size_t i=0;i<ny;++i)if(!sycl::isfinite(a[i])||!sycl::isfinite(b[i]))throw std::runtime_error("nonfinite output");
    if(std::memcmp(a.data(),b.data(),ny*4))throw std::runtime_error("original/rowbatch bits differ");
    owned.release();
    std::printf("HC_Q8_ROWBATCH_CASE tokens=%d K=%d M=%d words=%zu bitwise=1 device_bytes=%zu freed=1\n",T,K,M,ny,wb+(nx+2*ny)*4);
}
}
int main(int argc,char**argv){
    try{
        if(argc!=2||std::string(argv[1])!="--synthetic")throw std::invalid_argument("--synthetic required");
        const char*mask=std::getenv("ZE_AFFINITY_MASK"),*sel=std::getenv("ONEAPI_DEVICE_SELECTOR"),*mode=std::getenv("STRATA_HC_Q8_ROWBATCH");
        if(!mask||std::strcmp(mask,"0")||!sel||std::strcmp(sel,"level_zero:gpu")||!mode||std::strcmp(mode,"1"))throw std::invalid_argument("card0/LevelZero/rowbatch1 pin required");
        {
            sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
            if(q.get_backend()!=sycl::backend::ext_oneapi_level_zero)throw std::runtime_error("LevelZero required");
            for(int T=1;T<=8;++T){run(q,T,10240,320);run(q,T,320,10240);}
            q.wait_and_throw();
        }
        std::puts("HC_Q8_ROWBATCH_SYNTHETIC_PASS cases=16 context_retired=1 model_math_qualified=0 speed_qualified=0");return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"HC_Q8_ROWBATCH_ERROR %s\n",e.what());return 1;}
}
