// Model-independent operator fixture. Root-only GPU execution under card lease.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include "strata/kernels/hc_native_projection.hpp"
#include <cstdint>
#include <algorithm>
#include <new>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
struct DeviceMemory {
    sycl::queue& q;void* p;
    DeviceMemory(sycl::queue& queue,size_t bytes):q(queue),p(sycl::malloc_device(bytes,q)){if(!p)throw std::bad_alloc();}
    ~DeviceMemory(){sycl::free(p,q);}
};
static uint32_t rng(uint32_t& s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
static void need(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
static void run(sycl::queue& q,int k,int m,int tokens) {
    uint32_t seed=uint32_t(k+31*m+tokens);size_t blocks=size_t(m)*(k/32);
    std::vector<uint8_t> weights(blocks*34);std::vector<float> input(size_t(tokens)*k);
    for(size_t b=0;b<blocks;++b){uint16_t scale=uint16_t(0x2800+(rng(seed)%0x1800));weights[b*34]=uint8_t(scale);weights[b*34+1]=uint8_t(scale>>8);for(int i=0;i<32;++i)weights[b*34+2+i]=uint8_t(rng(seed));}
    for(float& f:input)f=float(int(rng(seed)%4097)-2048)/2048.0f;
    DeviceMemory w(q,weights.size()),x(q,input.size()*4),reference(q,size_t(tokens)*m*4),result(q,size_t(tokens)*m*4);
    q.memcpy(w.p,weights.data(),weights.size()).wait_and_throw();q.memcpy(x.p,input.data(),input.size()*4).wait_and_throw();
    auto baseline=[&]{for(int at=0;at<tokens;at+=8){int n=std::min(8,tokens-at);need(strata::kernels::hc_q8_0_project_f32(static_cast<const uint8_t*>(w.p),weights.size(),static_cast<const float*>(x.p)+size_t(at)*k,size_t(n)*k,static_cast<float*>(reference.p)+size_t(at)*m,size_t(n)*m,k,m,n,&q),"old scalar launch rejected");}q.wait_and_throw();};
    auto candidate=[&](int tile){need(strata::kernels::hc_q8_0_project_f32_token_reuse(static_cast<const uint8_t*>(w.p),weights.size(),static_cast<const float*>(x.p),input.size(),static_cast<float*>(result.p),size_t(tokens)*m,k,m,tokens,tile,&q),"new tile launch rejected");};
    std::vector<float> a(size_t(tokens)*m),b(a.size());
    auto compare=[&]{q.memcpy(a.data(),reference.p,a.size()*4).wait_and_throw();q.memcpy(b.data(),result.p,b.size()*4).wait_and_throw();need(std::memcmp(a.data(),b.data(),a.size()*4)==0,"token reuse is not bitwise scalar");};
    baseline();for(int tile:{2,4}){candidate(tile);q.wait_and_throw();compare();}
    if(tokens==512){
        namespace exp=sycl::ext::oneapi::experimental;
        exp::command_graph<exp::graph_state::modifiable> graph(q.get_context(),q.get_device());
        graph.begin_recording(q);candidate(4);graph.end_recording(q);auto executable=graph.finalize();
        q.ext_oneapi_graph(executable).wait_and_throw();compare();
        // Replay after original input allocation contents change; graph arguments stay owned by value.
        for(float& f:input)f=-f+0.125f;q.memcpy(x.p,input.data(),input.size()*4).wait_and_throw();baseline();q.ext_oneapi_graph(executable).wait_and_throw();compare();
    }
    need(!strata::kernels::hc_q8_0_project_f32_token_reuse(static_cast<const uint8_t*>(w.p),weights.size(),static_cast<const float*>(x.p),input.size(),static_cast<float*>(result.p),a.size(),k,m,tokens,3,&q),"invalid tile accepted");
    std::cout<<"HC_Q8_TOKEN_REUSE_BITWISE k="<<k<<" m="<<m<<" tokens="<<tokens<<" tiles=2,4 graph_replay="<<(tokens==512)<<"\n";
}
int main(){try{setenv("STRATA_HC_Q8_TOKEN_TILE","0",1);sycl::queue q(sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::in_order{}});for(int t:{1,3,7,8,512}){run(q,10240,320,t);run(q,320,10240,t);}q.wait_and_throw();std::cout<<"HC_Q8_TOKEN_REUSE_FIXTURE_PASS cases=10 model_payload=0\n";return 0;}catch(const std::exception& e){std::cerr<<"HC_Q8_TOKEN_REUSE_FIXTURE_FAIL "<<e.what()<<"\n";return 1;}}
