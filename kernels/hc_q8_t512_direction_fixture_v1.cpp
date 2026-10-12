// Source55 exact existing Q8 token kernels: T512 tile2 versus tile4, root GPU only.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include "strata/kernels/hc_native_projection.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <vector>
namespace ex=sycl::ext::oneapi::experimental;
struct Owner {
    sycl::queue& q;std::vector<void*> buffers;
    ex::command_graph<ex::graph_state::modifiable>* graph=nullptr;
    ex::command_graph<ex::graph_state::executable>* executable=nullptr;bool recording=false;
    std::vector<ex::command_graph<ex::graph_state::modifiable>*> graphs;
    std::vector<ex::command_graph<ex::graph_state::executable>*> executables;
    explicit Owner(sycl::queue& queue):q(queue){buffers.reserve(4);graphs.reserve(2);executables.reserve(2);}
    [[noreturn]] static void retain(){std::cerr<<"HC_Q8_OWNER_DRAIN_FAILED retained_graph_and_buffers=1\n";std::cerr.flush();std::_Exit(86);}
    ~Owner() noexcept {
        if(recording){try{graph->end_recording(q);recording=false;}catch(...){retain();}}
        try{q.wait_and_throw();}catch(...){retain();}
        // Only original drained queue retirement authorizes graph/context storage release.
        for(auto* x:executables)delete x;for(auto* x:graphs)delete x;
        for(void* p:buffers){try{sycl::free(p,q);}catch(...){std::cerr<<"HC_Q8_OWNER_FREE_FAILED\n";std::cerr.flush();std::_Exit(87);}}
    }
    void* alloc(size_t n){if(buffers.size()==4)throw std::logic_error("owner bound");void* p=sycl::malloc_device(n,q);if(!p)throw std::bad_alloc();buffers.push_back(p);return p;}
};
static uint32_t rng(uint32_t& s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
static void need(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
static void run(sycl::queue& q,int k,int m,int tokens) {
    uint32_t seed=uint32_t(k+31*m+tokens);size_t blocks=size_t(m)*(k/32);
    std::vector<uint8_t> weights(blocks*34);std::vector<float> input(size_t(tokens)*k);
    for(size_t b=0;b<blocks;++b){uint16_t scale=uint16_t(0x2800+(rng(seed)%0x1800));weights[b*34]=uint8_t(scale);weights[b*34+1]=uint8_t(scale>>8);for(int i=0;i<32;++i)weights[b*34+2+i]=uint8_t(rng(seed));}
    for(float& f:input)f=float(int(rng(seed)%4097)-2048)/2048.0f;
    Owner owner(q); // Host weight/input vectors outlive original queue retirement.
    auto* w=static_cast<uint8_t*>(owner.alloc(weights.size()));auto* x=static_cast<float*>(owner.alloc(input.size()*4));auto* reference=static_cast<float*>(owner.alloc(size_t(tokens)*m*4));auto* result=static_cast<float*>(owner.alloc(size_t(tokens)*m*4));
    q.memcpy(w,weights.data(),weights.size()).wait_and_throw();q.memcpy(x,input.data(),input.size()*4).wait_and_throw();
    auto baseline=[&]{need(strata::kernels::hc_q8_0_project_f32_token_reuse(w,weights.size(),x,input.size(),reference,size_t(tokens)*m,k,m,tokens,2,&q),"current whole-T512 tile2 rejected");};
    auto candidate=[&](int tile){need(strata::kernels::hc_q8_0_project_f32_token_reuse(w,weights.size(),x,input.size(),result,size_t(tokens)*m,k,m,tokens,tile,&q),"new tile launch rejected");};
    std::vector<float> a(size_t(tokens)*m),b(a.size());auto compare=[&]{try{q.memcpy(a.data(),reference,a.size()*4).wait_and_throw();q.memcpy(b.data(),result,b.size()*4).wait_and_throw();}catch(...){Owner::retain();}need(std::memcmp(a.data(),b.data(),a.size()*4)==0,"tile4 is not bitwise current tile2");};
    baseline();q.wait_and_throw();for(int tile:{2,4}){candidate(tile);q.wait_and_throw();compare();}
    if(tokens==512){
        for(int warm=0;warm<3;++warm)for(int tile:{2,4}){candidate(tile);q.wait_and_throw();}
        for(int repeat=0;repeat<9;++repeat){
            int order[2]={2,4};if(repeat&1)std::reverse(order,order+2);
            for(int tile:order){auto start=std::chrono::steady_clock::now();candidate(tile);q.wait_and_throw();auto end=std::chrono::steady_clock::now();compare();double us=std::chrono::duration<double,std::micro>(end-start).count();std::cout<<"HC_Q8_T512_LATENCY k="<<k<<" m="<<m<<" tokens="<<tokens<<" weights_bytes="<<weights.size()<<" input_bytes="<<input.size()*4<<" output_bytes="<<a.size()*4<<" repeat="<<repeat<<" tile="<<tile<<" host_submit_plus_queue_wait_us="<<us<<" device_time_qualified=0\n";}
        }
    }
    if(tokens==512){
        for(int tile:{2,4}) {
            owner.graph=new ex::command_graph<ex::graph_state::modifiable>(q.get_context(),q.get_device());owner.graphs.push_back(owner.graph);owner.recording=true;owner.graph->begin_recording(q);candidate(tile);owner.graph->end_recording(q);owner.recording=false;owner.executable=new ex::command_graph<ex::graph_state::executable>(owner.graph->finalize());owner.executables.push_back(owner.executable);q.ext_oneapi_graph(*owner.executable).wait_and_throw();compare();
            for(float& f:input)f=-f+0.125f;q.memcpy(x,input.data(),input.size()*4).wait_and_throw();baseline();q.wait_and_throw();q.ext_oneapi_graph(*owner.executable).wait_and_throw();compare();
        }
    }
    need(!strata::kernels::hc_q8_0_project_f32_token_reuse(w,weights.size(),x,input.size(),result,a.size(),k,m,tokens,3,&q),"invalid tile accepted");std::cout<<"HC_Q8_T512_DIRECTION_BITWISE k="<<k<<" m="<<m<<" tokens="<<tokens<<" tiles=2,4 graph_replay="<<(tokens==512)<<"\n";
}
int main(){try{setenv("STRATA_HC_Q8_TOKEN_TILE","2",1);sycl::queue q(sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::in_order{}});const char* expected=std::getenv("B70_HC_EXPECTED_PCI");need(expected && std::strlen(expected)==12,"Exact expected PCI BDF required");need(q.get_device().is_gpu() && q.get_backend()==sycl::backend::ext_oneapi_level_zero,"Original LevelZero GPU required");
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
need(q.get_device().has(sycl::aspect::ext_intel_pci_address),"Selected GPU lacks PCI address query");const auto actual=q.get_device().get_info<sycl::ext::intel::info::device::pci_address>();std::cout<<"HC_Q8_T512_DEVICE pci_address="<<actual<<" expected="<<expected<<" name="<<q.get_device().get_info<sycl::info::device::name>()<<"\n";std::cout.flush();need(actual==expected,"Selected GPU PCI differs from owned card");
#else
need(false,"Compiler lacks required PCI address extension");
#endif
for(int t:{512}){run(q,10240,320,t);run(q,320,10240,t);}q.wait_and_throw();std::cout<<"HC_Q8_T512_DIRECTION_FIXTURE_PASS cases=2 model_payload=0\n";return 0;}catch(const std::exception& e){std::cerr<<"HC_Q8_T512_DIRECTION_FIXTURE_FAIL "<<e.what()<<"\n";return 1;}}
