// Whole native HC composition: old8 slices vs explicit prefill512, root GPU only.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include "strata/kernels/hc_native_composition.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace ex=sycl::ext::oneapi::experimental;
namespace sk=strata::kernels;
struct Owner {
    sycl::queue& q;std::vector<void*> buffers;
    ex::command_graph<ex::graph_state::modifiable>* graph=nullptr;
    ex::command_graph<ex::graph_state::executable>* executable=nullptr;bool recording=false;
    explicit Owner(sycl::queue& queue):q(queue){buffers.reserve(32);}
    [[noreturn]] static void retain(){std::cerr<<"HC_PREFILL_DRAIN_FAILED retained=1\n";std::cerr.flush();std::_Exit(86);}
    ~Owner() noexcept {
        if(recording){try{graph->end_recording(q);recording=false;}catch(...){retain();}}
        try{q.wait_and_throw();}catch(...){retain();}
        delete executable;delete graph;
        for(void* p:buffers){try{sycl::free(p,q);}catch(...){std::cerr<<"HC_PREFILL_FREE_FAILED\n";std::cerr.flush();std::_Exit(87);}}
    }
    template<class T>T* alloc(size_t n){if(buffers.size()==32)throw std::logic_error("owner bound");auto* p=sycl::malloc_device<T>(n,q);if(!p)throw std::bad_alloc();buffers.push_back(p);return p;}
};
static void need(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
static uint32_t rng(uint32_t& s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
static std::vector<float> floats(size_t n,uint32_t& seed){std::vector<float> x(n);for(auto& v:x)v=float(int(rng(seed)%4097)-2048)/2048.0f;return x;}
static std::vector<uint8_t> q8(size_t blocks,uint32_t& seed){std::vector<uint8_t> w(blocks*34);for(size_t b=0;b<blocks;++b){uint16_t d=uint16_t(0x2000+rng(seed)%0x1400);w[b*34]=uint8_t(d);w[b*34+1]=uint8_t(d>>8);for(int j=0;j<32;++j)w[b*34+2+j]=uint8_t(rng(seed));}return w;}
struct Outputs {float *pending,*xn,*lo,*gate,*rs,*mixed,*inject,*written;};
static Outputs allocate(Owner& o,int t){return {o.alloc<float>(size_t(t)*10240),o.alloc<float>(size_t(t)*10240),o.alloc<float>(size_t(t)*320),o.alloc<float>(size_t(t)*10240),o.alloc<float>(size_t(t)*4),o.alloc<float>(size_t(t)*2560),o.alloc<float>(size_t(t)*4),o.alloc<float>(size_t(t)*10240)};}
static sk::HcNativeArgs args(const sk::HcNativeWeights& w,const Outputs& out,const float* R,const float* bo,const float* inj,int t,bool apply){
    sk::HcNativeArgs a;a.w=w;a.tokens=t;a.R=R;a.R_floats=size_t(t)*10240;
    a.xn=out.xn;a.xn_floats=size_t(t)*10240;a.lo=out.lo;a.lo_floats=size_t(t)*320;
    a.gate=out.gate;a.gate_floats=size_t(t)*10240;a.rs=out.rs;a.rs_floats=size_t(t)*4;
    a.mixed=out.mixed;a.mixed_floats=size_t(t)*2560;a.inject_out=out.inject;a.inject_out_floats=size_t(t)*4;
    a.apply=apply;if(apply){a.bo_prev=bo;a.bo_floats=size_t(t)*2560;a.inj_prev=inj;a.prev_floats=size_t(t)*4;a.R_out=out.pending;a.R_out_floats=size_t(t)*10240;}return a;
}
static void run(sycl::queue& q,int tokens,bool apply){
    uint32_t seed=71+tokens;auto down=q8(320*(10240/32),seed),up=q8(10240*(320/32),seed);
    auto norm=floats(10240,seed),inject=floats(4*10240,seed),residual=floats(size_t(tokens)*10240,seed),bo=floats(size_t(tokens)*2560,seed),prev=floats(size_t(tokens)*4,seed);
    for(auto& x:norm)x=x*.1f+1.0f;for(auto& x:inject)x*=.01f;
    Owner owner(q); // Host backing vectors above outlive owned queue retirement.
    auto* dw=owner.alloc<uint8_t>(down.size());auto* uw=owner.alloc<uint8_t>(up.size());auto* nw=owner.alloc<float>(norm.size());auto* iw=owner.alloc<float>(inject.size());
    auto* R=owner.alloc<float>(residual.size());auto* b=owner.alloc<float>(bo.size());auto* p=owner.alloc<float>(prev.size());
    q.memcpy(dw,down.data(),down.size());q.memcpy(uw,up.data(),up.size());q.memcpy(nw,norm.data(),norm.size()*4);q.memcpy(iw,inject.data(),inject.size()*4);q.memcpy(b,bo.data(),bo.size()*4);q.memcpy(p,prev.data(),prev.size()*4);
    auto old=allocate(owner,tokens),fresh=allocate(owner,tokens);sk::HcNativeWeights w{nw,norm.size(),dw,down.size(),uw,up.size(),iw,inject.size()};
    auto reset=[&]{q.memcpy(R,residual.data(),residual.size()*4);q.memcpy(old.written,residual.data(),residual.size()*4);q.memcpy(fresh.written,residual.data(),residual.size()*4);};
    auto baseline=[&]{for(int at=0;at<tokens;at+=8){int n=std::min(8,tokens-at);Outputs o{old.pending+size_t(at)*10240,old.xn+size_t(at)*10240,old.lo+size_t(at)*320,old.gate+size_t(at)*10240,old.rs+size_t(at)*4,old.mixed+size_t(at)*2560,old.inject+size_t(at)*4,old.written+size_t(at)*10240};auto a=args(w,o,R+size_t(at)*10240,b+size_t(at)*2560,p+size_t(at)*4,n,apply);need(sk::hc_native_read_f32(a,&q),"old8 read refused");need(sk::hc_native_write_f32(o.written,size_t(n)*10240,o.mixed,size_t(n)*2560,o.inject,size_t(n)*4,n,&q),"old8 write refused");}};
    auto candidate=[&]{auto a=args(w,fresh,R,b,p,tokens,apply);need(sk::hc_native_prefill_read_f32(a,&q),"prefill read refused");need(sk::hc_native_prefill_write_f32(fresh.written,size_t(tokens)*10240,fresh.mixed,size_t(tokens)*2560,fresh.inject,size_t(tokens)*4,tokens,&q),"prefill write refused");};
    auto equal=[&](const float* a,const float* b,size_t n,const char* field){std::vector<float> x(n),y(n);try{q.memcpy(x.data(),a,n*4).wait_and_throw();q.memcpy(y.data(),b,n*4).wait_and_throw();}catch(...){Owner::retain();}if(std::memcmp(x.data(),y.data(),n*4)){std::cerr<<"different_field="<<field<<"\n";need(false,"whole HC is not bitwise");}};
    auto compare=[&]{if(apply)equal(old.pending,fresh.pending,size_t(tokens)*10240,"pending");equal(old.xn,fresh.xn,size_t(tokens)*10240,"xn");equal(old.lo,fresh.lo,size_t(tokens)*320,"lo_silu");equal(old.gate,fresh.gate,size_t(tokens)*10240,"gate");equal(old.rs,fresh.rs,size_t(tokens)*4,"rs");equal(old.mixed,fresh.mixed,size_t(tokens)*2560,"mixed");equal(old.inject,fresh.inject,size_t(tokens)*4,"inject");equal(old.written,fresh.written,size_t(tokens)*10240,"written");};
    reset();baseline();candidate();q.wait_and_throw();compare();
    if(tokens==512 && apply){owner.graph=new ex::command_graph<ex::graph_state::modifiable>(q.get_context(),q.get_device());owner.recording=true;owner.graph->begin_recording(q);candidate();owner.graph->end_recording(q);owner.recording=false;owner.executable=new ex::command_graph<ex::graph_state::executable>(owner.graph->finalize());
        for(auto& x:residual)x=-x+.125f;reset();baseline();q.ext_oneapi_graph(*owner.executable).wait_and_throw();compare();}
    auto bad=args(w,fresh,R,b,p,tokens,apply);bad.tokens=513;need(!sk::hc_native_prefill_read_f32(bad,&q),"overbound accepted");
    bad=args(w,fresh,R,b,p,tokens,apply);bad.xn_floats=0;need(!sk::hc_native_prefill_read_f32(bad,&q),"short workspace accepted");
    if(tokens>8){bad=args(w,fresh,R,b,p,tokens,apply);need(!sk::hc_native_read_f32(bad,&q),"old guard widened");}
    std::cout<<"HC_PREFILL_BITWISE tokens="<<tokens<<" apply="<<apply<<" xn_lo_gate_rs_mixed_inject_written=1 graph_changed_input="<<(tokens==512&&apply)<<"\n";
}
int main(){try{setenv("STRATA_HC_Q8_TOKEN_TILE","0",1);sycl::queue q(sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::in_order{}});for(int t:{1,7,8,9,512})for(bool apply:{false,true})run(q,t,apply);std::cout<<"HC_PREFILL_FIXTURE_PASS cases=10 model_payload=0\n";return 0;}catch(const std::exception& e){std::cerr<<"HC_PREFILL_FIXTURE_FAIL "<<e.what()<<"\n";return 1;}}
