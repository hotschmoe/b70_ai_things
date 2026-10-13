// Original-packed-weight projection parity only. No conv/recurrence qualification.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include "prefill_gdn_decoder_projection_v1.hpp"
#include "prefill_gdn_post_store_generated_v1.inc"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace ex=sycl::ext::oneapi::experimental;
namespace pf=strata::kernels::pf_gdn_decoder_proj;
using namespace strata::kernels;
static void need(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
static std::vector<unsigned char> read(const std::filesystem::path&p,size_t n){std::ifstream f(p,std::ios::binary);need(bool(f),"Original exported file missing");std::vector<unsigned char> b(n);f.read(reinterpret_cast<char*>(b.data()),n);need(f.gcount()==std::streamsize(n)&&f.peek()==std::char_traits<char>::eof(),"Exact original export extent required");return b;}
struct Owner{
 sycl::queue&q;std::vector<void*> buffers;std::vector<ex::command_graph<ex::graph_state::modifiable>*> graphs;std::vector<ex::command_graph<ex::graph_state::executable>*> execs;ex::command_graph<ex::graph_state::modifiable>*active=nullptr;bool recording=false;
 explicit Owner(sycl::queue&x):q(x){buffers.reserve(32);graphs.reserve(2);execs.reserve(2);}
 [[noreturn]]static void retain(){std::cerr<<"PF_GDN_RETAIN drain_failed=1\n";std::cerr.flush();std::_Exit(86);}
 ~Owner()noexcept{if(recording){try{active->end_recording(q);}catch(...){retain();}}try{q.wait_and_throw();}catch(...){retain();}for(auto*g:execs)delete g;for(auto*g:graphs)delete g;for(auto*b:buffers){try{sycl::free(b,q);}catch(...){retain();}}}
 unsigned char* alloc(size_t n){need(buffers.size()<32,"Owner buffer bound");auto*p=sycl::malloc_device<unsigned char>(n+128,q);if(!p)throw std::bad_alloc();buffers.push_back(p);q.memset(p,0xA5,n+128).wait_and_throw();return p+64;}
 void upload(void*p,const void*h,size_t n){try{q.memcpy(p,h,n).wait_and_throw();}catch(...){retain();}}
 std::vector<unsigned char> copy(void*p,size_t n){std::vector<unsigned char> h(n+128);try{q.memcpy(h.data(),static_cast<unsigned char*>(p)-64,n+128).wait_and_throw();}catch(...){retain();}for(size_t i=0;i<64;i++)need(h[i]==0xA5&&h[n+64+i]==0xA5,"Projection canary overwritten");return {h.begin()+64,h.begin()+64+n};}
 template<class F>auto record(F f){active=new ex::command_graph<ex::graph_state::modifiable>(q.get_context(),q.get_device());graphs.push_back(active);recording=true;active->begin_recording(q);f();active->end_recording(q);recording=false;auto*e=new ex::command_graph<ex::graph_state::executable>(active->finalize());execs.push_back(e);return e;}
};
struct Outputs{float*qkv;float*z;float*gate;float*beta;float*block;void*scratch;void*input_packet;void*output_packet;};
static void decoder(const pf::Weights&w,const float*x,const float*y,Outputs o,int T,void*s){for(int t=0;t<T;t++){
 native_quantize_q8_1(x+t*2560,o.scratch,2560,1,s);
 if(t==T-1)strata::q_of(s)->memcpy(o.input_packet,o.scratch,2880);
 native_mmvq(8,w.qkv,o.scratch,o.qkv+t*10240,2560,10240,1,s);
 native_mmvq(8,w.z,o.scratch,o.z+t*6144,2560,6144,1,s);
 gdn_ab_multi(x+t*2560,w.alpha,w.beta,w.dt,w.ssm_a,o.gate+t*48,o.beta+t*48,2560,48,1,s);
 native_quantize_q8_1(y+t*6144,o.scratch,6144,1,s);
 if(t==T-1)strata::q_of(s)->memcpy(o.output_packet,o.scratch,6912);
 native_mmvq(8,w.out,o.scratch,o.block+t*2560,6144,2560,1,s);
}}
static void candidate(const pf::Weights&w,const float*x,const float*y,Outputs o,int T,void*s){pf::input(w,x,o.qkv,o.z,o.gate,o.beta,T,o.scratch,pf::scratch_bytes,s);const int last=(T-1)%8;strata::q_of(s)->memcpy(o.input_packet,static_cast<unsigned char*>(o.scratch)+last*2880,2880);pf::output(w,y,o.block,T,o.scratch,pf::scratch_bytes,s);strata::q_of(s)->memcpy(o.output_packet,static_cast<unsigned char*>(o.scratch)+last*6912,6912);}
static std::array<std::vector<unsigned char>,5> collect(Owner&o,Outputs a,int T){const std::array<float*,5> p{a.qkv,a.z,a.gate,a.beta,a.block};const std::array<size_t,5> dims{10240,6144,48,48,2560};std::array<std::vector<unsigned char>,5> r;for(size_t j=0;j<5;j++){r[j]=o.copy(p[j],T*dims[j]*4);for(size_t k=0;k<r[j].size();k+=4){float v;std::memcpy(&v,r[j].data()+k,4);need(std::isfinite(v),"Nonfinite projection output");}}return r;}
int main(int argc,char**argv){try{
 need(argc==2,"Usage: pf-gdn ORIGINAL_EXPORT_ROOT");const auto root=std::filesystem::path(argv[1]);need(pf::enabled(),"Explicit projection fixture optin required");need(native_mmvq_multi_exact(),"Original default exact MMVQ profile required");
 // All host source storage outlives Owner even on partial construction failure.
 const std::array<const char*,7> names{"qkv.q80","z.q80","out.q80","alpha.bf16","beta.bf16","dt.f32","ssm_a.f32"};const std::array<size_t,7> sizes{27852800,16711680,16711680,245760,245760,192,192};std::array<std::vector<unsigned char>,7> weights;for(size_t i=0;i<7;i++)weights[i]=read(root/names[i],sizes[i]);
 std::vector<float>gamma(128,1.0f);std::vector<float>x(9*2560),y(9*6144);for(size_t i=0;i<x.size();i++)x[i]=float(int(i*17%257)-128)*0.001953125f;for(size_t i=0;i<y.size();i++)y[i]=float(int(i*29%509)-254)*0.00390625f;
 sycl::queue q(sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::in_order{}});need(q.get_backend()==sycl::backend::ext_oneapi_level_zero,"Original LevelZero required");const char*pci=std::getenv("B70_PF_GDN_EXPECTED_PCI");need(pci&&std::strlen(pci)==12,"Leased PCI required");
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
 need(q.get_device().has(sycl::aspect::ext_intel_pci_address),"PCI capability required");auto actual=q.get_device().get_info<sycl::ext::intel::info::device::pci_address>();need(actual==pci,"Selected PCI differs from leased card");std::cout<<"PF_GDN_DEVICE pci="<<actual<<"\n";
#else
 need(false,"PCI query unavailable");
#endif
 Owner owner(q);std::array<void*,7> wp{};for(size_t i=0;i<7;i++){wp[i]=owner.alloc(sizes[i]);owner.upload(wp[i],weights[i].data(),sizes[i]);}pf::Weights w;w.qkv_type=w.z_type=w.out_type=8;w.qkv=wp[0];w.z=wp[1];w.out=wp[2];w.alpha=static_cast<uint16_t*>(wp[3]);w.beta=static_cast<uint16_t*>(wp[4]);w.dt=static_cast<float*>(wp[5]);w.ssm_a=static_cast<float*>(wp[6]);auto*X=reinterpret_cast<float*>(owner.alloc(x.size()*4));auto*Y=reinterpret_cast<float*>(owner.alloc(y.size()*4));owner.upload(X,x.data(),x.size()*4);owner.upload(Y,y.data(),y.size()*4);
 auto make=[&](){Outputs o;o.qkv=reinterpret_cast<float*>(owner.alloc(9*10240*4));o.z=reinterpret_cast<float*>(owner.alloc(9*6144*4));o.gate=reinterpret_cast<float*>(owner.alloc(9*48*4));o.beta=reinterpret_cast<float*>(owner.alloc(9*48*4));o.block=reinterpret_cast<float*>(owner.alloc(9*2560*4));o.scratch=owner.alloc(pf::scratch_bytes);o.input_packet=owner.alloc(2880);o.output_packet=owner.alloc(6912);return o;};auto a=make(),b=make();
 // Guard allocation extents are fixed at nine rows, not the active request T.
 auto check=[&](int T){auto aa=collect(owner,a,9),bb=collect(owner,b,9);const std::array<size_t,5>d{10240,6144,48,48,2560};for(size_t i=0;i<5;i++)need(std::memcmp(aa[i].data(),bb[i].data(),T*d[i]*4)==0,"Decoder/prefill projection differs bitwise");need(owner.copy(a.input_packet,2880)==owner.copy(b.input_packet,2880),"Input Q81 packet differs");need(owner.copy(a.output_packet,6912)==owner.copy(b.output_packet,6912),"Output Q81 packet differs");return bb;};
 for(int T:{1,7,8,9}){decoder(w,X,Y,a,T,&q);candidate(w,X,Y,b,T,&q);q.wait_and_throw();check(T);std::cout<<"PF_GDN_DIRECT T="<<T<<" fields=5 bitwise=1\n";}
 // Source-extracted post kernel: extra F32 store preserves original half outputs.
 auto*G=reinterpret_cast<float*>(owner.alloc(128*4));owner.upload(G,gamma.data(),128*4);auto*post=reinterpret_cast<float*>(owner.alloc(y.size()*4));auto*h0=reinterpret_cast<uint16_t*>(owner.alloc(9*6144*2));auto*h1=reinterpret_cast<uint16_t*>(owner.alloc(9*6144*2));owner.upload(post,y.data(),y.size()*4);pf_post::launch<false>(q,b.z,G,1e-6f,Y,h0,9);pf_post::launch<true>(q,b.z,G,1e-6f,post,h1,9);q.wait_and_throw();need(owner.copy(h0,9*6144*2)==owner.copy(h1,9*6144*2),"Added F32 post store altered original half output");need(owner.copy(Y,y.size()*4)!=owner.copy(post,y.size()*4),"F32 post store did not publish normalized/gated value");std::cout<<"PF_GDN_POST_STORE half_bitwise=1 F32_published=1 decoder_recurrence_parity=0\n";
 auto*ga=owner.record([&]{decoder(w,X,Y,a,9,&q);});auto*gb=owner.record([&]{candidate(w,X,Y,b,9,&q);});q.ext_oneapi_graph(*ga);q.ext_oneapi_graph(*gb);q.wait_and_throw();auto before=check(9);for(float&v:x)v=-v+0.0078125f;for(float&v:y)v=-v-0.015625f;owner.upload(X,x.data(),x.size()*4);owner.upload(Y,y.data(),y.size()*4);q.ext_oneapi_graph(*ga);q.ext_oneapi_graph(*gb);q.wait_and_throw();auto after=check(9);need(before!=after,"Changed input graph did not read new owner inputs");for(size_t i=0;i<7;i++){auto bytes=owner.copy(wp[i],sizes[i]);need(bytes==weights[i],"Original packed weights changed");}owner.copy(X,x.size()*4);owner.copy(Y,y.size()*4);owner.copy(a.scratch,pf::scratch_bytes);owner.copy(b.scratch,pf::scratch_bytes);
 std::cout<<"PF_GDN_OPERATOR_PASS original_weights=1 synthetic_F32_operands=1 direct_cases=4 graphs=2 changed_input=1 bitwise_fields=5 last_row_packets=2 guards=1 conv_recurrence_qualified=0 whole_model_qualified=0\n";return 0;
 }catch(const std::exception&e){std::cerr<<"PF_GDN_OPERATOR_FAIL "<<e.what()<<"\n";return 1;}}
