// Isolated current59/61 HcNorm source oracle. Never claims actual MAIN RMS data.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include "hc_native_composition.hpp"
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <array>
#include <cfenv>
#include <limits>
#include <stdexcept>
#include <vector>
#include "hc_rms3_generated_v2.inc"
namespace ex=sycl::ext::oneapi::experimental;
static void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static std::vector<float> read(const std::filesystem::path& path,size_t count){std::ifstream f(path,std::ios::binary);need(bool(f),"Original fixture input missing");std::vector<float> v(count);f.read(reinterpret_cast<char*>(v.data()),count*4);need(f.gcount()==std::streamsize(count*4)&&f.peek()==std::char_traits<char>::eof(),"Exact fixture input size required");for(float x:v)need(std::isfinite(x),"Nonfinite fixture input");return v;}
static void save(const std::filesystem::path& p,const std::vector<float>& v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*4);need(bool(f),"Oracle output write failed");}
struct Owner{
 sycl::queue& q;std::vector<float*> buffers;std::vector<ex::command_graph<ex::graph_state::modifiable>*> graphs;std::vector<ex::command_graph<ex::graph_state::executable>*> executables;ex::command_graph<ex::graph_state::modifiable>* active=nullptr;bool recording=false;
 explicit Owner(sycl::queue& Q):q(Q){buffers.reserve(10);graphs.reserve(3);executables.reserve(3);}
 [[noreturn]]static void retain(){std::cerr<<"HC_RMS3_RETAIN drain_failed=1\n";std::cerr.flush();std::_Exit(86);}
 ~Owner()noexcept{if(recording){try{active->end_recording(q);}catch(...){retain();}}try{q.wait_and_throw();}catch(...){retain();}for(auto* g:executables)delete g;for(auto* g:graphs)delete g;for(auto* b:buffers){try{sycl::free(b,q);}catch(...){retain();}}}
 float* allocate(size_t n){need(buffers.size()<10,"Owner buffer bound");auto* p=sycl::malloc_device<float>(n+32,q);if(!p)throw std::bad_alloc();buffers.push_back(p);q.fill(p,123456.75f,n+32).wait_and_throw();return p+16;}
 std::vector<float> copy(float* p,size_t n){std::vector<float> host(n+32);try{q.memcpy(host.data(),p-16,host.size()*4).wait_and_throw();}catch(...){retain();}for(size_t i=0;i<16;++i)need(host[i]==123456.75f&&host[n+16+i]==123456.75f,"Oracle guard overwritten");return {host.begin()+16,host.begin()+16+n};}
 ex::command_graph<ex::graph_state::executable>* record(const HcNativeArgs& a,float* ss,float* arg,int mode){active=new ex::command_graph<ex::graph_state::modifiable>(q.get_context(),q.get_device());graphs.push_back(active);recording=true;active->begin_recording(q);if(mode==2)explicit_sqrt_divide(q,a,ss,arg);else if(mode==1)observed(q,a,ss,arg);else baseline(q,a,ss,arg);active->end_recording(q);recording=false;auto* e=new ex::command_graph<ex::graph_state::executable>(active->finalize());executables.push_back(e);return e;}
};
static std::pair<std::vector<float>,std::array<float,4>> own_explicit_normalize(const std::vector<float>& r,const std::vector<float>& norm){
 need(std::numeric_limits<float>::is_iec559&&std::fegetround()==FE_TONEAREST,"Own host F32 RNE required");std::vector<float> xn(10240);std::array<float,4> rs{};
 for(size_t c=0;c<4;++c){std::array<float,32> lanes{};for(size_t lane=0;lane<32;++lane)for(size_t d=lane;d<2560;d+=32){float x=r[c*2560+d];lanes[lane]=std::fma(x,x,lanes[lane]);}for(unsigned mask=16;mask;mask>>=1){auto old=lanes;for(size_t lane=0;lane<32;++lane)lanes[lane]=old[lane]+old[lane^mask];}const float argument=lanes[0]/2560.0f+1e-6f;rs[c]=1.0f/std::sqrt(argument);for(size_t d=0;d<2560;++d)xn[c*2560+d]=(r[c*2560+d]*norm[c*2560+d])*rs[c];}
 return {xn,rs};
}
int main(int argc,char**argv){try{
 need(argc==3,"Usage: hc-rms3 ORIGINAL_INPUT_ROOT NEW_OUTPUT_ROOT");const std::filesystem::path input=argv[1],out=argv[2];need(!std::filesystem::exists(out),"New oracle output required");std::filesystem::create_directory(out);const char* enabled=std::getenv("B70_HC_RMS3_OBSERVE");need(enabled&&std::strcmp(enabled,"1")==0,"Explicit default-OFF isolated RMS observer optin required");
 const char* choice=std::getenv("B70_HC_RMS3_EXPLICIT_SQRT_DIVIDE");need(choice&&std::strcmp(choice,"1")==0,"Explicit default-OFF sqrt/divide candidate required");auto r=read(input/"own-residual_input.f32",10240),norm=read(input/"norm.f32",10240),native=read(input/"target-attn_hc_normalized.f32",10240);sycl::queue q(sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::in_order{}});need(q.get_backend()==sycl::backend::ext_oneapi_level_zero,"Original LevelZero backend required");const char* expected=std::getenv("B70_HC_EXPECTED_PCI");need(expected&&std::strlen(expected)==12,"Exact leased PCI required");
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
 need(q.get_device().has(sycl::aspect::ext_intel_pci_address),"PCI capability required");auto pci=q.get_device().get_info<sycl::ext::intel::info::device::pci_address>();need(pci==expected,"Selected GPU differs from leased card");std::cout<<"HC_RMS3_DEVICE pci="<<pci<<"\n";
#else
 need(false,"PCI query unavailable");
#endif
 Owner owner(q);auto* R=owner.allocate(10240);auto* W=owner.allocate(10240);auto* xn0=owner.allocate(10240);auto* rs0=owner.allocate(4);auto* xn1=owner.allocate(10240);auto* rs1=owner.allocate(4);auto* sum=owner.allocate(4);auto* arg=owner.allocate(4);auto* xn2=owner.allocate(10240);auto* rs2=owner.allocate(4);q.memcpy(R,r.data(),r.size()*4).wait_and_throw();q.memcpy(W,norm.data(),norm.size()*4).wait_and_throw();HcNativeArgs a;a.R=R;a.R_floats=10240;a.w.norm=W;a.w.norm_floats=10240;a.tokens=1;a.xn=xn0;a.xn_floats=10240;a.rs=rs0;a.rs_floats=4;auto b=a;b.xn=xn1;b.rs=rs1;auto candidate=a;candidate.xn=xn2;candidate.rs=rs2;
 baseline(q,a,sum,arg);observed(q,b,sum,arg);q.wait_and_throw();auto x0=owner.copy(xn0,10240),x1=owner.copy(xn1,10240),s0=owner.copy(rs0,4),s1=owner.copy(rs1,4),sums=owner.copy(sum,4),args=owner.copy(arg,4);save(out/"baseline-normalized.f32",x0);save(out/"observed-normalized.f32",x1);save(out/"native-square_sum.f32",sums);save(out/"native-rms_argument.f32",args);save(out/"native-rms_scale.f32",s1);need(std::memcmp(x0.data(),x1.data(),10240*4)==0&&std::memcmp(s0.data(),s1.data(),16)==0,"Observer altered operator outputs");const bool matches=std::memcmp(x0.data(),native.data(),10240*4)==0;std::cout<<"HC_RMS3_BEFORE_AFTER bitwise=1 saved_MAIN_normalized_match="<<matches<<" actual_MAIN_RMS_observed=0\n";
 explicit_sqrt_divide(q,candidate,sum,arg);q.wait_and_throw();auto x2=owner.copy(xn2,10240),s2=owner.copy(rs2,4);auto own_xn=read(input/"own-attn_hc_normalized.f32",10240),own_rs=read(input/"own-HC_rms_scale.f32",4);auto own_calc=own_explicit_normalize(r,norm);need(std::memcmp(own_calc.first.data(),own_xn.data(),40960)==0&&std::memcmp(own_calc.second.data(),own_rs.data(),16)==0,"Own independent explicit operations differ from executed CPU inputs");need(std::memcmp(x2.data(),own_calc.first.data(),40960)==0&&std::memcmp(s2.data(),own_calc.second.data(),16)==0,"Explicit device sqrt/divide differs from independent own operations");save(out/"explicit-sqrt-divide-normalized.f32",x2);save(out/"explicit-sqrt-divide-rms-scale.f32",s2);
 auto* g0=owner.record(a,sum,arg,0);auto* g1=owner.record(b,sum,arg,1);auto* g2=owner.record(candidate,sum,arg,2);q.ext_oneapi_graph(*g2).wait_and_throw();auto replay2=owner.copy(xn2,10240);need(std::memcmp(replay2.data(),x2.data(),40960)==0,"Explicit original graph replay differs");q.ext_oneapi_graph(*g0).wait_and_throw();q.ext_oneapi_graph(*g1).wait_and_throw();auto replay0=owner.copy(xn0,10240),replay1=owner.copy(xn1,10240);need(std::memcmp(replay0.data(),x0.data(),10240*4)==0&&std::memcmp(replay1.data(),x1.data(),10240*4)==0,"Original input graph replay differs");for(float& x:r)x=-x+0.03125f;q.memcpy(R,r.data(),r.size()*4).wait_and_throw();q.ext_oneapi_graph(*g0).wait_and_throw();q.ext_oneapi_graph(*g1).wait_and_throw();q.ext_oneapi_graph(*g2).wait_and_throw();auto changed2=owner.copy(xn2,10240);auto changed_expected=own_explicit_normalize(r,norm);need(std::memcmp(changed2.data(),changed_expected.first.data(),40960)==0,"Changed explicit graph differs from independent own operations");auto changed0=owner.copy(xn0,10240),changed1=owner.copy(xn1,10240);need(std::memcmp(changed0.data(),changed1.data(),10240*4)==0,"Changed input graph differs");need(std::memcmp(changed0.data(),x0.data(),10240*4)!=0,"Changed input graph did not read changed owner buffer");owner.copy(R,10240);owner.copy(W,10240);owner.copy(rs0,4);owner.copy(rs1,4);owner.copy(sum,4);owner.copy(arg,4);owner.copy(xn2,10240);owner.copy(rs2,4);need(matches,"Isolated operator normalized differs from saved MAIN; no intrinsic attribution authority");std::cout<<"HC_RMS3_EXPLICIT_OPERATOR_PASS original_inputs=1 baseline_observer_MAIN_match=1 explicit_own_normalized_RS_match=1 graphs=3 changed_input_bitwise=1 guards=1 whole48_qualified=0\n";return 0;
}catch(const std::exception& e){std::cerr<<"HC_RMS3_OPERATOR_FAIL "<<e.what()<<"\n";return 1;}}
