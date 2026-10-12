// Synthetic original native expert E2 control; separate OFF/ON processes, root GPU only.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include "strata/kernels/iq_kernels.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace k=strata::kernels;namespace ex=sycl::ext::oneapi::experimental;namespace fs=std::filesystem;
static void need(bool v,const char* s){if(!v)throw std::runtime_error(s);}
struct Owner {
 sycl::queue& q;std::vector<void*> memory;ex::command_graph<ex::graph_state::modifiable>* graph=nullptr;ex::command_graph<ex::graph_state::executable>* executable=nullptr;bool recording=false;
 explicit Owner(sycl::queue& x):q(x){memory.reserve(16);}
 [[noreturn]]static void retain(){std::cerr<<"IQ_PAIR_DRAIN_FAILED retained=1\n";std::cerr.flush();std::_Exit(86);}
 ~Owner()noexcept{if(recording){try{graph->end_recording(q);recording=false;}catch(...){retain();}}try{q.wait_and_throw();}catch(...){retain();}delete executable;delete graph;for(void* p:memory){try{sycl::free(p,q);}catch(...){std::cerr<<"IQ_PAIR_FREE_FAILED\n";std::cerr.flush();std::_Exit(87);}}}
 template<class T>T* alloc(size_t n,bool host=false){need(memory.size()<16,"owner cap");T* p=host?sycl::malloc_host<T>(n,q):sycl::malloc_device<T>(n,q);if(!p)throw std::bad_alloc();memory.push_back(p);return p;}
};
static std::vector<uint8_t> random_rows(int type,int rows,int n,std::mt19937& rng){
 const int qk=(type==7||type==8)?32:256;const size_t rb=k::iq_row_bytes(type,n),bs=k::iq_row_bytes(type,qk);need(rb && bs,"quant rows unsupported");std::vector<uint8_t> w(size_t(rows)*rb);std::uniform_int_distribution<int> byte(0,255),exponent(2,8),mantissa(0,1023),sign(0,1);for(auto& b:w)b=uint8_t(byte(rng));
 for(size_t at=0;at<w.size();at+=bs){int e=(type==12||type==13)?1+exponent(rng)%4:exponent(rng);uint16_t d=uint16_t((sign(rng)?0x8000:0)|(e<<10)|mantissa(rng));std::memcpy(w.data()+at,&d,2);if(type==12||type==13||type==7){int em=(type==7)?exponent(rng):1+exponent(rng)%4;uint16_t m=uint16_t((sign(rng)?0x8000:0)|(em<<10)|mantissa(rng));std::memcpy(w.data()+at+2,&m,2);}}
 return w;
}
struct Files {
 fs::path output,compare;
 void store(const std::string& name,const void* data,size_t bytes){
  if(!compare.empty()){fs::path p=compare/name;need(fs::is_regular_file(p)&&!fs::is_symlink(p)&&fs::file_size(p)==bytes,"OFF comparison file absent/size differs");std::vector<char> saved(bytes);std::ifstream in(p,std::ios::binary);in.exceptions(std::ios::failbit|std::ios::badbit);in.read(saved.data(),std::streamsize(bytes));need(std::memcmp(data,saved.data(),bytes)==0,"ON bytes differ from original OFF");}
  fs::path p=output/name;need(!fs::exists(p),"new file required");std::ofstream out(p,std::ios::binary);out.exceptions(std::ios::failbit|std::ios::badbit);out.write(static_cast<const char*>(data),std::streamsize(bytes));out.close();
 }
};
static std::vector<uint8_t> readback(sycl::queue& q,const void* p,size_t bytes){std::vector<uint8_t> out(bytes);try{q.memcpy(out.data(),p,bytes).wait_and_throw();}catch(...){Owner::retain();}return out;}
static void run(sycl::queue& q,Files& files,int gu,int down,bool pinned,int plan){
 constexpr int H=2560,FF=640,T=4,CAP=4;const auto L=k::native_expert_layout(gu,down,H,FF);need(k::native_expert_supported(gu,down,H,FF),"original expert layout unsupported");std::mt19937 rng(743+gu*31+down);
 // All source host backing vectors precede owner and survive its queue drain.
 std::vector<uint8_t> blobs;for(int expert=0;expert<2;++expert){auto b=random_rows(gu,2*FF,H,rng),d=random_rows(down,H,FF,rng);b.insert(b.end(),d.begin(),d.end());need(b.size()==L.bytes,"blob extent differs");blobs.insert(blobs.end(),b.begin(),b.end());}
 std::vector<float> input(size_t(T)*H);std::uniform_int_distribution<int> xdist(-2048,2048);for(auto& x:input)x=float(xdist(rng))/2048.f;
 std::vector<int32_t> starts,tokens={3,1,2,0},dest={3,0,2,1};std::vector<unsigned long long> ptr;int groups=0;
 if(plan==0||plan==1){starts={0,2};groups=1;if(plan==1)tokens={2,2,1,0};}
 else if(plan==2){starts={0,1,2};groups=2;}
 else if(plan==3){starts={0,3};groups=1;tokens={3,0,2,1};}
 else{need(plan==4,"plan differs");starts={0,1,3};groups=2;tokens={2,3,1,0};}
 const size_t scratch_bytes=k::native_expert_scratch_bytes(CAP,FF),input_q_bytes=size_t(T)*(H/32)*36,out_bytes=size_t(CAP)*H*4;
 Owner owner(q);auto* w=owner.alloc<uint8_t>(blobs.size(),pinned);auto* dx=owner.alloc<float>(input.size());auto* xq=owner.alloc<uint8_t>(input_q_bytes);auto* scratch=owner.alloc<uint8_t>(scratch_bytes);auto* out=owner.alloc<float>(size_t(CAP)*H);auto* gp=owner.alloc<unsigned long long>(2);auto* st=owner.alloc<int32_t>(3);auto* ng=owner.alloc<int32_t>(1);auto* tok=owner.alloc<int32_t>(CAP);auto* dst=owner.alloc<int32_t>(CAP);
 if(pinned)std::memcpy(w,blobs.data(),blobs.size());else q.memcpy(w,blobs.data(),blobs.size()).wait_and_throw();ptr={reinterpret_cast<unsigned long long>(w),reinterpret_cast<unsigned long long>(w+L.bytes)};
 q.memcpy(gp,ptr.data(),ptr.size()*8);q.memcpy(st,starts.data(),starts.size()*4);q.memcpy(ng,&groups,4);q.memcpy(tok,tokens.data(),tokens.size()*4);q.memcpy(dst,dest.data(),dest.size()*4);q.wait_and_throw();
 auto pack=[&]{q.memcpy(dx,input.data(),input.size()*4).wait_and_throw();k::quantize_q8_1_rows(dx,T,H,xq,&q);q.wait_and_throw();};
 auto reset=[&]{q.memset(scratch,0x3c,scratch_bytes);q.memset(out,0xa5,out_bytes);};
 auto invoke=[&]{k::native_expert_grouped(L,gp,st,ng,dst,tok,2,CAP,xq,scratch,out,&q,1);};
 const std::string prefix="g"+std::to_string(gu)+"d"+std::to_string(down)+"-"+(pinned?"host":"device")+"-p"+std::to_string(plan);
 auto snapshot=[&](const std::string& tag){auto s=readback(q,scratch,scratch_bytes),o=readback(q,out,out_bytes),x=readback(q,xq,input_q_bytes);for(size_t e=0;e<size_t(groups?starts.back():0);++e){float value;for(int r=0;r<H;++r){std::memcpy(&value,o.data()+(size_t(dest[e])*H+r)*4,4);need(std::isfinite(value),"nonfinite native output");}}files.store(prefix+"-"+tag+"-scratch.bin",s.data(),s.size());files.store(prefix+"-"+tag+"-output.bin",o.data(),o.size());files.store(prefix+"-"+tag+"-xq.bin",x.data(),x.size());return std::make_pair(std::move(s),std::move(o));};
 pack();reset();k::native_expert_set_mode(-1,0);invoke();q.wait_and_throw();auto original=snapshot("initial");
 if(plan==0 && !pinned){files.store("g"+std::to_string(gu)+"d"+std::to_string(down)+"-original-blobs.bin",blobs.data(),blobs.size());files.store("g"+std::to_string(gu)+"d"+std::to_string(down)+"-original-input.bin",input.data(),input.size()*4);}
 if(plan==0){for(int phase:{1,2}){k::native_expert_set_mode(-1,phase);for(int warm=0;warm<3;++warm){invoke();q.wait_and_throw();}for(int repeat=0;repeat<7;++repeat){auto begin=std::chrono::steady_clock::now();invoke();q.wait_and_throw();auto end=std::chrono::steady_clock::now();std::cout<<"IQ_PAIR_PHASE_LATENCY gu="<<gu<<" down="<<down<<" pinned="<<pinned<<" phase="<<phase<<" repeat="<<repeat<<" host_submit_wait_us="<<std::chrono::duration<double,std::micro>(end-begin).count()<<" device_time_qualified=0\n";}}k::native_expert_set_mode(-1,0);}
 reset();q.wait_and_throw();owner.graph=new ex::command_graph<ex::graph_state::modifiable>(q.get_context(),q.get_device());owner.recording=true;owner.graph->begin_recording(q);invoke();owner.graph->end_recording(q);owner.recording=false;owner.executable=new ex::command_graph<ex::graph_state::executable>(owner.graph->finalize());q.ext_oneapi_graph(*owner.executable).wait_and_throw();auto gs=readback(q,scratch,scratch_bytes),go=readback(q,out,out_bytes);need(gs==original.first&&go==original.second,"same-input graph differs from direct call");
 for(float& x:input)x=-x+.125f;pack();reset();invoke();q.wait_and_throw();auto direct=snapshot("changed");reset();q.ext_oneapi_graph(*owner.executable).wait_and_throw();gs=readback(q,scratch,scratch_bytes);go=readback(q,out,out_bytes);need(gs==direct.first&&go==direct.second,"changed-input graph differs from direct call");
 std::cout<<"IQ_PAIR_CASE gu="<<gu<<" down="<<down<<" pinned="<<pinned<<" plan="<<plan<<" groups="<<groups<<" cap_entries="<<CAP<<" bytes_per_expert="<<L.bytes<<" scratch_bytes="<<scratch_bytes<<" xq_bytes="<<input_q_bytes<<" graph_changed_input=1\n";
}
int main(int argc,char** argv){try{need(argc==4||argc==6,"args: FLAG OUTPUT [--compare OFFDIR]");const char* flag=std::getenv("STRATA_EXPERT_PAIR_REUSE");need(flag&&std::strlen(flag)==1&&(flag[0]=='0'||flag[0]=='1')&&std::string(argv[1])==flag,"actual cached flag/CLI differ");need(std::string(argv[2])=="--output","--output required");Files files{fs::absolute(argv[3]),{}};need(!fs::exists(files.output)&&files.output.parent_path()==fs::canonical(files.output.parent_path()),"new canonical output required");if(argc==6){need(std::string(argv[4])=="--compare","--compare required");files.compare=fs::absolute(argv[5]);need(files.compare==fs::canonical(files.compare)&&fs::is_directory(files.compare)&&flag[0]=='1',"original OFF directory required for ON");}else need(flag[0]=='0',"ON must compare original OFF");fs::create_directory(files.output);
 sycl::queue q(sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::in_order{}});need(q.get_device().has(sycl::aspect::usm_host_allocations),"pinned host USM unsupported");const char* expected=std::getenv("B70_IQ_EXPECTED_PCI");need(expected&&q.get_backend()==sycl::backend::ext_oneapi_level_zero,"expected LevelZero PCI required");
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
 need(q.get_device().has(sycl::aspect::ext_intel_pci_address),"PCI query unsupported");auto actual=q.get_device().get_info<sycl::ext::intel::info::device::pci_address>();std::cout<<"IQ_PAIR_DEVICE pci_address="<<actual<<" expected="<<expected<<" flag="<<flag<<"\n";std::cout.flush();need(actual==expected,"wrong original GPU selected");
#else
 need(false,"PCI query extension unavailable");
#endif
 k::iq_set_old_kernels(false);k::native_grouped_set_v1(false);for(auto pair:{std::pair<int,int>{12,7},{12,8},{13,7},{13,8}})for(bool pinned:{false,true})for(int plan=0;plan<5;++plan)run(q,files,pair.first,pair.second,pinned,plan);q.wait_and_throw();std::cout<<"IQ_PAIR_FIXTURE_PASS cases=40 changed_graphs=40 flag="<<flag<<" original_OFF_bitwise="<<(flag[0]=='1')<<" model_payload=0\n";return 0;}catch(const std::exception& e){std::cerr<<"IQ_PAIR_FIXTURE_FAIL "<<e.what()<<"\n";return 1;}}
