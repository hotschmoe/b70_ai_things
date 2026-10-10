// Synthetic actual compiled HC composition plus explicitly separate shadows.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "strata/kernels/hc_native_composition.hpp"
#include "strata/kernels/hc_native_projection.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
constexpr size_t N=2560,H=4,D=N*H,L=320;
std::vector<uint8_t> load(const fs::path& path,size_t bytes) {
 if(fs::is_symlink(path)||fs::file_size(path)!=bytes)throw std::runtime_error("synthetic input extent/symlink differs");
 std::ifstream f(path,std::ios::binary);std::vector<uint8_t> raw(bytes);f.read(reinterpret_cast<char*>(raw.data()),std::streamsize(bytes));if(!f||f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("synthetic short/changed input");return raw;
}
struct Owned {
 sycl::queue& q;std::vector<void*> ptrs;explicit Owned(sycl::queue& queue):q(queue){}
 void* make(size_t n){void* p=sycl::malloc_device(n,q);if(!p)throw std::runtime_error("USM allocation failed");ptrs.push_back(p);return p;}
 float* floats(size_t n){return static_cast<float*>(make(n*4));}
 void* input(const fs::path& p,size_t n){auto v=load(p,n);void* d=make(n);q.memcpy(d,v.data(),n).wait_and_throw();return d;}
 void release(){q.wait_and_throw();for(void* p:ptrs)sycl::free(p,q);ptrs.clear();}
 ~Owned(){try{release();}catch(...){}}
};
struct GraphOwner {
 sycl::queue& q;dpct::experimental::command_graph_ptr graph=nullptr;dpct::experimental::command_graph_exec_ptr executable=nullptr;bool recording=false;
 explicit GraphOwner(sycl::queue& queue):q(queue){}
 ~GraphOwner(){try{if(recording)dpct::experimental::end_recording(&q,&graph);q.wait_and_throw();}catch(...){}delete executable;delete graph;}
};
void dump(sycl::queue& q,const fs::path& dir,const std::string& field,const float* ptr,size_t n) {
 std::vector<float> v(n);q.memcpy(v.data(),ptr,n*4).wait_and_throw();auto p=dir/(field+".f32");if(fs::exists(p))throw std::runtime_error("new raw output required");std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(n*4));f.close();if(!f)throw std::runtime_error("raw write failed");
}
class NormShadow;class IntrinsicShadow;class MixShadow;class FPControlShadow;
void shadows(sycl::queue& q,const strata::kernels::HcNativeArgs& a,const float* residual,float* sums,float* args,float* rs,float* xn,float* pre,float* exp_low,float* silu,float* exp_gate,float* mix_sep,float* mix_fma,const float* fp_inputs,float* fp_outputs) {
 const size_t T=size_t(a.tokens);
 q.parallel_for<NormShadow>(sycl::nd_range<1>(sycl::range<1>(T*H*32),sycl::range<1>(32)),[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
  const size_t tc=item.get_group_linear_id(),t=tc/H,c=tc%H,lane=item.get_local_linear_id();float sum=0;
  for(size_t d=lane;d<N;d+=32){const float r=residual[t*D+c*N+d];sum=sycl::fma(r,r,sum);}
  for(unsigned mask=16;mask;mask>>=1)sum+=sycl::permute_group_by_xor(item.get_sub_group(),sum,mask);
  const float arg=sum/float(N)+a.eps,s=sycl::rsqrt(arg);if(!lane){sums[tc]=sum;args[tc]=arg;rs[tc]=s;}
  for(size_t d=lane;d<N;d+=32){const size_t i=c*N+d;xn[t*D+i]=(residual[t*D+i]*a.w.norm[i])*s;}
 });
 // PreSiLU is a SECOND actual compiled projection of actual synthetic xn.
 if(!strata::kernels::hc_q8_0_project_f32(a.w.down,a.w.down_bytes,a.xn,T*D,pre,T*L,D,L,a.tokens,&q))throw std::runtime_error("secondary preSiLU projection rejected");
 q.parallel_for<IntrinsicShadow>(sycl::range<1>(T*D),[=](sycl::id<1> id){const size_t i=id[0];exp_gate[i]=sycl::exp(-a.gate[i]);if(i<T*L){const float x=pre[i]/4.0f,e=sycl::exp(-x);exp_low[i]=e;silu[i]=x*(1.0f/(1.0f+e));}});
 q.parallel_for<MixShadow>(sycl::range<1>(T*N),[=](sycl::id<1> id){const size_t td=id[0],t=td/N,d=td%N;float separate=0,fused=0;
  for(size_t c=0;c<H;++c){const size_t i=t*D+c*N+d;const float s=1.0f/(1.0f+exp_gate[i]);const float product=a.xn[i]*s;
   // Volatile materializes the product: do not infer compiler contraction from this shadow.
   volatile float stored=product;separate+=stored;fused=sycl::fma(a.xn[i],s,fused);}
  mix_sep[td]=separate/4.0f;mix_fma[td]=fused/4.0f;
 });
 // Runtime USM inputs prevent constant folding. Capabilities are not FP mode.
 q.parallel_for<FPControlShadow>(sycl::range<1>(T*2),[=](sycl::id<1> id){const size_t t=id[0]/2,j=id[0]%2;fp_outputs[id[0]]=sycl::fma(fp_inputs[t*4+j*2],fp_inputs[t*4+j*2+1],0.0f);});
}
int main(int argc,char** argv) try {
 if(argc!=5||std::string(argv[1])!="--inputs"||std::string(argv[3])!="--output")throw std::runtime_error("leaf --inputs DIR --output NEWDIR");
 fs::path inputs(argv[2]),out(argv[4]);if(fs::is_symlink(inputs)||fs::exists(out))throw std::runtime_error("new output and nonsymlink inputs required");fs::create_directories(out);
 auto& q=dpct::get_in_order_queue();const char* affinity=std::getenv("ZE_AFFINITY_MASK"),*selector=std::getenv("ONEAPI_DEVICE_SELECTOR");
 if(!affinity||std::string(affinity)!="0"||!selector||std::string(selector)!="level_zero:gpu"||std::getenv("STRATA_VERIFY_EAGER")||q.get_backend()!=sycl::backend::ext_oneapi_level_zero||!q.has_property<sycl::property::queue::in_order>())throw std::runtime_error("exact physicalcard0 LevelZero in_order/EAGERabsent required");
 std::printf("HC35_COMPOSITION_DEVICE backend=level_zero vendor=%s driver=%s affinity=0 selector=level_zero:gpu\n",q.get_device().get_info<sycl::info::device::vendor>().c_str(),q.get_device().get_info<sycl::info::device::driver_version>().c_str());
 std::printf("HC35_COMPOSITION_FP_CONFIG flags=");for(auto flag:q.get_device().get_info<sycl::info::device::single_fp_config>())std::printf("%llu,",static_cast<unsigned long long>(flag));std::printf(" observed_device_flags_only=1 compiler_lowering_unobserved=1\n");
 std::printf("HC35_COMPOSITION_CONFIG synthetic=1 compiled_composition=1 shadows_separate=1 subgroup=32 eps=1e-6 graph_and_queue=1 normal_model_graph_qualified=0 device_intrinsics_qualified=0 model_math_qualified=0\n");
 std::ifstream manifest(inputs/"cases.tsv");std::string header;std::getline(manifest,header);if(header!="HC35_COMPOSITION_INPUTS_V1")throw std::runtime_error("manifest header differs");
 size_t cases=0,frames=0;std::set<std::string> seen;std::string id;int tokens=0;
 while(manifest>>id>>tokens){if((tokens!=1&&tokens!=2)||id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-")!=std::string::npos||!seen.insert(id).second||cases>=4)throw std::runtime_error("exact synthetic T1/T2 case roster differs");const size_t T=size_t(tokens);Owned mem(q);auto base=inputs/id;
  auto host_input=load(base.string()+".residual.f32",T*D*4);auto* R=mem.floats(T*D);auto* R_out=mem.floats(T*D);auto* standalone=mem.floats(T*D);auto* bo=static_cast<float*>(mem.input(base.string()+".bo.f32",T*N*4));auto* previous=static_cast<float*>(mem.input(base.string()+".previous.f32",T*H*4));
  strata::kernels::HcNativeArgs a;a.tokens=tokens;a.w.norm=static_cast<float*>(mem.input(base.string()+".norm.f32",D*4));a.w.norm_floats=D;a.w.down=static_cast<uint8_t*>(mem.input(base.string()+".down.q8_0",L*(D/32)*34));a.w.down_bytes=L*(D/32)*34;a.w.up=static_cast<uint8_t*>(mem.input(base.string()+".up.q8_0",D*(L/32)*34));a.w.up_bytes=D*(L/32)*34;a.w.inject=static_cast<float*>(mem.input(base.string()+".inject.f32",H*D*4));a.w.inject_floats=H*D;
  a.R=R;a.R_floats=T*D;a.xn=mem.floats(T*D);a.xn_floats=T*D;a.lo=mem.floats(T*L);a.lo_floats=T*L;a.gate=mem.floats(T*D);a.gate_floats=T*D;a.rs=mem.floats(T*H);a.rs_floats=T*H;a.mixed=mem.floats(T*N);a.mixed_floats=T*N;a.inject_out=mem.floats(T*H);a.inject_out_floats=T*H;
  auto* sum=mem.floats(T*H);auto* argument=mem.floats(T*H);auto* rs=mem.floats(T*H);auto* xn=mem.floats(T*D);auto* pre=mem.floats(T*L);auto* exp_low=mem.floats(T*L);auto* silu=mem.floats(T*L);auto* exp_gate=mem.floats(T*D);auto* mix_sep=mem.floats(T*N);auto* mix_fma=mem.floats(T*N);
  auto* fp_inputs=static_cast<float*>(mem.input(base.string()+".fp_control.f32",T*4*4));auto* fp_outputs=mem.floats(T*2);
  for(int apply=0;apply<3;++apply){a.apply=apply!=0;a.bo_prev=a.apply?bo:nullptr;a.bo_floats=a.apply?T*N:0;a.inj_prev=a.apply?previous:nullptr;a.prev_floats=a.apply?T*H:0;a.R_out=a.apply?(apply==2?R:R_out):nullptr;a.R_out_floats=a.apply?T*D:0;
   auto body=[&](){if(!strata::kernels::hc_native_read_f32(a,&q)||!strata::kernels::hc_native_write_f32(standalone,T*D,bo,T*N,previous,T*H,tokens,&q))throw std::runtime_error("actual compiled HC descriptor rejected");};
   GraphOwner graphs(q);
   for(int route=0;route<3;++route){q.memcpy(R,host_input.data(),host_input.size());q.memcpy(standalone,host_input.data(),host_input.size()).wait_and_throw();
    if(route==0){body();q.wait_and_throw();}else{if(route==1){dpct::experimental::begin_recording(&q);graphs.recording=true;body();dpct::experimental::end_recording(&q,&graphs.graph);graphs.recording=false;if(!graphs.graph)throw std::runtime_error("actual HC graph absent");graphs.executable=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graphs.graph->finalize());delete graphs.graph;graphs.graph=nullptr;}q.ext_oneapi_graph(*graphs.executable).wait_and_throw();}
    const std::string frame=id+"-a"+std::to_string(apply)+"-r"+std::to_string(route);auto directory=out/frame;fs::create_directory(directory);const float* residual=a.apply?a.R_out:a.R;
    shadows(q,a,residual,sum,argument,rs,xn,pre,exp_low,silu,exp_gate,mix_sep,mix_fma,fp_inputs,fp_outputs);
    for(auto pair:std::vector<std::pair<std::string,const float*>>{{"rs",a.rs},{"inject",a.inject_out},{"norm_square_sum_shadow",sum},{"norm_argument_shadow",argument},{"rs_shadow",rs}})dump(q,directory,pair.first,pair.second,T*H);
    for(auto pair:std::vector<std::pair<std::string,const float*>>{{"xn",a.xn},{"gate",a.gate},{"residual_after_pending",residual},{"standalone_write",standalone},{"xn_shadow",xn},{"exp_gate_shadow",exp_gate}})dump(q,directory,pair.first,pair.second,T*D);
    for(auto pair:std::vector<std::pair<std::string,const float*>>{{"post_silu_low",a.lo},{"pre_silu_secondary_projection",pre},{"exp_low_shadow",exp_low},{"silu_shadow",silu}})dump(q,directory,pair.first,pair.second,T*L);
    for(auto pair:std::vector<std::pair<std::string,const float*>>{{"mixed",a.mixed},{"mix_separate_shadow",mix_sep},{"mix_fma_shadow",mix_fma}})dump(q,directory,pair.first,pair.second,T*N);
    dump(q,directory,"fp_control_shadow",fp_outputs,T*2);
    std::printf("HC35_COMPOSITION_FRAME case=%s tokens=%d apply=%d alias=%d route=%d graph_replay=%d fields=19\n",id.c_str(),tokens,apply!=0,apply==2,route,route);++frames;
   }
   q.wait_and_throw(); // GraphOwner retires graphs before any owning USM is freed.
  }
  mem.release();++cases;
 }
 if(!manifest.eof()||cases!=4||frames!=36)throw std::runtime_error("complete4cases/36frames required");
 std::printf("HC35_COMPOSITION_RESULT cases=4 frames=36 execution_completed=1 all_owned_allocations_freed=1 numerical_comparison_done=0 device_intrinsics_qualified=0 normal_model_graph_qualified=0 model_math_qualified=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HC35_COMPOSITION_ERROR %s\n",e.what());return 2;}
