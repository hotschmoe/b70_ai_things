// Fixed independently owned inputs; unchanged linked public source43 quantizer.
#define main own_q81_fixture_utilities
#include "hc_composition_arithmetic35_gpu_v1.cpp"
#undef main
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include <cstring>
using namespace strata::kernels;
class OwnQ81MismatchShadow;
void q81_bytes(sycl::queue&q,const fs::path&p,const void*ptr,size_t n){if(fs::exists(p))throw std::runtime_error("exclusive output required");std::vector<uint8_t>b(n);if(n)q.memcpy(b.data(),ptr,n).wait_and_throw();std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.data()),n);f.close();if(!f)throw std::runtime_error("output failed");}
void q81_host(const fs::path&p,const std::vector<uint8_t>&b){if(fs::exists(p))throw std::runtime_error("exclusive echo required");std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.data()),b.size());f.close();if(!f)throw std::runtime_error("echo failed");}
void q81_maps(const fs::path&p){if(fs::exists(p))throw std::runtime_error("exclusive maps required");std::ifstream i("/proc/self/maps");std::ofstream o(p);o<<i.rdbuf();o.close();if(!i||!o)throw std::runtime_error("maps failed");}
int main(int argc,char**argv)try{
 if(argc!=7||std::string(argv[1])!="--inputs"||std::string(argv[3])!="--output"||std::string(argv[5])!="--maps-prefix")throw std::runtime_error("exact CLI required");
 fs::path inputs(argv[2]),out(argv[4]);if(fs::is_symlink(inputs)||fs::exists(out))throw std::runtime_error("regular inputs/new outputs required");fs::create_directories(out);
 auto config=load(inputs/"config.txt",64);std::string text(config.begin(),config.end());int cols=0;
 std::string mixed="OWN_Q81_43_V1 rows=4 cols=2560 role=hc_mixed ids=own4\n";mixed.resize(64,' ');std::string gated="OWN_Q81_43_V1 rows=4 cols=6144 role=output_gated ids=own4\n";gated.resize(64,' ');
 if(text==mixed)cols=2560;
 if(text==gated)cols=6144;
 if(!cols)throw std::runtime_error("exact role/geometry config required");
 const size_t count=size_t(cols)*4,blocks=count/32,packet_bytes=blocks*36;auto raw=load(inputs/"own-input.f32",count*4),expected=load(inputs/"cpu-native.q81",packet_bytes);
 for(size_t i=0;i<count;++i){float x;std::memcpy(&x,raw.data()+i*4,4);if(!std::isfinite(x))throw std::runtime_error("finite original raw required");}
 auto&q=dpct::get_in_order_queue();const char*a=std::getenv("ZE_AFFINITY_MASK"),*s=std::getenv("ONEAPI_DEVICE_SELECTOR");if(!a||std::string(a)!="0"||!s||std::string(s)!="level_zero:gpu"||!q.get_device().is_gpu()||q.get_backend()!=sycl::backend::ext_oneapi_level_zero||!q.has_property<sycl::property::queue::in_order>()||!q.get_device().has(sycl::aspect::fp16))throw std::runtime_error("card0 LevelZero in_order fp16 required");
 for(const char*k:{"STRATA_VERIFY_EAGER","STRATA_ROPE_TABLE","STRATA_NO_NORM_ROPE","STRATA_ATTN_LANECELL"})if(std::getenv(k))throw std::runtime_error("alternative presence flag refused");
 auto vendor=q.get_device().get_info<sycl::info::device::vendor>(),driver=q.get_device().get_info<sycl::info::device::driver_version>(),name=q.get_device().get_info<sycl::info::device::name>();auto fp=q.get_device().get_info<sycl::info::device::single_fp_config>();std::printf("OWNQ81_DEVICE backend=level_zero affinity=0 fp16=1 vendor=%s driver=%s name=%s\n",vendor.c_str(),driver.c_str(),name.c_str());std::printf("OWNQ81_FP_CONFIG flags=");for(auto f:fp)std::printf("%llu,",static_cast<unsigned long long>(f));std::printf(" observed_device_flags_only=1\n");std::printf("OWNQ81_CONFIG rows=4 cols=%d source_function_unchanged=1 shadow_not_internal=1\n",cols);
 q81_host(out/"consumed-raw.f32",raw);q81_host(out/"consumed-cpu.q81",expected);Owned mem(q);auto*x=mem.floats(count);auto*y=static_cast<uint8_t*>(mem.make(packet_bytes));auto*selected=static_cast<int32_t*>(mem.make(blocks*4));auto*shadow=mem.floats(blocks*35);GraphOwner graph(q);q81_maps(std::string(argv[6])+".before");
 auto reset=[&](){q.memcpy(x,raw.data(),raw.size());q.memset(y,0,packet_bytes);q.memset(shadow,0,blocks*35*4).wait_and_throw();};auto body=[&](){native_quantize_q8_1(x,y,cols,4,&q);};
 for(int route=0;route<3;++route){reset();auto frame=out/("route"+std::to_string(route));fs::create_directory(frame);q81_bytes(q,frame/"packet_before.q81",y,packet_bytes);
  if(route==0){body();q.wait_and_throw();}else{if(route==1){dpct::experimental::begin_recording(&q);graph.recording=true;body();dpct::experimental::end_recording(&q,&graph.graph);graph.recording=false;if(!graph.graph)throw std::runtime_error("graph missing");graph.executable=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph.graph->finalize());delete graph.graph;graph.graph=nullptr;}q.ext_oneapi_graph(*graph.executable).wait_and_throw();}
  std::vector<uint8_t> actual(packet_bytes);q.memcpy(actual.data(),y,packet_bytes).wait_and_throw();std::vector<int32_t> ids;for(size_t b=0;b<blocks;++b)if(std::memcmp(actual.data()+b*36,expected.data()+b*36,36)!=0)ids.push_back(int32_t(b));const size_t mismatches=ids.size();
  if(mismatches){q.memcpy(selected,ids.data(),mismatches*4).wait_and_throw();q.parallel_for<OwnQ81MismatchShadow>(sycl::nd_range<1>(sycl::range<1>(mismatches*32),sycl::range<1>(32)),[=](sycl::nd_item<1>item)[[sycl::reqd_sub_group_size(32)]]{size_t b=item.get_group_linear_id(),lane=item.get_local_linear_id();float xi=x[size_t(selected[b])*32+lane],amax=sycl::fabs(xi),sum=xi;for(int o=16;o>0;o>>=1){amax=sycl::fmax(amax,dpct::experimental::permute_sub_group_by_xor(0xffffffffu,item.get_sub_group(),amax,o));sum+=dpct::experimental::permute_sub_group_by_xor(0xffffffffu,item.get_sub_group(),sum,o);}float d=q8_1_finite(amax/127.0f);if(!lane){shadow[b*35]=amax;shadow[b*35+1]=d;shadow[b*35+2]=sum;}shadow[b*35+3+lane]=amax==0?0:xi/d;}).wait_and_throw();}
  q81_bytes(q,frame/"input.f32",x,count*4);q81_bytes(q,frame/"packet.q81",y,packet_bytes);q81_bytes(q,frame/"mismatch_blocks.i32",selected,mismatches*4);q81_bytes(q,frame/"shadow.f32",shadow,mismatches*35*4);std::printf("OWNQ81_FRAME route=%d graph_replay=%d fields=5 mismatch_blocks=%zu input_restored=1 packet_zero_restored=1\n",route,route,mismatches);std::fflush(stdout);
 }
 q.wait_and_throw();delete graph.executable;graph.executable=nullptr;mem.release();q81_maps(std::string(argv[6])+".after");std::printf("OWNQ81_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 shadow_not_internal=1 full_model_math_qualified=0\n");std::fflush(stdout);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"OWNQ81_ERROR %s\n",e.what());return 2;}
