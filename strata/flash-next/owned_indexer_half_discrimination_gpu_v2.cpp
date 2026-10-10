// Diagnostic only: exact source37 expression versus an observable F16 store/load.
#define main half37_utility_fixture
#include "hc_composition_arithmetic35_gpu_v1.cpp"
#undef main
#include <cstring>
constexpr size_t HALF37_N=512;
void half37_maps(const fs::path& p){if(fs::exists(p))throw std::runtime_error("exclusive mapped library output required");std::ifstream in("/proc/self/maps");std::ofstream out(p);out<<in.rdbuf();out.close();if(!in||!out)throw std::runtime_error("maps write failed");}
class Half37Expression;class Half37Store;class Half37Load;
void half37_bytes(sycl::queue& q,const fs::path& path,const void* ptr,size_t bytes){
 if(fs::exists(path))throw std::runtime_error("exclusive output required");std::vector<uint8_t> raw(bytes);q.memcpy(raw.data(),ptr,bytes).wait_and_throw();std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(raw.data()),bytes);f.close();if(!f)throw std::runtime_error("raw output failed");
}
int main(int argc,char** argv) try {
 if(argc!=7||std::string(argv[1])!="--raw"||std::string(argv[3])!="--output"||std::string(argv[5])!="--maps-prefix")throw std::runtime_error("exact own raw/output CLI required");
 auto host=load(argv[2],HALF37_N*4);fs::path out(argv[4]);if(fs::exists(out))throw std::runtime_error("new output required");fs::create_directories(out);
 for(size_t i=0;i<HALF37_N;++i){float x;std::memcpy(&x,host.data()+i*4,4);if(!std::isfinite(x)||std::abs(x)>65504.0f)throw std::runtime_error("finite bounded own F32 values required");}
 std::ofstream echo(out/"consumed-raw.f32",std::ios::binary);echo.write(reinterpret_cast<const char*>(host.data()),host.size());echo.close();if(!echo)throw std::runtime_error("own input echo failed");
 auto& q=dpct::get_in_order_queue();const char* affinity=std::getenv("ZE_AFFINITY_MASK"),*selector=std::getenv("ONEAPI_DEVICE_SELECTOR");
 if(!affinity||std::string(affinity)!="0"||!selector||std::string(selector)!="level_zero:gpu"||!q.get_device().is_gpu()||q.get_backend()!=sycl::backend::ext_oneapi_level_zero||!q.has_property<sycl::property::queue::in_order>()||!q.get_device().has(sycl::aspect::fp16))throw std::runtime_error("card0 LevelZero in-order fp16 required");
 std::printf("HALF37_DEVICE backend=level_zero affinity=0 fp16=1 vendor=%s driver=%s name=%s\n",q.get_device().get_info<sycl::info::device::vendor>().c_str(),q.get_device().get_info<sycl::info::device::driver_version>().c_str(),q.get_device().get_info<sycl::info::device::name>().c_str());
 half37_maps(std::string(argv[6])+".before");std::printf("HALF37_FP_CONFIG flags=");for(auto flag:q.get_device().get_info<sycl::info::device::single_fp_config>())std::printf("%llu,",static_cast<unsigned long long>(flag));std::printf(" observed_device_flags_only=1 compiler_lowering_unobserved=1\n");
 Owned mem(q);auto* raw=mem.floats(HALF37_N);auto* expr=mem.floats(HALF37_N);auto* widened=mem.floats(HALF37_N);auto* half=static_cast<sycl::half*>(mem.make(HALF37_N*2));
 auto reset=[&](){q.memcpy(raw,host.data(),host.size());q.memset(expr,0,HALF37_N*4);q.memset(widened,0,HALF37_N*4);q.memset(half,0,HALF37_N*2).wait_and_throw();};
 auto body=[&](){
  q.parallel_for<Half37Expression>(sycl::range<1>(HALF37_N),[=](sycl::id<1> id){const size_t i=id[0];expr[i]=sycl::vec<sycl::half,1>(sycl::vec<float,1>(raw[i]).convert<sycl::half,sycl::rounding_mode::rte>()[0]).convert<float,sycl::rounding_mode::automatic>()[0];});
  q.parallel_for<Half37Store>(sycl::range<1>(HALF37_N),[=](sycl::id<1> id){const size_t i=id[0];half[i]=sycl::vec<float,1>(raw[i]).convert<sycl::half,sycl::rounding_mode::rte>()[0];});
  q.parallel_for<Half37Load>(sycl::range<1>(HALF37_N),[=](sycl::id<1> id){const size_t i=id[0];widened[i]=sycl::vec<sycl::half,1>(half[i]).convert<float,sycl::rounding_mode::automatic>()[0];});
 };GraphOwner graph(q);
 for(int route=0;route<3;++route){reset();if(route==0){body();q.wait_and_throw();}else{if(route==1){dpct::experimental::begin_recording(&q);graph.recording=true;body();dpct::experimental::end_recording(&q,&graph.graph);graph.recording=false;if(!graph.graph)throw std::runtime_error("graph missing");graph.executable=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph.graph->finalize());delete graph.graph;graph.graph=nullptr;}q.ext_oneapi_graph(*graph.executable).wait_and_throw();}
  auto frame=out/("route"+std::to_string(route));fs::create_directory(frame);half37_bytes(q,frame/"expression.f32",expr,HALF37_N*4);half37_bytes(q,frame/"materialized.f32",widened,HALF37_N*4);half37_bytes(q,frame/"materialized.f16",half,HALF37_N*2);half37_bytes(q,frame/"input.f32",raw,HALF37_N*4);
  std::printf("HALF37_FRAME route=%d graph_replay=%d fields=4 own_input_restored=1 values=512\n",route,route);std::fflush(stdout);
 }
 q.wait_and_throw();delete graph.executable;graph.executable=nullptr;mem.release();half37_maps(std::string(argv[6])+".after");std::printf("HALF37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 compiler_lowering_qualified=0 full_model_math_qualified=0\n");std::fflush(stdout);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HALF37_ERROR %s\n",e.what());return 2;}
