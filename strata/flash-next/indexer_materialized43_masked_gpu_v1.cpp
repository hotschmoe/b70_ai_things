// Linked source43 materializer control. Only independently owned raw/gamma inputs.
#define main indexer43_fixture_utilities
#include "hc_composition_arithmetic35_gpu_v1.cpp"
#undef main
#include "strata/kernels/native_qsa_indexer.hpp"
#include <array>
#include <cstring>
using namespace strata::kernels;
constexpr size_t I43_STATE=3076;
void i43_dump(sycl::queue& q,const fs::path& path,const void* p,size_t n){
 if(fs::exists(path))throw std::runtime_error("exclusive output required");std::vector<uint8_t>b(n);q.memcpy(b.data(),p,n).wait_and_throw();std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(b.data()),n);f.close();if(!f)throw std::runtime_error("dump failed");
}
int main(int argc,char**argv)try{
 if(argc!=9||std::string(argv[1])!="--inputs"||std::string(argv[3])!="--output"||std::string(argv[5])!="--mode"||std::string(argv[7])!="--maps-prefix")throw std::runtime_error("exact masked43 CLI required");
 const int mode=std::string(argv[6])=="0"?0:std::string(argv[6])=="1"?1:-1;const char*env=std::getenv("STRATA_QSA_INDEXER_MATERIALIZED_HALF");if(mode<0||!env||std::string(env)!=argv[6])throw std::runtime_error("exact separate-process resolved mode required");
 fs::path input(argv[2]),out(argv[4]);if(fs::is_symlink(input)||fs::exists(out))throw std::runtime_error("regular input/new output required");auto rh=load(input/"raw.f32",2048),gh=load(input/"gamma.f32",512);
 for(auto*b:{&rh,&gh})for(size_t i=0;i<b->size();i+=4){float x;std::memcpy(&x,b->data()+i,4);if(!std::isfinite(x)||std::abs(x)>65504)throw std::runtime_error("finite bounded independent inputs required");}
 auto&q=dpct::get_in_order_queue();const char*a=std::getenv("ZE_AFFINITY_MASK"),*d=std::getenv("ONEAPI_DEVICE_SELECTOR");if(!a||std::string(a)!="0"||!d||std::string(d)!="level_zero:gpu"||q.get_backend()!=sycl::backend::ext_oneapi_level_zero||!q.get_device().is_gpu()||!q.get_device().has(sycl::aspect::fp16)||!q.has_property<sycl::property::queue::in_order>())throw std::runtime_error("card0 LevelZero fp16 ordered queue required");
 const QsaShapes shape=qsa_real_shapes();if(shape.idx_dim!=128||shape.idx_block!=4||shape.n_rot!=64||qsa_rms_eps()!=1e-6f)throw std::runtime_error("actual43 geometry/epsilon differs");RopeScaling scaling;if(scaling.type!=RopeScalingType::None||scaling.freq_base!=1e7||scaling.factor!=1.||scaling.freq_scale_in!=0.||scaling.orig_ctx!=262144.||scaling.ext_factor!=0.||scaling.attn_factor!=1.||scaling.beta_fast!=32.||scaling.beta_slow!=1.)throw std::runtime_error("actual default RopeScaling differs");rope_scaling_set(scaling);native_qsa_indexer_set_enabled(true);if(native_qsa_indexer_materialized_half_enabled()!=bool(mode))throw std::runtime_error("actual43 mode differs");
 fs::create_directories(out);auto maps=[&](const std::string&suffix){std::ifstream in("/proc/self/maps");std::ofstream file(std::string(argv[8])+suffix);file<<in.rdbuf();file.close();if(!in||!file)throw std::runtime_error("actual maps failed");};maps(".before");Owned memory(q);auto*raw=memory.floats(512);auto*gamma=memory.floats(128);auto*half=static_cast<uint16_t*>(memory.make(1024));auto*wide=memory.floats(512);auto*positions=static_cast<int32_t*>(memory.make(16));auto*steps=static_cast<int32_t*>(memory.make(4*kStepCount*4));
 std::array<QsaIndexerBuffers,2>state;for(auto&b:state){b.tail=memory.floats(384);b.dead=memory.floats(128);b.pooled=memory.floats(256);b.block_pos=static_cast<int32_t*>(memory.make(4));}
 auto*result=static_cast<uint8_t*>(memory.make(2*I43_STATE));auto*before=static_cast<uint8_t*>(memory.make(2*I43_STATE));
 auto snap=[&](uint8_t*dest){for(int owner=0;owner<2;++owner){auto&b=state[owner];auto*p=dest+owner*I43_STATE;q.memcpy(p,b.tail,1536);q.memcpy(p+1536,b.dead,512);q.memcpy(p+2048,b.pooled,1024);q.memcpy(p+3072,b.block_pos,4);}};
 const char*names[]={"single4","steps4","batch4","accept0","accept1","accept2","grouped2","grouped_scalar2"};
 for(int scenario=0;scenario<8;++scenario){std::array<int32_t,4>ps{{0,1,2,3}};if(scenario>=3&&scenario<=5){ps={{-1,-1,-1,-1}};for(int i=0;i<scenario-3;++i)ps[i]=i;}if(scenario>=6)ps={{0,1,0,1}};
  std::array<int32_t,4*kStepCount>st{};for(int i=0;i<4;++i){qsa_step_fill(st.data()+i*kStepCount,i,shape);st[i*kStepCount+kStepPos]=ps[i];}
  auto reset=[&](){q.memcpy(raw,rh.data(),2048);q.memcpy(gamma,gh.data(),512);q.memcpy(positions,ps.data(),16);q.memcpy(steps,st.data(),sizeof(st));q.memset(half,0x5a,1024);q.memset(wide,0x3d,2048);for(auto&b:state){q.memset(b.tail,0,1536);q.memset(b.dead,0,512);q.memset(b.pooled,0,1024);q.memset(b.block_pos,0,4);}q.memset(result,0,2*I43_STATE);q.memset(before,0,2*I43_STATE).wait_and_throw();};
  auto feed=[&](int start,int rows,const int32_t*pos,int stride,int64_t p0){return native_qsa_indexer_materialize_input(raw+start*128,pos,stride,rows,p0,4,128,mode?half:nullptr,mode?wide:nullptr,size_t(start)*128,512,&q);};
  auto body=[&](){snap(before);if(scenario==0){for(int i=0;i<4;++i){auto*f=feed(i,1,positions+i,0,0);native_qsa_indexer_append(f,positions+i,0,gamma,qsa_rms_eps(),state[0],shape,4,scaling,&q);}}
   else if(scenario==1){auto*f=feed(0,4,steps+kStepPos,kStepCount,0);native_qsa_indexer_append_steps(f,steps+kStepPos,kStepCount,4,0,gamma,qsa_rms_eps(),state[0],shape,4,scaling,&q);}
   else if(scenario==2){auto*f=feed(0,4,nullptr,0,0);native_qsa_indexer_append_batch(f,4,0,0,gamma,qsa_rms_eps(),state[0],shape,4,scaling,&q);}
   else if(scenario<=5){auto*f=feed(0,2,positions,1,0);native_qsa_indexer_append_steps(f,positions,1,2,0,gamma,qsa_rms_eps(),state[0],shape,4,scaling,&q);}
   else for(int owner=0;owner<2;++owner){const int start=owner*2;if(scenario==6){auto*f=feed(start,2,positions+start,1,0);native_qsa_indexer_append_steps(f,positions+start,1,2,0,gamma,qsa_rms_eps(),state[owner],shape,4,scaling,&q);}else for(int j=0;j<2;++j){const int i=start+j;auto*f=feed(i,1,positions+i,0,0);native_qsa_indexer_append(f,positions+i,0,gamma,qsa_rms_eps(),state[owner],shape,4,scaling,&q);}}snap(result);};
  GraphOwner graph(q);for(int route=0;route<3;++route){reset();if(route==0){body();q.wait_and_throw();}else{if(route==1){dpct::experimental::begin_recording(&q);graph.recording=true;body();dpct::experimental::end_recording(&q,&graph.graph);graph.recording=false;if(!graph.graph)throw std::runtime_error("owned graph missing");graph.executable=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph.graph->finalize());delete graph.graph;graph.graph=nullptr;}q.ext_oneapi_graph(*graph.executable).wait_and_throw();}
   auto frame=out/(std::string(names[scenario])+"-route"+std::to_string(route));fs::create_directory(frame);i43_dump(q,frame/"raw.f32",raw,2048);i43_dump(q,frame/"gamma.f32",gamma,512);i43_dump(q,frame/"positions.i32",positions,16);i43_dump(q,frame/"steps.i32",steps,sizeof(st));i43_dump(q,frame/"actual-half.f16",half,1024);i43_dump(q,frame/"actual-expanded.f32",wide,2048);i43_dump(q,frame/"before.bin",before,2*I43_STATE);i43_dump(q,frame/"after.bin",result,2*I43_STATE);std::printf("INDEXER43_MASK_FRAME mode=%d scenario=%s route=%d fields=8 restored=1\n",mode,names[scenario],route);std::fflush(stdout);
  }q.wait_and_throw();delete graph.executable;graph.executable=nullptr;
 }
 q.wait_and_throw();memory.release();maps(".after");std::printf("INDEXER43_MASK_RESULT mode=%d frames=24 graph_retired=1 owned_freed=1 model_math=0\n",mode);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"INDEXER43_MASK_ERROR %s\n",e.what());return 2;}
