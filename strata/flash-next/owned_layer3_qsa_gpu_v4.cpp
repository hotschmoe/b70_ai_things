// Own original projection operands only; unchanged linked source37 functions.
#define main qsa37_owned_fixture_utilities
#include "hc_composition_arithmetic35_gpu_v1.cpp"
#undef main
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include <array>
#include <cstring>
#include <unistd.h>

using namespace strata::kernels;
constexpr size_t QT=4,QH=24,KH=2,QD=256,IH=4,ID=128,CAP=4,POOL=2048;
void raw_dump(sycl::queue& q,const fs::path& out,const std::string& field,const void* p,size_t bytes){
 std::vector<uint8_t> raw(bytes);q.memcpy(raw.data(),p,bytes).wait_and_throw();auto path=out/field;
 if(fs::exists(path))throw std::runtime_error("exclusive own QSA output required");std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(raw.data()),bytes);f.close();if(!f)throw std::runtime_error("own QSA raw output failed");
}
void maps_dump(const fs::path& p){if(fs::exists(p))throw std::runtime_error("exclusive maps required");std::ifstream in("/proc/self/maps");std::ofstream out(p);out<<in.rdbuf();out.close();if(!in||!out)throw std::runtime_error("mapped libraries receipt failed");}
void echo_loaded(const fs::path& dir,const std::string& name,const std::vector<uint8_t>& raw){auto p=dir/name;if(fs::exists(p))throw std::runtime_error("exclusive consumed own input echo required");std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(raw.data()),raw.size());f.close();if(!f)throw std::runtime_error("actual consumed own input echo failed");}
int main(int argc,char** argv) try{
 if(argc!=7||std::string(argv[1])!="--inputs"||std::string(argv[3])!="--output"||std::string(argv[5])!="--maps-prefix")throw std::runtime_error("exact own-input QSA CLI required");
 fs::path inputs(argv[2]),out(argv[4]);std::string maps=argv[6];if(fs::is_symlink(inputs)||fs::exists(out))throw std::runtime_error("new output and regular own inputs required");fs::create_directories(out);
 auto config=load(inputs/"config.txt",210);std::string expected="QSA37_OWN_INPUTS_V4\nids 248045 8678 198 15666\ngeometry 24 2 256 4 128 64 4 2048\nrope none 10000000 1 0 262144 0 1 32 1\nepsilon source_qsa_rms_eps\nwindows 0:2 2:1 3:1\norigin independent_original_projection_only\n";
 if(std::string(config.begin(),config.end())!=expected)throw std::runtime_error("exact source geometry/RopeScaling/own input manifest required");auto echo=out/"input_echo";fs::create_directory(echo);echo_loaded(echo,"config.txt",config);
 for(const char* key:{"STRATA_VERIFY_EAGER","STRATA_ROPE_TABLE","STRATA_NO_NORM_ROPE","STRATA_ATTN_LANECELL"})if(std::getenv(key))throw std::runtime_error("alternative source presence flag forbidden");
 auto& q=dpct::get_in_order_queue();const char* affinity=std::getenv("ZE_AFFINITY_MASK"),*selector=std::getenv("ONEAPI_DEVICE_SELECTOR");if(!affinity||std::string(affinity)!="0"||!selector||std::string(selector)!="level_zero:gpu"||!q.get_device().is_gpu()||q.get_backend()!=sycl::backend::ext_oneapi_level_zero||!q.has_property<sycl::property::queue::in_order>())throw std::runtime_error("card0 LevelZero in_order required");
 const QsaShapes shape=qsa_real_shapes();if(shape.n_head!=24||shape.n_head_kv!=2||shape.head_dim!=256||shape.idx_n_head!=4||shape.idx_dim!=128||shape.n_rot!=64||shape.page_size!=4||shape.idx_top_k!=2048)throw std::runtime_error("actual linked source geometry differs");
 RopeScaling scaling;if(scaling.type!=RopeScalingType::None||scaling.freq_base!=1e7||scaling.factor!=1.0||scaling.freq_scale_in!=0.0||scaling.orig_ctx!=262144.0||scaling.ext_factor!=0.0||scaling.attn_factor!=1.0||scaling.beta_fast!=32.0||scaling.beta_slow!=1.0||qsa_rms_eps()!=1e-6f)throw std::runtime_error("actual resolved source scaling/epsilon differs");rope_scaling_set(scaling);native_qsa_set_enabled(true);native_rope_set_enabled(true);native_qsa_indexer_set_enabled(true);if(!native_qsa_enabled()||!native_rope_enabled()||!native_qsa_indexer_enabled()||!native_norm_rope_usable(256,64)||!native_norm_rope_usable(128,64))throw std::runtime_error("actual source native/fused capabilities absent");
 Owned mem(q);auto hostq=load(inputs/"qfull.f32",QT*QH*2*QD*4),hostk=load(inputs/"k.f32",QT*KH*QD*4),hostiq=load(inputs/"iq.f32",QT*IH*ID*4);
 auto finite_input=[](const std::vector<uint8_t>& raw){for(size_t i=0;i<raw.size();i+=4){float x;std::memcpy(&x,raw.data()+i,4);if(!std::isfinite(x))throw std::runtime_error("finite own F32 input required");}};finite_input(hostq);finite_input(hostk);finite_input(hostiq);
 echo_loaded(echo,"qfull.f32",hostq);echo_loaded(echo,"k.f32",hostk);echo_loaded(echo,"iq.f32",hostiq);
 std::vector<float> plain(QT*QH*QD);for(size_t r=0;r<QT*QH;++r)std::memcpy(plain.data()+r*QD,hostq.data()+r*2*QD*4,QD*4);
 auto* qfull=mem.floats(QT*QH*2*QD);auto* k=mem.floats(QT*KH*QD);auto* iq=mem.floats(QT*IH*ID);auto* qplain=mem.floats(QT*QH*QD);q.memcpy(qfull,hostq.data(),hostq.size());q.memcpy(qplain,plain.data(),plain.size()*4);
 auto own_input=[&](const char* name,size_t count){auto bytes=load(inputs/name,count*4);finite_input(bytes);echo_loaded(echo,name,bytes);auto* p=mem.floats(count);q.memcpy(p,bytes.data(),bytes.size()).wait_and_throw();return p;};
 auto* v=own_input("v.f32",QT*KH*QD);auto* raw=own_input("raw.f32",QT*ID);auto* qgamma=own_input("qgamma.f32",QD);auto* kgamma=own_input("kgamma.f32",QD);auto* iqgamma=own_input("iqgamma.f32",ID);auto* ikgamma=own_input("ikgamma.f32",ID);
 auto* qrot=mem.floats(QT*QH*QD);auto* qnorm=mem.floats(QT*QH*QD);auto* knorm=mem.floats(QT*KH*QD);auto* iqnorm=mem.floats(QT*IH*ID);auto* attention=mem.floats(QT*QH*QD);auto* gated=mem.floats(QT*QH*QD);
 auto* kp=static_cast<uint16_t*>(mem.make(POOL*2));auto* vp=static_cast<uint16_t*>(mem.make(POOL*2));auto* page=static_cast<int32_t*>(mem.make(4));int32_t zero=0;q.memcpy(page,&zero,4);
 auto* steps=static_cast<int32_t*>(mem.make(QT*kStepCount*4));std::array<int32_t,QT*kStepCount> hs{};for(int t=0;t<4;++t)qsa_step_fill(hs.data()+t*kStepCount,t,shape);q.memcpy(steps,hs.data(),sizeof(hs));
 auto* ids=static_cast<int32_t*>(mem.make(QT*CAP*4));std::array<int32_t,QT*CAP> hi{};for(int t=0;t<4;++t)for(int c=0;c<4;++c)hi[t*4+c]=c<=t?c:-1;q.memcpy(ids,hi.data(),sizeof(hi));
 auto positions=[&](size_t heads){std::vector<int32_t> hp(QT*heads);for(size_t t=0;t<QT;++t)for(size_t h=0;h<heads;++h)hp[t*heads+h]=int32_t(t);auto* p=static_cast<int32_t*>(mem.make(hp.size()*4));q.memcpy(p,hp.data(),hp.size()*4).wait_and_throw();return p;};auto* pq=positions(QH);auto* pk=positions(KH);auto* pi=positions(IH);
 QsaIndexerBuffers ib;ib.tail=mem.floats(3*ID);ib.dead=mem.floats(ID);ib.pooled=mem.floats(2*ID);ib.block_pos=static_cast<int32_t*>(mem.make(4));
 auto* tails=mem.floats(3*3*ID);auto* deads=mem.floats(3*ID);auto* pooled=mem.floats(3*2*ID);auto* blocks=static_cast<int32_t*>(mem.make(3*4));auto* kw=static_cast<uint16_t*>(mem.make(3*POOL*2));auto* vw=static_cast<uint16_t*>(mem.make(3*POOL*2));
 const size_t scratch_stride=qsa_decode_attn_scratch_floats(CAP,shape);if(!scratch_stride||scratch_stride>65536)throw std::runtime_error("bounded actual scratch geometry required");auto* scratch=mem.floats(QT*scratch_stride);
 QsaAttnPools pools;pools.k_pool=kp;pools.v_pool=vp;pools.page_table=page;
 auto body=[&](){
  const int starts[3]={0,2,3},counts[3]={2,1,1};
  for(int w=0;w<3;++w){const int p=starts[w],n=counts[w];
   native_qsa_rms_norm_weighted(qplain+p*QH*QD,qgamma,qnorm+p*QH*QD,QD,n*QH,qsa_rms_eps(),&q);
   native_qsa_rms_norm_weighted(k+p*KH*QD,kgamma,knorm+p*KH*QD,QD,n*KH,qsa_rms_eps(),&q);
   native_qsa_rms_norm_weighted(iq+p*IH*ID,iqgamma,iqnorm+p*IH*ID,ID,n*IH,qsa_rms_eps(),&q);
   native_qsa_rms_norm_rope(k+p*KH*QD,QD,kgamma,k+p*KH*QD,n*KH,QD,64,qsa_rms_eps(),scaling,pk+p*KH,&q);
   for(int t=p;t<p+n;++t)kv_append_step(kp,vp,page,steps+t*kStepCount,k+t*KH*QD,v+t*KH*QD,shape,&q,nullptr);
   native_qsa_indexer_append_steps(raw+p*ID,steps+p*kStepCount+kStepPos,kStepCount,n,0,ikgamma,qsa_rms_eps(),ib,shape,4,scaling,&q);
   native_qsa_rms_norm_rope(qfull+p*QH*2*QD,2*QD,qgamma,qrot+p*QH*QD,n*QH,QD,64,qsa_rms_eps(),scaling,pq+p*QH,&q);
   native_qsa_rms_norm_rope(iq+p*IH*ID,ID,iqgamma,iq+p*IH*ID,n*IH,ID,64,qsa_rms_eps(),scaling,pi+p*IH,&q);
   qsa_decode_attn_batch(qrot+p*QH*QD,pools,ids+p*CAP,steps+p*kStepCount,CAP,shape,scratch+p*scratch_stride,attention+p*QH*QD,n,&q);
   native_qsa_gate_apply(attention+p*QH*QD,qfull+p*QH*2*QD,gated+p*QH*QD,n*QH,QD,&q);
   q.memcpy(tails+w*3*ID,ib.tail,3*ID*4);q.memcpy(deads+w*ID,ib.dead,ID*4);q.memcpy(pooled+w*2*ID,ib.pooled,2*ID*4);q.memcpy(blocks+w,ib.block_pos,4);q.memcpy(kw+w*POOL,kp,POOL*2);q.memcpy(vw+w*POOL,vp,POOL*2);
  }
 };
 auto reset=[&](){q.memcpy(k,hostk.data(),hostk.size());q.memcpy(iq,hostiq.data(),hostiq.size());q.memset(kp,0,POOL*2);q.memset(vp,0,POOL*2);q.memset(ib.tail,0,3*ID*4);q.memset(ib.dead,0,ID*4);q.memset(ib.pooled,0,2*ID*4);q.memset(ib.block_pos,0,4);q.memset(scratch,0,QT*scratch_stride*4).wait_and_throw();};
 maps_dump(maps+".before");std::printf("OWNQSA37_DEVICE backend=level_zero affinity=0 vendor=%s driver=%s name=%s\n",q.get_device().get_info<sycl::info::device::vendor>().c_str(),q.get_device().get_info<sycl::info::device::driver_version>().c_str(),q.get_device().get_info<sycl::info::device::name>().c_str());std::printf("OWNQSA37_FP_CONFIG flags=");for(auto flag:q.get_device().get_info<sycl::info::device::single_fp_config>())std::printf("%llu,",static_cast<unsigned long long>(flag));std::printf(" observed_device_flags_only=1 compiler_lowering_unobserved=1\n");std::printf("OWNQSA37_CONFIG rope=none freq_base=10000000 factor=1 freq_scale_in=0 orig_ctx=262144 ext_factor=0 attn_factor=1 beta_fast=32 beta_slow=1 epsilon=source_qsa_rms_eps native_flags=1,1,1 actual_resolved_checked=1\n");std::printf("OWNQSA37_INPUTS files=10 own_projection_only=1 input_echo_bound=1\n");GraphOwner graph(q);
 for(int route=0;route<3;++route){reset();if(route==0){body();q.wait_and_throw();}else{if(route==1){dpct::experimental::begin_recording(&q);graph.recording=true;body();dpct::experimental::end_recording(&q,&graph.graph);graph.recording=false;if(!graph.graph)throw std::runtime_error("own QSA graph missing");graph.executable=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph.graph->finalize());delete graph.graph;graph.graph=nullptr;}q.ext_oneapi_graph(*graph.executable).wait_and_throw();}
  auto frame=out/("route"+std::to_string(route));fs::create_directory(frame);
  for(auto item:std::vector<std::pair<std::string,const float*>>{{"q_normalized",qnorm},{"q_RoPE",qrot},{"attention",attention},{"gated",gated}})dump(q,frame,item.first,item.second,QT*QH*QD);
  dump(q,frame,"k_normalized",knorm,QT*KH*QD);dump(q,frame,"k_RoPE",k,QT*KH*QD);dump(q,frame,"iq_normalized",iqnorm,QT*IH*ID);dump(q,frame,"iq_RoPE",iq,QT*IH*ID);dump(q,frame,"indexer_tail_windows",tails,3*3*ID);dump(q,frame,"indexer_dead_windows",deads,3*ID);dump(q,frame,"indexer_pooled_windows",pooled,3*2*ID);
  raw_dump(q,frame,"indexer_block_pos_windows.i32",blocks,3*4);raw_dump(q,frame,"K_pool_windows.f16",kw,3*POOL*2);raw_dump(q,frame,"V_pool_windows.f16",vw,3*POOL*2);
  std::printf("OWNQSA37_FRAME route=%d graph_replay=%d fields=14 own_zero_restored=1 windows=2,1,1\n",route,route);std::fflush(stdout);
 }
 q.wait_and_throw();delete graph.executable;graph.executable=nullptr;mem.release();maps_dump(maps+".after");std::printf("OWNQSA37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 internal_norm_argument_observed=0 internal_scores_softmax_observed=0 whole_model_math_qualified=0\n");std::fflush(stdout);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"OWNQSA37_ERROR %s\n",e.what());return 2;}
