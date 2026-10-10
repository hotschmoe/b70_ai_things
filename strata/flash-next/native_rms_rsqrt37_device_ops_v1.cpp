// Fresh isolated caller of UNCHANGED compiled SDK37 HC. No model weights.
#define main hc35_frozen_composition_fixture_main
#include "hc_composition_arithmetic35_gpu_v1.cpp"
#undef main
class RmsArgumentShadow37;
class RmsDeviceOperationShadow37;
int rms37_frozen_proposal_main(int argc,char** argv) try {
 if(argc!=5||std::string(argv[1])!="--residual"||std::string(argv[3])!="--output")throw std::runtime_error("--residual OWNED_F32 --output NEWDIR");
 const fs::path input(argv[2]),out(argv[4]);if(fs::exists(out)||fs::is_symlink(out))throw std::runtime_error("new output required");auto raw=load(input,D*4);fs::create_directories(out);
 auto& q=dpct::get_in_order_queue();const char* affinity=std::getenv("ZE_AFFINITY_MASK"),*selector=std::getenv("ONEAPI_DEVICE_SELECTOR");
 if(!affinity||std::string(affinity)!="0"||!selector||std::string(selector)!="level_zero:gpu"||std::getenv("STRATA_VERIFY_EAGER")||q.get_backend()!=sycl::backend::ext_oneapi_level_zero||!q.has_property<sycl::property::queue::in_order>())throw std::runtime_error("physicalcard0 LevelZero in_order EAGERabsent required");
 Owned mem(q);strata::kernels::HcNativeArgs a;a.tokens=1;a.eps=1e-6f;
 auto* R=mem.floats(D);auto* norm=mem.floats(D);std::vector<float> ones(D,1.0f);q.memcpy(norm,ones.data(),D*4).wait_and_throw();
 a.w.norm=norm;a.w.norm_floats=D;a.w.down=static_cast<uint8_t*>(mem.make(L*(D/32)*34));a.w.down_bytes=L*(D/32)*34;a.w.up=static_cast<uint8_t*>(mem.make(D*(L/32)*34));a.w.up_bytes=D*(L/32)*34;q.memset((void*)a.w.down,0,a.w.down_bytes);q.memset((void*)a.w.up,0,a.w.up_bytes);
 a.R=R;a.R_floats=D;a.xn=mem.floats(D);a.xn_floats=D;a.lo=mem.floats(L);a.lo_floats=L;a.gate=mem.floats(D);a.gate_floats=D;a.rs=mem.floats(H);a.rs_floats=H;a.mixed=mem.floats(N);a.mixed_floats=N;
 auto* sum=mem.floats(H);auto* argument=mem.floats(H);
 auto* device_rsqrt=mem.floats(H);auto* device_native_rsqrt=mem.floats(H);auto* device_recip_sqrt=mem.floats(H);
 GraphOwner graph(q);
 auto body=[&](){if(!strata::kernels::hc_native_read_f32(a,&q))throw std::runtime_error("compiled SDK HC rejected descriptor");};
 for(int route=0;route<3;++route){q.memcpy(R,raw.data(),raw.size());q.memset(a.rs,0xff,H*4);q.memset(a.xn,0xff,D*4).wait_and_throw();
  if(route==0){body();q.wait_and_throw();}else{if(route==1){dpct::experimental::begin_recording(&q);graph.recording=true;body();dpct::experimental::end_recording(&q,&graph.graph);graph.recording=false;if(!graph.graph)throw std::runtime_error("graph capture absent");graph.executable=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph.graph->finalize());delete graph.graph;graph.graph=nullptr;}q.ext_oneapi_graph(*graph.executable).wait_and_throw();}
  // Separate leaf shadow: it is NOT an observed internal argument of HC.
  q.parallel_for<RmsArgumentShadow37>(sycl::nd_range<1>(sycl::range<1>(H*32),sycl::range<1>(32)),[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
   const size_t c=item.get_group_linear_id(),lane=item.get_local_linear_id();float s=0;for(size_t d=lane;d<N;d+=32){float r=R[c*N+d];s=sycl::fma(r,r,s);}for(unsigned mask=16;mask;mask>>=1)s+=sycl::permute_group_by_xor(item.get_sub_group(),s,mask);if(!lane){sum[c]=s;argument[c]=s/float(N)+1e-6f;}
  }).wait_and_throw();
  // Same separately observed argument; these are eager leaf candidates, not HC internals.
  q.parallel_for<RmsDeviceOperationShadow37>(sycl::range<1>(H),[=](sycl::id<1> index){const size_t c=index[0];const float x=argument[c];device_rsqrt[c]=sycl::rsqrt(x);device_native_rsqrt[c]=sycl::native::rsqrt(x);device_recip_sqrt[c]=1.0f/sycl::sqrt(x);}).wait_and_throw();auto dir=out/('r'+std::to_string(route));fs::create_directory(dir);dump(q,dir,"actual_hc_rs",a.rs,H);dump(q,dir,"actual_hc_xn_normones",a.xn,D);dump(q,dir,"separate_square_sum_shadow",sum,H);dump(q,dir,"separate_argument_shadow",argument,H);dump(q,dir,"device_rsqrt_shadow",device_rsqrt,H);dump(q,dir,"device_native_rsqrt_shadow",device_native_rsqrt,H);dump(q,dir,"device_recip_sqrt_shadow",device_recip_sqrt,H);
  std::printf("RMS37_FRAME route=%d graph_replay=%d actual_compiled_hc=1 fields=7 synthetic_norm=1 synthetic_zero_down_up=1 original_model_math_qualified=0\n",route,route);
 }
 q.wait_and_throw();delete graph.executable;graph.executable=nullptr;mem.release();
 std::printf("RMS37_RESULT routes=3 owned_allocations_freed=1 graph_retired=1 device_intrinsics_qualified=0 full_model_math_qualified=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"RMS37_ERROR %s\n",e.what());return 2;}
