// NEW CPU bulk interface; exact frozen scalar body retained as a dependency.
#include <algorithm>
#define main hc35_frozen_scalar_main
#include "hc_f32_arithmetic35_host_v1.cpp"
#undef main
int main(int argc,char** argv) try {
 if(argc==8)return hc35_frozen_scalar_main(argc,argv);
 if(argc!=10)throw std::runtime_error("bulk OP K M MODE A B C-or-dash OUT TAG");
 const std::string op=argv[1],mode=argv[4];size_t usedK=0,usedM=0,K=std::stoull(argv[2],&usedK),M=std::stoull(argv[3],&usedM);
 if(usedK!=std::strlen(argv[2])||usedM!=std::strlen(argv[3])||K<1||K>10240||M<1||M>10240||M>(64u<<20)/(K*4)||std::string(argv[9])!="source35_hc_bulk_fma_v1")throw std::runtime_error("bounded bulk source/shape differs");
 if(std::fesetround(FE_TONEAREST)!=0||std::fegetround()!=FE_TONEAREST)throw std::runtime_error("RNE unavailable");
#if defined(__SSE__)
 _mm_setcsr(_mm_getcsr()&~(unsigned(1)<<15)&~(unsigned(1)<<6));if(_mm_getcsr()&((unsigned(1)<<15)|(unsigned(1)<<6)))throw std::runtime_error("gradual host mode unavailable");
#else
 throw std::runtime_error("x86 SSE host flag evidence required");
#endif
 std::vector<float> result;
 if(op=="matrixdot"){
  if(K%32||std::string(argv[7])!="-"||(mode!="shared"&&mode!="paired"))throw std::runtime_error("exact block32 matrix mode required");
  auto weights=floats(argv[5],M*K),input=floats(argv[6],mode=="shared"?K:M*K);result.resize(M);
  std::vector<float> w(K),x(K);if(mode=="shared")x=input;
  for(size_t row=0;row<M;++row){std::copy(weights.begin()+row*K,weights.begin()+(row+1)*K,w.begin());if(mode=="paired")std::copy(input.begin()+row*K,input.begin()+(row+1)*K,x.begin());result[row]=dot32(w,x);}
 }else if(op=="fma"){
  if(mode!="paired")throw std::runtime_error("FMA paired mode required");auto a=floats(argv[5],M*K),b=floats(argv[6],M*K),c=floats(argv[7],M*K);result.resize(M*K);for(size_t i=0;i<M*K;++i)result[i]=::fmaf(a[i],b[i],c[i]);
 }else throw std::runtime_error("only matrixdot/fma bulk implemented");
 for(float value:result)if(!std::isfinite(value))throw std::runtime_error("nonfinite output");if(fs::exists(argv[8]))throw std::runtime_error("new output required");std::ofstream out(argv[8],std::ios::binary);out.write(reinterpret_cast<const char*>(result.data()),std::streamsize(result.size()*4));out.close();if(!out)throw std::runtime_error("output write failed");
 std::cout<<"{\"schema\":2,\"op\":\""<<op<<"\",\"K\":"<<K<<",\"M\":"<<M<<",\"mode\":\""<<mode<<"\",\"output_floats\":"<<result.size()<<",\"rounding\":\"FE_TONEAREST\",\"flush_to_zero\":false,\"denormals_are_zero\":false,\"device_intrinsics_qualified\":false,\"model_math_qualified\":false}"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<"HC35_BULK_ERROR "<<e.what()<<std::endl;return 2;}
