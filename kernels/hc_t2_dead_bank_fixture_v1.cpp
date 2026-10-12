#include "strata/kernels/hc_native_composition.hpp"
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include <functional>
#include <tuple>
#include <stdexcept>
#include <cstdio>
#include <cstring>
using namespace strata::kernels;
struct Owner {
 sycl::queue& q;std::vector<void*> allocations;std::vector<std::function<void()>> graphs;
 explicit Owner(sycl::queue& s):q(s){}
 template<class T>T* take(size_t n){auto* p=sycl::malloc_device<T>(n,q);if(!p)throw std::bad_alloc();allocations.push_back(p);return p;}
 template<class T>T* graph(T* p){graphs.emplace_back([p]{delete p;});return p;}
 ~Owner(){try{q.wait_and_throw();for(auto i=graphs.rbegin();i!=graphs.rend();++i)(*i)();for(void* p:allocations)sycl::free(p,q.get_context());}catch(...){std::fprintf(stderr,"HC_T2 failed drain/retirement retains remaining resources\n");std::fflush(stderr);std::_Exit(86);}}
};
int main()try{
 auto& q=dpct::get_in_order_queue();Owner own(q);constexpr int D=10240,N=2560,H=4,L=320,K=10;
 auto fl=[&](int n){return own.take<float>(n);};
 auto weights=[&](int k,int m){std::vector<unsigned char> w(size_t(k/32)*m*34);for(size_t off=0;off<w.size();off+=34){uint16_t s=0x2400;std::memcpy(w.data()+off,&s,2);for(int c=0;c<32;++c)w[off+2+c]=static_cast<unsigned char>(int((off/34+c)%7)-3);}auto* p=own.take<unsigned char>(w.size());q.memcpy(p,w.data(),w.size()).wait_and_throw();return p;};
 auto* norm=fl(D);auto* inject=fl(H*D);std::vector<float> gamma(D,1),inj(H*D);for(size_t i=0;i<inj.size();++i)inj[i]=float(int(i%9)-4)/1024;q.memcpy(norm,gamma.data(),D*4).wait_and_throw();q.memcpy(inject,inj.data(),inj.size()*4).wait_and_throw();
 HcNativeWeights W{norm,D,weights(D,L),size_t(L)*(D/32)*34,weights(L,D),size_t(D)*(L/32)*34,inject,H*D};
 auto args=[&]{HcNativeArgs a;a.w=W;a.tokens=2;a.R=fl(2*D);a.R_floats=2*D;a.xn=fl(2*D);a.xn_floats=2*D;a.lo=fl(2*L);a.lo_floats=2*L;a.gate=fl(2*K*N);a.gate_floats=2*D;a.rs=fl(2*H);a.rs_floats=2*H;a.mixed=fl(2*N);a.mixed_floats=2*N;a.inject_out=fl(2*H);a.inject_out_floats=2*H;return a;};
 auto reference=args(),candidate=args();auto* bo=fl(2*N);auto* previous=fl(2*H);std::vector<float> input(2*D),block(2*N),prev(2*H,.125f);for(size_t i=0;i<input.size();++i)input[i]=float(int(i%101)-50)/64;for(size_t i=0;i<block.size();++i)block[i]=float(int(i%17)-8)/32;q.memcpy(bo,block.data(),block.size()*4).wait_and_throw();q.memcpy(previous,prev.data(),prev.size()*4).wait_and_throw();
 auto read=[&](const float* p,size_t n){std::vector<unsigned char> out(n*4);q.memcpy(out.data(),p,out.size()).wait_and_throw();return out;};
 auto exact=[&]{for(auto v:std::vector<std::tuple<const float*,const float*,size_t>>{{reference.R,candidate.R,2*D},{reference.xn,candidate.xn,2*D},{reference.lo,candidate.lo,2*L},{reference.gate,candidate.gate,2*D},{reference.rs,candidate.rs,2*H},{reference.mixed,candidate.mixed,2*N},{reference.inject_out,candidate.inject_out,2*H}})if(read(std::get<0>(v),std::get<2>(v))!=read(std::get<1>(v),std::get<2>(v)))throw std::runtime_error("T2 intermediate bytes differ");};
 for(int apply=0;apply<2;++apply){
  q.memcpy(const_cast<float*>(reference.R),input.data(),input.size()*4);q.memcpy(const_cast<float*>(candidate.R),input.data(),input.size()*4);
  const uint32_t sentinel=0x3f800000;std::vector<uint32_t> bank(2*K*N,sentinel);q.memcpy(candidate.gate,bank.data(),bank.size()*4);
  if(apply){reference.apply=candidate.apply=true;reference.bo_prev=candidate.bo_prev=bo;reference.bo_floats=candidate.bo_floats=2*N;reference.inj_prev=candidate.inj_prev=previous;reference.prev_floats=candidate.prev_floats=2*H;reference.R_out=const_cast<float*>(reference.R);candidate.R_out=const_cast<float*>(candidate.R);reference.R_out_floats=candidate.R_out_floats=2*D;}
  for(int t=0;t<2;++t){auto a=reference;a.tokens=1;a.R+=t*D;a.R_floats=D;a.xn+=t*D;a.xn_floats=D;a.lo+=t*L;a.lo_floats=L;a.gate+=t*D;a.gate_floats=D;a.rs+=t*H;a.rs_floats=H;a.mixed+=t*N;a.mixed_floats=N;a.inject_out+=t*H;a.inject_out_floats=H;if(apply){a.R_out+=t*D;a.R_out_floats=D;a.bo_prev+=t*N;a.bo_floats=N;a.inj_prev+=t*H;a.prev_floats=H;}if(!hc_native_read_f32(a,&q))throw std::runtime_error("reference rejected");}
  namespace g=sycl::ext::oneapi::experimental;auto* graph=own.graph(new g::command_graph<g::graph_state::modifiable>(q.get_context(),q.get_device()));if(!hc_native_read_f32_t2(candidate,&q))throw std::runtime_error("direct T2 candidate rejected");q.wait_and_throw();exact();q.memcpy(const_cast<float*>(candidate.R),input.data(),input.size()*4);q.wait_and_throw();graph->begin_recording(q);if(!hc_native_read_f32_t2(candidate,&q))throw std::runtime_error("T2 candidate rejected");graph->end_recording(q);auto* exec=own.graph(new decltype(graph->finalize())(graph->finalize()));q.ext_oneapi_graph(*exec).wait_and_throw();
  for(int replay=0;replay<3;++replay){if(replay){q.memcpy(const_cast<float*>(candidate.R),input.data(),input.size()*4);q.ext_oneapi_graph(*exec).wait_and_throw();}
   exact();
   auto tail=read(candidate.gate+2*D,2*K*N-2*D);if(std::memcmp(tail.data(),bank.data()+2*D,tail.size()))throw std::runtime_error("borrowed gate overran parts bank");
  }
 }
 std::printf("HC_T2_OPERATOR_PASS pending=2 replays=4 fields=7 borrowed_tail_exact=1 model_qualified=0 speed_qualified=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HC_T2_OPERATOR_ERROR %s\n",e.what());return 1;}
