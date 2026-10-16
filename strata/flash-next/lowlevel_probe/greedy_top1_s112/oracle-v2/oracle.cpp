// SOURCE PREPARATION ONLY: exact-ID/bytes oracle, no model weights.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "private_argmax.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::kernels;
namespace {
constexpr size_t GUARD=128, RESIDUAL=4*2560;
constexpr unsigned char CANARY=0xa5;
bool cleanup_error=false;
void need(bool b,const char* what){if(!b)throw std::runtime_error(what);}
float from_bits(uint32_t x){float f;std::memcpy(&f,&x,4);return f;}
struct Device {
 sycl::queue& q;size_t bytes;uint8_t* raw;uint8_t* p;
 Device(sycl::queue& qq,size_t n):q(qq),bytes(n),raw(sycl::malloc_device<uint8_t>(n+2*GUARD,q)),p(raw?raw+GUARD:nullptr){
  need(raw!=nullptr,"device allocation");q.memset(raw,CANARY,n+2*GUARD).wait_and_throw();
 }
 ~Device(){try{if(raw)sycl::free(raw,q);}catch(...){cleanup_error=true;}}
 Device(const Device&)=delete;Device& operator=(const Device&)=delete;
 template<class T>T* as(){return reinterpret_cast<T*>(p);}
 void upload(const void* h){q.memcpy(p,h,bytes).wait_and_throw();}
 void fill(unsigned char x){q.memset(p,x,bytes).wait_and_throw();}
 std::vector<uint8_t> read(){std::vector<uint8_t>b(bytes+2*GUARD);q.memcpy(b.data(),raw,b.size()).wait_and_throw();
  need(std::all_of(b.begin(),b.begin()+GUARD,[](auto x){return x==CANARY;}),"device prefix guard");
  need(std::all_of(b.end()-GUARD,b.end(),[](auto x){return x==CANARY;}),"device suffix guard");
  return std::vector<uint8_t>(b.begin()+GUARD,b.end()-GUARD);
 }
};
struct Mapped {
 sycl::queue& q;size_t bytes;uint8_t* raw;uint8_t* p;
 Mapped(sycl::queue& qq,size_t n):q(qq),bytes(n),raw(static_cast<uint8_t*>(strata::host_malloc_polled(n+2*GUARD,q))),p(raw?raw+GUARD:nullptr){
  need(raw!=nullptr,"polled host allocation");std::memset(raw,CANARY,n+2*GUARD);
 }
 ~Mapped(){try{if(raw)strata::host_free_polled(raw,q);}catch(...){cleanup_error=true;}}
 template<class T>T* as(){return reinterpret_cast<T*>(p);}
 void check()const{for(size_t i=0;i<GUARD;++i)need(raw[i]==CANARY&&p[bytes+i]==CANARY,"mapped guard");}
};
struct Route {
 sycl::queue& q;bool candidate;int n;
 Device logits,arg,top,ids,prob,rsrc,rdst,row,token;
 Mapped published,published_prob;
 dpct::experimental::command_graph_exec_ptr graph=nullptr;
 Route(sycl::queue& qq,bool c,int nv):q(qq),candidate(c),n(nv),
 logits(q,size_t(n)*4),arg(q,argmax_rows_scratch_bytes(4)),top(q,row_top_prob_scratch_bytes(4)),
 ids(q,8*4),prob(q,8*4),rsrc(q,RESIDUAL*4),rdst(q,RESIDUAL*4),row(q,4),token(q,4),published(q,8),published_prob(q,8){
  need(arg.bytes==4*sizeof(s112_private::ArgRow),"arg scratch source layout");arg.fill(0);top.fill(0);int zero=0;row.upload(&zero);
 }
 ~Route(){try{q.wait_and_throw();}catch(...){cleanup_error=true;}if(graph)delete graph;}
 void record(){
  if(candidate)s112_private::launch(logits.as<float>(),n,arg.p,ids.as<int32_t>(),q);
  else{SamplerParams sp;sp.greedy=true;sp.temperature=0.0f;sample_tokens(logits.as<float>(),1,n,nullptr,0,sp,ids.as<int32_t>(),&q);}
  // Identical downstream functions/arguments, including all original probability work.
  row_top_prob_split(logits.as<float>(),1,n,ids.as<int32_t>(),prob.as<float>(),top.p,&q);
  mtp_select(rsrc.as<float>(),RESIDUAL,ids.as<int32_t>(),row.as<int32_t>(),rdst.as<float>(),token.as<int32_t>(),published.as<int32_t>(),0,&q,prob.as<float>(),published_prob.as<float>());
 }
 void capture(){
  need(!graph,"capture once");dpct::experimental::command_graph_ptr g=nullptr;
  bool begin_attempted=false,end_completed=false;
  try{
   begin_attempted=true;dpct::experimental::begin_recording(&q);
   record();dpct::experimental::end_recording(&q,&g);end_completed=true;
   need(g!=nullptr,"capture ended without graph");
   graph=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());
   delete g;g=nullptr;q.wait_and_throw();
  }catch(...){
   const auto original=std::current_exception();
   // DPCT end sets g and erases its manager entry BEFORE calling native end.
   // Balance a still-owned/uncertain recording before any drain; never recapture.
   if(begin_attempted&&!end_completed){
    try{if(g)g->end_recording();else dpct::experimental::end_recording(&q,&g);}
    catch(...){cleanup_error=true;std::fprintf(stderr,"CLEANUP HOLD capture balance failed\n");}
   }
   try{q.wait_and_throw();}catch(...){cleanup_error=true;std::fprintf(stderr,"CLEANUP HOLD capture drain failed\n");}
   delete g;std::rethrow_exception(original);
  }
 }
 void poison_outputs(){ids.fill(0xcc);prob.fill(0xcc);rdst.fill(0xcc);token.fill(0xcc);
  std::memset(published.p,0xcc,published.bytes);std::memset(published_prob.p,0xcc,published_prob.bytes);
  reinterpret_cast<volatile int32_t*>(published.p)[0]=-1;std::atomic_thread_fence(std::memory_order_seq_cst);
 }
 void run(bool captured){if(captured){need(graph!=nullptr,"missing graph");q.ext_oneapi_graph(*graph);}else record();}
};
struct Snap{std::vector<uint8_t> ids,prob,residual,token,pub,pubprob;};
void zero_counters(Route& r){
 auto a=r.arg.read(),t=r.top.read();const size_t as=argmax_rows_scratch_bytes(1),ts=row_top_prob_scratch_bytes(1);
 for(int j=0;j<4;++j){uint32_t ac,tc;std::memcpy(&ac,a.data()+j*as,4);std::memcpy(&tc,t.data()+j*ts,4);need(ac==0&&tc==0,"scratch counter not reset");}
 need(std::all_of(a.begin()+as,a.end(),[](auto x){return x==0;}),"unused arg rows changed");
 need(std::all_of(t.begin()+ts,t.end(),[](auto x){return x==0;}),"unused top rows changed");
}
Snap snapshot(Route& r,const std::vector<float>& h,const std::vector<float>& residual,int expected){
 auto l=r.logits.read(),src=r.rsrc.read(),row=r.row.read();
 need(l.size()==h.size()*4&&std::memcmp(l.data(),h.data(),l.size())==0,"logits modified");
 need(src.size()==residual.size()*4&&std::memcmp(src.data(),residual.data(),src.size())==0,"residual source modified");
 int ri;std::memcpy(&ri,row.data(),4);need(ri==0,"selector row changed");
 Snap s{r.ids.read(),r.prob.read(),r.rdst.read(),r.token.read(),std::vector<uint8_t>(r.published.p,r.published.p+8),std::vector<uint8_t>(r.published_prob.p,r.published_prob.p+8)};
 int id,tok,pub;std::memcpy(&id,s.ids.data(),4);std::memcpy(&tok,s.token.data(),4);std::memcpy(&pub,s.pub.data(),4);
 need(id==expected&&tok==expected&&pub==expected,"exact host/ID/token/publication mismatch");
 need(std::memcmp(s.residual.data(),residual.data(),s.residual.size())==0,"selected residual mismatch");
 need(std::memcmp(s.prob.data(),s.pubprob.data(),4)==0,"published probability mismatch");
 for(size_t j=4;j<s.ids.size();++j)need(s.ids[j]==0xcc,"ID unused output changed");
 for(size_t j=4;j<s.prob.size();++j)need(s.prob[j]==0xcc,"prob unused output changed");
 for(size_t j=4;j<8;++j)need(s.pub[j]==0xcc&&s.pubprob[j]==0xcc,"mapped unused output changed");
 r.published.check();r.published_prob.check();zero_counters(r);return s;
}
void equal(const Snap&a,const Snap&b){need(a.ids==b.ids&&a.prob==b.prob&&a.residual==b.residual&&a.token==b.token&&a.pub==b.pub&&a.pubprob==b.pubprob,"strict downstream bytes differ");}
int host_id(const std::vector<float>& x){float best=-INFINITY;int id=int(x.size());for(int i=0;i<int(x.size());++i)if(x[i]>best){best=x[i];id=i;}return id<int(x.size())?id:0;}
std::vector<float> fixture(int n,int owner,int kind){
 std::vector<float>x(n,-4.0f);
 const int nb=std::min(128,std::max(1,(n+4095)/4096)),per=(n+nb-1)/nb;
 need(nb>=2,"fixture requires multiple WG chunks");
 const int last=(nb-1)*per;
 const int a=owner?per:n-1,b=owner?n-1:1;
 switch(kind){
 case 0:for(int i=0;i<n;++i)x[i]=float((i*17+owner*31)%257)-128.0f;x[a]=1000.0f;break;
 case 1:x[owner?0:per-1]=x[owner?last:per]=1000.0f;break;
 case 2:x[owner?17:4]=from_bits(0x80000000);x[owner?4:17]=0.0f;break;
 case 3:for(int i=0;i<n;++i)x[i]=i%7? -INFINITY:from_bits(0x7fc01234);x[a]=x[b]=-2.0f;break;
 case 4:for(int i=0;i<n;++i)x[i]=i%7? -INFINITY:from_bits(0x7fc01234);break;
 case 5:x[a]=x[b]=INFINITY;break;
 case 6:x[0]=0.0f;x[owner?n-1:per]=from_bits(1);break;
 case 7:x[owner?per-1:last-1]=x[owner?per:last]=1000.0f;break;
 default:throw std::runtime_error("fixture kind");
 }return x;
}
std::vector<float> residual_fixture(int owner,int kind){std::vector<float>x(RESIDUAL);for(size_t i=0;i<x.size();++i)x[i]=float(int(i%193)-96)+float(owner*8+kind)*0.125f;return x;}
void stage(Route&r,const std::vector<float>&h,const std::vector<float>&rs){r.logits.upload(h.data());r.rsrc.upload(rs.data());r.poison_outputs();}
}
int main(int argc,char**argv)try{
 bool timing=false;for(int i=1;i<argc;++i){need(std::string(argv[i])=="--timing","usage: oracle [--timing]");timing=true;}
 auto& q=dpct::get_in_order_queue();need(q.has_property<sycl::property::queue::in_order>(),"in-order queue required");
 std::printf("CONFIG S112_PRIVATE kernel=GreedyTop1S112 leader_acquire=1 device=%s\n",q.get_device().get_info<sycl::info::device::name>().c_str());
 size_t cases=0;
 for(int n:{4097,248320}){
  std::array<std::unique_ptr<Route>,2> ref,cand;
  for(int o=0;o<2;++o){ref[o]=std::make_unique<Route>(q,false,n);cand[o]=std::make_unique<Route>(q,true,n);ref[o]->capture();cand[o]->capture();}
  // Each triplet changes fixture data, returns to A, and reuses persistent owner scratch.
  // Eager first, then captured; no reset of arg/top scratch after initial allocation.
  for(bool captured:{false,true})for(const std::array<int,3>& seq:{std::array<int,3>{0,1,0},std::array<int,3>{2,3,2},std::array<int,3>{4,5,4},std::array<int,3>{6,7,6}}){
   for(int kind:seq){
    std::array<std::vector<float>,2> h,rs;std::array<int,2> expected;
    for(int o=0;o<2;++o){h[o]=fixture(n,o,kind);rs[o]=residual_fixture(o,kind);expected[o]=host_id(h[o]);stage(*ref[o],h[o],rs[o]);stage(*cand[o],h[o],rs[o]);}
    // Both owners have independent allocations; only ordering/queue is shared, as in the pair tail.
    for(int o=0;o<2;++o)ref[o]->run(captured);
    for(int o=0;o<2;++o)cand[o]->run(captured);q.wait_and_throw();
    for(int o=0;o<2;++o)equal(snapshot(*ref[o],h[o],rs[o],expected[o]),snapshot(*cand[o],h[o],rs[o],expected[o]));
    ++cases;
   }
  }
  std::printf("RESULT n=%d eager_and_capture=PASS owner_routes=4 cases=%zu\n",n,cases);
  if(timing&&n==248320){
   need(cases==48,"all strict cases must pass before timing");
   // Balanced diagnostic whole-chain timings ONLY after all correctness cases for this shape.
   // Inputs already staged; reset/readback/guards are outside each measured two-owner chain.
   std::array<std::vector<float>,2> h,rs;for(int o=0;o<2;++o){h[o]=fixture(n,o,0);rs[o]=residual_fixture(o,0);}
   for(int repeat=0;repeat<2;++repeat)for(int route:{0,1,1,0}){
    auto& selected=route?cand:ref;for(int o=0;o<2;++o)stage(*selected[o],h[o],rs[o]);
    auto a=std::chrono::steady_clock::now();for(int o=0;o<2;++o)selected[o]->run(true);q.wait_and_throw();auto b=std::chrono::steady_clock::now();
    for(int o=0;o<2;++o)(void)snapshot(*selected[o],h[o],rs[o],host_id(h[o]));
    std::printf("TIMING n=%d repeat=%d route=%s two_owner_chain_host_us=%.3f\n",n,repeat,route?"candidate":"reference",std::chrono::duration<double,std::micro>(b-a).count());
   }
  }
  q.wait_and_throw();
 }
 need(!cleanup_error,"cleanup or drain failure");
 std::printf("VERDICT PASS_STRICT_PRIVATE_ID_DOWNSTREAM_BYTES_GUARDS_COUNTERS_CAPTURE cases=%zu speed_qualified=0\n",cases);return 0;
}catch(const std::exception&e){try{dpct::get_in_order_queue().wait_and_throw();}catch(...){}std::fprintf(stderr,"VERDICT HOLD %s\n",e.what());return 1;}
