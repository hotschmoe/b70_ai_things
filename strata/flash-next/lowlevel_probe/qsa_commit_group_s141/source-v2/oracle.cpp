// SOURCE ONLY. Existing public QSA kernels; no new device code.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace K=strata::kernels;namespace fs=std::filesystem;
using Bytes=std::vector<unsigned char>;using Clock=std::chrono::steady_clock;
constexpr int D=128,CAP=32,LAYERS=12,OWNERS=2,GROUPS=LAYERS*OWNERS;constexpr size_t GUARD=256;
bool cleanup_error=false;
void need(bool x,const std::string&m){if(!x)throw std::runtime_error(m);}
void eq(const Bytes&a,const Bytes&b,const std::string&m){if(a!=b){size_t i=0;while(i<std::min(a.size(),b.size())&&a[i]==b[i])++i;throw std::runtime_error(m+" first_byte="+std::to_string(i));}}
struct Buffer{
 sycl::queue&q;size_t n;unsigned char*p;std::string name;
 Buffer(sycl::queue&qq,size_t nn,std::string s):q(qq),n(nn),p(sycl::malloc_device<unsigned char>(n+2*GUARD,q)),name(s){need(p!=nullptr,"alloc "+name);q.memset(p,0xd7,n+2*GUARD);}
 ~Buffer(){try{q.wait_and_throw();sycl::free(p,q);}catch(...){cleanup_error=true;}}
 template<class T>T*ptr()const{return reinterpret_cast<T*>(p+GUARD);}
 void put(const Bytes&b){need(b.size()==n,"put size");q.memcpy(p+GUARD,b.data(),n).wait_and_throw();}
 template<class T>void putv(const std::vector<T>&v){need(v.size()*sizeof(T)==n,"putv size");q.memcpy(p+GUARD,v.data(),n).wait_and_throw();}
 Bytes get()const{Bytes b(n+2*GUARD);q.memcpy(b.data(),p,b.size()).wait_and_throw();for(size_t i=0;i<GUARD;++i)need(b[i]==0xd7&&b[GUARD+n+i]==0xd7,"guard "+name);return Bytes(b.begin()+GUARD,b.begin()+GUARD+n);}
};
void disjoint(const Buffer&a,const Buffer&b){auto x=reinterpret_cast<uintptr_t>(a.p),y=reinterpret_cast<uintptr_t>(b.p);need(x+a.n+2*GUARD<=y||y+b.n+2*GUARD<=x,"USM alias");}
struct Graph{
 sycl::queue&q;dpct::experimental::command_graph_exec_ptr e=nullptr;
 explicit Graph(sycl::queue&qq):q(qq){}
 ~Graph(){try{q.wait_and_throw();}catch(...){cleanup_error=true;}delete e;}
 void capture(const std::function<void()>&f){q.wait_and_throw();need(!e,"capture once");bool begun=false,ended=false;dpct::experimental::command_graph_ptr g=nullptr;
  try{begun=true;dpct::experimental::begin_recording(&q);f();dpct::experimental::end_recording(&q,&g);ended=true;need(g!=nullptr,"null graph");e=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());delete g;g=nullptr;}
  catch(...){auto original=std::current_exception();if(begun&&!ended){try{if(g)g->end_recording();else dpct::experimental::end_recording(&q,&g);}catch(...){cleanup_error=true;std::fprintf(stderr,"HOLD capture balance\n");}}try{q.wait_and_throw();}catch(...){cleanup_error=true;}delete g;std::rethrow_exception(original);}}
 void replay(){need(e!=nullptr,"graph absent");q.ext_oneapi_graph(*e);}
};
struct State{
 Buffer tail,dead,pool,bpos;
 explicit State(sycl::queue&q):tail(q,3*D*4,"tail"),dead(q,D*4,"dead"),pool(q,(CAP/4+1)*D*4,"pooled"),bpos(q,4,"block_pos"){}
 std::array<Buffer*,4>buffers(){return{&tail,&dead,&pool,&bpos};}
 K::QsaIndexerBuffers args(){return{tail.ptr<float>(),dead.ptr<float>(),pool.ptr<float>(),bpos.ptr<int32_t>()};}
 std::array<Bytes,4>get(){std::array<Bytes,4>x{tail.get(),dead.get(),pool.get(),bpos.get()};
  for(int field=0;field<3;++field)for(size_t i=0;i<x[field].size();i+=4){float value;std::memcpy(&value,x[field].data()+i,4);need(std::isfinite(value),"nonfinite state float field "+std::to_string(field)+" byte "+std::to_string(i));}return x;}
 void put(const std::array<Bytes,4>&x){auto b=buffers();for(int i=0;i<4;++i)b[i]->put(x[i]);}
 void init(int tag){std::vector<float>t(3*D),d(D),p((CAP/4+1)*D);for(size_t i=0;i<t.size();++i)t[i]=float(int((i+tag)%29)-14)/32;for(size_t i=0;i<d.size();++i)d[i]=float(int((i+tag)%31)-15)/16;for(size_t i=0;i<p.size();++i)p[i]=float(int((i+tag)%43)-21)/8;tail.putv(t);dead.putv(d);pool.putv(p);bpos.putv(std::vector<int32_t>{-1});}
};
struct Fixture{
 sycl::queue&q;K::QsaShapes shapes{};K::RopeScaling rope{};
 Buffer raw,gamma,pos,tail_snapshot,pre_raw,onepos;
 std::array<std::array<std::unique_ptr<State>,GROUPS>,2>states;
 std::array<std::unique_ptr<Graph>,2>graphs;
 Bytes raw_ref,gamma_ref,pos_ref,tail_ref,pre_ref,one_ref;
 Fixture(sycl::queue&qq):q(qq),raw(q,GROUPS*3*D*4,"raw"),gamma(q,GROUPS*D*4,"gamma"),pos(q,OWNERS*3*4,"positions"),tail_snapshot(q,GROUPS*3*D*4,"tail_snapshot"),pre_raw(q,D*4,"prefix_raw"),onepos(q,4,"prefix_pos"){
  shapes.idx_dim=D;shapes.idx_block=4;shapes.n_rot=64;
  for(auto&arm:states)for(auto&s:arm)s=std::make_unique<State>(q);
  std::vector<Buffer*>all{&raw,&gamma,&pos,&tail_snapshot,&pre_raw,&onepos};for(auto&arm:states)for(auto&s:arm)for(auto*b:s->buffers())all.push_back(b);
  for(size_t i=0;i<all.size();++i)for(size_t j=i+1;j<all.size();++j)disjoint(*all[i],*all[j]);
 }
 void append(State&s,int g,const float*r,const int32_t*p,int stride,int n){K::native_qsa_indexer_append_steps(r,p,stride,n,0,gamma.ptr<float>()+g*D,1e-6f,s.args(),shapes,CAP,rope,&q);}
 void singles(State&s,int g,const float*r,const int32_t*p){for(int j=0;j<3;++j)K::native_qsa_indexer_append(r+j*D,p+j,0,gamma.ptr<float>()+g*D,1e-6f,s.args(),shapes,CAP,rope,&q);}
 void chain(int arm){for(int g=0;g<GROUPS;++g){auto&s=*states[arm][g];q.memcpy(s.tail.ptr<float>(),tail_snapshot.ptr<float>()+g*3*D,3*D*4);const float*r=raw.ptr<float>()+g*3*D;const auto*p=pos.ptr<int32_t>()+(g%OWNERS)*3;if(arm)append(s,g,r,p,1,3);else singles(s,g,r,p);}}
 void capture(){for(int arm=0;arm<2;++arm){graphs[arm]=std::make_unique<Graph>(q);graphs[arm]->capture([&,arm]{chain(arm);});}}
 void readonly(){eq(raw.get(),raw_ref,"raw read-only");eq(gamma.get(),gamma_ref,"gamma read-only");eq(pos.get(),pos_ref,"positions read-only");eq(tail_snapshot.get(),tail_ref,"tail snapshot read-only");eq(pre_raw.get(),pre_ref,"prefix raw read-only");eq(onepos.get(),one_ref,"prefix position read-only");}
 using Seed=std::array<std::array<Bytes,4>,GROUPS>;
 Seed setup(int start,int keep,bool changed){
  std::vector<float>r(GROUPS*3*D),gm(GROUPS*D),pr(D);for(size_t i=0;i<r.size();++i)r[i]=float(int((i*17+start*3)%257)-128)/64*(changed?-.7f:1.f)+float(i%7)/4096;for(size_t i=0;i<gm.size();++i)gm[i]=.75f+float((i+start)%11)/32;for(int i=0;i<D;++i)pr[i]=float((i*3)%37-18)/32;raw.putv(r);gamma.putv(gm);pre_raw.putv(pr);
  std::vector<int32_t>ps(6);for(int owner=0;owner<2;++owner){int n=owner?4-keep:keep,p0=owner?(start+1):start;for(int j=0;j<3;++j)ps[owner*3+j]=j<n?p0+j:-1;}pos.putv(ps);
  std::vector<float>tails(GROUPS*3*D);Seed seed;
  for(int g=0;g<GROUPS;++g){auto&s=*states[0][g];s.init(g+start);int p0=(g%2)?start+1:start;
   for(int p=0;p<p0;++p){onepos.putv(std::vector<int32_t>{p});K::native_qsa_indexer_append(pre_raw.ptr<float>(),onepos.ptr<int32_t>(),0,gamma.ptr<float>()+g*D,1e-6f,s.args(),shapes,CAP,rope,&q);}
   auto tail=s.tail.get();std::memcpy(tails.data()+g*3*D,tail.data(),tail.size());
   // Speculation writes ALL3 positions; partial-commit must restore only saved tail.
   for(int j=0;j<3;++j){onepos.putv(std::vector<int32_t>{p0+j});K::native_qsa_indexer_append(raw.ptr<float>()+g*3*D+j*D,onepos.ptr<int32_t>(),0,gamma.ptr<float>()+g*D,1e-6f,s.args(),shapes,CAP,rope,&q);}
   seed[g]=s.get();
  }
  tail_snapshot.putv(tails);q.wait_and_throw();raw_ref=raw.get();gamma_ref=gamma.get();pos_ref=pos.get();tail_ref=tail_snapshot.get();pre_ref=pre_raw.get();one_ref=onepos.get();return seed;
 }
 void stage(const Seed&s){q.wait_and_throw();for(int a=0;a<2;++a)for(int g=0;g<GROUPS;++g)states[a][g]->put(s[g]);q.wait_and_throw();}
 std::vector<Bytes>snapshot(int arm){q.wait_and_throw();std::vector<Bytes>b;for(auto&s:states[arm])for(auto&x:s->get())b.push_back(std::move(x));return b;}
 void compare(){auto a=snapshot(0),b=snapshot(1);need(a.size()==96&&a.size()==b.size(),"snapshot fields");for(size_t i=0;i<a.size();++i)eq(a[i],b[i],"complete state field "+std::to_string(i));readonly();}
};
void save(const fs::path&p,const std::vector<Bytes>&v){std::ofstream f(p,std::ios::binary);for(auto&b:v)f.write(reinterpret_cast<const char*>(b.data()),b.size());f.close();need(bool(f),"save snapshot");}
int run(int argc,char**argv){need(argc==3||argc==4,"usage: oracle NEW_DIR EXPECTED_PCI [--timing]");bool timing=argc==4;need(!timing||std::string(argv[3])=="--timing","timing option");fs::path out=argv[1];need(fs::create_directory(out),"new output directory");need(dpct::device_count()==1,"one pinned device");const sycl::device dev=dpct::dev_mgr::instance().get_device(0);need(dev.has(sycl::aspect::ext_intel_pci_address),"PCI extension");auto pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();need(pci==argv[2],"PCI mismatch");auto rt=K::rope_table_for(K::RopeScaling{});need(rt.cos==nullptr,"analytic current-profile route only");need(K::mrope_table()==nullptr,"text no IMRoPE map");
 sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(),dev,sycl::property_list{sycl::property::queue::in_order()});
 {Fixture f(q);f.capture();int count=0;
  for(int start:{0,2,3,7})for(int keep:{1,2,3}){std::vector<Bytes>first;
   for(int phase=0;phase<3;++phase){bool changed=phase==1;auto seed=f.setup(start,keep,changed);f.stage(seed);f.chain(0);f.chain(1);f.compare();auto eager=f.snapshot(0);f.stage(seed);f.graphs[0]->replay();f.graphs[1]->replay();f.compare();auto captured=f.snapshot(0);for(size_t i=0;i<eager.size();++i)eq(eager[i],captured[i],"eager/captured");if(phase==0)first=eager;else if(phase==2)for(size_t i=0;i<eager.size();++i)eq(first[i],eager[i],"A/B/A restoration");save(out/("case"+std::to_string(count)+"-phase"+std::to_string(phase)+".bin"),captured);std::printf("{\"type\":\"strict\",\"start\":%d,\"keepA\":%d,\"keepB\":%d,\"phase\":%d,\"fields\":96,\"complete_bytes\":true,\"guards\":true,\"readonly\":true,\"eager_captured\":true}\n",start,keep,4-keep,phase);}
   ++count;
  }
  need(count==12,"finite cases");
  if(timing){auto seed=f.setup(3,2,false);std::array<std::vector<Bytes>,2>expected;for(int a=0;a<2;++a){f.stage(seed);f.graphs[a]->replay();expected[a]=f.snapshot(a);}for(size_t i=0;i<expected[0].size();++i)eq(expected[0][i],expected[1][i],"timing reference");
   for(int a:{0,1}){f.stage(seed);f.graphs[a]->replay();auto actual=f.snapshot(a);for(size_t i=0;i<actual.size();++i)eq(actual[i],expected[a][i],"warm state");f.readonly();}
   const int order[8]={0,1,1,0,1,0,0,1};for(int i=0;i<8;++i){int a=order[i];f.stage(seed);auto t0=Clock::now();f.graphs[a]->replay();q.wait_and_throw();auto t1=Clock::now();auto actual=f.snapshot(a);for(size_t j=0;j<actual.size();++j)eq(actual[j],expected[a][j],"timed state");f.readonly();std::printf("{\"type\":\"timing\",\"iteration\":%d,\"arm\":%d,\"completed_host_chain_us\":%.6f,\"groups\":24,\"appends\":%d,\"restores\":24}\n",i,a,std::chrono::duration<double,std::micro>(t1-t0).count(),a?24:72);}}
  q.wait_and_throw();
 }
 need(!cleanup_error,"cleanup failure");std::printf("{\"status\":\"PASS_PRIVATE_QSA_THREE_STEPS\",\"base_cases\":12,\"ABA_phases\":36,\"eager_and_captured\":true,\"groups\":24,\"snapshots\":36,\"timing_enabled\":%s,\"serving_qualified\":false}\n",timing?"true":"false");return 0;
}
int main(int argc,char**argv){try{return run(argc,argv);}catch(const std::exception&e){std::fprintf(stderr,"HOLD %s\n",e.what());return 1;}}
