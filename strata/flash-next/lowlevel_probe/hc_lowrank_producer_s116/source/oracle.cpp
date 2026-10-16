// S116 SOURCE ONLY. No build/run authorization from this file.
#define DPCT_PROFILING_ENABLED
#include "private_hc.hpp"
#include "anchor_names.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace H=s116_private;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
using Bytes=std::vector<uint8_t>;
constexpr int N=2560,HC=4,D=N*HC,LR=320,TM=8,ROWS=324,SLICES=80,SPLITS=4;
constexpr size_t G=256;
bool cleanup_error=false;
static void need(bool b,const std::string& m){if(!b)throw std::runtime_error(m);}
static uint16_t bf16(float x){uint32_t u;std::memcpy(&u,&x,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
static void equal(const Bytes&a,const Bytes&b,const char*m){need(a==b,m);}
static uint32_t word(const Bytes&x,size_t i){uint32_t u;need((i+1)*4<=x.size(),"word range");std::memcpy(&u,x.data()+i*4,4);return u;}
static void finite(const Bytes&x,size_t n,const char*m,size_t offset=0){need((offset+n)*4<=x.size(),"finite range");for(size_t i=0;i<n;++i){float f;std::memcpy(&f,x.data()+(offset+i)*4,4);if(!std::isfinite(f))throw std::runtime_error(m);}}
static void poison_after(const Bytes&x,size_t used,const char*m){need(used<=x.size(),"tail bound");for(size_t i=used;i<x.size();++i)need(x[i]==0xff,m);}
struct Buffer {
 sycl::queue& q;size_t bytes;uint8_t* raw;const char* name;
 Buffer(sycl::queue&qq,size_t n,const char*label):q(qq),bytes(n),raw(sycl::malloc_device<uint8_t>(n+2*G,q)),name(label){
  need(raw!=nullptr,std::string("allocation ")+name);q.memset(raw,0xd7,n+2*G);q.memset(raw+G,0xff,n);
 }
 ~Buffer(){try{q.wait_and_throw();if(raw)sycl::free(raw,q);}catch(...){cleanup_error=true;}}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
 template<class T>T* ptr()const{return reinterpret_cast<T*>(raw+G);}
 void poison(){q.memset(raw+G,0xff,bytes);}
 template<class T>void put(const std::vector<T>&v){need(v.size()*sizeof(T)==bytes,"upload size");q.memcpy(raw+G,v.data(),bytes);}
 Bytes get()const{Bytes v(bytes+2*G);q.memcpy(v.data(),raw,v.size()).wait_and_throw();
  for(size_t i=0;i<G;++i)need(v[i]==0xd7&&v[bytes+G+i]==0xd7,std::string("guard ")+name);
  return Bytes(v.begin()+G,v.begin()+G+bytes);
 }
 template<class T>void readonly(const std::vector<T>&v)const{auto b=get();need(b.size()==v.size()*sizeof(T)&&!std::memcmp(b.data(),v.data(),b.size()),std::string("read-only ")+name);}
};
static void disjoint(const Buffer&a,const Buffer&b){const auto x=reinterpret_cast<uintptr_t>(a.raw),y=reinterpret_cast<uintptr_t>(b.raw);need(x+a.bytes+2*G<=y||y+b.bytes+2*G<=x,"private USM spans alias");}
struct Graph {
 sycl::queue& q;dpct::experimental::command_graph_exec_ptr exec=nullptr;
 explicit Graph(sycl::queue&qq):q(qq){}
 ~Graph(){try{q.wait_and_throw();}catch(...){cleanup_error=true;}delete exec;}
 void capture(const std::function<void()>&record){
  q.wait_and_throw();need(!exec,"capture once");dpct::experimental::command_graph_ptr g=nullptr;bool begun=false,ended=false;
  try{begun=true;dpct::experimental::begin_recording(&q);record();dpct::experimental::end_recording(&q,&g);ended=true;
   need(g!=nullptr,"capture null");exec=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());delete g;g=nullptr;q.wait_and_throw();
  }catch(...){const auto original=std::current_exception();
   if(begun&&!ended){try{if(g)g->end_recording();else dpct::experimental::end_recording(&q,&g);}catch(...){cleanup_error=true;std::fprintf(stderr,"HOLD capture balance failed\n");}}
   try{q.wait_and_throw();}catch(...){cleanup_error=true;std::fprintf(stderr,"HOLD capture drain failed\n");}
   delete g;std::rethrow_exception(original);
  }
 }
 void replay(){need(exec!=nullptr,"graph absent");q.ext_oneapi_graph(*exec);}
};
struct Fixture {
 std::vector<float> R,bo,inj,wn,Rb,bob,injb;
 std::vector<uint16_t> wd,wu,wi;
 Fixture():R(size_t(TM)*D),bo(size_t(TM)*N),inj(size_t(TM)*HC),wn(D),wd(size_t(LR)*D),wu(size_t(D)*LR),wi(size_t(HC)*D){
  // Exact S39/S73 generation order, seed, distribution, scales, BF16 conversion.
  std::mt19937 rng(7);std::normal_distribution<float>nd(0.f,1.f);
  auto fill=[&](std::vector<float>&v,float scale){for(auto&x:v)x=scale*nd(rng);};
  fill(R,1.f);fill(bo,.5f);fill(inj,1.f);fill(wn,.3f);for(auto&x:wn)x+=1.f;
  for(auto&x:wd)x=bf16(.02f*nd(rng));for(auto&x:wu)x=bf16(.05f*nd(rng));for(auto&x:wi)x=bf16(.02f*nd(rng));
  Rb=R;bob=bo;injb=inj;
  for(size_t i=0;i<Rb.size();++i)Rb[i]=-.7f*Rb[i]+.001f*float(i%17);
  for(auto&x:bob)x=-.4f*x+.02f;for(auto&x:injb)x+=.3f;
 }
};
struct Shared {
 sycl::queue&q;const Fixture&h;Buffer bo,inj,wn,wd,wu,wi;
 Shared(sycl::queue&qq,const Fixture&hh):q(qq),h(hh),bo(q,h.bo.size()*4,"bo"),inj(q,h.inj.size()*4,"inj_prev"),wn(q,h.wn.size()*4,"w_norm"),wd(q,h.wd.size()*2,"w_down"),wu(q,h.wu.size()*2,"w_up"),wi(q,h.wi.size()*2,"w_inject"){
  wn.put(h.wn);wd.put(h.wd);wu.put(h.wu);wi.put(h.wi);stage(false);
 }
 void stage(bool b){bo.put(b?h.bob:h.bo);inj.put(b?h.injb:h.inj);q.wait_and_throw();}
 void inputs(bool b){bo.readonly(b?h.bob:h.bo);inj.readonly(b?h.injb:h.inj);}
 void weights(){wn.readonly(h.wn);wd.readonly(h.wd);wu.readonly(h.wu);wi.readonly(h.wi);}
};
struct Snap {Bytes R,lo,rs,io,mix,xn,part,part2;};
static Bytes anchored(const Snap&s){Bytes out;out.reserve(747776);for(const auto*b:{&s.R,&s.lo,&s.rs,&s.io,&s.mix,&s.xn})out.insert(out.end(),b->begin(),b->end());need(out.size()==747776,"S39 snapshot fields/size");return out;}
static void equal_final(const Snap&a,const Snap&b){equal(a.R,b.R,"R bytes");equal(a.lo,b.lo,"lo bytes");equal(a.rs,b.rs,"rs bytes");equal(a.io,b.io,"inject bytes");equal(a.mix,b.mix,"mixed bytes");equal(a.xn,b.xn,"xn bytes");equal(a.part,b.part,"raw part bytes");equal(a.part2,b.part2,"raw part2 bytes");}
struct Route {
 sycl::queue&q;Shared&shared;bool candidate;int T,apply,inject;H::HCMultiS116 m{};
 Buffer R,xn,lo,rs,io,mix,part,part2;Graph whole_graph,synthetic_graph;
 Route(sycl::queue&qq,Shared&s,bool c,int t,int a,int i):q(qq),shared(s),candidate(c),T(t),apply(a),inject(i),R(q,size_t(TM)*D*4,"R"),xn(q,size_t(TM)*D*4,"xn"),lo(q,size_t(TM)*LR*4,"lo"),rs(q,size_t(TM)*HC*4,"rs"),io(q,size_t(TM)*HC*4,"inject_out"),mix(q,size_t(TM)*N*4,"mixed"),part(q,size_t(SPLITS)*TM*LR*4,"part"),part2(q,size_t(SLICES)*TM*ROWS*4,"part2"),whole_graph(q),synthetic_graph(q){
  need(T==1||T==2||T==6,"only reviewed T1/2/6");
  m.T=T;m.nsplit=1;m.xn=xn.ptr<float>();m.part=part.ptr<float>();m.part2=part2.ptr<float>();
  for(int k=0;k<T;++k){auto&a=m.a[k];a.R=R.ptr<float>()+size_t(k)*D;a.R_out=R.ptr<float>()+size_t(k)*D;a.apply=apply;
   a.bo_prev=s.bo.ptr<float>()+size_t(k)*N;a.inj_prev=s.inj.ptr<float>()+size_t(k)*HC;a.w_norm=s.wn.ptr<float>();a.w_down=s.wd.ptr<uint16_t>();a.w_up=s.wu.ptr<uint16_t>();a.w_inject=inject?s.wi.ptr<uint16_t>():nullptr;
   a.lo=lo.ptr<float>()+size_t(k)*LR;a.rs=rs.ptr<float>()+size_t(k)*HC;a.inject_out=io.ptr<float>()+size_t(k)*HC;a.mixed=mix.ptr<float>()+size_t(k)*N;}
  const auto own=buffers();for(size_t u=0;u<own.size();++u){for(size_t v=u+1;v<own.size();++v)disjoint(*own[u],*own[v]);for(Buffer*p:{&s.bo,&s.inj,&s.wn,&s.wd,&s.wu,&s.wi})disjoint(*own[u],*p);}
 }
 std::array<Buffer*,8>buffers(){return {&R,&xn,&lo,&rs,&io,&mix,&part,&part2};}
 ~Route(){try{q.wait_and_throw();}catch(...){cleanup_error=true;}}
 void stage(bool b){q.wait_and_throw();R.put(b?shared.h.Rb:shared.h.R);for(Buffer*p:{&xn,&lo,&rs,&io,&mix,&part,&part2})p->poison();q.wait_and_throw();}
 void prefix(){H::prefix(q,m,candidate);}void up(){H::up(q,m,candidate);}void whole(){H::whole(q,m,candidate);}
 void reduce(){H::reduce_only(q,m,candidate);}
 Snap snapshot(bool b,bool final,bool ordinary=true){
  q.wait_and_throw();Snap s{R.get(),lo.get(),rs.get(),io.get(),mix.get(),xn.get(),part.get(),part2.get()};
  const auto&input=b?shared.h.Rb:shared.h.R;
  need(!std::memcmp(s.R.data()+size_t(T)*D*4,input.data()+size_t(T)*D,(TM-T)*D*4),"inactive R tail");
  if(!final||!apply)need(!std::memcmp(s.R.data(),input.data(),s.R.size()),"R premature/untaken write");
  if(final&&apply)need(std::memcmp(s.R.data(),input.data(),size_t(T)*D*4)!=0,"pending write not exercised");
  poison_after(s.lo,(final||candidate)?size_t(T)*LR*4:0,"lo tail/readiness");
  poison_after(s.rs,size_t(T)*HC*4,"rs tail");poison_after(s.xn,size_t(T)*D*4,"xn tail");
  poison_after(s.io,inject?size_t(T)*HC*4:0,"inject tail/untaken");poison_after(s.mix,final?size_t(T)*N*4:0,"mixed tail/readiness");
  poison_after(s.part,size_t(T)*LR*4,"raw part inactive/split tail");
  for(int sl=0;sl<SLICES;++sl)for(int k=0;k<TM;++k)for(int r=0;r<ROWS;++r){size_t pos=(size_t(sl)*TM+k)*ROWS+r;
   if(k>=T||(r>=LR&&!inject))need(word(s.part2,pos)==0xffffffffu,"raw part2 inactive/untaken");}
  if(ordinary){finite(s.R,size_t(T)*D,"finite R");finite(s.rs,size_t(T)*HC,"finite rs");finite(s.xn,size_t(T)*D,"finite xn");finite(s.part,size_t(T)*LR,"finite part");
   if(inject)finite(s.io,size_t(T)*HC,"finite inject");if(final||candidate)finite(s.lo,size_t(T)*LR,"finite lo");if(final)finite(s.mix,size_t(T)*N,"finite mixed");
   for(int sl=0;sl<SLICES;++sl)for(int k=0;k<T;++k)finite(s.part2,inject?ROWS:LR,"finite part2",(size_t(sl)*TM+k)*ROWS);
  }
  return s;
 }
};
static void owners_disjoint(Route&a,Route&b){for(auto*x:a.buffers())for(auto*y:b.buffers())disjoint(*x,*y);}
static std::set<std::string>anchors_checked;
static void anchor(const std::string&name,const Snap&s){
 if(std::find(s116_anchor_names.begin(),s116_anchor_names.end(),name)==s116_anchor_names.end())return;
 if(anchors_checked.count(name))return;
 std::ifstream f(fs::path(s116_anchor_dir)/name,std::ios::binary);need(bool(f),"historical anchor absent "+name);
 Bytes expected{std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};equal(anchored(s),expected,"historical S39 anchor bytes");anchors_checked.insert(name);
 std::printf("{\"type\":\"S39_anchor\",\"name\":\"%s\",\"bytes\":%zu,\"PASS\":true}\n",name.c_str(),expected.size());
}
static void pre_equal(const Snap&a,const Snap&b){equal(a.R,b.R,"prefix R");equal(a.rs,b.rs,"prefix rs");equal(a.xn,b.xn,"prefix xn");equal(a.io,b.io,"prefix inject");equal(a.part,b.part,"prefix raw part");equal(a.part2,b.part2,"prefix raw part2");equal(a.mix,b.mix,"prefix mixed poison");}
static void producer_lo(const Snap&pre,const Snap&post){equal(pre.lo,post.lo,"candidate pre-UP lo vs control post-UP lo");}
static constexpr std::array<uint32_t,14>special={0u,0x80000000u,1u,0x80000001u,0x007fffffu,0x807fffffu,0x00800000u,0x80800000u,0x3f800000u,0xbf800000u,0x7f800000u,0xff800000u,0x7fc12345u,0xffc54321u};
static std::vector<uint32_t>partials(int T,bool changed){
 std::vector<uint32_t>x(size_t(SLICES)*TM*ROWS,0xffffffffu);
 for(int sl=0;sl<SLICES;++sl)for(int k=0;k<T;++k)for(int r=0;r<ROWS;++r){
  const size_t i=(size_t(sl)*TM+k)*ROWS+r;const int kind=(r+k*3)%int(special.size());
  x[i]=sl==0?special[kind]:(changed&&sl==1?special[(kind+5)%special.size()]:0u);
 }
 return x;
}
static void print_bits(const char*label,const Bytes&x,size_t n){std::printf("{\"type\":\"%s\",\"bits\":[",label);for(size_t i=0;i<n;++i)std::printf("%s\"%08x\"",i?",":"",word(x,i));std::printf("]}\n");}
static int run(int argc,char**argv){
 need(argc==4||argc==5,"usage: oracle NEW_OUTPUT_DIR --pci PHYSICAL_PCI [--timing]");need(std::string(argv[2])=="--pci","require explicit physical PCI");
 const bool timing=argc==5;need(!timing||std::string(argv[4])=="--timing","only optional --timing");
 const fs::path out=argv[1];need(fs::create_directory(out),"new output directory required");
 need(dpct::device_count()==1,"one exposed physical device required");const sycl::device dev=dpct::dev_mgr::instance().get_device(0);
 need(dev.has(sycl::aspect::ext_intel_pci_address),"PCI unavailable");std::string pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();
 std::transform(pci.begin(),pci.end(),pci.begin(),[](unsigned char c){return char(std::tolower(c));});need(pci==argv[3],"wrong physical PCI");
 sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(),dev,[](sycl::exception_list errors){for(const auto&e:errors)std::rethrow_exception(e);},sycl::property_list{sycl::property::queue::in_order()});
 std::printf("CONFIG S116_PRIVATE sixHCclasses normalC16 sliced=1 MORE=0 ILP=0 GRF=default nsplit=1 pci=%s\n",pci.c_str());
 try{
  Fixture h;Shared shared(q,h);std::array<std::unique_ptr<Route>,2>timed;Snap timed_reference;size_t cases=0,eager=0,captured=0;
  for(int T:{1,2,6})for(int apply:{0,1})for(int inject:{0,1}){
   std::array<std::unique_ptr<Route>,2>route;for(int a=0;a<2;++a){route[a]=std::make_unique<Route>(q,shared,a!=0,T,apply,inject);route[a]->whole_graph.capture([&r=*route[a]]{r.whole();});}owners_disjoint(*route[0],*route[1]);
   const std::string stem="T"+std::to_string(T)+"-a"+std::to_string(apply)+"-i"+std::to_string(inject);std::array<Snap,2>initial;
   for(int state=0;state<3;++state){bool b=state==1;shared.stage(b);std::array<Snap,2>pre,post;
    for(int a=0;a<2;++a){auto&r=*route[a];r.stage(b);r.prefix();pre[a]=r.snapshot(b,false);r.up();post[a]=r.snapshot(b,true);++eager;}
    pre_equal(pre[0],pre[1]);equal_final(post[0],post[1]);producer_lo(pre[1],post[0]);
    if(state==0){initial=post;if(T==6&&apply&&inject)timed_reference=post[0];}
    if(state==1)need(post[0].mix!=initial[0].mix,"changed ordinary mixed not observable");
    if(state==2)for(int a=0;a<2;++a)equal_final(post[a],initial[a]);
    if(state<2)anchor(stem+(b?"-direct-b.bin":"-direct-a.bin"),post[0]);
    for(int a=0;a<2;++a){auto&r=*route[a];r.stage(b);r.whole_graph.replay();const auto replay=r.snapshot(b,true);equal_final(replay,post[a]);++captured;if(a==0&&state<2)anchor(stem+(b?"-graph-b.bin":"-graph-a.bin"),replay);}
    shared.inputs(b);
   }
   ++cases;std::printf("{\"type\":\"strict_case\",\"T\":%d,\"apply\":%d,\"inject\":%d,\"states\":\"A-B-A\",\"eager_and_captured\":true,\"raw_parts_final_bytes_guards_readonly\":true,\"PASS\":true}\n",T,apply,inject);
   if(T==6&&apply&&inject)timed=std::move(route);
  }
  need(cases==12&&eager==72&&captured==72&&anchors_checked.size()==36,"strict coverage/anchor count");shared.weights();
  // The SAME selected reduce+UP kernel classes, with synthetic part2 inputs.
  // Norm/down are bypassed in this diagnostic, never in whole-read timing.
  std::array<std::unique_ptr<Route>,2>diag;for(int a=0;a<2;++a){diag[a]=std::make_unique<Route>(q,shared,a!=0,2,0,1);diag[a]->synthetic_graph.capture([&r=*diag[a]]{r.reduce();r.up();});}owners_disjoint(*diag[0],*diag[1]);
  std::array<Snap,2>first_diag;size_t diagnostics=0;
  for(int state=0;state<3;++state){bool b=state==1;const auto p=partials(2,b);shared.stage(false);std::array<Snap,2>pre,post;
   for(int a=0;a<2;++a){auto&r=*diag[a];r.stage(false);r.part2.put(p);
    // Finite independent rs/xn are irrelevant to lo but make the UP inputs defined.
    std::vector<float>rr(size_t(TM)*HC);std::vector<float>xx(size_t(TM)*D);for(int k=0;k<2;++k){for(int c=0;c<HC;++c)rr[k*HC+c]=1.f;for(int i=0;i<D;++i)xx[size_t(k)*D+i]=0.f;}
    q.memcpy(r.rs.ptr<float>(),rr.data(),size_t(2)*HC*4);q.memcpy(r.xn.ptr<float>(),xx.data(),size_t(2)*D*4);r.reduce();pre[a]=r.snapshot(false,false,false);r.up();post[a]=r.snapshot(false,true,false);++diagnostics;
   }
   pre_equal(pre[0],pre[1]);equal_final(post[0],post[1]);producer_lo(pre[1],post[0]);
   print_bits("synthetic_reduce_actual_raw_part",pre[0].part,special.size());print_bits("synthetic_candidate_preUP_lo",pre[1].lo,special.size());
   if(state==0)first_diag=post;if(state==1)need(pre[0].part!=first_diag[0].part,"changed synthetic raw part not observable");if(state==2)for(int a=0;a<2;++a)equal_final(post[a],first_diag[a]);
   for(int a=0;a<2;++a){auto&r=*diag[a];r.stage(false);r.part2.put(p);std::vector<float>rr(size_t(TM)*HC,1.f),xx(size_t(TM)*D,0.f);q.memcpy(r.rs.ptr<float>(),rr.data(),size_t(2)*HC*4);q.memcpy(r.xn.ptr<float>(),xx.data(),size_t(2)*D*4);r.synthetic_graph.replay();auto s=r.snapshot(false,true,false);equal_final(s,post[a]);++diagnostics;}
  }
  // Direct control-UP-only raw-part probe. NOT candidate cross-context proof.
  shared.stage(false);auto&r=*diag[0];r.stage(false);std::vector<uint32_t>raw(size_t(SPLITS)*TM*LR,0xffffffffu);for(int k=0;k<2;++k)for(int i=0;i<LR;++i)raw[size_t(k)*LR+i]=special[(i+k*3)%special.size()];
  r.part.put(raw);std::vector<float>rr(size_t(TM)*HC,1.f);q.memcpy(r.rs.ptr<float>(),rr.data(),size_t(2)*HC*4);r.up();q.wait_and_throw();r.part.readonly(raw);auto direct_lo=r.lo.get();poison_after(direct_lo,size_t(2)*LR*4,"direct control lo tail");print_bits("DIRECT_CONTROL_UP_ONLY_raw_input",r.part.get(),special.size());print_bits("DIRECT_CONTROL_UP_ONLY_lo_NOT_candidate_proof",direct_lo,special.size());
  r.R.readonly(h.R);poison_after(r.mix.get(),size_t(2)*N*4,"direct mixed tail");poison_after(r.rs.get(),size_t(2)*HC*4,"direct rs tail");poison_after(r.io.get(),0,"direct io untaken");poison_after(r.xn.get(),0,"direct xn untaken");poison_after(r.part2.get(),0,"direct part2 untaken");
  shared.weights();shared.inputs(false);need(diagnostics==12,"synthetic eager/captured A-B-A coverage");
  std::printf("{\"type\":\"strict_complete_before_timing\",\"base_cases\":12,\"ordinary_eager_route_calls\":72,\"ordinary_captured_route_calls\":72,\"historical_unique_anchors\":36,\"synthetic_route_calls\":12,\"direct_control_only_probes\":1,\"HC_counters\":\"NONE_NO_RESET_LOGIC\",\"PASS\":true}\n");
  if(timing){shared.stage(false);for(int round=0;round<2;++round)for(int arm:{0,1,1,0}){auto&t=*timed[arm];t.stage(false);q.wait_and_throw();const auto begin=Clock::now();t.whole_graph.replay();q.wait_and_throw();const auto end=Clock::now();auto s=t.snapshot(false,true);equal_final(s,timed_reference);double usec=std::chrono::duration<double,std::micro>(end-begin).count();need(std::isfinite(usec)&&usec>0,"finite timing");std::printf("{\"type\":\"timing\",\"round\":%d,\"arm\":%d,\"whole_reads\":1,\"graph_submissions\":1,\"completed_host_us\":%.6f}\n",round,arm,usec);}}
  for(auto&t:timed)t.reset();for(auto&t:diag)t.reset();q.wait_and_throw();shared.weights();shared.inputs(false);

 }
 catch(...){const auto original=std::current_exception();try{q.wait_and_throw();}catch(...){cleanup_error=true;std::fprintf(stderr,"HOLD final drain failed\n");}std::rethrow_exception(original);}
 q.wait_and_throw();need(!cleanup_error,"cleanup error");
  std::ofstream result(out/"verdict.json");result<<"{\"status\":\"PASS_PRIVATE_STRICT_FIXTURES_ONLY\",\"base_cases\":12,\"anchors\":36,\"timing\":"<<(timing?"true":"false")<<",\"production_unchanged\":true}\n";need(bool(result),"verdict write");
 return 0;
}
int main(int argc,char**argv){setvbuf(stdout,nullptr,_IONBF,0);try{return run(argc,argv);}catch(const std::exception&e){std::fprintf(stderr,"S116 HOLD: %s; no retry/reset/tolerance\n",e.what());return 5;}}
