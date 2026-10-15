#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/qsa_select.hpp"
#include "private_tiled_scores.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace K=strata::kernels;
namespace fs=std::filesystem;
using Bytes=std::vector<uint8_t>;
using Clock=std::chrono::steady_clock;
constexpr int STRIDE=50002,POOL=50001,DIM=128,CAP=2051;
constexpr size_t GUARD=4096;
struct Strict:std::runtime_error{using std::runtime_error::runtime_error;};
static void need(bool v,const std::string& s){if(!v)throw std::runtime_error(s);}
static void check(bool v,const std::string& s){if(!v)throw Strict(s);}
static void save(const fs::path& p,const Bytes& b){std::ofstream f(p,std::ios::binary);need(bool(f),"open "+p.string());f.write(reinterpret_cast<const char*>(b.data()),b.size());need(bool(f),"write");f.close();need(bool(f),"close");}
static void text(const fs::path& p,const std::string& s){save(p,Bytes(s.begin(),s.end()));}
static double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
struct Buffer{
 sycl::queue& q;size_t bytes;uint8_t* p=nullptr;std::string name;
 Buffer(sycl::queue& s,size_t n,const char* label):q(s),bytes(n),name(label){p=sycl::malloc_device<uint8_t>(bytes+2*GUARD,q);need(p!=nullptr,"allocation "+name);q.memset(p,0xd7,bytes+2*GUARD);}
 template<class T>T* ptr()const{return reinterpret_cast<T*>(p+GUARD);}
 void poison(){q.memset(p+GUARD,0xff,bytes);}
 void put(const void* h){q.memcpy(p+GUARD,h,bytes);}
 Bytes get()const{Bytes b(bytes);q.memcpy(b.data(),p+GUARD,bytes).wait_and_throw();return b;}
 void guards()const{std::array<uint8_t,2*GUARD>b;q.memcpy(b.data(),p,GUARD);q.memcpy(b.data()+GUARD,p+GUARD+bytes,GUARD).wait_and_throw();for(auto v:b)check(v==0xd7,"guard "+name);}
 void readonly(const void* h)const{Bytes b=get();check(!std::memcmp(b.data(),h,bytes),"readonly "+name);}
 ~Buffer(){if(p)try{q.wait_and_throw();sycl::free(p,q);}catch(...){}}
};
struct Data{
 sycl::queue& q;std::vector<float> pooled,dead;Buffer pk,dk;
 Data(sycl::queue& s):q(s),pooled(size_t(POOL)*DIM),dead(DIM),pk(q,pooled.size()*4,"pooled"),dk(q,dead.size()*4,"dead"){
  uint32_t r=0x5130a177u;auto val=[&](){r^=r<<13;r^=r>>17;r^=r<<5;int z=int(r%513)-256;float v=float(z)*0x1p-8f;if((r&63)==0)v=std::bit_cast<float>((r&64)?0x80000000u:0u);return v;};
  for(auto& v:pooled)v=val();for(auto& v:dead)v=val();pooled[0]=1.f;
  pk.put(pooled.data());dk.put(dead.data());q.wait_and_throw();
 }
 void memory(){pk.guards();dk.guards();pk.readonly(pooled.data());dk.readonly(dead.data());}
};
struct Case{const char* tag;int nq,last;};
constexpr Case CASES[]={{"near16k",256,16384}};
struct Packed{Bytes scores,ids;bool operator==(const Packed&)const=default;};
struct State{
 sycl::queue& q;Data& data;Case c;fs::path dir;std::vector<float>a,b;std::vector<int32_t>steps;int active,current=-1;
 Buffer query,step,score,ids;Packed expected[2];
 State(sycl::queue& s,Data& d,Case v,const fs::path& out):q(s),data(d),c(v),dir(out/c.tag),a(size_t(c.nq)*512),b(a.size()),steps(size_t(c.nq)*K::kStepCount),active(c.last/4+1),
  query(q,a.size()*4,"query"),step(q,steps.size()*4,"steps"),score(q,size_t(c.nq)*STRIDE*4,"scores"),ids(q,size_t(c.nq)*CAP*4,"ids"){
  need(fs::create_directory(dir),"fresh case directory");need(c.nq>8&&c.last<=200000&&c.last-c.nq>=0&&active<=POOL,"case bounds");
  uint32_t r=0x7d213acbu;for(auto& v:a){r^=r<<13;r^=r>>17;r^=r<<5;v=float(int(r%513)-256)*0x1p-8f;if((r&31)==0)v=std::bit_cast<float>((r&32)?0x80000000u:0u);}
  // Last query's block0 is a complete pooled block: known score1 -> score0 after sign flip.
  for(int j=0;j<512;++j)a[size_t(c.nq-1)*512+j]=(j&1)?-0.f:0.f;a[size_t(c.nq-1)*512]=1.f;
  for(size_t i=0;i<a.size();++i){need(std::isfinite(a[i]),"finite input");b[i]=std::bit_cast<float>(std::bit_cast<uint32_t>(a[i])^0x80000000u);}
  for(int i=0;i<c.nq;++i){const int nk=c.last-c.nq+i+1;auto* st=steps.data()+i*K::kStepCount;st[K::kStepPos]=nk-1;st[K::kStepNKv]=nk;st[K::kStepNBid]=nk/4;st[K::kStepWidth]=std::min(nk,CAP);need(st[K::kStepNBid]>=0&&st[K::kStepNBid]+1<=active&&active<STRIDE,"exact step/active");}
  // All four remainders must occur, including empty tail and n_bid=50000 spare row.
  unsigned mods=0;for(int i=0;i<c.nq;++i)mods|=1u<<(steps[i*K::kStepCount+K::kStepNKv]%4);need(mods==15,"all nkv mod4");
  step.put(steps.data());q.wait_and_throw();
  std::printf("{\"type\":\"case\",\"case\":\"%s\",\"nq\":%d,\"first_nkv\":%d,\"last_nkv\":%d,\"active\":%d,\"stride\":50002,\"pool_rows\":50001,\"nkv_mod4_mask\":15}\n",c.tag,c.nq,c.last-c.nq+1,c.last,active);
 }
 void reset(){q.wait_and_throw();score.poison();ids.poison();q.wait_and_throw();}
 void input(int state){need(state==0||state==1,"state");q.wait_and_throw();query.put((state?b:a).data());current=state;reset();}
 void call(int arm){need(current>=0,"input before call");
  if(arm==0)K::qsa_block_scores(data.pk.ptr<float>(),data.dk.ptr<float>(),query.ptr<float>(),step.ptr<int32_t>(),c.nq,STRIDE,K::qsa_real_shapes(),score.ptr<float>(),&q,active);
  else{need(arm==1,"arm");s84::launch(data.pk.ptr<float>(),data.dk.ptr<float>(),query.ptr<float>(),step.ptr<int32_t>(),c.nq,STRIDE,score.ptr<float>(),q);}
  K::qsa_block_topk(score.ptr<float>(),step.ptr<int32_t>(),c.nq,STRIDE,CAP,K::qsa_real_shapes(),ids.ptr<int32_t>(),&q,active);
 }
 Packed observe(const std::string& tag,bool export_data){
  auto sc=score.get(),id=ids.get();Packed p;
  for(int i=0;i<c.nq;++i){const int nb=steps[i*K::kStepCount+K::kStepNBid]+1,w=steps[i*K::kStepCount+K::kStepWidth];
   p.scores.insert(p.scores.end(),sc.begin()+size_t(i)*STRIDE*4,sc.begin()+(size_t(i)*STRIDE+nb)*4);
   p.ids.insert(p.ids.end(),id.begin()+size_t(i)*CAP*4,id.begin()+(size_t(i)*CAP+w)*4);
  }
  // Export complete valid cells before finite/poison/comparison gates; never export padding.
  if(export_data){save(dir/(tag+".scores.f32"),p.scores);save(dir/(tag+".ids.i32"),p.ids);}
  for(int i=0;i<c.nq;++i){const int nb=steps[i*K::kStepCount+K::kStepNBid]+1,w=steps[i*K::kStepCount+K::kStepWidth];
   for(int j=0;j<nb;++j){float v;std::memcpy(&v,sc.data()+(size_t(i)*STRIDE+j)*4,4);check(std::isfinite(v),"finite active scores");}
   for(size_t j=(size_t(i)*STRIDE+nb)*4;j<size_t(i+1)*STRIDE*4;++j)check(sc[j]==0xff,"inactive score poison");
   for(int j=0;j<w;++j){int32_t v;std::memcpy(&v,id.data()+(size_t(i)*CAP+j)*4,4);check(v>=0&&v<steps[i*K::kStepCount+K::kStepNKv],"ID range");if(j){int32_t prev;std::memcpy(&prev,id.data()+(size_t(i)*CAP+j-1)*4,4);check(v>prev,"ascending IDs");}}
   for(size_t j=(size_t(i)*CAP+w)*4;j<size_t(i+1)*CAP*4;++j)check(id[j]==0xff,"inactive ID poison");
  }
  for(Buffer* v:{&query,&step,&score,&ids})v->guards();query.readonly((current?b:a).data());step.readonly(steps.data());data.memory();
  // Known pooled block0 last-query score distinguishes actual work from no-op.
  float last;std::memcpy(&last,sc.data()+size_t(c.nq-1)*STRIDE*4,4);check(last==float(current?0:1),"known observable score");
  return p;
 }
 Packed run(int arm,const std::string& tag,bool exp){reset();call(arm);q.wait_and_throw();return observe(tag,exp);}
 void correctness(){
  for(int state=0;state<2;++state){input(state);const std::string t=state?"changed":"initial";
   expected[state]=run(0,"public-"+t,true);check(run(1,"candidate-"+t,true)==expected[state],"candidate exact "+t);
   check(run(0,"",false)==expected[state],"public repeat "+t);check(run(1,"",false)==expected[state],"candidate repeat "+t);
  }
  check(expected[0].scores!=expected[1].scores,"changed score observable");
  input(0);check(run(0,"",false)==expected[0],"public restored");check(run(1,"",false)==expected[0],"candidate restored");
  std::printf("{\"type\":\"correctness\",\"case\":\"%s\",\"status\":\"PASS\",\"snapshots\":4,\"valid_score_bytes\":%zu,\"valid_ID_bytes\":%zu,\"repeat_changed_restored\":true,\"inactive_poison_readonly_guards\":true}\n",c.tag,expected[0].scores.size(),expected[0].ids.size());
 }
 void timing(){
  input(0);for(int warm=0;warm<3;++warm)for(int arm:{0,1})check(run(arm,"",false)==expected[0],"warm full bytes");
  constexpr int order[4][2]={{0,1},{1,0},{1,0},{0,1}};
  for(int pair=0;pair<4;++pair)for(int slot=0;slot<2;++slot){const int arm=order[pair][slot];reset();
   const auto t0=Clock::now();call(arm);const auto t1=Clock::now();q.wait_and_throw();const auto t2=Clock::now();
   check(observe("",false)==expected[0],"timed full bytes");
   std::printf("{\"type\":\"timing\",\"case\":\"%s\",\"pair\":%d,\"slot\":%d,\"arm\":\"%s\",\"order\":\"%s\",\"submit_us\":%.6f,\"complete_us\":%.6f,\"full_scorer_plus_topk\":true,\"full_bytes_checked\":true}\n",c.tag,pair,slot,arm?"candidate":"public",order[pair][0]?"BA":"AB",us(t0,t1),us(t0,t2));
  }
 }
};
static int run(const fs::path& out){
 for(const char* v:{"STRATA_TOPK_OLD","STRATA_TOPK_ACTIVE_ANY","STRATA_TOPK_CAPACITY_GUARD","STRATA_QSA_CLUSTER","STRATA_QSA_WARP","STRATA_SCORES_MULTI","STRATA_QSA_EARLY_EXIT"})need(!std::getenv(v),std::string("require absent ")+v);
 const auto geometry=K::qsa_real_shapes();need(geometry.idx_dim==128&&geometry.idx_n_head==4&&geometry.idx_block==4&&geometry.idx_top_k==2048,"exact geometry");
 need(dpct::device_count()==1,"one visible device");const auto& dev=dpct::dev_mgr::instance().get_device(0);
 need(dev.has(sycl::aspect::ext_intel_pci_address),"PCI unavailable");auto pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();std::transform(pci.begin(),pci.end(),pci.begin(),[](unsigned char c){return char(std::tolower(c));});need(pci=="0000:0b:00.0","physical card0 required");
 std::printf("{\"type\":\"identity\",\"pci\":\"%s\",\"visible_devices\":1,\"device\":\"%s\",\"driver\":\"%s\",\"execution\":\"eager_only_no_graph\"}\n",pci.c_str(),dev.get_info<sycl::info::device::name>().c_str(),dev.get_info<sycl::info::device::driver_version>().c_str());
 sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(),dev,sycl::property_list{sycl::property::queue::in_order()});
 Data data(q);std::vector<std::unique_ptr<State>> states;
 for(auto c:CASES){states.emplace_back(std::make_unique<State>(q,data,c,out));states.back()->correctness();}
 // Every case passes all numerical/memory gates before the first warm/timed sample.
 for(auto& s:states)s->timing();q.wait_and_throw();states.clear();q.wait_and_throw();
 text(out/"result.json","{\"status\":\"PASS\",\"cases\":1,\"snapshots\":4,\"data_files\":8,\"eager_only\":true,\"graphs\":0,\"warm_per_arm\":3,\"balanced_pairs_per_case\":4,\"timed_samples\":8,\"serving_qualified\":false,\"speed_qualified\":false}\n");return 0;
}
int main(int argc,char** argv){setvbuf(stdout,nullptr,_IONBF,0);fs::path out;bool made=false;try{need(argc==2,"usage scorer_leaf NEW_OUTPUT");out=argv[1];need(fs::create_directory(out),"fresh output");made=true;return run(out);}catch(const std::exception& e){int rc=dynamic_cast<const Strict*>(&e)?5:2;std::fprintf(stderr,"S84 failure: %s\n",e.what());if(made)try{text(out/"result.json","{\"status\":\"FAIL\",\"exit\":"+std::to_string(rc)+"}\n");}catch(...){return 2;}return rc;}}
