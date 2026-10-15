// S98 SOURCE PREPARATION ONLY: QT17 GU, public anchor and one natural E4/E2/1 candidate.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "private_iq2_gu.hpp"
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace K=strata::kernels; namespace fs=std::filesystem;
using Bytes=std::vector<uint8_t>;using Clock=std::chrono::steady_clock;
constexpr int N=2560,FF=640,T=6,CG=8,CE=32,GRID=2,BATCH=20,WARM=3,ROUNDS=4;
constexpr size_t G=4096,TAIL=512,PLANE=473600,GU=2*PLANE,BLOB=1868800;
constexpr size_t F=size_t(CE)*FF*4,FA=(F+255)&~size_t(255);
struct Strict:std::runtime_error{using std::runtime_error::runtime_error;};
static void need(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
static void strict(bool b,const std::string& s){if(!b)throw Strict(s);}
static double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
static void save(const fs::path& p,const void* b,size_t n){need(!fs::exists(p),"refuse export overwrite");std::ofstream f(p,std::ios::binary);need(bool(f),"open export");f.write(static_cast<const char*>(b),n);f.close();need(bool(f),"export close");}
static uint64_t fnv(const Bytes& b){uint64_t h=0xcbf29ce484222325ull;for(auto v:b){h^=v;h*=0x100000001b3ull;}return h;}
struct Buffer {
 sycl::queue& q;size_t bytes;uint8_t* raw;const char* name;
 Buffer(sycl::queue& s,size_t n,const char* label):q(s),bytes(n),name(label){raw=sycl::malloc_device<uint8_t>(bytes+2*G,q);need(raw!=nullptr,"allocate");q.memset(raw,0xd7,bytes+2*G);q.memset(raw+G,0xff,bytes);}
 template<class V=uint8_t>V* ptr()const{return reinterpret_cast<V*>(raw+G);}
 void put(const void* b,size_t n){need(n==bytes,"upload size");q.memcpy(ptr(),b,n);}
 Bytes get()const{Bytes b(bytes);q.memcpy(b.data(),ptr(),bytes).wait_and_throw();return b;}
 void guards()const{std::array<uint8_t,2*G> b;q.memcpy(b.data(),raw,G);q.memcpy(b.data()+G,raw+G+bytes,G).wait_and_throw();for(auto v:b)strict(v==0xd7,std::string("guard ")+name);}
 void readonly(const void* b)const{auto actual=get();strict(!std::memcmp(actual.data(),b,bytes),std::string("readonly ")+name);}
 ~Buffer(){try{q.wait_and_throw();sycl::free(raw,q);}catch(...){}}
};
// Same drained ownership pattern as retained S65. Graphs die before their buffers/queue.
struct Graph {
 sycl::queue& q;dpct::experimental::command_graph_ptr graph=nullptr;dpct::experimental::command_graph_exec_ptr exec=nullptr;
 ~Graph(){try{q.wait_and_throw();delete exec;delete graph;}catch(...){}}
 template<class C>void capture(C call,const fs::path& dot){q.wait_and_throw();dpct::experimental::begin_recording(&q);call();dpct::experimental::end_recording(&q,&graph);need(graph!=nullptr,"null graph");graph->print_graph(dot.string(),true);need(fs::file_size(dot)>0,"DOT empty");exec=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());}
 void run(){q.ext_oneapi_graph(*exec).wait_and_throw();}
};
struct Inputs {
 sycl::queue& q;fs::path out;std::array<Bytes,2> hostw;std::array<std::unique_ptr<Buffer>,2>w;
 std::array<std::vector<float>,2>x;std::array<Bytes,2>expectedq;Buffer dx,xq;int current=-1;
 Inputs(sycl::queue& s,const fs::path& compact,const fs::path& output):q(s),out(output),dx(q,size_t(T)*N*4,"FP32 input"),xq(q,size_t(T)*(N/32)*36+TAIL,"Q8 input"){
  need(fs::file_size(compact)==28160000,"S42 compact file size");std::ifstream f(compact,std::ios::binary);need(bool(f),"open frozen S42 compact weights");
  for(int i=0;i<2;++i){hostw[i].assign(BLOB,0x5a);f.seekg(i?20992000:3072000);f.read(reinterpret_cast<char*>(hostw[i].data()),GU);need(size_t(f.gcount())==GU,"short GU slices");w[i]=std::make_unique<Buffer>(q,BLOB,"actual expert GU + unread down sentinel");w[i]->put(hostw[i].data(),BLOB);std::printf("{\"type\":\"weight_identity\",\"layer\":0,\"expert\":%d,\"gu_type\":17,\"down_type\":20,\"compact_offset\":%d,\"read_bytes\":%zu,\"GU_fnv\":\"%016llx\",\"down_payload_read\":false}\n",i?211:41,i?20992000:3072000,GU,(unsigned long long)fnv(Bytes(hostw[i].begin(),hostw[i].begin()+GU)));}
  f.close();need(bool(f),"close compact weights");x[0].resize(size_t(T)*N);x[1].resize(x[0].size());std::mt19937 rng(7);std::normal_distribution<float> nd(0,1);
  for(size_t i=0;i<x[0].size();++i){x[0][i]=nd(rng);x[1][i]=-.7f*x[0][i]+.001f*float(i%17);need(std::isfinite(x[0][i])&&std::isfinite(x[1][i])&&std::abs(x[0][i])<64&&std::abs(x[1][i])<64,"bounded input");}
 }
 void input(int b){need(b==0||b==1,"input index");q.wait_and_throw();current=b;dx.put(x[b].data(),dx.bytes);q.memset(xq.ptr(),0xff,xq.bytes);K::native_quantize_q8_1(dx.ptr<float>(),xq.ptr(),N,T,&q);q.wait_and_throw();auto v=xq.get();for(size_t i=v.size()-TAIL;i<v.size();++i)strict(v[i]==0xff,"quantizer tail");if(expectedq[b].empty()){expectedq[b]=v;save(out/(b?"input-b.bin":"input-a.bin"),x[b].data(),dx.bytes);save(out/(b?"canonical-b-q8.bin":"canonical-a-q8.bin"),v.data(),v.size());}else strict(expectedq[b]==v,"canonical Q8 restore");}
 void check(){dx.guards();xq.guards();dx.readonly(x[current].data());xq.readonly(expectedq[current].data());for(int i=0;i<2;++i){w[i]->guards();w[i]->readonly(hostw[i].data());}}
};
struct State {
 sycl::queue& q;Inputs& d;int layout,ng,end;fs::path out;K::NativeExpertLayout L;
 std::array<unsigned long long,CG> hp{};std::array<int32_t,CG+1> hs{};std::array<int32_t,CE> ht{},hd{};int32_t hn;
 Buffer p,s,t,dst,n,gate,up,scratch,dummy;std::vector<float> sentinel;std::array<Bytes,2>expected;
 Graph orig,multi,orig20,multi20;
 State(sycl::queue& queue,Inputs& data,int which,const fs::path& base):q(queue),d(data),layout(which),ng(which==1?0:6),end(which==1?0:which==2?8:23),out(base/(which==1?"zero":which==2?"singleton":"active")),L(K::native_expert_layout(17,20,N,FF)),hn(ng),p(q,sizeof(hp),"group pointers"),s(q,sizeof(hs),"group starts"),t(q,sizeof(ht),"entry tokens"),dst(q,sizeof(hd),"entry destinations"),n(q,sizeof(hn),"group count"),gate(q,F,"gate"),up(q,F,"up"),scratch(q,K::native_expert_scratch_bytes(CE,FF),"public phase1 scratch"),dummy(q,size_t(CE)*N*4,"phase1 down output untouched"),sentinel(size_t(CE)*FF,123.0f),orig{q},multi{q},orig20{q},multi20{q}{
  need(fs::create_directory(out),"new layout directory");need(L.gu_row==740&&L.up_off==PLANE&&L.down_off==GU&&L.bytes==BLOB&&FA==F,"exact QT17 layout");
  for(int g=0;g<CG;++g)hp[g]=(unsigned long long)d.w[g%2]->ptr();hs.fill(which==1?5:end);hs[0]=which==1?5:2;for(int g=0;g<ng;++g)hs[g+1]=hs[g]+(which==2?1:g+1);
  for(int e=0;e<CE;++e){ht[e]=(e*5+e/3)%T;hd[e]=(e*7)%CE;}p.put(hp.data(),p.bytes);s.put(hs.data(),s.bytes);t.put(ht.data(),t.bytes);dst.put(hd.data(),dst.bytes);n.put(&hn,n.bytes);
  std::printf("{\"type\":\"route\",\"layout\":%d,\"groups\":%d,\"cap_groups\":8,\"grid_groups\":2,\"cap_entries\":32,\"first_entry\":%d,\"end_entry\":%d,\"sizes\":%s,\"input_tokens\":6,\"repeated_group_pointers_synthetic\":true}\n",layout,ng,which==1?5:2,end,which==1?"[]":which==2?"[1,1,1,1,1,1]":"[1,2,3,4,5,6]");
 }
 void reset(){q.wait_and_throw();gate.put(sentinel.data(),F);up.put(sentinel.data(),F);q.memset(scratch.ptr(),0,scratch.bytes);q.memcpy(scratch.ptr(),sentinel.data(),F);q.memcpy(scratch.ptr()+FA,sentinel.data(),F);q.memset(dummy.ptr(),0xff,dummy.bytes);q.wait_and_throw();}
 void call(int arm){if(arm==0)K::native_expert_grouped(L,p.ptr<unsigned long long>(),s.ptr<int32_t>(),n.ptr<int32_t>(),dst.ptr<int32_t>(),t.ptr<int32_t>(),CG,CE,d.xq.ptr(),scratch.ptr(),dummy.ptr<float>(),&q,GRID);else if(arm==1)s98::launch_original(p.ptr<unsigned long long>(),s.ptr<int32_t>(),n.ptr<int32_t>(),t.ptr<int32_t>(),d.xq.ptr<block_q8_1>(),L,gate.ptr<float>(),up.ptr<float>(),GRID,&q);else if(arm==2)s98::launch_multi(p.ptr<unsigned long long>(),s.ptr<int32_t>(),n.ptr<int32_t>(),t.ptr<int32_t>(),d.xq.ptr<block_q8_1>(),L,gate.ptr<float>(),up.ptr<float>(),GRID,&q);else need(false,"arm");}
 Bytes snapshot(int arm,const std::string& phase,bool export_bytes){
  q.wait_and_throw();Bytes a(2*F);if(arm==0){q.memcpy(a.data(),scratch.ptr(),F);q.memcpy(a.data()+F,scratch.ptr()+FA,F).wait_and_throw();}else{q.memcpy(a.data(),gate.ptr(),F);q.memcpy(a.data()+F,up.ptr(),F).wait_and_throw();}
  if(export_bytes)save(out/(phase+".bin"),a.data(),a.size());
  for(auto b:{&p,&s,&t,&dst,&n,&gate,&up,&scratch,&dummy})b->guards();d.check();p.readonly(hp.data());s.readonly(hs.data());t.readonly(ht.data());dst.readonly(hd.data());n.readonly(&hn);
  auto dy=dummy.get();for(auto v:dy)strict(v==0xff,"phase1 wrote down output");
  for(int plane=0;plane<2;++plane)for(int e=0;e<CE;++e)for(int col=0;col<FF;++col){float v;const size_t i=(size_t(plane)*CE*FF+size_t(e)*FF+col)*4;std::memcpy(&v,a.data()+i,4);strict(std::isfinite(v),"nonfinite GU");if(!ng||e<2||e>=end)strict(!std::memcmp(&v,&sentinel[size_t(e)*FF+col],4),"inactive prefix/capacity overwritten");}
  std::printf("{\"type\":\"checked_snapshot\",\"layout\":%d,\"phase\":\"%s\",\"input\":%d,\"arm\":%d,\"bytes\":%zu,\"guards\":true,\"readonly\":true,\"finite\":true,\"inactive\":true}\n",layout,phase.c_str(),d.current,arm,a.size());return a;
 }
 void equal(int arm,const std::string& phase,bool save_bytes){auto v=snapshot(arm,phase,save_bytes);if(v!=expected[d.current])save(out/(phase+"-mismatch.bin"),v.data(),v.size());strict(v==expected[d.current],"GU full byte equality "+phase);}
 void correctness(){
  for(int b:{0,1}){d.input(b);std::string z=b?"b":"a";reset();call(0);expected[b]=snapshot(0,"eager-public-"+z,true);for(int arm:{1,2}){reset();call(arm);equal(arm,"eager-"+std::to_string(arm)+"-"+z,true);reset();call(arm);equal(arm,"repeat-"+std::to_string(arm)+"-"+z,false);}}
  strict(d.expectedq[0]!=d.expectedq[1],"changed Q8 observable");if(ng)strict(expected[0]!=expected[1],"changed GU observable");
  d.input(0);orig.capture([&]{call(1);},out/"single-original.dot");multi.capture([&]{call(2);},out/"single-multi.dot");
  if(ng){orig20.capture([&]{for(int i=0;i<BATCH;++i)call(1);},out/"batch20-original.dot");multi20.capture([&]{for(int i=0;i<BATCH;++i)call(2);},out/"batch20-multi.dot");}
  for(int b:{0,1,0}){d.input(b);std::string z=b?"b":"a";for(int arm:{1,2}){reset();(arm==1?orig:multi).run();equal(arm,"graph-"+std::to_string(arm)+"-"+z,false);if(ng){reset();(arm==1?orig20:multi20).run();equal(arm,"batch20-"+std::to_string(arm)+"-"+z,false);}}}
  std::printf("{\"type\":\"correctness_pass\",\"layout\":%d,\"initial_changed_restored\":true,\"zero_observability_vacuous\":%s}\n",layout,ng?"false":"true");
 }
 void timing(){
  need(ng&&d.current==0,"active initial timing only");for(int i=0;i<WARM;++i)for(int arm:{1,2}){reset();(arm==1?orig20:multi20).run();equal(arm,"warm",false);}
  for(int r=0;r<ROUNDS;++r)for(int pos=0;pos<2;++pos){int arm=(r%2?2-pos:1+pos);reset();auto tb=q.ext_oneapi_submit_barrier();tb.wait_and_throw();auto a=Clock::now();q.ext_oneapi_graph(*(arm==1?orig20.exec:multi20.exec));auto b=Clock::now();auto te=q.ext_oneapi_submit_barrier();te.wait_and_throw();auto c=Clock::now();auto x=tb.get_profiling_info<sycl::info::event_profiling::command_end>(),y=te.get_profiling_info<sycl::info::event_profiling::command_end>();need(y>=x,"profiling order");equal(arm,"timed-r"+std::to_string(r)+"-p"+std::to_string(pos),false);std::printf("{\"type\":\"timing_sample\",\"layout\":%d,\"round\":%d,\"position\":%d,\"arm\":%d,\"calls\":20,\"enqueue_us\":%.6f,\"completed_us\":%.6f,\"barrier_span_us\":%.6f,\"own_full_bytes\":true,\"speed_qualified\":false}\n",layout,r,pos,arm,us(a,b),us(a,c),double(y-x)/1000.0);}
 }
};
static int run(const fs::path& out,const fs::path& compact){
 need(dpct::device_count()==1,"one visible device");const sycl::device dev=dpct::dev_mgr::instance().get_device(0);need(dev.has(sycl::aspect::ext_intel_pci_address),"PCI required");auto pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();std::transform(pci.begin(),pci.end(),pci.begin(),[](unsigned char c){return char(std::tolower(c));});need(pci=="0000:0b:00.0","physical card0 required");
 sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(),dev,{sycl::property::queue::in_order(),sycl::property::queue::enable_profiling()});
 K::native_expert_set_mode(0,1);K::native_grouped_set_v1(false);K::iq_set_old_kernels(false);
 Inputs data(q,compact,out);State active(q,data,0,out),single(q,data,2,out),zero(q,data,1,out);active.correctness();single.correctness();zero.correctness();data.input(0);active.timing();single.timing();q.wait_and_throw();
 std::string result="{\"status\":\"PASS_PRIMITIVE_ONLY\",\"layouts\":3,\"inputs\":2,\"public_anchor\":true,\"host_binding_preflight_required\":true,\"speed_qualified\":false,\"serving_qualified\":false}\n";save(out/"result.json",result.data(),result.size());return 0;
}
int main(int argc,char** argv){setvbuf(stdout,nullptr,_IONBF,0);fs::path out;bool created=false;try{need(argc==3,"usage: routed_iq2_leaf NEW_OUT S42_WEIGHTS");out=argv[1];need(fs::create_directory(out),"new output directory");created=true;return run(out,argv[2]);}catch(const std::exception& e){int rc=dynamic_cast<const Strict*>(&e)?5:2;std::fprintf(stderr,"S98 %s: %s\n",rc==5?"STRICT_FAILURE":"RUNTIME_OR_IO_FAILURE",e.what());if(created)try{std::string s="{\"status\":\"FAIL\",\"exit\":"+std::to_string(rc)+"}\n";save(out/"result.json",s.data(),s.size());}catch(...){return 2;}return rc;}}
