// S41 source-only: current real Q6 bytes + production quantize/MMVQ/pack; no kernel math.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace K=strata::kernels; namespace fs=std::filesystem;
constexpr int MAXC=8;constexpr size_t GUARD=256;
static const int columns[]={6,1,2,3,4,8};
static void need(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
struct Fixture{const char* name;int ni,no;uint64_t offset,bytes;};
static const Fixture fixtures[]={
 {"blk.3.attn_output.weight",6144,2560,3572274688ull,12902400},
 {"blk.4.attn_qkv.weight",2560,10240,4312859136ull,21504000},
 {"blk.27.attn_q.weight",2560,12288,24116060928ull,25804800},
 {"blk.1.attn_gate.weight",2560,6144,1752472832ull,12902400}};
static void save_input(const fs::path& path,const void* data,size_t bytes){std::ofstream stream(path,std::ios::binary);stream.write(reinterpret_cast<const char*>(data),bytes);stream.close();need(bool(stream),"input artifact write failed");}
static uint64_t fnv(const void* p,size_t n){uint64_t h=0xcbf29ce484222325ull;for(size_t i=0;i<n;++i){h^=static_cast<const uint8_t*>(p)[i];h*=0x100000001b3ull;}return h;}
struct Header {
 FILE* f;uint64_t size;std::vector<uint64_t> found;
 Header(const char* file):f(std::fopen(file,"rb")),size(fs::file_size(file)),found(4,0){
  need(f&&size==47039860096ull,"expected current shard1 size");char magic[4];read(magic,4);need(!std::memcmp(magic,"GGUF",4)&&u32()==3,"GGUFv3 required");
  auto nt=u64(),nk=u64();uint64_t align=32;need(nt==1223&&nk<100000,"unexpected tensor/metadata count");
  for(uint64_t i=0;i<nk;++i){auto key=str();auto type=u32();if(key=="general.alignment"){need(type==4,"bad alignment type");align=u32();}else skip(type);}
  std::vector<uint64_t> rel(4,0);
  for(uint64_t i=0;i<nt;++i){auto name=str();auto nd=u32();need(nd>=1&&nd<=4,"bad dimensions");std::vector<uint64_t>d(nd);for(auto& x:d)x=u64();auto type=u32();auto off=u64();
   for(size_t j=0;j<4;++j)if(name==fixtures[j].name){need(found[j]==0&&type==14&&nd==2&&d[0]==uint64_t(fixtures[j].ni)&&d[1]==uint64_t(fixtures[j].no),"tensor type/shape/duplicate");found[j]=1;rel[j]=off;}}
  need(align>0&&align<=4096&&(align&(align-1))==0,"bad alignment");auto at=uint64_t(ftello(f));auto base=(at+align-1)/align*align;need(base==11024256,"unexpected header/data base");
  for(size_t j=0;j<4;++j){need(found[j]==1&&rel[j]<=size-base,"missing/out-of-bounds tensor");found[j]=base+rel[j];need(found[j]==fixtures[j].offset&&fixtures[j].bytes<=size-found[j],"payload lookup/bounds mismatch");}
 }
 void read(void* p,size_t n){need(std::fread(p,1,n,f)==n,"truncated file");}
 uint32_t u32(){uint32_t x;read(&x,4);return x;}uint64_t u64(){uint64_t x;read(&x,8);return x;}
 std::string str(){auto n=u64();need(n<=1<<24,"oversized header string");std::string s(n,'\0');read(s.data(),n);return s;}
 void jump(uint64_t n){auto at=uint64_t(ftello(f));need(at<=size&&n<=size-at,"header skip bounds");need(fseeko(f,off_t(n),SEEK_CUR)==0,"header seek");}
 void skip(uint32_t type,int depth=0){need(depth<8,"metadata depth");if(type==8){jump(u64());return;}if(type==9){auto e=u32();auto n=u64();need(n<=size,"metadata array bounds");for(uint64_t i=0;i<n;++i)skip(e,depth+1);return;}
  static const int sizes[]={1,1,2,2,4,4,4,1,0,0,8,8,8};need(type<13&&sizes[type]>0,"unknown metadata type");jump(sizes[type]);}
 std::vector<uint8_t> payload(size_t i){const auto& a=fixtures[i];need(fseeko(f,off_t(found[i]),SEEK_SET)==0,"payload seek");std::vector<uint8_t>w(a.bytes);read(w.data(),w.size());return w;}
 ~Header(){if(f)std::fclose(f);}
};
struct Buffer {
 sycl::queue& q;size_t bytes;uint8_t* raw;const char* name;
 Buffer(sycl::queue& s,size_t n,const char* tag):q(s),bytes(n),name(tag){raw=sycl::malloc_device<uint8_t>(bytes+2*GUARD,q);need(raw!=nullptr,"USM allocation");q.memset(raw,0xd7,bytes+2*GUARD);q.memset(raw+GUARD,0xff,bytes);}
 template<class T>T* ptr()const{return reinterpret_cast<T*>(raw+GUARD);}
 void put(const void* p){q.memcpy(raw+GUARD,p,bytes);}
 std::vector<uint8_t> get(){std::vector<uint8_t>h(bytes);q.memcpy(h.data(),raw+GUARD,bytes).wait_and_throw();return h;}
 void guard(){uint8_t h[2*GUARD];q.memcpy(h,raw,GUARD);q.memcpy(h+GUARD,raw+GUARD+bytes,GUARD).wait_and_throw();for(auto b:h)need(b==0xd7,std::string("guard ")+name);}
 ~Buffer(){try{q.wait_and_throw();sycl::free(raw,q);}catch(...){}}
};
struct Packed {sycl::queue& q;void* w;bool live=false;~Packed(){try{q.wait_and_throw();if(live)K::native_q6_k_unpack(w);}catch(...){}}};
struct Graph {sycl::queue& q;dpct::experimental::command_graph_ptr g=nullptr;dpct::experimental::command_graph_exec_ptr e=nullptr;
 ~Graph(){try{q.wait_and_throw();delete e;delete g;}catch(...){}}
 void capture(const auto& call){dpct::experimental::begin_recording(&q);call();dpct::experimental::end_recording(&q,&g);need(g!=nullptr,"null graph");e=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());}
 void replay(){q.ext_oneapi_graph(*e).wait_and_throw();}};
static void eq(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b,const std::string& what){
 if(a!=b){size_t at=0;while(at<std::min(a.size(),b.size())&&a[at]==b[at])++at;throw std::runtime_error(what+" first_byte="+std::to_string(at));}}
static int run(int argc,char** argv){
 need(argc==3,"usage: q6_dense_oracle CURRENT_SHARD1 NEW_OUTPUT_DIR");Header file(argv[1]);const fs::path out=argv[2];need(fs::create_directory(out),"new output dir required");
 for(const char* v:{"STRATA_Q6P_SELFTEST","STRATA_TSUM"})need(!std::getenv(v)||std::string(std::getenv(v))=="0",std::string("disable ")+v);
 auto env=[](const char* k,const char* value){auto v=std::getenv(k);need(v&&std::string(v)==value,std::string("require ")+k+"="+value);};
 env("STRATA_Q6_PACKED","1");env("STRATA_Q6P_GRID","640");env("STRATA_MMVQ_WIDE","1");env("STRATA_MMVQ_A2","1");env("STRATA_MMVQ_WIDE_RPW","1");env("STRATA_MMVQ_PF","0");env("STRATA_MMVQ_WIDE_MIN","0");env("STRATA_MMVQ_WIDE_MAXC","8");
 need(K::native_mmvq_multi_exact()&&dpct::device_count()==1,"exact mode and one device required");
 const sycl::device dev=dpct::dev_mgr::instance().get_device(0);need(dev.has(sycl::aspect::ext_intel_pci_address),"PCI unavailable");auto pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();need(pci=="0000:44:00.0","physical card1 required");
 sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(),dev,sycl::property_list{sycl::property::queue::in_order(),sycl::property::queue::enable_profiling()});
 for(size_t fi=0;fi<4;++fi){const auto& f=fixtures[fi];auto weights=file.payload(fi);need(K::native_mmvq_weight_bytes(14,f.ni,f.no)==weights.size(),"production weight-byte mismatch");
  std::printf("{\"type\":\"fixture\",\"tensor\":\"%s\",\"n_in\":%d,\"n_out\":%d,\"offset\":%llu,\"bytes\":%zu,\"fnv1a64\":\"%016llx\"}\n",f.name,f.ni,f.no,(unsigned long long)f.offset,weights.size(),(unsigned long long)fnv(weights.data(),weights.size()));
  std::mt19937 rng(7);std::normal_distribution<float> nd(0.f,1.f);std::vector<float>x(size_t(MAXC)*f.ni);for(auto& v:x)v=nd(rng);auto xb=x;for(size_t i=0;i<xb.size();++i)xb[i]=-.7f*xb[i]+.001f*float(i%17);
  Buffer w(q,weights.size(),"weights"),dx(q,x.size()*4,"input"),xq(q,K::native_q8_1_bytes(f.ni,MAXC),"q8_1"),y(q,size_t(MAXC)*f.no*4,"output");
  w.put(weights.data());dx.put(x.data());q.wait_and_throw();Packed packed{q,w.ptr<void>()};Graph original{q},candidate{q};
  auto guards=[&]{for(auto* p:{&w,&dx,&xq,&y})p->guard();};
  auto poison=[&]{q.wait_and_throw();guards();q.memset(y.ptr<void>(),0xff,y.bytes).wait_and_throw();};
  auto quant=[&](int C){K::native_quantize_q8_1(dx.ptr<float>(),xq.ptr<void>(),f.ni,C,&q);};
  auto mm=[&](int C){K::native_mmvq(14,w.ptr<void>(),xq.ptr<void>(),y.ptr<float>(),f.ni,f.no,C,&q);};
  auto observe=[&](int C,const char* phase){poison();const auto a=std::chrono::steady_clock::now();mm(C);const auto b=std::chrono::steady_clock::now();q.wait_and_throw();const auto z=std::chrono::steady_clock::now();
   std::printf("{\"type\":\"call_observation\",\"tensor\":\"%s\",\"C\":%d,\"phase\":\"%s\",\"host_enqueue_us\":%.3f,\"host_complete_us\":%.3f,\"timing_not_comparable\":true}\n",f.name,C,phase,std::chrono::duration<double,std::micro>(b-a).count(),std::chrono::duration<double,std::micro>(z-a).count());};
  auto snapshot=[&](int C,const std::string& phase){q.wait_and_throw();guards();auto data=y.get();
   for(size_t i=0;i<size_t(C)*f.no;++i){float v;std::memcpy(&v,data.data()+i*4,4);need(std::isfinite(v),"nonfinite output");}
   for(size_t i=size_t(C)*f.no*4;i<data.size();++i)need(data[i]==0xff,"output inactive/tail overwritten");
   std::ofstream s(out/("f"+std::to_string(fi)+"-C"+std::to_string(C)+"-"+phase+".bin"),std::ios::binary);s.write(reinterpret_cast<const char*>(data.data()),data.size());s.close();need(bool(s),"snapshot write");return data;};
  quant(MAXC);q.wait_and_throw();const auto qbytes=xq.get();std::vector<std::vector<uint8_t>> base;
  save_input(out/("f"+std::to_string(fi)+"-input-f32.bin"),x.data(),x.size()*sizeof(float));
  save_input(out/("f"+std::to_string(fi)+"-input-q8_1.bin"),qbytes.data(),qbytes.size());
  for(int C:columns){observe(C,"original");base.push_back(snapshot(C,"original"));eq(xq.get(),qbytes,"quantized input modified");}
  eq(w.get(),weights,"original weight bytes modified before pack");auto original_x=dx.get();need(!std::memcmp(original_x.data(),x.data(),original_x.size()),"FP32 input modified before pack");
  // Record original dispatch before registration; do not execute graph until direct parity passes.
  // Both graphs die before unpack/free. Registry changes cannot alter an already captured graph.
  auto chain=[&]{quant(6);mm(6);};original.capture(chain);
  // Native pack uses the SAME dpct device/context default queue; prior uploads are drained.
  q.wait_and_throw();need(K::native_q6_k_pack(w.ptr<void>(),f.ni,f.no,f.name),"existing pack rejected");packed.live=true;eq(w.get(),weights,"original weight bytes modified by pack");
  size_t ci=0;for(int C:columns){observe(C,"packed");auto got=snapshot(C,"packed");eq(xq.get(),qbytes,"packed input modified");eq(got,base[ci++],std::string("STRICT original/packed ")+f.name+" C"+std::to_string(C));}
  // Only after every direct comparison passes, execute C6 changed-input graphs.
  candidate.capture(chain);poison();original.replay();eq(snapshot(6,"original-graph-original"),base[0],"original initial graph");
  poison();candidate.replay();eq(snapshot(6,"packed-graph-original"),base[0],"packed initial graph");
  dx.put(xb.data());poison();original.replay();const auto changed=snapshot(6,"original-graph-changed");const auto changed_q=xq.get();
  need(changed!=base[0],"original changed input not observable");need(changed_q!=qbytes,"changed quantized input not observable");
  poison();candidate.replay();auto changed_packed=snapshot(6,"packed-graph-changed");eq(changed_packed,changed,"packed changed graph");eq(xq.get(),changed_q,"changed graph quantized input mismatch");
  eq(w.get(),weights,"original weight bytes modified");auto gotx=dx.get();need(!std::memcmp(gotx.data(),xb.data(),gotx.size()),"FP32 input modified");guards();
  std::printf("{\"type\":\"fixture_PASS\",\"tensor\":\"%s\",\"columns\":[6,1,2,3,4,8],\"changed_C6_graph\":true,\"full_memcmp\":true,\"guards\":true,\"inputs_unchanged\":true}\n",f.name);
 }
 q.wait_and_throw();std::printf("{\"status\":\"PASS_FOUR_REAL_Q6_SHAPES_STRICT\",\"fixtures\":4,\"column_pairs\":24,\"pci\":\"%s\"}\n",pci.c_str());return 0;
}
int main(int argc,char** argv){setvbuf(stdout,nullptr,_IONBF,0);try{return run(argc,argv);}catch(const std::exception& e){std::fprintf(stderr,"q6_dense_oracle FAIL_FAST: %s\n",e.what());return 5;}}
