// SOURCE ONLY: same-context two-queue graph/barrier ordering; no model math or speed claim.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_doorbell.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
namespace xq {
constexpr size_t N=257,G=32,S=N+2*G,DEPTH=8,WORDS=(1+3*DEPTH)*S;
constexpr uint32_t GUARD=0xa5a5a5a5u;
class PreA;class PreB;class TailData;class TailAdvance;
static bool cleanup_error=false;
static void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void save(const fs::path& p,const void* b,size_t n){need(!fs::exists(p),"refuse overwrite");std::ofstream f(p,std::ios::binary);need(bool(f),"open output");f.write(static_cast<const char*>(b),n);f.close();need(bool(f),"close output");}
static uint32_t input(size_t i,unsigned phase){return (uint32_t(i)*747796405u+(phase==1?0x1234567bu:0x91e10da5u))^0xa341316cu;}
static uint32_t pre(uint32_t previous,uint32_t value,unsigned role,unsigned depth){return (previous^(role?0x55aa33ccu:0u))*1664525u+(value^((2*depth+role)*0x9e3779b9u))+1013904223u;}
static uint32_t tail(uint32_t a,uint32_t b){return (a*2246822519u)^(b+3266489917u);}
static void drain(sycl::queue& a,sycl::queue& b)noexcept{try{a.wait();}catch(...){cleanup_error=true;}try{b.wait();}catch(...){cleanup_error=true;}}
template<unsigned ROLE,class Name>static void producer(sycl::queue& q,const uint32_t* mapped,uint32_t* data){
 q.parallel_for<Name>(sycl::nd_range<1>(512,256),[=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
  const size_t i=it.get_global_id(0);if(i>=N)return;const unsigned d=data[G];
  if(d>=DEPTH){if(i==0)data[G+1+ROLE]=1;return;}
  const uint32_t previous=d?data[(1+3*(d-1)+2)*S+G+i]:uint32_t(i);
  const uint32_t value=strata::sys_load_mapped(mapped+i);
  data[(1+3*d+ROLE)*S+G+i]=(previous^(ROLE?0x55aa33ccu:0u))*1664525u+(value^((2*d+ROLE)*0x9e3779b9u))+1013904223u;
 });
}
static void consumer(sycl::queue& q,uint32_t* data){
 q.parallel_for<TailData>(sycl::nd_range<1>(512,256),[=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
  const size_t i=it.get_global_id(0);if(i>=N)return;const unsigned d=data[G];
  if(d>=DEPTH){if(i==0)data[G+3]=1;return;}
  const uint32_t a=data[(1+3*d)*S+G+i],b=data[(1+3*d+1)*S+G+i];
  data[(1+3*d+2)*S+G+i]=(a*2246822519u)^(b+3266489917u);
 });
 // Separate ordered node: no work-group/global race between a counter update and tail readers.
 q.parallel_for<TailAdvance>(sycl::nd_range<1>(32,32),[=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{if(it.get_global_id(0)==0&&data[G]<DEPTH)++data[G];});
}
struct Storage{
 sycl::queue& a;sycl::queue& b;uint32_t* mapped=nullptr;uint32_t* data=nullptr;
 Storage(sycl::queue& qa,sycl::queue& qb):a(qa),b(qb){
  mapped=static_cast<uint32_t*>(strata::host_malloc_polled(S*sizeof(uint32_t),a));
  try{need(mapped!=nullptr,"mapped allocation");{std::lock_guard<std::mutex> lock(strata::detail::uncached_mu());need(strata::detail::uncached_set().count(mapped)==1,"require uncached zeMemAllocHost, no fallback");}
   data=sycl::malloc_device<uint32_t>(WORDS,a);need(data!=nullptr,"device allocation");
  }catch(...){if(mapped)strata::host_free_polled(mapped,a);throw;}
 }
 void close()noexcept{if(!data&&!mapped)return;drain(a,b);if(data){try{sycl::free(data,a);}catch(...){cleanup_error=true;}data=nullptr;}if(mapped){try{strata::host_free_polled(mapped,a);}catch(...){cleanup_error=true;}mapped=nullptr;}}
 ~Storage(){close();}
};
struct Graph{
 sycl::queue& q;sycl::queue& other;dpct::experimental::command_graph_exec_ptr exec=nullptr;
 Graph(sycl::queue& own,sycl::queue& peer):q(own),other(peer){}
 template<class F>void capture(F record,const fs::path& dot){
  q.wait_and_throw();need(exec==nullptr,"capture once");dpct::experimental::command_graph_ptr graph=nullptr;bool begun=false,ended=false;
  try{begun=true;dpct::experimental::begin_recording(&q);record();dpct::experimental::end_recording(&q,&graph);ended=true;need(graph!=nullptr,"null graph");graph->print_graph(dot.string(),true);need(fs::is_regular_file(dot)&&fs::file_size(dot)>0,"missing DOT");exec=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());delete graph;graph=nullptr;q.wait_and_throw();}
  catch(...){const auto original=std::current_exception();if(begun&&!ended){try{if(graph)graph->end_recording();else dpct::experimental::end_recording(&q,&graph);}catch(...){cleanup_error=true;}}drain(q,other);delete graph;std::rethrow_exception(original);}
 }
 sycl::event submit(){return q.ext_oneapi_graph(*exec);}
 ~Graph(){drain(q,other);delete exec;}
};
static void run(unsigned mode,const fs::path& out){
 need(dpct::device_count()==1,"one visible GPU");const sycl::device dev=dpct::dev_mgr::instance().get_device(0);need(dev.has(sycl::aspect::ext_intel_pci_address),"PCI required");
 auto pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();std::transform(pci.begin(),pci.end(),pci.begin(),[](unsigned char c){return char(std::tolower(c));});need(pci=="0000:0b:00.0","physical card0 required");
 const auto ctx=dpct::dev_mgr::instance().get_device(0).get_context();
 auto async=[](sycl::exception_list list){for(auto e:list)std::rethrow_exception(e);};
 sycl::queue a(ctx,dev,async,{sycl::property::queue::in_order()}),b(ctx,dev,async,{sycl::property::queue::in_order()});
 need(a!=b&&a.get_context()==b.get_context()&&a.get_device()==b.get_device(),"distinct queues/same context/device required");need(a.has_property<sycl::property::queue::in_order>()&&b.has_property<sycl::property::queue::in_order>(),"in-order required");need(a.get_backend()==sycl::backend::ext_oneapi_level_zero&&b.get_backend()==sycl::backend::ext_oneapi_level_zero,"Level Zero required");
 Storage data(a,b);
 {Graph ga(a,b),gb(b,a),gt(a,b);ga.capture([&]{producer<0,PreA>(a,data.mapped+G,data.data);},out/"pre-a.dot");gb.capture([&]{producer<1,PreB>(b,data.mapped+G,data.data);},out/"pre-b.dot");gt.capture([&]{consumer(a,data.data);},out/"tail.dot");
  std::vector<uint32_t> first;
  for(unsigned phase=0;phase<3;++phase){
   a.wait_and_throw();b.wait_and_throw();std::vector<uint32_t> host(S,GUARD);for(size_t i=0;i<N;++i)host[G+i]=input(i,phase);std::memcpy(data.mapped,host.data(),S*4);std::atomic_thread_fence(std::memory_order_seq_cst);
   a.memset(data.data,0xa5,WORDS*4);a.memset(data.data+G,0,4*sizeof(uint32_t)).wait_and_throw();
   sycl::event previous_tail;
   for(unsigned d=0;d<DEPTH;++d){
    if(mode&&d>0)(void)b.ext_oneapi_submit_barrier(std::vector<sycl::event>{previous_tail});
    const auto ea=ga.submit();const auto eb=gb.submit();
    if(mode)(void)a.ext_oneapi_submit_barrier(std::vector<sycl::event>{ea,eb});
    else{a.wait_and_throw();b.wait_and_throw();}
    previous_tail=gt.submit();if(!mode)a.wait_and_throw();
   }
   // API discriminator enqueues the complete dependency window; production h_out/CPU PLE still needs a separate decision.
   a.wait_and_throw();b.wait_and_throw();std::vector<uint32_t> actual(WORDS),expected(WORDS,GUARD);a.memcpy(actual.data(),data.data,WORDS*4).wait_and_throw();save(out/("phase-"+std::to_string(phase)+".bin"),actual.data(),WORDS*4);
   need(std::memcmp(data.mapped,host.data(),S*4)==0,"mapped input/guards/read-only mismatch");expected[G]=DEPTH;expected[G+1]=expected[G+2]=expected[G+3]=0;
   for(size_t i=0;i<N;++i){uint32_t prior=uint32_t(i);for(unsigned d=0;d<DEPTH;++d){const uint32_t va=pre(prior,host[G+i],0,d),vb=pre(prior,host[G+i],1,d);const uint32_t v=tail(va,vb);expected[(1+3*d)*S+G+i]=va;expected[(1+3*d+1)*S+G+i]=vb;expected[(1+3*d+2)*S+G+i]=v;prior=v;}}
   if(actual!=expected)save(out/("phase-"+std::to_string(phase)+"-expected.bin"),expected.data(),WORDS*4);need(actual==expected,"strict EVERY iteration producer/tail/counter/guard mismatch");if(phase==0)first=actual;if(phase==1)need(actual!=first,"B must change output");if(phase==2)need(actual==first,"restored A differs");
   std::printf("{\"type\":\"phase\",\"mode\":%u,\"phase\":%u,\"iterations_checked\":8,\"graph_submits_source\":24,\"crossqueue_barriers_source\":%u,\"intermediates_full_bytes\":%zu,\"strict\":true,\"readonly\":true,\"guards\":true}\n",mode,phase,mode?15u:0u,WORDS*4);
  }
  a.wait_and_throw();b.wait_and_throw();
 }
 data.close();need(!cleanup_error,"graph/storage cleanup failure");
 const std::string result="{\"status\":\"PASS_TWO_QUEUE_GRAPH_BARRIER_ORDERING_ONLY\",\"mode\":"+std::to_string(mode)+",\"same_context_device_inorder\":true,\"graphs\":3,\"graph_kernel_nodes\":[1,1,2],\"phases\":3,\"depth_per_phase\":8,\"graph_submits_source\":72,\"kernel_operations_source\":96,\"barrier_submits_source\":"+std::to_string(mode?45:0)+",\"dependency_host_waits_source\":"+std::to_string(mode?0:72)+",\"common_setup_fills\":6,\"common_full_readbacks\":3,\"bytes_per_readback\":32100,\"mapped_A_B_A\":true,\"physical_card\":\"0000:0b:00.0\",\"actual_LevelZero_counts_measured\":false,\"overlap_measured\":false,\"MTP_numeric_qualified\":false,\"speed_qualified\":false}\n";save(out/"result.json",result.data(),result.size());
}
}
int main(int argc,char** argv){setvbuf(stdout,nullptr,_IONBF,0);try{xq::need(argc==3,"usage: cross_queue_graph MODE NEW_OUT");xq::need(std::string(argv[1])=="0"||std::string(argv[1])=="1","mode0/1 only");xq::need(fs::create_directory(argv[2]),"new output required");xq::run(unsigned(argv[1][0]-'0'),argv[2]);return 0;}catch(const std::exception& e){std::fprintf(stderr,"CROSS_QUEUE STOP: %s\n",e.what());return 5;}}
