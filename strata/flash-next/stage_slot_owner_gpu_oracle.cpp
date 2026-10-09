// No-weight slot owner oracle. Normal init calls actual20 helper/owner paths.
// Fault seams are TU-only; production patch/header bytes remain unchanged.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "strata/core/session.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/qsa.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace oracle {
enum class Fault { none, null_device, partial_zero, partial_throw };
struct Frame {std::string label;int dev=0;int64_t lo=0,hi=48;uint64_t bytes=0;Fault fault=Fault::none;unsigned registered=0,init_calls=0;};
Frame* active=nullptr;unsigned total_owners=0;uint64_t live_bytes=0;constexpr uint64_t LIMIT=1ull<<30;
void require(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
std::string quote(const std::string& x){std::string r="\"";char b[8];for(unsigned char c:x){if(c=='"'||c=='\\'){r+='\\';r+=char(c);}else if(c<32||c>=127){std::snprintf(b,sizeof b,"\\u%04x",unsigned(c));r+=b;}else r+=char(c);}return r+'"';}
std::string address(const void* p){char b[40];std::snprintf(b,sizeof b,"%p",p);return quote(b);}
void mark(const std::string& stage,const char* event){std::fprintf(stderr,"UPLOAD_USM {\"event\":%s,\"stage\":%s%s}\n",quote(event).c_str(),quote(stage).c_str(),std::string(event)=="destroy_end"?",\"owning_destructor_returned\":true":"");std::fflush(stderr);}
void registration(Frame& f,const void* p,uint64_t bytes,const char* role,sycl::queue& queue){
 require(p && bytes,"Missing owned allocation descriptor");const auto type=sycl::get_pointer_type(p,queue.get_context());
 const bool host=std::string(role).find("host_")==0;
 require(type==(host?sycl::usm::alloc::host:sycl::usm::alloc::device),"Live USM kind differs");
 if(!host)require(sycl::get_pointer_device(p,queue.get_context())==queue.get_device(),"Live device owner differs");
 const auto context=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
 std::fprintf(stderr,"UPLOAD_USM {\"event\":\"owner_register\",\"stage\":%s,\"device\":%d,\"ze_context\":%s,\"pointer\":%s,\"bytes\":%llu,\"role\":%s,\"name\":%s}\n",quote(f.label).c_str(),f.dev,address(context).c_str(),address(p).c_str(),(unsigned long long)bytes,quote(role).c_str(),quote(role).c_str());
 std::fflush(stderr);++f.registered;++total_owners;
}
void register_session(void* base,strata::core::SessionState& state,const strata::core::ModelGeometry& g){
 auto& f=*active;auto& queue=dpct::get_in_order_queue();registration(f,base,f.bytes,"slot_arena",queue);
 const bool partial=f.fault==Fault::partial_zero || f.fault==Fault::partial_throw;
 require(state.qsa_ord0==f.lo/4 && state.qsa_alloc==(partial?1:(f.hi-f.lo)/4),"Global QSA ordinal/layout differs");
 require(state.gdn_alloc==(partial?3:(f.hi-f.lo)-(f.hi-f.lo)/4),"Owned GDN layer count differs");
 const auto start=reinterpret_cast<uintptr_t>(base),gd=reinterpret_cast<uintptr_t>(state.gdn_state);
 require(gd>=start && gd+64<=start+f.bytes,"GDN sentinel outside owning arena");
 for(int64_t j=0;j<state.qsa_alloc;++j){auto& q=state.qsa_states[state.qsa_ord0+j];
  registration(f,q.host_step,strata::kernels::qsa_step_bytes()+sizeof(int32_t),"host_step",queue);
  registration(f,q.host_pos,uint64_t(g.n_head)*4,"host_pos",queue);
  if(q.owned_host_kv)registration(f,q.owned_host_kv,q.owned_host_kv_bytes,"host_kv",queue);
 }
}
}
namespace sycl {
void* slot_oracle_allocate(size_t bytes,const queue& q){
 if(oracle::active && oracle::active->fault==oracle::Fault::null_device)return nullptr;
 return sycl::malloc_device(bytes,q);
}
}
namespace strata::core {
uint64_t slot_oracle_session_init(const ModelGeometry& g,int64_t cells,int64_t experts,void* base,SessionState& state,int64_t lo,int64_t hi){
 using namespace oracle;require(active!=nullptr,"Unlabelled oracle init");++active->init_calls;
 // Deliberately initialize one valid four-layer prefix into the full-sized arena,
 // then report failure. No malformed shape, oversized allocation or device OOM.
 const bool partial=active->fault==Fault::partial_zero || active->fault==Fault::partial_throw;
 const uint64_t result=session_init(g,cells,experts,base,state,lo,partial?std::min<int64_t>(hi,lo+4):hi);
 require(result!=0,"Actual session_init unexpectedly failed");register_session(base,state,g);
 if(partial){mark(active->label,"destroy_begin");if(active->fault==Fault::partial_throw)throw std::runtime_error("TU-only bounded partial-init injection");return 0;}
 return result;
}
}
// Only these two calls inside the actual owner header are interposed. All
// allocation success, geometry/carving, queue waits, release and rebind are real.
#define malloc_device slot_oracle_allocate
#define session_init slot_oracle_session_init
#include "strata/core/slot_session_arena.hpp"
#undef session_init
#undef malloc_device

namespace oracle {
struct Slot {Frame frame;std::unique_ptr<strata::core::SlotSessionArena> owner;
 ~Slot(){if(owner){mark(frame.label,"destroy_begin");owner.reset();live_bytes-=frame.bytes;mark(frame.label,"destroy_end");}}
};
void retire(Slot& s){mark(s.frame.label,"destroy_begin");s.owner.reset();live_bytes-=s.frame.bytes;mark(s.frame.label,"destroy_end");}
std::unique_ptr<Slot> allocate(const std::string& label,int dev,int64_t lo,int64_t hi,int64_t cells,Fault fault,const strata::core::ModelGeometry& g){
 auto s=std::make_unique<Slot>();s->frame={label,dev,lo,hi,strata::core::session_bytes(g,cells,10,lo,hi),fault};
 require(s->frame.bytes>0 && live_bytes+s->frame.bytes<=LIMIT,"Bounded device arena budget exceeded");
 dpct::select_device(0); // Match foreign-construction pattern corrected by owned init.
 s->owner=std::make_unique<strata::core::SlotSessionArena>(dev);active=&s->frame;std::string why;
 const bool initialized=s->owner->initialize(g,cells,10,lo,hi,s->frame.bytes,why);active=nullptr;
 if(fault==Fault::none){require(initialized,"Normal owner init: "+why);const strata::core::OnDevice owning(dev);auto& q=dpct::get_in_order_queue();strata::core::session_zero(*s->owner,g,nullptr,(void*)&q);q.wait_and_throw();live_bytes+=s->frame.bytes;return s;}
 require(!initialized && !s->owner->base() && s->owner->allocation_bytes()==0,"Injected failure retained device arena");
 require((fault==Fault::null_device && s->frame.registered==0 && s->frame.init_calls==0) ||
         (fault!=Fault::null_device && s->frame.registered>=3 && s->frame.init_calls==1),"Bounded fault path was not exercised");
 s->owner.reset();if(fault!=Fault::null_device)mark(s->frame.label,"destroy_end");return s;
}
void fresh_probe(int dev,const std::string& label){
 const strata::core::OnDevice device(dev);auto& q=dpct::get_in_order_queue();Frame frame;frame.label=label;frame.dev=dev;auto* p=sycl::malloc_device<uint8_t>(65536,q);require(p,"Fresh probe allocation failed");registration(frame,p,65536,"probe",q);
 q.memset(p,0xa5,65536).wait_and_throw();uint8_t actual=0;q.memcpy(&actual,p,1).wait_and_throw();require(actual==0xa5,"Fresh probe data mismatch");mark(label,"destroy_begin");sycl::free(p,q);q.wait_and_throw();mark(label,"destroy_end");
}
}
int main(int argc,char** argv){
 using namespace oracle;std::vector<std::array<int64_t,3>> stages;std::string output;int64_t cells=64;
 try {
  for(int i=1;i<argc;++i){std::string a=argv[i];require(i+1<argc,"Missing argument");std::string value=argv[++i];if(a=="--output")output=value;else if(a=="--cells")cells=std::stoll(value);else if(a=="--stage"){std::array<int64_t,3> s{};char c;std::istringstream in(value);require(bool(in>>s[0]>>c) && c==':' && bool(in>>s[1]>>c) && c==':' && bool(in>>s[2]) && in.eof(),"Stage must be lo:hi:device");stages.push_back(s);}else require(false,"Unknown argument");}
  require(!output.empty() && cells>=32 && cells<=256 && stages.size()>=1 && stages.size()<=2,"Bounded oracle arguments differ");
  strata::core::ModelGeometry g;unsigned completed=0;std::set<int64_t> covered;std::set<int> devices;
  for(const auto& range:stages){const int64_t lo=range[0],hi=range[1];const int dev=int(range[2]);require(devices.insert(dev).second,"Each stage needs its own device in this oracle");require(dev>=0 && dev<=1 && lo>=0 && lo%4==0 && hi<=48 && hi%4==0 && lo<hi,"Invalid stage bounds");for(auto l=lo;l<hi;++l)require(covered.insert(l).second,"Duplicate layer ownership");
   std::vector<std::unique_ptr<Slot>> slots;
   for(int n=0;n<3;++n)slots.push_back(allocate("normal-"+std::to_string(dev)+"-"+std::to_string(n),dev,lo,hi,cells,Fault::none,g));
   const strata::core::OnDevice owning(dev);auto& q=dpct::get_in_order_queue();
   for(unsigned n=0;n<slots.size();++n){require(slots[n]->owner->gdn_alloc>0,"Missing GDN ownership");std::array<uint32_t,16> pattern{};pattern.fill(0x3f800000u+n);q.memcpy(slots[n]->owner->gdn_state,pattern.data(),sizeof pattern).wait_and_throw();}
   for(unsigned n=0;n<slots.size();++n){std::array<uint32_t,16> actual{};q.memcpy(actual.data(),slots[n]->owner->gdn_state,sizeof actual).wait_and_throw();for(auto x:actual)require(x==0x3f800000u+n,"Private slot state alias or overwrite");}
   // Actual vector shrink invokes each wrapper and actual owner destructor.
   slots.resize(1);
   strata::core::qsa_state_register_existing_rope(slots[0]->owner->qsa_states[slots[0]->owner->qsa_ord0],g);
   retire(*slots[0]);slots.clear();
   allocate("null-"+std::to_string(dev),dev,lo,hi,cells,Fault::null_device,g);
   allocate("partial-zero-"+std::to_string(dev),dev,lo,hi,cells,Fault::partial_zero,g);
   allocate("partial-throw-"+std::to_string(dev),dev,lo,hi,cells,Fault::partial_throw,g);
   fresh_probe(dev,"probe-"+std::to_string(dev));++completed;
  }
  require(covered.size()==48 && live_bytes==0,"Incomplete full layer roster or live device arenas");
  std::ofstream report(output,std::ios::out|std::ios::trunc);require(bool(report),"Cannot open oracle report");report<<"{\"schema\":1,\"owner_and_probe_passed\":true,\"passed\":false,\"awaiting_logical_free_trace\":true,\"stages\":"<<completed<<",\"owner_registrations\":"<<total_owners<<",\"cells\":"<<cells<<",\"model_weights_loaded\":false,\"inference_or_graph_retirement_qualified\":false,\"fault_seams\":\"TU-only allocator-null and actual valid four-layer partial init zero/throw\"}\n";report.close();std::puts("PASS bounded no-weight owner/probe; logical-free trace still required");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL slot owner oracle: %s\n",e.what());return 1;}
}
