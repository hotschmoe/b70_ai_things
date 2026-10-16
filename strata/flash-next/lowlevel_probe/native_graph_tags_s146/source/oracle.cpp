// S145 SOURCE ONLY. PUBLIC production control, private exact-body layer wrappers.
#include "private_gdn.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#include <optional>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace K=strata::kernels;namespace fs=std::filesystem;using Clock=std::chrono::steady_clock;using Bytes=std::vector<unsigned char>;
constexpr int L=36,OWN=2,C=10240,HV=48,HK=16,MT=8,ROWS=3,S=128;
constexpr size_t REC=size_t(S)*HV*S,CONV=size_t(C)*3,STRIDE=REC+CONV,STATE=L*STRIDE,G=256;
bool cleanup_error=false;
void need(bool b,const std::string&m){if(!b)throw std::runtime_error(m);}
void eq(const Bytes&a,const Bytes&b,const std::string&m){if(a!=b){size_t i=0;while(i<std::min(a.size(),b.size())&&a[i]==b[i])++i;throw std::runtime_error(m+" first_byte="+std::to_string(i));}}
uint64_t fnv(const Bytes&x){uint64_t h=14695981039346656037ull;for(auto b:x){h^=b;h*=1099511628211ull;}return h;}
struct Buffer {
 sycl::queue&q;size_t n;unsigned char*p;std::string name;
 Buffer(sycl::queue&qq,size_t nn,std::string label):q(qq),n(nn),p(sycl::malloc_device<unsigned char>(n+2*G,q)),name(label){need(p!=nullptr,"allocate "+name);q.memset(p,0xd7,n+2*G);}
 ~Buffer(){try{q.wait_and_throw();sycl::free(p,q);}catch(...){cleanup_error=true;}}
 template<class T>T*ptr()const{return reinterpret_cast<T*>(p+G);}
 void put(const Bytes&b){need(b.size()==n,"put bytes");q.memcpy(p+G,b.data(),n).wait_and_throw();}
 template<class T>void putv(const std::vector<T>&v){need(v.size()*sizeof(T)==n,"put elements");q.memcpy(p+G,v.data(),n).wait_and_throw();}
 Bytes get(bool floats=false)const{Bytes out(n);std::array<unsigned char,2*G>ends{};
  q.memcpy(out.data(),p+G,n);q.memcpy(ends.data(),p,G);q.memcpy(ends.data()+G,p+G+n,G).wait_and_throw();
  for(size_t i=0;i<2*G;++i)need(ends[i]==0xd7,"guard "+name);
  if(floats){need(n%4==0,"float alignment");for(size_t i=0;i<n;i+=4){float x;std::memcpy(&x,out.data()+i,4);if(!std::isfinite(x))throw std::runtime_error("nonfinite "+name+" byte "+std::to_string(i));}}return out;
 }
 template<class T>void readonly(const std::vector<T>&v){auto b=get();need(b.size()==v.size()*sizeof(T)&&!std::memcmp(b.data(),v.data(),n),"read-only "+name);}
};
void disjoint(const Buffer&a,const Buffer&b){auto x=reinterpret_cast<uintptr_t>(a.p),y=reinterpret_cast<uintptr_t>(b.p);need(x+a.n+2*G<=y||y+b.n+2*G<=x,"USM spans alias");}
struct Graph {
 sycl::queue&q;dpct::experimental::command_graph_exec_ptr e=nullptr;
 explicit Graph(sycl::queue&qq):q(qq){}
 ~Graph(){try{q.wait_and_throw();}catch(...){cleanup_error=true;}delete e;}
 void capture(const std::function<void()>&f){q.wait_and_throw();need(!e,"capture once");bool begun=false,ended=false;dpct::experimental::command_graph_ptr g=nullptr;
  try{begun=true;dpct::experimental::begin_recording(&q);f();dpct::experimental::end_recording(&q,&g);ended=true;need(g!=nullptr,"null graph");e=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(g->finalize());delete g;g=nullptr;}
  catch(...){auto original=std::current_exception();if(begun&&!ended){try{if(g)g->end_recording();else dpct::experimental::end_recording(&q,&g);}catch(...){cleanup_error=true;std::fprintf(stderr,"HOLD capture balance\n");}}try{q.wait_and_throw();}catch(...){cleanup_error=true;std::fprintf(stderr,"HOLD capture drain\n");}delete g;std::rethrow_exception(original);}}
 void replay(){need(e!=nullptr,"graph absent");q.ext_oneapi_graph(*e);}
};
using Snap=std::array<Bytes,OWN>;
void equal_snap(const Snap&a,const Snap&b,const std::string&why){for(int o=0;o<OWN;++o)eq(a[o],b[o],why+" owner "+std::to_string(o));}
struct Fixture {
 sycl::queue&q;Buffer qkv,h,gate,beta,keep,z,gamma,y;
 std::array<std::array<std::unique_ptr<Buffer>,OWN>,2>state;
 std::array<std::unique_ptr<Graph>,2>graphs;
 std::vector<float>qkv_h,h_h,gate_h,beta_h,z_h,gamma_h;std::vector<int32_t>keep_h;Bytes y_h;
 std::array<std::vector<float>,OWN>seed;
 Fixture(sycl::queue&qq):q(qq),qkv(q,L*MT*C*4,"qkv"),h(q,L*MT*C*4,"h"),gate(q,L*MT*HV*4,"gate"),beta(q,L*MT*HV*4,"beta"),keep(q,OWN*4,"keep"),z(q,MT*HV*S*4,"unused_z"),gamma(q,S*4,"unused_gamma"),y(q,MT*HV*S*4,"output_sentinel"),qkv_h(L*MT*C),h_h(L*MT*C),gate_h(L*MT*HV),beta_h(L*MT*HV),z_h(MT*HV*S,.25f),gamma_h(S,1.f),keep_h(OWN),y_h(MT*HV*S*4,0xa5){
  for(int a=0;a<2;++a)for(int o=0;o<OWN;++o)state[a][o]=std::make_unique<Buffer>(q,STATE*4,"state_arm"+std::to_string(a)+"_owner"+std::to_string(o));
  for(auto&v:seed)v.resize(STATE);
  std::vector<Buffer*>all{&qkv,&h,&gate,&beta,&keep,&z,&gamma,&y};for(auto&a:state)for(auto&b:a)all.push_back(b.get());for(size_t i=0;i<all.size();++i)for(size_t j=i+1;j<all.size();++j)disjoint(*all[i],*all[j]);
  z.putv(z_h);gamma.putv(gamma_h);y.put(y_h);q.wait_and_throw();
 }
 void chain(int a){if(a){for(int o=0;o<OWN;++o){int first=o*ROWS;s145::commit_layers(q,state[a][o]->ptr<float>(),qkv.ptr<float>()+first*C,h.ptr<float>()+first*C,gate.ptr<float>()+first*HV,beta.ptr<float>()+first*HV,keep.ptr<int32_t>()+o);}}
  else for(int l=0;l<L;++l)for(int o=0;o<OWN;++o){int first=o*ROWS;float*st=state[a][o]->ptr<float>()+size_t(l)*STRIDE;const int32_t*k=keep.ptr<int32_t>()+o;
   K::gdn_conv_commit(st+REC,qkv.ptr<float>()+size_t(l)*MT*C+first*C,C,k,&q);
   K::gdn_step_norm_multi(st,h.ptr<float>()+size_t(l)*MT*C+first*C,C,gate.ptr<float>()+size_t(l)*MT*HV+first*HV,beta.ptr<float>()+size_t(l)*MT*HV+first*HV,z.ptr<float>()+first*HV*S,gamma.ptr<float>(),1e-6f,y.ptr<float>()+first*HV*S,HK,HV,ROWS,k,&q,ROWS,nullptr);
  }
 }
 void capture(){for(int a=0;a<2;++a){graphs[a]=std::make_unique<Graph>(q);graphs[a]->capture([&,a]{chain(a);});}}
 void setup(int acceptedA,bool changed){
  keep_h={acceptedA,4-acceptedA};keep.putv(keep_h);
  for(size_t i=0;i<qkv_h.size();++i){qkv_h[i]=float(int((i*13+7)%101)-50)/128;h_h[i]=float(int((i*17+11)%89)-44)/256;if(changed){qkv_h[i]=-.7f*qkv_h[i]+.002f;h_h[i]=-.6f*h_h[i]-.001f;}}
  for(size_t i=0;i<gate_h.size();++i){gate_h[i]=-.05f-float(i%17)/128;beta_h[i]=.1f+float(i%11)/128;if(changed){gate_h[i]-=.07f;beta_h[i]=.2f+.25f*beta_h[i];}}
  for(int o=0;o<OWN;++o)for(size_t i=0;i<STATE;++i){float x=float(int((i*7+o*19)%127)-63)/2048;if(changed)x=-.7f*x+.0001f;
   // Sparse bounded finite edge values, not an all-input arithmetic proof.
   if(i%65537==0)x=(i&1)?-0.f:0.f;else if(i%65539==0)x=(i&1)?-0x1p-149f:0x1p-149f;seed[o][i]=x;}
  qkv.putv(qkv_h);h.putv(h_h);gate.putv(gate_h);beta.putv(beta_h);q.wait_and_throw();
 }
 void stage(){q.wait_and_throw();for(auto&a:state)for(int o=0;o<OWN;++o)a[o]->putv(seed[o]);q.wait_and_throw();}
 void restore(const Snap&s){q.wait_and_throw();for(auto&a:state)for(int o=0;o<OWN;++o)a[o]->put(s[o]);q.wait_and_throw();}
 void readonly(){qkv.readonly(qkv_h);h.readonly(h_h);gate.readonly(gate_h);beta.readonly(beta_h);keep.readonly(keep_h);z.readonly(z_h);gamma.readonly(gamma_h);eq(y.get(),y_h,"state-only output sentinel changed");}
 Snap snapshot(int a){q.wait_and_throw();return{state[a][0]->get(true),state[a][1]->get(true)};}
 Snap compare(const std::string&why){auto control=snapshot(0),candidate=snapshot(1);equal_snap(control,candidate,why);readonly();return control;}
};
void receipt(const Snap&s,int pair,int phase,const char*kind){std::printf("{\"type\":\"snapshot\",\"pair\":%d,\"phase\":%d,\"kind\":\"%s\",\"complete_bytes_compared\":%zu,\"owner0_fnv1a64\":\"%016llx\",\"owner1_fnv1a64\":\"%016llx\",\"digest_not_numeric_comparator\":true}\n",pair,phase,kind,s[0].size()+s[1].size(),(unsigned long long)fnv(s[0]),(unsigned long long)fnv(s[1]));}
// S146 host-only method discriminator; same S145 device classes/images and whole-state comparator.
int native_tags(sycl::queue&q){
 need(q.is_in_order(),"S146 in-order queue required");
 need(!q.has_property<sycl::property::queue::enable_profiling>(),"S146 no whole-queue profiling");
 {Fixture f(q);f.capture();f.setup(2,false);f.stage();f.chain(0);f.chain(1);auto expected=f.compare("S146 public/private reference");
  receipt(expected,2,0,"S146 reference");
  for(int a:{0,1}){f.stage();f.graphs[a]->replay();auto actual=f.snapshot(a);equal_snap(actual,expected,"S146 warm state");f.readonly();}
  // Two samples per arm x tag cell, mirrored around the middle. No 16-sample/new numeric campaign.
  const int arms[8]={0,1,1,0,0,1,1,0};const bool tagged[8]={false,false,true,true,true,true,false,false};
  for(int i=0;i<8;++i){const int a=arms[i];const bool tags=tagged[i];f.stage();
   std::optional<sycl::event> begin_tag,end_tag;
   auto wall_begin=Clock::now();
   if(tags)begin_tag=sycl::ext::oneapi::experimental::submit_profiling_tag(q);
   auto submit_begin=Clock::now();f.graphs[a]->replay();auto submit_end=Clock::now();
   if(tags)end_tag=sycl::ext::oneapi::experimental::submit_profiling_tag(q);
   q.wait_and_throw();auto wall_end=Clock::now();
   // All queries/readbacks are AFTER the common existing completion wait and wall endpoint.
   uint64_t begin_ns=0,end_ns=0;
   if(tags){begin_ns=begin_tag->get_profiling_info<sycl::info::event_profiling::command_end>();end_ns=end_tag->get_profiling_info<sycl::info::event_profiling::command_start>();need(end_ns>begin_ns,"S146 unordered/zero device tag span");}
   auto actual=f.snapshot(a);equal_snap(actual,expected,"S146 timed full state");f.readonly();
   std::printf("{\"type\":\"native_tag_timing\",\"iteration\":%d,\"arm\":%d,\"tags\":%s,\"completed_host_chain_us\":%.6f,\"host_graph_submit_us\":%.6f,\"device_envelope_valid\":%s,\"device_envelope_ns\":%llu,\"begin_command_end_ns\":%llu,\"end_command_start_ns\":%llu,\"complete_state_bytes_checked\":%zu,\"guards\":true,\"readonly\":true,\"finite\":true,\"output_sentinel\":true,\"conv_calls_source\":%d,\"state_calls_source\":%d}\n",i,a,tags?"true":"false",std::chrono::duration<double,std::micro>(wall_end-wall_begin).count(),std::chrono::duration<double,std::micro>(submit_end-submit_begin).count(),tags?"true":"false",(unsigned long long)(tags?end_ns-begin_ns:0),(unsigned long long)begin_ns,(unsigned long long)end_ns,size_t(OWN)*STATE*4,a?2:72,a?2:72);
  }
  q.wait_and_throw();
 }
 need(!cleanup_error,"S146 cleanup failure");
 std::printf("{\"status\":\"PASS_S146_NATIVE_TAG_METHOD_ONLY\",\"native_aspect\":true,\"queue_profiling\":false,\"tags_outside_capture\":true,\"samples\":8,\"samples_per_arm_tag_cell\":2,\"accepted_pair\":[2,2],\"S145_full_numeric_campaign_not_repeated\":true,\"per_kernel_occupancy_or_readiness_qualified\":false,\"serving_qualified\":false}\n");return 0;
}
int run(int argc,char**argv){need(argc==3||argc==4,"NEW_DIR EXPECTED_PCI [--timing|--native-tags]");bool tagmode=argc==4&&std::string(argv[3])=="--native-tags";bool timing=argc==4&&!tagmode;need(!timing||std::string(argv[3])=="--timing","timing arg");need(fs::create_directory(argv[1]),"new output directory");
 for(const auto&v:std::array<std::pair<const char*,const char*>,3>{{{"STRATA_GDN_SPLIT","0"},{"STRATA_GDN_COMMIT_SPLIT","1"},{"STRATA_DBG_GDN","0"}}}){const char*e=std::getenv(v.first);need(e&&std::string(e)==v.second,std::string("require ")+v.first+"="+v.second);}
 need(dpct::device_count()==1,"one pinned device");const sycl::device dev=dpct::dev_mgr::instance().get_device(0);need(dev.has(sycl::aspect::ext_intel_pci_address),"PCI extension");auto pci=dev.get_info<sycl::ext::intel::info::device::pci_address>();need(pci==argv[2],"PCI mismatch");if(tagmode&&!dev.has(sycl::aspect::ext_oneapi_queue_profiling_tag)){std::printf("{\"status\":\"UNSUPPORTED_S146_NATIVE_PROFILING_TAG\",\"native_aspect\":false,\"fallback_attempted\":false}\n");return 3;}sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(),dev,sycl::property_list{sycl::property::queue::in_order()});
 if(tagmode)return native_tags(q);
 {Fixture f(q);f.capture();int strict=0;
  for(int accepted:{1,2,3}){Snap initialA;
   for(int phase=0;phase<3;++phase){f.setup(accepted,phase==1);f.stage();f.chain(0);f.chain(1);auto eager=f.compare("eager control/candidate");receipt(eager,accepted,phase,"eager");
    f.stage();f.graphs[0]->replay();f.graphs[1]->replay();auto captured=f.compare("captured control/candidate");equal_snap(eager,captured,"eager/captured");captured={};
    if(phase==0)initialA=eager;else if(phase==2)equal_snap(initialA,eager,"A/B/A state restoration");
    // Accepted-state bytes are restored unchanged, then both routes commit the next
    // seeded window; this is byte-format continuation, not full serving RAM restore.
    f.restore(eager);f.chain(0);f.chain(1);auto next=f.compare("snapshot continuation eager");receipt(next,accepted,phase,"continuation-eager");
    f.restore(eager);f.graphs[0]->replay();f.graphs[1]->replay();auto next_graph=f.compare("snapshot continuation captured");equal_snap(next,next_graph,"snapshot continuation eager/captured");next_graph={};
    ++strict;std::printf("{\"type\":\"strict\",\"acceptedA\":%d,\"acceptedB\":%d,\"phase\":%d,\"layers\":36,\"owners\":2,\"rows\":3,\"all_state_bytes\":%zu,\"guards\":true,\"finite\":true,\"readonly\":true,\"output_sentinel\":true,\"snapshot_continuation\":true}\n",accepted,4-accepted,phase,size_t(OWN)*STATE*4);
   }
  }
  need(strict==9,"finite strict phase count");
  if(timing){f.setup(2,false);f.stage();f.chain(0);f.chain(1);auto expected=f.compare("timing reference");
   for(int a:{0,1}){f.stage();f.graphs[a]->replay();auto actual=f.snapshot(a);equal_snap(actual,expected,"warm state");f.readonly();}
   const int order[8]={0,1,1,0,1,0,0,1};for(int i=0;i<8;++i){int a=order[i];f.stage();auto start=Clock::now();f.graphs[a]->replay();q.wait_and_throw();auto end=Clock::now();auto actual=f.snapshot(a);equal_snap(actual,expected,"timed state");f.readonly();std::printf("{\"type\":\"timing\",\"iteration\":%d,\"arm\":%d,\"completed_host_chain_us\":%.6f,\"conv_calls\":%d,\"state_calls\":%d}\n",i,a,std::chrono::duration<double,std::micro>(end-start).count(),a?2:72,a?2:72);}}
  q.wait_and_throw();
 }
 need(!cleanup_error,"cleanup failure");std::printf("{\"status\":\"PASS_PRIVATE_GDN_LAYER_BATCH\",\"accepted_pairs\":3,\"ABA_phases\":9,\"eager_and_captured\":true,\"complete_state_bytes\":%zu,\"snapshot_receipts\":18,\"fullsnapshotfiles\":0,\"timing_enabled\":%s,\"serving_qualified\":false}\n",size_t(OWN)*STATE*4,timing?"true":"false");return 0;
}
int main(int argc,char**argv){try{return run(argc,argv);}catch(const std::exception&e){std::fprintf(stderr,"HOLD %s\n",e.what());return 1;}}
