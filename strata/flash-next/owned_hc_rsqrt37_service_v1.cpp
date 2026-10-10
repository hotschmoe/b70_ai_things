// Independent arithmetic reference: own positive F32 arguments, no HC operands.
#define main hc37_reference_fixture_helpers
#include "hc_composition_arithmetic35_gpu_v1.cpp"
#undef main
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <array>
#include <iostream>
#include <cmath>
#include <cstdint>
class OwnedHcRsqrtReference37;
uint32_t bits_of(float value){uint32_t bits;std::memcpy(&bits,&value,4);return bits;}
float value_of(const std::string& text){if(text.size()!=8||text.find_first_not_of("0123456789abcdef")!=std::string::npos)throw std::runtime_error("exact F32 hexadecimal input required");uint32_t bits=uint32_t(std::stoul(text,nullptr,16));float value;std::memcpy(&value,&bits,4);if(!std::isfinite(value)||value<=0)throw std::runtime_error("positive finite own argument required");return value;}
void save_maps(const fs::path& path){if(fs::exists(path))throw std::runtime_error("new maps receipt required");std::ifstream input("/proc/self/maps",std::ios::binary);std::ofstream output(path,std::ios::binary);output<<input.rdbuf();output.close();if(!input||!output)throw std::runtime_error("actual mapped-library receipt failed");}
int main(int argc,char** argv) try {
 if(argc!=5||std::string(argv[1])!="--maximum-requests"||std::string(argv[2])!="512"||std::string(argv[3])!="--maps-prefix")throw std::runtime_error("exact maximum512/newmaps CLI required");const std::string maps=argv[4];
 const char* affinity=std::getenv("ZE_AFFINITY_MASK"),*selector=std::getenv("ONEAPI_DEVICE_SELECTOR");auto& q=dpct::get_in_order_queue();if(!affinity||std::string(affinity)!="0"||!selector||std::string(selector)!="level_zero:gpu"||std::getenv("STRATA_VERIFY_EAGER")||!q.get_device().is_gpu()||q.get_backend()!=sycl::backend::ext_oneapi_level_zero||!q.has_property<sycl::property::queue::in_order>())throw std::runtime_error("card0 GPU LevelZero in_order EAGERabsent required");
 Owned mem(q);auto* argument=mem.floats(4);auto* result=mem.floats(4);GraphOwner graph(q);auto body=[&](){q.parallel_for<OwnedHcRsqrtReference37>(sycl::range<1>(4),[=](sycl::id<1> id){result[id[0]]=sycl::rsqrt(argument[id[0]]);});};
 save_maps(maps+".before");std::printf("OWNRS37_READY pid=%ld backend=level_zero affinity=0 width=4 maximum=512\n",long(getpid()));std::fflush(stdout);std::string line;size_t count=0;bool quit=false;
 while(std::getline(std::cin,line)){
  if(line=="QUIT"){quit=true;break;}
  std::istringstream input(line);std::string verb,tag,extra;std::array<std::string,4> words;input>>verb>>tag;for(auto& word:words)input>>word;size_t serial=0;try{serial=std::stoul(tag);}catch(...){throw std::runtime_error("request ordinal required");}
  if(verb!="RSQRT"||tag!=std::to_string(count+1)||serial!=count+1||count>=512||!input||input>>extra)throw std::runtime_error("exact next own request/width/budget required");
  std::array<float,4> owned,direct,replay;for(size_t i=0;i<4;++i)owned[i]=value_of(words[i]);q.memcpy(argument,owned.data(),16).wait_and_throw();body();q.wait_and_throw();q.memcpy(direct.data(),result,16).wait_and_throw();
  if(!graph.executable){dpct::experimental::begin_recording(&q);graph.recording=true;body();dpct::experimental::end_recording(&q,&graph.graph);graph.recording=false;if(!graph.graph)throw std::runtime_error("own RS graph absent");graph.executable=new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph.graph->finalize());delete graph.graph;graph.graph=nullptr;}
  for(int repeat=0;repeat<2;++repeat){q.ext_oneapi_graph(*graph.executable).wait_and_throw();q.memcpy(replay.data(),result,16).wait_and_throw();if(std::memcmp(direct.data(),replay.data(),16))throw std::runtime_error("own argument direct/replay RS differs");}
  for(float value:direct)if(!std::isfinite(value)||value<=0)throw std::runtime_error("positive finite own RS result required");
  ++count;std::printf("OWNRS37_RESPONSE seq=%zu",count);for(float value:owned)std::printf(" arg=%08x",bits_of(value));for(float value:direct)std::printf(" rs=%08x",bits_of(value));std::printf(" direct_replays_bitwise=1\n");std::fflush(stdout);
 }
 if(!quit||count==0)throw std::runtime_error("explicit owned QUIT after actual work required");q.wait_and_throw();delete graph.executable;graph.executable=nullptr;mem.release();save_maps(maps+".after");std::printf("OWNRS37_DONE requests=%zu graph_retired=1 owned_allocations_freed=1\n",count);std::fflush(stdout);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"OWNRS37_ERROR %s\n",error.what());return 2;}
