// Metadata/finite-input wrapper only; frozen proposal and linked HC math unchanged.
#include "native_rms_rsqrt37_device_ops_v1.cpp"
#include <unistd.h>
#include <cstring>
void ascii_text(const std::string& s){for(unsigned char c:s){if(c>=32&&c<127)std::putchar(c);else std::printf("\\x%02x",unsigned(c));}}
int main(int argc,char** argv) try {
 if(argc!=5||std::string(argv[1])!="--residual"||std::string(argv[3])!="--output")throw std::runtime_error("exact residual/output CLI required");
 auto raw=load(argv[2],D*4);std::vector<float> values(D);std::memcpy(values.data(),raw.data(),D*4);for(float v:values)if(!std::isfinite(v))throw std::runtime_error("finite own residual required");
 const char* affinity=std::getenv("ZE_AFFINITY_MASK"),*selector=std::getenv("ONEAPI_DEVICE_SELECTOR");if(!affinity||std::string(affinity)!="0"||!selector||std::string(selector)!="level_zero:gpu"||std::getenv("STRATA_VERIFY_EAGER"))throw std::runtime_error("physicalcard0/GPU/EAGERabsent required");
 auto& q=dpct::get_in_order_queue();if(!q.get_device().is_gpu()||q.get_backend()!=sycl::backend::ext_oneapi_level_zero||!q.has_property<sycl::property::queue::in_order>())throw std::runtime_error("actual in_order LevelZero GPU required");
 std::printf("RMS37_DEVICE pid=%ld backend=level_zero affinity=0 name=",long(getpid()));ascii_text(q.get_device().get_info<sycl::info::device::name>());std::printf(" vendor=");ascii_text(q.get_device().get_info<sycl::info::device::vendor>());std::printf(" driver=");ascii_text(q.get_device().get_info<sycl::info::device::driver_version>());std::printf("\n");std::fflush(stdout);
 fs::path maps=std::string(argv[4])+".loaded-maps-before";if(fs::exists(maps))throw std::runtime_error("new actual mapped-library receipt required");std::ifstream in("/proc/self/maps",std::ios::binary);std::ofstream out(maps,std::ios::binary);out<<in.rdbuf();out.close();if(!in||!out)throw std::runtime_error("actual maps receipt failed");
 int code=rms37_frozen_proposal_main(argc,argv);if(code!=0)return code;
 fs::path after=std::string(argv[4])+".loaded-maps";if(fs::exists(after))throw std::runtime_error("new actual postexecution mapped-library receipt required");std::ifstream post("/proc/self/maps",std::ios::binary);std::ofstream saved(after,std::ios::binary);saved<<post.rdbuf();saved.close();if(!post||!saved)throw std::runtime_error("postexecution actual maps receipt failed");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"RMS37_OWNED_ERROR %s\n",e.what());return 2;}
