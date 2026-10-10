// Public source35 HC projection leaf; synthetic inputs only, no exp/rsqrt/fullHC.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "strata/kernels/hc_native_projection.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
std::vector<uint8_t> read_bytes(const fs::path& path,size_t bytes) {
 if(fs::is_symlink(path)||fs::file_size(path)!=bytes)throw std::runtime_error("input extent/symlink differs");
 std::ifstream stream(path,std::ios::binary);std::vector<uint8_t> raw(bytes);stream.read(reinterpret_cast<char*>(raw.data()),std::streamsize(bytes));
 if(!stream||stream.peek()!=std::char_traits<char>::eof())throw std::runtime_error("input changed during read");return raw;
}
void save(const fs::path& path,const std::vector<float>& values) {
 if(fs::exists(path))throw std::runtime_error("new output required");
 std::ofstream stream(path,std::ios::binary);stream.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*4));stream.close();if(!stream)throw std::runtime_error("raw output write failed");
}
struct Buffers {
 sycl::queue& q;std::vector<void*> owned;
 explicit Buffers(sycl::queue& queue):q(queue){}
 void* make(size_t n){void* p=sycl::malloc_device(n,q);if(!p)throw std::runtime_error("USM allocation failed");owned.push_back(p);return p;}
 void release(){q.wait_and_throw();for(auto& p:owned)if(p){sycl::free(p,q);p=nullptr;}owned.clear();}
 ~Buffers(){try{release();}catch(...){}}
};
int main(int argc,char** argv) try {
 if(argc!=5||std::string(argv[1])!="--inputs"||std::string(argv[3])!="--output")throw std::runtime_error("usage: leaf --inputs DIR --output NEWDIR");
 fs::path inputs(argv[2]),out(argv[4]);if(fs::exists(out)||fs::is_symlink(inputs))throw std::runtime_error("new output/nonsymlink input required");fs::create_directories(out);
 auto& q=dpct::get_in_order_queue();if(!q.has_property<sycl::property::queue::in_order>())throw std::runtime_error("in_order queue required");
 auto context=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());auto device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
 const std::string name=q.get_device().get_info<sycl::info::device::name>();
 const std::string vendor=q.get_device().get_info<sycl::info::device::vendor>(),driver=q.get_device().get_info<sycl::info::device::driver_version>();
 const char* affinity=std::getenv("ZE_AFFINITY_MASK"),*selector=std::getenv("ONEAPI_DEVICE_SELECTOR");
 if(!affinity||std::string(affinity)!="0"||!selector||std::string(selector)!="level_zero:gpu")throw std::runtime_error("exact physicalcard0 selector/affinity required");
 if(q.get_backend()!=sycl::backend::ext_oneapi_level_zero)throw std::runtime_error("actual LevelZero backend required");
 std::printf("HC35_PROJECTION_DEVICE backend=level_zero vendor=%s driver=%s affinity=0 selector=level_zero:gpu physical_card_parent_mapping_required=1\n",vendor.c_str(),driver.c_str());
 std::printf("HC35_PROJECTION_CONFIG synthetic=1 device=%s queue=%p ze_context=%p ze_device=%p in_order=1 device_intrinsics_qualified=0 model_math_qualified=0\n",name.c_str(),(void*)&q,(void*)context,(void*)device);
 std::ifstream manifest(inputs/"cases.tsv");std::string header;std::getline(manifest,header);if(header!="HC35_PROJECTION_INPUTS_V1")throw std::runtime_error("synthetic manifest version differs");
 std::set<std::string> seen;size_t cases=0;
 { Buffers memory(q);std::string id,type;int k=0,m=0,t=0;
 while(manifest>>id>>type>>k>>m>>t) {
  if(id.empty()||id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-")!=std::string::npos||!seen.insert(id).second||cases>=16||(type!="F32"&&type!="Q8_0")||k<32||k>10240||k%32||m<1||m>4||t<1||t>2)throw std::runtime_error("bounded synthetic projection descriptor differs");
  const size_t weight_bytes=size_t(m)*(type=="F32"?size_t(k)*4:size_t(k/32)*34),input_bytes=size_t(t)*k*4;
  auto weights=read_bytes(inputs/(id+".weights.bin"),weight_bytes),input=read_bytes(inputs/(id+".input.bin"),input_bytes);
  auto* dw=memory.make(weight_bytes);auto* dx=static_cast<float*>(memory.make(input_bytes));auto* dy=static_cast<float*>(memory.make(size_t(m)*t*4));
  q.memcpy(dw,weights.data(),weights.size());q.memcpy(dx,input.data(),input.size());
  const bool ok=type=="F32"?strata::kernels::hc_f32_project_f32(static_cast<const float*>(dw),weight_bytes/4,dx,size_t(t)*k,dy,size_t(t)*m,k,m,t,&q):strata::kernels::hc_q8_0_project_f32(static_cast<const uint8_t*>(dw),weight_bytes,dx,size_t(t)*k,dy,size_t(t)*m,k,m,t,&q);
  if(!ok)throw std::runtime_error("public HC projection descriptor rejected");
  std::vector<float> actual(size_t(m)*t);q.memcpy(actual.data(),dy,actual.size()*4).wait_and_throw();save(out/(id+".gpu.f32"),actual);
  std::printf("HC35_PROJECTION_ROW case=%s type=%s k=%d m=%d t=%d bytes=%zu\n",id.c_str(),type.c_str(),k,m,t,actual.size()*4);++cases;
 }
 if(!manifest.eof()||cases==0)throw std::runtime_error("incomplete/empty manifest");memory.release(); }
 std::printf("HC35_PROJECTION_RESULT cases=%zu execution_completed=1 all_owned_allocations_freed=1 numerical_comparison_done=0 device_intrinsics_qualified=0 model_math_qualified=0\n",cases);return 0;
} catch(const std::exception& error){std::fprintf(stderr,"HC35_PROJECTION_ERROR %s\n",error.what());return 2;}
