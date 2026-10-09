// Parent-leased pure SYCL health option for the exact39992d70 model runtime.
// No model files, RNG, PyTorch or CCL. This is a distinct controlled workload.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void need(bool v,const char* msg){if(!v) throw std::runtime_error(msg);}
class HealthPattern;class HealthMatmul;
}
int main(int argc,char** argv){try{
    need(argc>=2 && argc<=3,"usage: matched-sycl-health receipt.json [N=256]");int n=argc==3 ? std::stoi(argv[2]):256;need(n>=16 && n<=2048 && n%16==0,"N must be16..2048 multiple16");
    sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
    const size_t words=4u*1024u*1024u;auto* pattern=sycl::malloc_device<uint32_t>(words,q);auto* copied=sycl::malloc_device<uint32_t>(words,q);
    float* a=sycl::malloc_device<float>(size_t(n)*n,q);float* c=sycl::malloc_device<float>(size_t(n)*n,q);need(pattern && copied && a && c,"device allocation failed");
    q.parallel_for<HealthPattern>(sycl::range<1>(words),[=](sycl::id<1> index){pattern[index[0]]=uint32_t(index[0])^0xb70a5a5au;});q.memcpy(copied,pattern,words*4).wait_and_throw();
    std::vector<uint32_t> back(words);q.memcpy(back.data(),copied,words*4).wait_and_throw();for(size_t i=0;i<words;++i) need(back[i]==(uint32_t(i)^0xb70a5a5au),"exact read/write/copy mismatch");
    std::vector<float> input(size_t(n)*n);for(int row=0;row<n;++row) for(int col=0;col<n;++col) input[size_t(row)*n+col]=float((row*17+col*13)%31-15)/32;
    q.memcpy(a,input.data(),input.size()*4).wait_and_throw();
    q.submit([&](sycl::handler& h){sycl::local_accessor<float,2> left(sycl::range<2>(16,16),h),right(sycl::range<2>(16,16),h);h.parallel_for<HealthMatmul>(sycl::nd_range<2>(sycl::range<2>(n,n),sycl::range<2>(16,16)),[=](sycl::nd_item<2> item){int row=item.get_global_id(0),col=item.get_global_id(1),lr=item.get_local_id(0),lc=item.get_local_id(1);float sum=0;for(int block=0;block<n;block+=16){left[lr][lc]=a[size_t(row)*n+block+lc];right[lr][lc]=a[size_t(block+lr)*n+col];item.barrier(sycl::access::fence_space::local_space);for(int k=0;k<16;++k) sum=sycl::fma(left[lr][k],right[k][lc],sum);item.barrier(sycl::access::fence_space::local_space);}c[size_t(row)*n+col]=sum;});});q.wait_and_throw();
    std::vector<float> result(input.size());q.memcpy(result.data(),c,result.size()*4).wait_and_throw();for(float value:result) need(std::isfinite(value),"nonfinite matrix result");
    int checked=0;for(int row:{0,n/2,n-1}) for(int col:{0,n/2,n-1}){double reference=0;for(int k=0;k<n;++k) reference+=double(input[size_t(row)*n+k])*input[size_t(k)*n+col];need(result[size_t(row)*n+col]==float(reference),"exact rational FP64 matmul sample mismatch");++checked;}
    q.wait_and_throw();sycl::free(pattern,q);sycl::free(copied,q);sycl::free(a,q);sycl::free(c,q);q.wait_and_throw();
    std::ofstream output(argv[1]);need(bool(output),"receipt open failed");output<<"{\"schema\":1,\"passed\":true,\"backend\":\"pure_SYCL\",\"pattern_bytes\":16777216,\"pattern_byte_exact\":true,\"matmul_n\":"<<n<<",\"finite_matrix\":true,\"cpu_fp64_exact_samples\":"<<checked<<",\"normal_frees_returned\":true,\"model_files_opened\":false,\"collective_executed\":false,\"cause_proven\":false}\n";return 0;
}catch(const std::exception& e){std::cerr<<"MATCHED_HEALTH_CONTROL FAIL: "<<e.what()<<'\n';return 2;}}
