#include "strata/kernels/original_mtp53_typed_primitives.hpp"
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <new>
#include "strata/core/original_mtp53_owned_runtime.hpp"
using Owned53=OwnedRuntime53;
int main(int argc,char** argv) try {
    // OFF is checked before creating a queue, opening a file or allocating anything.
    if(!strata::core::original_mtp53::enabled()){std::fprintf(stderr,"MTP53 default OFF\n");return 2;}
    if(argc==3 && std::strcmp(argv[1],"--inventory-only")==0){
        strata::core::original_mtp53::TensorViews views;std::string error;
        if(!views.load(argv[2],error))throw std::runtime_error(error);
        std::printf("MTP53_SCHEMA_PASS tensors=34 packed_bytes=2775258112 payload_identity_verified=0\n");return 0;
    }
    if(argc!=2 || std::strcmp(argv[1],"--f32-synthetic")!=0)throw std::invalid_argument("use --inventory-only FILE or --f32-synthetic");
    auto& q=dpct::get_in_order_queue();if(!q.get_device().is_gpu() || !q.is_in_order())throw std::runtime_error("owned ordered GPU queue required");
    Owned53 owned(q);std::vector<float> weights(512*2560),input(2560),expected(512),actual(512);
    for(int i=0;i<2560;++i)input[i]=float(i%19-9)/32.0f;
    for(int r=0;r<512;++r){float sum=0;for(int i=0;i<2560;++i){float v=float((r+i)%17-8)/16.0f;weights[r*2560+i]=v;sum=std::fma(v,input[i],sum);}expected[r]=sum;}
    float* w=owned.take<float>(weights.size());float* x=owned.take<float>(input.size());float* y=owned.take<float>(512);
    q.memcpy(w,weights.data(),weights.size()*sizeof(float));q.memcpy(x,input.data(),input.size()*sizeof(float));q.wait_and_throw();
    auto invoke=[&]{strata::kernels::original_mtp53::f32_linear(w,x,y,512,1,&q);};
    auto compare=[&]{q.memcpy(actual.data(),y,actual.size()*sizeof(float)).wait_and_throw();if(std::memcmp(actual.data(),expected.data(),512*sizeof(float))!=0)throw std::runtime_error("F32 primitive differs from declared serial-FMA synthetic contract");};
    q.memset(y,0xa5,512*sizeof(float));invoke();q.wait_and_throw();compare();
    {
        namespace graph=sycl::ext::oneapi::experimental;
        auto* capture=owned.keep_graph(new graph::command_graph<graph::graph_state::modifiable>(q.get_context(),q.get_device()));capture->begin_recording(q);invoke();capture->end_recording(q);auto* executable=owned.keep_graph(new decltype(capture->finalize())(capture->finalize()));
        for(int replay=0;replay<2;++replay){q.memset(y,0xa5,512*sizeof(float));q.ext_oneapi_graph(*executable).wait_and_throw();compare();}
        q.wait_and_throw();
    }
    owned.retire();std::printf("MTP53_F32_SYNTHETIC_PASS direct=1 replays=2 rows=512 full_draft_qualified=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MTP53_ERROR %s\n",e.what());return 1;}
