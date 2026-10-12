#include "strata/kernels/original_mtp53_eh_hc_front.hpp"
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <new>
#include "strata/core/original_mtp53_owned_runtime.hpp"
using Owner53Front=OwnedRuntime53;
static void save(const std::string& path,const std::vector<float>& values){std::ofstream f(path,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));}
int main(int argc,char** argv)try{
    if(!strata::core::original_mtp53::enabled()){std::fprintf(stderr,"MTP53_FRONT default OFF\n");return 2;}
    if(argc!=3)throw std::invalid_argument("usage ORIGINAL_MTP.gguf EXISTING_NEW_OUTPUT_DIR (root freshly verifies publisher SHA first)");
    // Source/schema remains immutable; this helper is a front-segment fixture, NOT a complete draft step.
    strata::core::original_mtp53::TensorViews tensors;std::string error;if(!tensors.load(argv[1],error))throw std::runtime_error(error);
    auto& q=dpct::get_in_order_queue();if(!q.get_device().is_gpu()||!q.is_in_order())throw std::runtime_error("ordered owned GPU queue required");
    Owner53Front owner(q);using namespace strata::kernels::original_mtp53;
    EhHcWeights w{};w.own_embedding_q4k=owner.upload(tensors,"token_embd.weight");w.enorm=static_cast<const float*>(owner.upload(tensors,"blk.48.nextn.enorm.weight"));w.hnorm=static_cast<const float*>(owner.upload(tensors,"blk.48.nextn.hnorm.weight"));w.eh_proj_q4k=owner.upload(tensors,"blk.48.nextn.eh_proj.weight");w.hc_norm=static_cast<const float*>(owner.upload(tensors,"blk.48.hc_attn_norm.weight"));w.hc_down_q4k=owner.upload(tensors,"blk.48.hc_attn_down.weight");w.hc_up_q5_0=owner.upload(tensors,"blk.48.hc_attn_up.weight");w.hc_inject_q4k=owner.upload(tensors,"blk.48.hc_attn_inject.weight");
    EhHcScratch s{};s.embedding=owner.take<float>(2560);s.e_norm=owner.take<float>(2560);s.h_norm=owner.take<float>(10240);s.concat_eh=owner.take<float>(20480);s.residual=owner.take<float>(10240);s.hc_normalized=owner.take<float>(10240);s.low=owner.take<float>(320);s.gate=owner.take<float>(10240);s.weighted_streams=owner.take<float>(10240);s.mixed=owner.take<float>(2560);s.injection=owner.take<float>(4);s.q8_1=owner.take<unsigned char>(strata::kernels::native_q8_1_bytes(5120,4));
    std::vector<float> hidden(10240);for(size_t i=0;i<hidden.size();++i)hidden[i]=float(int((i*13)%101)-50)/64.0f;
    float* h=owner.take<float>(hidden.size());int32_t* token=owner.take<int32_t>(1);const int32_t id=248045;q.memcpy(h,hidden.data(),hidden.size()*4);q.memcpy(token,&id,4);q.wait_and_throw();save(std::string(argv[2])+"/own-hidden.f32",hidden);
    std::vector<std::pair<const char*,std::pair<float*,size_t>>> fields={{"embedding",{s.embedding,2560}},{"e_norm",{s.e_norm,2560}},{"h_norm",{s.h_norm,10240}},{"concat_eh",{s.concat_eh,20480}},{"residual",{s.residual,10240}},{"hc_normalized",{s.hc_normalized,10240}},{"low",{s.low,320}},{"gate",{s.gate,10240}},{"weighted_streams",{s.weighted_streams,10240}},{"mixed",{s.mixed,2560}},{"injection",{s.injection,4}}};std::vector<std::vector<float>> reference;
    auto fill_outputs=[&]{for(auto& f:fields)q.memset(f.second.first,0xa5,f.second.second*4);};auto invoke=[&]{eh_hc_front(token,h,w,s,&q);};
    auto observe=[&](const char* label,bool first){
        for(size_t i=0;i<fields.size();++i){std::vector<float> current(fields[i].second.second);q.memcpy(current.data(),fields[i].second.first,current.size()*4).wait_and_throw();for(float value:current)if(!std::isfinite(value))throw std::runtime_error("nonfinite original front output");if(first)reference.push_back(current);else if(std::memcmp(current.data(),reference[i].data(),current.size()*4))throw std::runtime_error("front-segment direct/replay or rejected rerun differs");save(std::string(argv[2])+"/"+label+"-"+fields[i].first+".f32",current);}
        std::vector<float> current(hidden.size());q.memcpy(current.data(),h,current.size()*4).wait_and_throw();if(std::memcmp(current.data(),hidden.data(),current.size()*4))throw std::runtime_error("immutable incoming state changed");
    };
    fill_outputs();invoke();q.wait_and_throw();observe("direct",true);
    {
        namespace g=sycl::ext::oneapi::experimental;auto* graph=owner.keep_graph(new g::command_graph<g::graph_state::modifiable>(q.get_context(),q.get_device()));graph->begin_recording(q);invoke();graph->end_recording(q);auto* executable=owner.keep_graph(new decltype(graph->finalize())(graph->finalize()));
        for(int n=0;n<2;++n){fill_outputs();q.ext_oneapi_graph(*executable).wait_and_throw();observe(n==0?"replay1":"replay2",false);}q.wait_and_throw();
    }
    // A rejected front result has no cache/state side effects: discard scratch, recompute from ORIGINAL h.
    // This proves ONLY front-input preservation/rerun; full MTP KV/indexer rollback is still unimplemented.
    fill_outputs();invoke();q.wait_and_throw();observe("rejected-rerun",false);
    owner.retire();std::printf("MTP53_FRONT_PASS token=248045 fields=11 direct=1 replays=2 rejected_rerun=1 input_state_unchanged=1 full_draft_qualified=0\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MTP53_FRONT_ERROR %s\n",e.what());return 1;}
