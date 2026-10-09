// Bounded actual Verifier stream-control/capture oracle. No model weights/math.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/verify.hpp"
#include "strata/core/verifier_stream_contract.hpp"
#include "strata/core/stage_expert_mirror.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/resident_plan_mirror.hpp"
#include <array>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace graph=sycl::ext::oneapi::experimental;
namespace {
void need(bool ok,const char* message){if(!ok) throw std::runtime_error(message);}
class ConsumeActualMirrorCapturePlan;
struct Storage {
    sycl::queue q;int32_t* res=nullptr;int32_t* ids=nullptr;int32_t* plan=nullptr;uint32_t* error=nullptr;uint32_t* output=nullptr;
    std::shared_ptr<strata::core::StageExpertMirror> mirror;
    explicit Storage(const sycl::queue& queue,int stage):q(queue){
        res=sycl::malloc_device<int32_t>(4,q);ids=sycl::malloc_device<int32_t>(2,q);plan=sycl::malloc_device<int32_t>(128,q);error=sycl::malloc_device<uint32_t>(1,q);output=sycl::malloc_device<uint32_t>(2,q);
        need(res&&ids&&plan&&error&&output,"control allocation failed");q.memset(res,0xff,16).wait_and_throw();int32_t e[2]={0,0};q.memcpy(ids,e,8).wait_and_throw();
        mirror=std::make_shared<strata::core::StageExpertMirror>(q);mirror->lb=stage;mirror->le=stage+1;mirror->n_layers=2;mirror->n_expert=2;mirror->bytes=256;
        mirror->segments.push_back({sycl::malloc_host<uint8_t>(256,q),256});need(mirror->segments[0].base,"control host segment failed");auto* tag=reinterpret_cast<uint32_t*>(mirror->segments[0].base);*tag=0xb7000000u+stage;
        mirror->table=sycl::malloc_device<unsigned long long>(4,q);need(mirror->table,"control mirror table failed");unsigned long long table[4]={};table[stage*2]=reinterpret_cast<unsigned long long>(tag);q.memcpy(mirror->table,table,32).wait_and_throw();std::string error;need(mirror->bind(res,2,2,q,error),error.c_str());
    }
    ~Storage(){q.wait_and_throw();mirror->release();mirror.reset();for(void* p:{static_cast<void*>(res),static_cast<void*>(ids),static_cast<void*>(plan),static_cast<void*>(error),static_cast<void*>(output)}) if(p) sycl::free(p,q);}
};
void capture_and_replay(strata::core::Verifier& verifier,Storage& storage,int stage,int tokens,int caller_device){
    const int before=int(dpct::get_current_device_id());need(before==caller_device,"caller device setup mismatch");
    {
        const strata::core::OnDevice owning(verifier.device());auto* stream=verifier.stream();need(stream && int(dpct::get_current_device_id())==verifier.device(),"owning capture scope failed");
        need(storage.mirror->matches(storage.res,2,2,*stream),"actual mirror queue/residency capture guard failed");
        dpct::experimental::begin_recording(stream);stream->memset(storage.error,0,4);
        strata::kernels::resident_plan_bound(storage.ids,tokens,1,storage.res+stage*2,2,nullptr,nullptr,256,storage.plan,tokens,nullptr,0,stream,storage.mirror->layer_table(stage),storage.error);
        auto* plan=storage.plan;auto* error=storage.error;auto* out=storage.output;size_t offset=((4+(tokens+1)+2*tokens)+1)&~size_t(1);
        stream->parallel_for<ConsumeActualMirrorCapturePlan>(sycl::range<1>(tokens),[=](sycl::id<1> index){if(plan[0]==1 && *error==0){auto* pointers=reinterpret_cast<unsigned long long*>(plan+offset);out[index[0]]=*reinterpret_cast<const uint32_t*>(pointers[0]);}});
        dpct::experimental::command_graph_ptr captured=nullptr;dpct::experimental::end_recording(stream,&captured);need(captured,"captured graph missing");
        auto executable=captured->finalize();delete captured;for(int repeat=0;repeat<2;++repeat){stream->ext_oneapi_graph(executable).wait_and_throw();uint32_t values[2]={},error_value=1;stream->memcpy(values,storage.output,tokens*4).wait_and_throw();stream->memcpy(&error_value,storage.error,4).wait_and_throw();need(error_value==0,"captured plan reported error");for(int t=0;t<tokens;++t) need(values[t]==0xb7000000u+stage,"actual captured mirror pointer/tag differs");}
    }
    need(int(dpct::get_current_device_id())==before,"owning capture scope did not restore caller device");
}
}
int main(int argc,char** argv){try{
    need(argc==2,"usage: oracle receipt.json");std::ofstream report(argv[1]);need(bool(report),"receipt open failed");
    dpct::select_device(0);
    // Real Verifier objects are constructed before GPU1 selection, just like
    // actual GpuStage embeds them before its OnDevice scope.
    std::array<strata::core::Verifier,2> verifiers;need(!verifiers[0].stream() && !verifiers[1].stream(),"Verifier construction captured a default queue");
    std::array<std::unique_ptr<Storage>,2> storage;std::string error;int negatives=0;
    for(int stage=0;stage<2;++stage){dpct::select_device(stage);need(verifiers[stage].init_capture_streams(error),error.c_str());need(verifiers[stage].device()==stage && verifiers[stage].stream()->get_device()==dpct::get_device(stage).in_order_queue().get_device(),"actual initializer selected foreign device");storage[stage]=std::make_unique<Storage>(*verifiers[stage].stream(),stage);}
    for(int stage=0;stage<2;++stage){auto& own=*storage[stage];auto& foreign=*storage[1-stage];need(!strata::core::verifier_stream_contract(verifiers[1-stage].stream(),own.q,error),"explicit foreign queue accepted");++negatives;need(!own.mirror->matches(foreign.res,2,2,own.q),"foreign residency accepted");++negatives;need(!own.mirror->matches(own.res,2,2,foreign.q),"foreign context/device accepted");++negatives;need(!own.mirror->matches(own.res,3,2,own.q),"changed geometry accepted");++negatives;}
    // Capture stage0/stage1 T2 then T1 while the caller deliberately holds the
    // opposite current device. The exact scoped queue contract must survive.
    for(int tokens:{2,1}) for(int stage=0;stage<2;++stage){int caller=1-stage;dpct::select_device(caller);capture_and_replay(verifiers[stage],*storage[stage],stage,tokens,caller);}
    dpct::select_device(0);{strata::core::Verifier invalid;invalid.set_stream(verifiers[1].stream());need(!invalid.init_capture_streams(error) && !invalid.stream(),"real initializer accepted supplied foreign stream");++negatives;}
    dpct::select_device(1);{strata::core::Verifier borrowed;borrowed.set_stream(verifiers[1].stream());need(borrowed.init_capture_streams(error) && borrowed.stream()==verifiers[1].stream(),"explicit compatible stream was not borrowed");}
    for(auto& item:storage) item.reset();dpct::select_device(0);
    report<<"{\"schema\":1,\"passed\":true,\"actual_verifier_objects\":2,\"constructed_on_device\":0,\"init_devices\":[0,1],\"capture_token_shapes\":[2,1],\"captured_graphs\":4,\"graph_replays\":8,\"negative_checks\":"<<negatives<<",\"caller_device_restored\":true,\"actual_mirror_guard_preserved\":true,\"model_math_executed\":false,\"full_model_qualified\":false}\n";return 0;
}catch(const std::exception& error){std::fprintf(stderr,"VERIFY_CAPTURE_ORACLE FAIL: %s\n",error.what());return 2;}}
