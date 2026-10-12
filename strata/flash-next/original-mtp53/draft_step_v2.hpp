#pragma once
#include "strata/kernels/original_mtp53_qsa_state.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/core/verify.hpp"
#include <array>
namespace strata::kernels::original_mtp53 {
struct HcWeights53{const float* norm;int down_type;const void* down;const void* up_q5;const void* inject_q4;};
struct HcScratch53{float* normalized;float* low;float* gate;float* products;float* mixed;float* injection;void* q8;};
struct FfnWeights53{const float* router_f32;const float* shared_gate_f32;const void* expert_blobs;const void* shared_gate_q4;const void* shared_up_q4;const void* shared_down_q8;};
struct FfnScratch53{float* router;int32_t* ids;float* weights;unsigned long long* groups;int32_t* starts;int32_t* count;int32_t* destinations;int32_t* tokens;void* native_scratch;void* q8;float* parts;float* shared_gate;float* shared_gate_rows;float* shared_up;void* shared_uq;float* shared;float* shared_gated;float* output;};
class HcProducts53;class HcMean53;class HcWriteWeights53;class HcWriteProducts53;class HcWriteSum53;class ExpertGroups53;class SharedGate53;
inline void hc_read53(const float* residual,const HcWeights53& w,const HcScratch53& s,void* stream){
    require_on();native_qsa_rms_norm_grouped(residual,w.norm,s.normalized,2560,4,4,1e-6f,stream);
    quant_linear(w.down_type,w.down,s.normalized,s.q8,s.low,10240,320,1,stream);scale_inplace(s.low,320,0.25f,stream);silu_inplace(s.low,320,stream);
    quant_linear(6,w.up_q5,s.low,s.q8,s.gate,320,10240,1,stream);auto& q=*strata::q_of(stream);
    const float* xn=s.normalized;const float* g=s.gate;float* products=s.products;float* mixed=s.mixed;
    q.parallel_for<HcProducts53>(sycl::range<1>(10240),[=](sycl::id<1> i){const size_t j=i[0];products[j]=xn[j]*(1.0f/(1.0f+sycl::exp(-g[j])));});
    q.parallel_for<HcMean53>(sycl::range<1>(2560),[=](sycl::id<1> i){size_t j=i[0];float v=products[j];v=v+products[2560+j];v=v+products[5120+j];v=v+products[7680+j];mixed[j]=v*0.25f;});
    if(w.inject_q4)quant_linear(12,w.inject_q4,s.normalized,s.q8,s.injection,10240,4,1,stream);
}
inline void hc_write53(const float* residual,const float* block,const float* inject,float* weights,float* products,float* out,void* stream){
    require_on();auto& q=*strata::q_of(stream);
    q.parallel_for<HcWriteWeights53>(sycl::range<1>(4),[=](sycl::id<1> i){const float v=inject[i[0]]*0.25f;const float sigmoid=1.0f/(1.0f+sycl::exp(-v));weights[i[0]]=sigmoid*2.0f;});
    q.parallel_for<HcWriteProducts53>(sycl::range<1>(10240),[=](sycl::id<1> i){const size_t j=i[0];products[j]=block[j%2560]*weights[j/2560];});
    q.parallel_for<HcWriteSum53>(sycl::range<1>(10240),[=](sycl::id<1> i){const size_t j=i[0];out[j]=residual[j]+products[j];});
}
inline void original_ffn53(const float* mixed,const FfnWeights53& w,const FfnScratch53& s,void* stream){
    require_on();const auto layout=native_expert_layout(12,8,2560,640);if(!native_expert_supported(12,8,2560,640))throw std::runtime_error("original MTP53 Q4_K/Q8_0 expert kernels missing");
    f32_linear(w.router_f32,mixed,s.router,512,1,stream);native_router_top10(s.router,s.ids,s.weights,stream);
    auto& q=*strata::q_of(stream);const int32_t* ids=s.ids;auto* ptr=s.groups;auto* starts=s.starts;auto* count=s.count;auto* destinations=s.destinations;auto* tokens=s.tokens;const auto base=reinterpret_cast<unsigned long long>(w.expert_blobs);const auto bytes=layout.bytes;
    q.single_task<ExpertGroups53>([=]{for(int j=0;j<10;++j){ptr[j]=base+static_cast<unsigned long long>(ids[j])*bytes;starts[j]=j;destinations[j]=j;tokens[j]=0;}starts[10]=10;count[0]=10;});
    native_quantize_q8_1(mixed,s.q8,2560,1,stream);native_expert_grouped(layout,s.groups,s.starts,s.count,s.destinations,s.tokens,10,10,s.q8,s.native_scratch,s.parts,stream,10);
    one_expert(w.shared_gate_q4,w.shared_up_q4,w.shared_down_q8,mixed,s.q8,s.shared_gate_rows,s.shared_up,s.shared_uq,s.shared,stream);
    f32_linear(w.shared_gate_f32,mixed,s.shared_gate,1,1,stream);const float* gate=s.shared_gate;const float* shared=s.shared;float* gated=s.shared_gated;
    q.parallel_for<SharedGate53>(sycl::range<1>(2560),[=](sycl::id<1> i){gated[i[0]]=shared[i[0]]*(1.0f/(1.0f+sycl::exp(-gate[0])));});
    native_moe_combine(s.parts,s.weights,s.shared_gated,s.output,2560,10,stream);
}
struct StepWeights53{EhHcWeights front;QsaWeights attention;HcWeights53 ffn_hc;FfnWeights53 ffn;HcWeights53 head_hc;const void* own_head_q6;};
struct StepScratch53{EhHcScratch front;QsaScratch attention;float* after_attention;HcScratch53 ffn_hc;FfnScratch53 ffn;float* next_hidden;HcScratch53 head_hc;float* logits;float* write_weights;float* write_products;void* head_q8;};
class OriginalDraftStep53{
    DraftQsaTransaction& state_;float* committed_hidden_;float* tentative_hidden_;void* stream_;int32_t token_=-1;bool active_=false;bool main_failed_=false;
public:
    OriginalDraftStep53(DraftQsaTransaction& state,float* committed_hidden,float* tentative_hidden,void* stream):state_(state),committed_hidden_(committed_hidden),tentative_hidden_(tentative_hidden),stream_(stream){require_on();if(!committed_hidden || !tentative_hidden || committed_hidden==tentative_hidden || !stream)throw std::invalid_argument("own draft53 hidden/queue required");}
    void prepare_candidate(int32_t candidate){require_on();if(main_failed_ || active_ || candidate<0 || candidate>=248320)throw std::runtime_error("draft53 invalid/active candidate");state_.prepare(stream_);token_=candidate;active_=true;}
    void record(const int32_t* device_token,const float* hidden,const StepWeights53& w,const StepScratch53& s){
        require_on();if(!active_ || s.next_hidden!=tentative_hidden_)throw std::runtime_error("draft53 recording outside own transaction");
        eh_hc_front(device_token,hidden,w.front,s.front,stream_);state_.record(s.front.mixed,w.attention,s.attention,stream_);
        hc_write53(s.front.residual,s.attention.output,s.front.injection,s.write_weights,s.write_products,s.after_attention,stream_);
        hc_read53(s.after_attention,w.ffn_hc,s.ffn_hc,stream_);original_ffn53(s.ffn_hc.mixed,w.ffn,s.ffn,stream_);
        hc_write53(s.after_attention,s.ffn.output,s.ffn_hc.injection,s.write_weights,s.write_products,s.next_hidden,stream_);
        hc_read53(s.next_hidden,w.head_hc,s.head_hc,stream_);own_head(w.own_head_q6,s.head_hc.mixed,s.head_q8,s.logits,1,stream_);
    }
    // Synthetic fixture calls exercise state machinery ONLY; they cannot assert actual MAIN validation.
    void settle_fixture(bool accepted){if(!active_)throw std::runtime_error("draft53 no candidate");state_.settle_after_main(accepted,stream_);if(accepted)strata::q_of(stream_)->memcpy(committed_hidden_,tentative_hidden_,10240*4);active_=false;}
    // Actual MAIN window/catch-up adapter. The MTP cache cell i holds the pair
    // (MAIN hidden after input i, MAIN predicted NEXT token i+1), as MtpDrafter::draft.
    // At MAIN base position p the draft has committed cells [0,p), NOT p+1.
    // MAIN [base,guess] commits1 on reject or2 on accept. In BOTH cases catch up
    // using MAIN final_R(row) and actual_picks[row], never the guessed hidden state.
    bool verify_and_catchup(strata::core::Verifier& main,int32_t previous_accepted,int32_t guessed_next,int64_t base_position,
                            strata::core::PoolMultiFn pool,void* user,std::array<int32_t,2>& actual_picks,
                            int32_t* device_token,int32_t* persistent_host_token,const StepWeights53& w,const StepScratch53& s,
                            bool& accepted,std::string& error) try {
        const char* mirror=std::getenv("STRATA_VERIFY_MIRROR_STAGE_T1");
        if(mirror && std::strcmp(mirror,"1")==0){error="original MTP53 requires mirrorT1 OFF; existing CLI guard preserved";return false;}
        if(main_failed_ || base_position<0 || previous_accepted<0 || previous_accepted>=248320 || guessed_next<0 || guessed_next>=248320 || base_position!=state_.committed_cells() || !device_token || !persistent_host_token){error="draft53 main/pair position or pending failure differs";return false;}
        const std::array<int32_t,2> inputs{previous_accepted,guessed_next};
        if(!main.run(2,inputs.data(),base_position,pool,user,actual_picks.data(),error)){main_failed_=true;return false;}
        accepted=actual_picks[0]==guessed_next;const int keep=accepted?2:1;
        if(!main.commit(keep,error) || !main.wait_commit(error)){main_failed_=true;return false;}
        auto& q=*strata::q_of(stream_);q.wait_and_throw();
        // A speculative MTP step may have used its own hidden; discard it even
        // when guess matched, then recompute from authoritative MAIN hidden.
        if(active_)settle_fixture(false);
        for(int row=0;row<keep;++row){
            if(state_.committed_cells()!=base_position+row || !main.final_R_all())throw std::runtime_error("draft53 catch-up source/position missing");
            *persistent_host_token=actual_picks[row];prepare_candidate(*persistent_host_token);
            q.memcpy(device_token,persistent_host_token,4);record(device_token,main.final_R_all()+size_t(row)*10240,w,s);q.wait_and_throw();settle_fixture(true);q.wait_and_throw();
        }
        // Next MAIN base is p+keep, token actual_picks[keep-1]. At EOS/budget
        // stop caller skips further generation; no further emitted-token inference here.
        if(state_.committed_cells()!=base_position+keep){main_failed_=true;error="draft53 catch-up did not reach target prefix";return false;}return true;
    }catch(const std::exception& e){main_failed_=true;error=e.what();return false;}

};
}
