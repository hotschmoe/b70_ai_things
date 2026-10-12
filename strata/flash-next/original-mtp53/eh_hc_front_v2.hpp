#pragma once
// Original qwen4exp MTP input/EH and attention-HC read segment, T=1.
// Implements graph_mtp/build_hc_mix order, not the historical split-FC substitute.
#include "strata/kernels/original_mtp53_typed_primitives.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/elementwise.hpp"
namespace strata::kernels::original_mtp53 {
struct EhHcWeights {
    const void* own_embedding_q4k;       // token_embd.weight [2560,248320]
    const float* enorm;                 // blk.48.nextn.enorm.weight [2560]
    const float* hnorm;                 // blk.48.nextn.hnorm.weight [10240], per stream
    const void* eh_proj_q4k;            // blk.48.nextn.eh_proj.weight [5120,2560]
    const float* hc_norm;               // blk.48.hc_attn_norm.weight [10240]
    const void* hc_down_q4k;            // blk.48.hc_attn_down.weight [10240,320]
    const void* hc_up_q5_0;             // blk.48.hc_attn_up.weight [320,10240]
    const void* hc_inject_q4k;          // blk.48.hc_attn_inject.weight [10240,4]
};
struct EhHcScratch {
    float* embedding;                  // 2560
    float* e_norm;                     // 2560
    float* h_norm;                     // 10240
    float* concat_eh;                  // 4 rows of5120, E FIRST then H per stream
    float* residual;                   // 10240; tentative EH result
    float* hc_normalized;              // 10240
    float* low;                        // 320; scaled then SiLU
    float* gate;                       // 10240
    float* weighted_streams;           // 10240; materialized product before mean
    float* mixed;                      // 2560
    float* injection;                  // 4, raw HC write gate, not sigmoid yet
    void* q8_1;                        // >= max(native_q8_1_bytes(5120,4),(...10240,1))
};
class ConcatEH53;
class GateProducts53;
class MeanStreams53;
// All inputs/weights/scratch must be legal USM in stream context, properly aligned
// and pairwise disjoint. token_id is one validated device ID in[0,248320).
// hidden is immutable [stream][channel]; this segment never modifies cache/state.
// F32 norm weights already carry the GGUF converter's(1+w): do NOT add1.
inline void eh_hc_front(const int32_t* token_id,const float* hidden,const EhHcWeights& w,
                        const EhHcScratch& s,void* stream) {
    require_on();
    if (!token_id || !hidden || !stream || !w.own_embedding_q4k || !w.enorm || !w.hnorm || !w.eh_proj_q4k ||
        !w.hc_norm || !w.hc_down_q4k || !w.hc_up_q5_0 || !w.hc_inject_q4k ||
        !s.embedding || !s.e_norm || !s.h_norm || !s.concat_eh || !s.residual || !s.hc_normalized ||
        !s.low || !s.gate || !s.weighted_streams || !s.mixed || !s.injection || !s.q8_1)
        throw std::invalid_argument("original MTP53 EH/HC arguments");
    auto& q=*strata::q_of(stream);
    own_embedding(w.own_embedding_q4k,token_id,s.embedding,1,stream);
    // Original build_norm h reshaped[N,HC,T]: one RMS perN, gamma slice stream.
    native_qsa_rms_norm_weighted(s.embedding,w.enorm,s.e_norm,2560,1,1e-6f,stream);
    native_qsa_rms_norm_grouped(hidden,w.hnorm,s.h_norm,2560,4,4,1e-6f,stream);
    const float* e=s.e_norm;const float* h=s.h_norm;float* concat=s.concat_eh;
    q.parallel_for<ConcatEH53>(sycl::range<1>(20480),[=](sycl::id<1> id){
        const size_t at=id[0],stream_id=at/5120,channel=at%5120;
        concat[at]=channel<2560?e[channel]:h[stream_id*2560+channel-2560];
    });
    // ONE combined original Q4_K matrix; never split into FC_embedding/FC_hidden.
    quant_linear(12,w.eh_proj_q4k,s.concat_eh,s.q8_1,s.residual,5120,2560,4,stream);
    native_qsa_rms_norm_grouped(s.residual,w.hc_norm,s.hc_normalized,2560,4,4,1e-6f,stream);
    quant_linear(12,w.hc_down_q4k,s.hc_normalized,s.q8_1,s.low,10240,320,1,stream);
    scale_inplace(s.low,320,0.25f,stream);silu_inplace(s.low,320,stream);
    quant_linear(6,w.hc_up_q5_0,s.low,s.q8_1,s.gate,320,10240,1,stream);
    const float* xn=s.hc_normalized;const float* gate=s.gate;float* products=s.weighted_streams;
    // Materialize sigmoid product as the original unfused ggml_mul graph does.
    // Numeric backend exp/reduction parity remains an actual fixture question.
    q.parallel_for<GateProducts53>(sycl::range<1>(10240),[=](sycl::id<1> id){
        const size_t at=id[0];const float sigmoid=1.0f/(1.0f+sycl::exp(-gate[at]));products[at]=xn[at]*sigmoid;
    });
    const float* product=s.weighted_streams;float* mixed=s.mixed;
    q.parallel_for<MeanStreams53>(sycl::range<1>(2560),[=](sycl::id<1> id){
        const size_t d=id[0];float sum=product[d];sum=sum+product[2560+d];sum=sum+product[5120+d];sum=sum+product[7680+d];mixed[d]=sum*0.25f;
    });
    quant_linear(12,w.hc_inject_q4k,s.hc_normalized,s.q8_1,s.injection,10240,4,1,stream);
}
}
