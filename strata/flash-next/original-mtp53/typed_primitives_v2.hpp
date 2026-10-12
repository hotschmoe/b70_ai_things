#pragma once
// Source-only building blocks, NOT a complete/original-parity MTP implementation.
#include "strata/core/original_mtp53_tensor_views.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
namespace strata::kernels::original_mtp53 {
inline void require_on() {
    if (!strata::core::original_mtp53::enabled()) throw std::runtime_error("original MTP53 is default OFF");
}
// Caller-owned, disjoint USM pointers in the stream's context; raw GGUF mapped
// pointers must first be copied/uploaded to legal USM. No allocations/waits/queries.
// scratch_q8_1 must hold native_q8_1_bytes(n_in,ncols). No weight requantization.
inline void quant_linear(int type,const void* weights,const float* x,void* scratch_q8_1,
                         float* y,int n_in,int n_out,int ncols,void* stream) {
    require_on();
    if (!weights || !x || !scratch_q8_1 || !y || !stream || ncols<1 || ncols>8 || n_in<=0 || n_out<=0 ||
        (type!=6 && type!=8 && type!=12 && type!=14) || native_mmvq_weight_bytes(type,n_in,n_out)==0)
        throw std::invalid_argument("original MTP53 quant-linear geometry/type");
    native_quantize_q8_1(x,scratch_q8_1,n_in,ncols,stream);
    native_mmvq(type,weights,scratch_q8_1,y,n_in,n_out,ncols,stream);
}
inline void own_embedding(const void* q4k_table,const int32_t* token_ids,float* output,int ncols,void* stream) {
    require_on();if (!q4k_table || !token_ids || !output || !stream || ncols<1 || ncols>8)
        throw std::invalid_argument("original MTP53 own embedding arguments");
    // Original self-contained table; never set/use the global main NativeEmbed.
    iq_embed_rows(12,q4k_table,1440,token_ids,ncols,2560,output,stream);
}
inline void own_head(const void* q6k_head,const float* x,void* scratch,float* logits,int ncols,void* stream) {
    quant_linear(14,q6k_head,x,scratch,logits,2560,248320,ncols,stream);
}
inline void bf16_linear(const uint16_t* raw_weights,const float* x,float* y,int n_out,int ncols,void* stream) {
    require_on();if (!raw_weights || !x || !y || !stream || (n_out!=128 && n_out!=512) || ncols<1 || ncols>8)
        throw std::invalid_argument("original MTP53 BF16 indexer arguments");
    bf16_gemv_fp32_mmvf_multi(x,2560,raw_weights,y,n_out,2560,n_out,ncols,stream);
}
// Missing type in native_mmvq: original F32 router [2560,512] and shared gate [2560].
// This new bounded primitive keeps F32 weights; declared serial FMA order is a
// diagnostic contract, not claimed equivalent to another backend's reduction.
class F32Linear53;
inline void f32_linear(const float* raw_weights,const float* x,float* y,int n_out,int ncols,void* stream) {
    require_on();if (!raw_weights || !x || !y || !stream || (n_out!=1 && n_out!=512) || ncols<1 || ncols>8)
        throw std::invalid_argument("original MTP53 F32 router arguments");
    auto& q=*strata::q_of(stream);
    q.parallel_for<F32Linear53>(sycl::range<1>(static_cast<size_t>(n_out)*ncols),[=](sycl::id<1> id) {
        const size_t index=id[0],row=index%n_out,col=index/n_out;float sum=0.0f;
        for (int i=0;i<2560;++i) sum=sycl::fma(raw_weights[row*2560+i],x[col*2560+i],sum);
        y[index]=sum;
    });
}
// Exact original expert types/shapes; caller selects one expert's raw block views.
// gate/up each640 rows x2560 Q4_K (921600B); down2560 rows x640 Q8_0 (1740800B).
inline void one_expert(const void* gate_q4k,const void* up_q4k,const void* down_q8,
                       const float* x,void* xq,float* gate,float* up,void* uq,float* out,void* stream) {
    require_on();if (!gate_q4k || !up_q4k || !down_q8 || !x || !xq || !gate || !up || !uq || !out || !stream)
        throw std::invalid_argument("original MTP53 expert pointers");
    native_quantize_q8_1(x,xq,2560,1,stream);
    native_mmvq(12,gate_q4k,xq,gate,2560,640,1,stream);
    native_mmvq(12,up_q4k,xq,up,2560,640,1,stream);
    native_swiglu_quantize_q8_1(gate,up,uq,640,1,stream);
    native_mmvq(8,down_q8,uq,out,640,2560,1,stream);
}
}
