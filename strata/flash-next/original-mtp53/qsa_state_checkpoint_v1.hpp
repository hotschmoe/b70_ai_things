#pragma once
// Isolated draft-owned T1 compress4 QSA state/attention segment. No main state/globals.
#include "strata/kernels/original_mtp53_eh_hc_front.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <algorithm>
namespace strata::kernels::original_mtp53 {
struct DraftBank {
    uint16_t* key;uint16_t* value; // [page][2KVheads][4cells][256], max_cells*512 each
    float* tail;float* dead;float* pooled;int32_t* block_pos; // 384,128,(max_cells/4+2)*128,1
};
struct QsaWeights {
    const void* query_q4k;const void* key_q4k;const void* value_q6k;const void* output_q4k;
    const float* query_norm;const float* key_norm;
    const uint16_t* index_query_bf16;const uint16_t* index_key_bf16;
    const float* index_query_norm;const float* index_key_norm;
};
struct QsaScratch {
    float* query_full;float* query;float* key;float* value; //12288,6144,512,512
    float* index_raw;uint16_t* index_half;float* index_expanded;float* index_query; //128,128,128,512
    float* scores;int32_t* selected;float* attention_scratch;float* attention;float* gated;float* output;
    void* q8_1; // native_q8_1_bytes(6144,1); output projection is largest input
};
class StoreRawHalf53;class LoadRawHalf53;class SelectOriginalBlocks53;
// Full-bank copying is a deliberately simple diagnostic transaction, NOT a latency optimization.
inline void copy_bank(const DraftBank& from,const DraftBank& to,int max_cells,sycl::queue& q){
    require_on();q.memcpy(to.key,from.key,size_t(max_cells)*512*2);q.memcpy(to.value,from.value,size_t(max_cells)*512*2);
    q.memcpy(to.tail,from.tail,384*4);q.memcpy(to.dead,from.dead,128*4);q.memcpy(to.pooled,from.pooled,size_t(max_cells/4+2)*128*4);q.memcpy(to.block_pos,from.block_pos,4);
}
// Original selection budget is512 COMPLETED blocks (=2048cells), plus incomplete tail.
// This preserves compress4 rather than substituting current state's constant2051-cell budget.
// Ties use lowest block ID, explicit diagnostic tie contract; original backend parity must be tested.
inline void select_original_blocks(const float* scores,const int32_t* step,int32_t* ids,int max_blocks,void* stream){
    require_on();if(!stream)throw std::invalid_argument("draft53 explicit stream required");auto& q=*strata::q_of(stream);
    q.single_task<SelectOriginalBlocks53>([=]{
        int chosen[512];float values[512];int have=0;const int n_bid=step[kStepNBid],n_kv=step[kStepNKv];
        for(int block=0;block<n_bid && block<max_blocks;++block){
            const float score=scores[block];int at=have;
            while(at>0 && (score>values[at-1] || (score==values[at-1] && block<chosen[at-1])))--at;
            if(at<512){const int last=have<512?have:511;for(int j=last;j>at;--j){chosen[j]=chosen[j-1];values[j]=values[j-1];}chosen[at]=block;values[at]=score;if(have<512)++have;}
        }
        // Attention selection rows are emitted in cache order; output order does not change softmax set.
        for(int i=1;i<have;++i){int value=chosen[i],j=i;while(j>0 && chosen[j-1]>value){chosen[j]=chosen[j-1];--j;}chosen[j]=value;}
        int out=0;for(int i=0;i<have;++i)for(int member=0;member<4;++member)ids[out++]=chosen[i]*4+member;
        for(int cell=n_bid*4;cell<n_kv;++cell)ids[out++]=cell;
        // step's width is prepared explicitly to this SAME cardinality.
    });
}
class DraftQsaTransaction {
    DraftBank committed_,working_;int max_cells_;int cells_=0;bool pending_=false;
    // Persistent caller-owned USM-host staging survives graph capture/replay. Never stack-copy sources.
    int32_t* host_step_;int32_t* host_position_;int32_t* step_;int32_t* positions_;const int32_t* own_pages_;
public:
    DraftQsaTransaction(DraftBank committed,DraftBank working,int max_cells,int32_t* host_step,int32_t* host_positions,
                        int32_t* step,int32_t* positions,const int32_t* own_pages):committed_(committed),working_(working),max_cells_(max_cells),host_step_(host_step),host_position_(host_positions),step_(step),positions_(positions),own_pages_(own_pages){
        require_on();if(max_cells<4 || max_cells>4096 || max_cells%4 || !host_step || !host_positions || !step || !positions || !own_pages)throw std::invalid_argument("draft53 bounded own state/staging required");
        struct Span{uintptr_t first,last;};std::vector<Span> spans;
        auto add=[&](const void* p,size_t bytes){const auto start=reinterpret_cast<uintptr_t>(p);if(!p || start%4 || bytes>UINTPTR_MAX-start)throw std::invalid_argument("draft53 aligned bank span");Span next{start,start+bytes};for(const auto& old:spans)if(next.first<old.last && old.first<next.last)throw std::invalid_argument("draft53 state banks alias");spans.push_back(next);};
        for(const auto& bank:{committed,working}){add(bank.key,size_t(max_cells)*512*2);add(bank.value,size_t(max_cells)*512*2);add(bank.tail,384*4);add(bank.dead,128*4);add(bank.pooled,size_t(max_cells/4+2)*128*4);add(bank.block_pos,4);}
    }
    // Caller zero-initializes BOTH banks and owns a complete identity page table independently of main.
    // Prepare only after previous draft and actual main verifier decisions are terminal; no fake callback.
    void prepare(void* stream){
        require_on();if(!stream)throw std::invalid_argument("draft53 explicit stream required");if(pending_ || cells_>=max_cells_)throw std::runtime_error("draft53 transaction active/capacity exhausted");auto& q=*strata::q_of(stream);if(!q.is_in_order())throw std::invalid_argument("draft53 ordered queue required");copy_bank(committed_,working_,max_cells_,q);
        host_step_[kStepPos]=cells_;host_step_[kStepNKv]=cells_+1;host_step_[kStepNBid]=(cells_+1)/4;
        host_step_[kStepWidth]=4*std::min((cells_+1)/4,512)+(cells_+1)%4;
        for(int h=0;h<24;++h)host_position_[h]=cells_;pending_=true;
    }
    // Record this inside a graph: pointers fixed, counts/positions are refreshed in owned persistent staging.
    void record(const float* mixed,const QsaWeights& w,const QsaScratch& s,void* stream){
        require_on();if(!stream || !mixed || !w.query_q4k || !w.key_q4k || !w.value_q6k || !w.output_q4k || !w.query_norm || !w.key_norm || !w.index_query_bf16 || !w.index_key_bf16 || !w.index_query_norm || !w.index_key_norm || !s.query_full || !s.query || !s.key || !s.value || !s.index_raw || !s.index_half || !s.index_expanded || !s.index_query || !s.scores || !s.selected || !s.attention_scratch || !s.attention || !s.gated || !s.output || !s.q8_1)throw std::invalid_argument("draft53 own QSA arguments");if(!pending_)throw std::runtime_error("draft53 record without prepare");auto& q=*strata::q_of(stream);q.memcpy(step_,host_step_,kStepCount*4);q.memcpy(positions_,host_position_,24*4);
        const auto shape=qsa_real_shapes();const RopeScaling scaling{};const int max_blocks=max_cells_/4+2;
        quant_linear(12,w.query_q4k,mixed,s.q8_1,s.query_full,2560,12288,1,stream);
        quant_linear(12,w.key_q4k,mixed,s.q8_1,s.key,2560,512,1,stream);
        quant_linear(14,w.value_q6k,mixed,s.q8_1,s.value,2560,512,1,stream);
        native_qsa_rms_norm_rope(s.query_full,512,w.query_norm,s.query,24,256,64,1e-6f,scaling,positions_,stream);
        native_qsa_rms_norm_rope(s.key,256,w.key_norm,s.key,2,256,64,1e-6f,scaling,positions_,stream);
        // V is NEVER normalized/rotated. Fixed FP16 pools; no global KV setting mutations.
        kv_append_step(working_.key,working_.value,own_pages_,step_,s.key,s.value,shape,stream);
        bf16_gemv_fp32_mmvf(mixed,w.index_key_bf16,s.index_raw,2560,128,stream);
        const float* raw=s.index_raw;uint16_t* half=s.index_half;float* expanded=s.index_expanded;
        q.parallel_for<StoreRawHalf53>(sycl::range<1>(128),[=](sycl::id<1> id){reinterpret_cast<sycl::half*>(half)[id[0]]=sycl::half(raw[id[0]]);});
        q.parallel_for<LoadRawHalf53>(sycl::range<1>(128),[=](sycl::id<1> id){expanded[id[0]]=float(reinterpret_cast<const sycl::half*>(half)[id[0]]);});
        const QsaIndexerBuffers indexer{working_.tail,working_.dead,working_.pooled,working_.block_pos};
        native_qsa_indexer_append(s.index_expanded,step_+kStepPos,0,w.index_key_norm,1e-6f,indexer,shape,max_cells_,scaling,stream);
        bf16_gemv_fp32_mmvf(mixed,w.index_query_bf16,s.index_query,2560,512,stream);
        native_qsa_rms_norm_rope(s.index_query,128,w.index_query_norm,s.index_query,4,128,64,1e-6f,scaling,positions_,stream);
        // Existing F32 block scores; common positive1/sqrt128 scale does not change mathematical ranking.
        // Floating/tie parity is an explicit unresolved numeric gate, not hidden as original equivalence.
        qsa_block_scores(working_.pooled,working_.dead,s.index_query,step_,1,max_blocks,shape,s.scores,stream);
        select_original_blocks(s.scores,step_,s.selected,max_blocks,stream);
        QsaAttnPools pools{};pools.k_pool=working_.key;pools.v_pool=working_.value;pools.page_table=own_pages_;
        qsa_decode_attn_step(s.query,pools,s.selected,step_,2051,shape,s.attention_scratch,s.attention,stream);
        native_qsa_gate_apply(s.attention,s.query_full,s.gated,24,256,stream);
        quant_linear(12,w.output_q4k,s.gated,s.q8_1,s.output,6144,2560,1,stream);
    }
    // OUTSIDE captured graphs, only after authoritative main acceptance/rejection is actually available.
    // The component fixture may exercise synthetic decisions but must not claim main verification.
    void settle_after_main(bool accepted,void* stream){
        require_on();if(!stream)throw std::invalid_argument("draft53 explicit stream required");if(!pending_)throw std::runtime_error("draft53 no pending state");if(accepted){copy_bank(working_,committed_,max_cells_,*strata::q_of(stream));++cells_;}pending_=false;
    }
    // Complete original checkpoint import only AFTER caller copied both full
    // banks and drained. This changes host transaction metadata, not kernel math.
    void restore_committed_cells(int cells){require_on();if(pending_ || cells<0 || cells>max_cells_)throw std::runtime_error("original checkpoint pending/count invalid");cells_=cells;}
    int committed_cells()const noexcept{return cells_;}
};
}
