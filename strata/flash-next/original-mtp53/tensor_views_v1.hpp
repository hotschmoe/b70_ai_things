#pragma once
// Prototype only: read-only original publisher tensor views, not an admitted MTP graph.
#include "strata/artifact/gguf_reader.hpp"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
namespace strata::core::original_mtp53 {
inline bool enabled() noexcept { const char* s=std::getenv("STRATA_ORIGINAL_MTP53"); return s && std::strcmp(s,"1")==0; }
struct Spec { const char* name; uint32_t type; std::vector<uint64_t> shape; uint64_t bytes; };
inline const std::vector<Spec>& specs() {
    static const std::vector<Spec> values={
        {"output.weight", 14u, {2560ull, 248320ull}, 521472000ull},
        {"token_embd.weight", 12u, {2560ull, 248320ull}, 357580800ull},
        {"blk.48.attn_k.weight", 12u, {2560ull, 512ull}, 737280ull},
        {"blk.48.attn_k_norm.weight", 0u, {256ull}, 1024ull},
        {"blk.48.attn_output.weight", 12u, {6144ull, 2560ull}, 8847360ull},
        {"blk.48.attn_q.weight", 12u, {2560ull, 12288ull}, 17694720ull},
        {"blk.48.attn_q_norm.weight", 0u, {256ull}, 1024ull},
        {"blk.48.attn_v.weight", 14u, {2560ull, 512ull}, 1075200ull},
        {"blk.48.ffn_down_exps.weight", 8u, {640ull, 2560ull, 512ull}, 891289600ull},
        {"blk.48.ffn_down_shexp.weight", 8u, {640ull, 2560ull}, 1740800ull},
        {"blk.48.ffn_gate_exps.weight", 12u, {2560ull, 640ull, 512ull}, 471859200ull},
        {"blk.48.ffn_gate_inp.weight", 0u, {2560ull, 512ull}, 5242880ull},
        {"blk.48.ffn_gate_inp_shexp.weight", 0u, {2560ull}, 10240ull},
        {"blk.48.ffn_gate_shexp.weight", 12u, {2560ull, 640ull}, 921600ull},
        {"blk.48.ffn_up_exps.weight", 12u, {2560ull, 640ull, 512ull}, 471859200ull},
        {"blk.48.ffn_up_shexp.weight", 12u, {2560ull, 640ull}, 921600ull},
        {"blk.48.hc_attn_down.weight", 12u, {10240ull, 320ull}, 1843200ull},
        {"blk.48.hc_attn_inject.weight", 12u, {10240ull, 4ull}, 23040ull},
        {"blk.48.hc_attn_norm.weight", 0u, {10240ull}, 40960ull},
        {"blk.48.hc_attn_up.weight", 6u, {320ull, 10240ull}, 2252800ull},
        {"blk.48.hc_ffn_down.weight", 14u, {10240ull, 320ull}, 2688000ull},
        {"blk.48.hc_ffn_inject.weight", 12u, {10240ull, 4ull}, 23040ull},
        {"blk.48.hc_ffn_norm.weight", 0u, {10240ull}, 40960ull},
        {"blk.48.hc_ffn_up.weight", 6u, {320ull, 10240ull}, 2252800ull},
        {"blk.48.indexer.k_norm.weight", 0u, {128ull}, 512ull},
        {"blk.48.indexer.k_proj.weight", 30u, {2560ull, 128ull}, 655360ull},
        {"blk.48.indexer.q_norm.weight", 0u, {128ull}, 512ull},
        {"blk.48.indexer.q_proj.weight", 30u, {2560ull, 512ull}, 2621440ull},
        {"blk.48.nextn.eh_proj.weight", 12u, {5120ull, 2560ull}, 7372800ull},
        {"blk.48.nextn.enorm.weight", 0u, {2560ull}, 10240ull},
        {"blk.48.nextn.hc_head_down.weight", 12u, {10240ull, 320ull}, 1843200ull},
        {"blk.48.nextn.hc_head_norm.weight", 0u, {10240ull}, 40960ull},
        {"blk.48.nextn.hc_head_up.weight", 6u, {320ull, 10240ull}, 2252800ull},
        {"blk.48.nextn.hnorm.weight", 0u, {10240ull}, 40960ull},
    };
    return values;
}
struct View { const void* readonly_gguf_bytes; uint32_t type; std::vector<uint64_t> shape; uint64_t bytes; };
class TensorViews {
    std::unique_ptr<strata::GgufFile> file_;
public:
    // Root must freshly verify the locked whole-file SHA before this call. This method
    // checks the original schema/bounds; it does not claim cryptographic payload identity.
    // GGUF uses PROT_READ mapping. There is no conversion, device upload, head borrowing,
    // +1 norm adjustment, query, graph capture, or model-state modification here.
    bool load(const std::string& path, std::string& err) try {
        if (!enabled()) { err="original MTP53 is default OFF"; return false; }
        if (file_) { err="original MTP53 already loaded"; return false; }
        auto next=std::make_unique<strata::GgufFile>(path);
        strata::Qwen4ExpGuard guard; guard.block_count=49;guard.experts=512;guard.experts_used=10;
        err=strata::check_architecture(*next,guard); if (!err.empty()) return false;
        const auto* count=next->get("qwen4exp.nextn_predict_layers");
        if (!count || count->u!=1 || next->tensors().size()!=34 || next->file_size()!=2786204800ull) {
            err="original publisher MTP53 requires exact one-layer/34-tensor self-contained schema";return false;
        }
        std::set<std::string> seen;
        for (const auto& t:next->tensors()) if (!seen.insert(t.name).second) { err="duplicate MTP53 tensor";return false; }
        for (const auto& want:specs()) {
            const auto* t=next->find(want.name);
            if (!t || t->type!=want.type || t->shape!=want.shape || strata::tensor_payload_bytes(*t)!=want.bytes ||
                t->offset>next->file_size()-next->data_start() || want.bytes>next->file_size()-next->data_start()-t->offset) {
                err=std::string("original MTP53 exact tensor mismatch: ")+want.name;return false;
            }
        }
        file_=std::move(next);return true;
    } catch(const std::exception& e) { err=e.what();return false; }
    View at(const std::string& name) const {
        if (!enabled() || !file_) throw std::runtime_error("original MTP53 not enabled/loaded");
        const auto* t=file_->find(name);if (!t) throw std::runtime_error("original MTP53 unknown tensor");
        return {file_->tensor_data(*t),t->type,t->shape,strata::tensor_payload_bytes(*t)};
    }
    static constexpr uint64_t conservative_weight_reserve_bytes=2775258112ull;
    // Caller owns any USM uploads/copies until all dependent operations retire.
    // This object must outlive users of returned read-only host views.
};
}
