#pragma once
#include "strata/kernels/original_mtp53_checkpoint_contract.hpp"
namespace strata::kernels::original_mtp53 {
struct ShiftedOriginalCheckpoint53 {
 std::vector<int32_t> target_inputs;int32_t lookahead=-1;
 OriginalCheckpoint53 current,pre_last;
 std::vector<uint8_t> last_target_hidden;
 std::string envelope_sha;
};
inline void validate_shifted53(const ShiftedOriginalCheckpoint53& c){
 validate_checkpoint53(c.current);validate_checkpoint53(c.pre_last);
 if(c.target_inputs.empty() || c.target_inputs.size()>4096 || c.current.cells!=c.target_inputs.size() || c.pre_last.cells+1!=c.current.cells || c.lookahead<0 || c.lookahead>=248320 || c.last_target_hidden.size()!=40960 || c.current.source_rid!=c.pre_last.source_rid || c.current.source_generation!=c.pre_last.source_generation || c.current.publisher_sha!=c.pre_last.publisher_sha || c.current.draft_contract_sha!=c.pre_last.draft_contract_sha || c.current.target_contract_sha!=c.pre_last.target_contract_sha || c.current.payload_sha!=checkpoint_digest53(c.current) || c.pre_last.payload_sha!=checkpoint_digest53(c.pre_last))throw std::runtime_error("shifted original complete envelope differs");
 for(int32_t t:c.target_inputs)if(t<0 || t>=248320)throw std::runtime_error("shifted actual target input invalid");
 std::vector<int32_t> known(c.target_inputs.begin()+1,c.target_inputs.end());if(c.pre_last.prefix!=known)throw std::runtime_error("shifted pre-last consumed NEXT prefix differs");known.push_back(c.lookahead);if(c.current.prefix!=known)throw std::runtime_error("shifted current consumed NEXT prefix differs");
 for(size_t i=0;i<c.last_target_hidden.size();i+=4){float f;std::memcpy(&f,c.last_target_hidden.data()+i,4);if(!std::isfinite(f))throw std::runtime_error("shifted authoritative MAIN hidden nonfinite");}
}
inline std::string shifted_digest53(const ShiftedOriginalCheckpoint53& c){validate_shifted53(c);strata::core::prefix_lifecycle::Sha256 h;const char schema[]="ORIGINAL-SHIFTED-NEXT-CP1";h.update(reinterpret_cast<const uint8_t*>(schema),sizeof(schema)-1);strata::core::prefix_lifecycle::digest_tokens(h,c.target_inputs);strata::core::prefix_lifecycle::number(h,uint32_t(c.lookahead));h.update(reinterpret_cast<const uint8_t*>(c.current.payload_sha.data()),64);h.update(reinterpret_cast<const uint8_t*>(c.pre_last.payload_sha.data()),64);h.update(c.last_target_hidden.data(),c.last_target_hidden.size());char out[65];h.finish(out);return out;}
}
