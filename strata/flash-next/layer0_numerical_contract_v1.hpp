// 0021 bounded numerical snapshot contract; no GPU dependencies.
#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
namespace strata::core::layer0_diag {
struct Field { const char* name; const char* encoding; std::size_t bytes; };
inline constexpr std::size_t maximum_frames=4, byte_cap=64u*1024u*1024u;
// One final T1 verifier row per raw GEN prefix1/2/4/8. Source state is physical
// [i,head,j] and convolution is [channel,history]. No raw fused hidden invented.
inline constexpr std::array<Field,33> fields{{
 {"residual_input","LE_F32[4,2560]",40960},
 {"attn_hc_normalized","LE_F32[4,2560]",40960},
 {"attn_hc_low","LE_F32[320]",1280},
 {"attn_hc_gate","LE_F32[4,2560]",40960},
 {"attn_hc_inject","LE_F32[4]",16},
 {"attn_mixed","LE_F32[2560]",10240},
 {"attn_input_q81","Q8_1_LE[80,36]",2880},
 {"gdn_state_before","LE_F32[128,48,128]",3145728},
 {"gdn_conv_before","LE_F32[10240,3]",122880},
 {"gdn_qkv","LE_F32[10240]",40960},
 {"gdn_z","LE_F32[6144]",24576},
 {"gdn_decay_beta","LE_F32[2,48]",384},
 {"gdn_normalized_qkv","LE_F32[10240]",40960},
 {"gdn_output_gated","LE_F32[6144]",24576},
 {"gdn_output_q81","Q8_1_LE[192,36]",6912},
 {"gdn_block_output","LE_F32[2560]",10240},
 {"gdn_state_after","LE_F32[128,48,128]",3145728},
 {"gdn_conv_after","LE_F32[10240,3]",122880},
 {"residual_after_attn","LE_F32[4,2560]",40960},
 {"ffn_mixed","LE_F32[2560]",10240},
 {"ffn_input_q81","Q8_1_LE[80,36]",2880},
 {"router_logits","LE_F32[512]",2048},
 {"router_ids","LE_I32[10]",40},
 {"router_weights","LE_F32[10]",40},
 {"shared_gate_up","LE_F32[2,640]",5120},
 {"shared_hidden_DERIVED","LE_F32[640]",2560},
 {"shared_hq81","Q8_1_LE[20,36]",720},
 {"expert_entry_map","LE_I32[10,3]",120},
 {"expert_gate_up","LE_F32[10,2,640]",51200},
 {"expert_hidden_DERIVED","LE_F32[10,640]",25600},
 {"expert_hq81","Q8_1_LE[10,20,36]",7200},
 {"ffn_block_output","LE_F32[2560]",10240},
 {"residual_after_ffn","LE_F32[4,2560]",40960}
}};
constexpr std::size_t frame_bytes(){std::size_t n=0;for(auto f:fields)n+=f.bytes;return n;}
static_assert(frame_bytes()*maximum_frames<=byte_cap,"Bounded layer0 capture overflow");
// Seven shared/expert fields remain explicitly UNOBSERVED in initial0021.
inline void require_enabled_geometry(bool enabled,int global_layer,int rows,int prefix_tokens){
 if(!enabled)return;
 if(global_layer!=0 || rows!=1 || !(prefix_tokens==1||prefix_tokens==2||prefix_tokens==4||prefix_tokens==8))
  throw std::runtime_error("Layer0 draft capture shape/position scope differs");
}
} // namespace strata::core::layer0_diag

namespace strata::core::layer0_diag {
inline std::size_t index(const char* name){for(std::size_t i=0;i<fields.size();++i)if(std::string(fields[i].name)==name)return i;throw std::invalid_argument("Unknown layer0 field");}
inline std::size_t offset(std::size_t field){if(field>=fields.size())throw std::invalid_argument("Layer0 field outside layout");std::size_t n=0;for(std::size_t i=0;i<field;++i)n+=fields[i].bytes;return n;}
inline void copy_extent(std::size_t field,std::size_t bytes){if(field>=fields.size()||bytes!=fields[field].bytes)throw std::invalid_argument("Layer0 copy extent differs");}
inline bool first_position(std::size_t tokens,int rows,std::int64_t position,int token,const std::vector<std::int64_t>& ids){return rows==1&&tokens==ids.size()&&(tokens==1||tokens==2||tokens==4||tokens==8)&&position==std::int64_t(tokens)-1&&token==ids.back();}
}
