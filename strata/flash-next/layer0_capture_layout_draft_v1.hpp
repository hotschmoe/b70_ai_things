// Source-only bounded layout draft; no live observer hooks or GPU dependencies.
#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
namespace original_layer0_capture_draft {
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
// Binding each field to an actual producer/queue, graph and owner is still TODO.
// Implementations must remain default off and add no allocation/copy nodes off.
inline void require_enabled_geometry(bool enabled,int global_layer,int rows,int prefix_tokens){
 if(!enabled)return;
 if(global_layer!=0 || rows!=1 || !(prefix_tokens==1||prefix_tokens==2||prefix_tokens==4||prefix_tokens==8))
  throw std::runtime_error("Layer0 draft capture shape/position scope differs");
}
} // namespace original_layer0_capture_draft
