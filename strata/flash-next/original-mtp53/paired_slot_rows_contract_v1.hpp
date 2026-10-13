#pragma once
#include <cstdint>
namespace strata::core::original_mtp53 {
// Only the exact original paired serve graph owns an eight-row handoff. This
// capability never widens stock grouped/pipe paths or changes a Verifier layout.
inline bool paired_slot_rows53(bool serving,bool original,bool full_chain,int device,int begin,int end,int layers,int max_rows,int slots,int handoff_base,int next_device,int next_begin,int next_end,int next_max_rows,int next_slots,bool same_geometry,bool has_handoff){
 return serving && original && full_chain && device==0 && begin==0 && end==24 && layers==48 && max_rows==8 && (slots==2 || slots==4 || slots==6) && handoff_base==0 && next_device==1 && next_begin==24 && next_end==48 && next_max_rows==8 && next_slots==slots && same_geometry && has_handoff;
}
inline bool grouped_t1_t2_rows53(const int* rows,const int64_t* positions,int count,int slots){
 if(!rows || !positions || count<1 || count>8)return false;
 for(int t=0;t<count;++t){if(rows[t]<0 || rows[t]>=slots || positions[t]<0 || positions[t]>=4096)return false;
  if(t && rows[t]==rows[t-1] && positions[t]!=positions[t-1]+1)return false;
  if(t>1 && rows[t]==rows[t-1] && rows[t]==rows[t-2])return false;
  for(int u=0;u<t-1;++u)if(rows[u]==rows[t] && rows[t-1]!=rows[t])return false;
 }
 return true;
}
}
