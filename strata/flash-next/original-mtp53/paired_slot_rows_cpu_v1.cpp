#include "strata/core/original_mtp53_paired_slot_rows.hpp"
#include <cstdio>
using namespace strata::core::original_mtp53;
int main(){auto valid=[](int next_capacity,int slot_count,int device,bool enabled){return paired_slot_rows53(enabled,true,true,device,0,24,48,8,slot_count,0,1,24,48,next_capacity,slot_count,true,true);};
 if(!valid(8,4,0,true) || !valid(8,2,0,true) || !valid(8,6,0,true))return 1;
 if(valid(4,4,0,true) || valid(8,4,1,true) || valid(8,4,0,false))return 2;
 int rows[8]={0,0,1,1,2,2,3,3};int64_t pos[8]={11,12,50,51,99,100,4094,4095};if(!grouped_t1_t2_rows53(rows,pos,8,4))return 3;
 int triple[3]={0,0,0};int64_t tp[3]={1,2,3};if(grouped_t1_t2_rows53(triple,tp,3,4))return 4;
 int broken[3]={0,1,0};if(grouped_t1_t2_rows53(broken,tp,3,4))return 5;
 pos[1]=13;if(grouped_t1_t2_rows53(rows,pos,8,4))return 6;pos[1]=12;pos[7]=4096;if(grouped_t1_t2_rows53(rows,pos,8,4))return 7;
 std::puts("MTP53_PAIRED_SLOT_ROWS_CPU_PASS controls=7 both_capacity8=1 private_contiguous_t1_t2=1 stock_scope_refused=1");}
