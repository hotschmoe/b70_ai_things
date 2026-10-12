#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::kernels::original_mtp53 {
inline bool decode_head_skip53(){
 static const bool enabled=[](){const char* value=std::getenv("STRATA_ORIGINAL_MTP53_DECODE_FIRST_HEAD_SKIP");if(!value || std::strcmp(value,"0")==0)return false;if(std::strcmp(value,"1")==0)return true;throw std::runtime_error("decode first head skip must0/1");}();return enabled;
}
inline bool decode_head_compute53(bool enabled,int keep,int row){
 if((keep!=1 && keep!=2) || row<0 || row>=keep)throw std::runtime_error("decode head row/keep differs");
 return !(enabled && keep==2 && row==0);
}
}
