// CPU-only reduction: same head signature/blocks/static names, no GPU or model.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#ifndef UNIQUE_C6
#define UNIQUE_C6 0
#endif
#if UNIQUE_C6
#define C6_LOGGED logged_c6
#else
#define C6_LOGGED logged
#endif
namespace strata::kernels {
namespace {
int readword(const char* key){const char* e=std::getenv(key);if(!e||std::strcmp(e,"0")==0)return 0;char* end=nullptr;unsigned long n=std::strtoul(e,&end,16);if(!end||*end||n!=0x01010101ul)throw std::invalid_argument("bad word");return int(n);}
int q5_head_runtime_ones_word(){static int word=[] {return readword("STRATA_Q5_HEAD_RUNTIME_ONES");}();return word;}
int q5_head_runtime_ones_c6_word(){static int word=[] {return readword("STRATA_Q5_HEAD_RUNTIME_ONES_C6");}();return word;}
bool wide_k(){return true;}int wide_k_rpw(){return 1;}
volatile unsigned sink=0;
}
void native_mmvq_head(int ggml_type,const void* weights,const void* x_q8_1,float* y,int n_in,int n_out,int ncols,void* stream){
 (void)weights;(void)x_q8_1;(void)y;(void)stream;
 static const bool head_rpw2=[] {const char* e=std::getenv("STRATA_HEAD_Q5K_RPW2");return e&&std::strcmp(e,"1")==0;}();(void)head_rpw2;
 if(ggml_type==13&&n_in==2560&&n_out==248320&&ncols==2&&wide_k()&&wide_k_rpw()==1){const int ones_word=q5_head_runtime_ones_word();if(ones_word!=0){
  static const bool logged=[] {std::fprintf(stderr,"strata Q5 head runtime ones: enabled=1 word=01010101 N=2560 M=248320 C=2 RPW=1 host-selection-only\n");return true;}();(void)logged;sink=sink+2;return;
 }}
 if(ggml_type==13&&n_in==2560&&n_out==248320&&ncols==6&&wide_k()&&wide_k_rpw()==1){const int ones_word=q5_head_runtime_ones_c6_word();if(ones_word!=0){
  static const bool C6_LOGGED=[] {std::fprintf(stderr,"strata Q5 head runtime ones C6: enabled=1 word=01010101 N=2560 M=248320 C=6 RPW=1 host-selection-only\n");return true;}();(void)C6_LOGGED;sink=sink+6;return;
 }}
}
}
int main(){const int cols[]={2,6,2,6};for(int c:cols)strata::kernels::native_mmvq_head(13,nullptr,nullptr,nullptr,2560,248320,c,nullptr);return strata::kernels::sink==16?0:2;}
