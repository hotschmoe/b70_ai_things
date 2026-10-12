#include "strata/kernels/original_mtp53_exact_vector_bytes.hpp"
#include <cstdio>
#include "strata/kernels/original_mtp53_native_token_boundary.hpp"
#include <cstdint>
using strata::kernels::original_mtp53::exact_vector_bytes53;
int main(){std::vector<float> positive{0.0f},negative{-0.0f};if(positive!=negative || exact_vector_bytes53(positive,negative))return 1;
 std::vector<float> a{1.0f,2.0f},b=a;if(!exact_vector_bytes53(a,b))return 2;b[1]=3.0f;if(exact_vector_bytes53(a,b))return 3;
 if(exact_vector_bytes53(a,std::vector<float>{1.0f}) || !exact_vector_bytes53(std::vector<float>{},std::vector<float>{}))return 4;
 uint32_t bits=0x7fc00001u;float nan;std::memcpy(&nan,&bits,4);std::vector<float> n{nan};if(!exact_vector_bytes53(n,n))return 5;
 ++bits;std::memcpy(&nan,&bits,4);if(exact_vector_bytes53(n,std::vector<float>{nan}))return 6;
 if(exact_vector_bytes53(std::vector<int64_t>{1},std::vector<int64_t>{2}))return 7;
 using namespace strata::kernels::original_mtp53;
 if(original_tokens32_53({0,248319})!=std::vector<int32_t>{0,248319})return 8;
 for(const int64_t bad:{int64_t(-1),int64_t(248320),int64_t(4294967494)}){bool refused=false;try{(void)original_token32_53(bad);}catch(const std::runtime_error&){refused=true;}if(!refused)return 9;}
 std::puts("MTP53_EXACT_VECTOR_CPU_PASS signed_zero_refused=1 nan_payload_exact=1 controls=11 token64_range_checked=1");}
