#pragma once
#include <cstring>
#include <vector>
namespace strata::kernels::original_mtp53 {
template<class T> inline bool exact_vector_bytes53(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
}
