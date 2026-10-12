#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace strata::kernels::original_mtp53 {
inline int32_t original_token32_53(int64_t token){if(token<0 || token>=248320)throw std::runtime_error("original native token outside publisher vocabulary");return static_cast<int32_t>(token);}
inline std::vector<int32_t> original_tokens32_53(const std::vector<int64_t>& tokens){std::vector<int32_t> result;result.reserve(tokens.size());for(const auto token:tokens)result.push_back(original_token32_53(token));return result;}
}
