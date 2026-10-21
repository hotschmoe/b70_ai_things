#pragma once
#include <bit>
#include <cstdint>
namespace b70_q1_packcompact_r2 {
inline uint64_t packed_key(float score, uint32_t index) {
  const uint32_t bits = std::bit_cast<uint32_t>(score);
  const uint32_t mask = (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  const uint32_t order = bits ^ mask;
  return (static_cast<uint64_t>(order) << 32) |
         static_cast<uint64_t>(0xffffffffu ^ index);
}
inline int64_t mapped_key(float score, uint32_t index) {
  return std::bit_cast<int64_t>(packed_key(score, index) ^ (1ull << 63));
}
constexpr int64_t kInvalid = (-9223372036854775807ll - 1);
}
