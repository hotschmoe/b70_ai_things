// pattern.hpp -- exact-byte freshness pattern shared by host and SYCL device code.
// No std:: calls so the same functions compile inside SYCL kernels.
//
// Contract:
//   * every pattern byte is ODD (low bit forced to 1), therefore it can never
//     equal POISON (0xA4, even), GUARD (0x5A, even) or a fresh zero page.
//   * the pattern depends on (byte index, salt); every timed sample uses a new
//     salt, so stale content from a previous epoch is detected word by word
//     (chance of a single-word coincidence ~2^-28, a stale buffer is certain).
#pragma once
#include <cstddef>
#include <cstdint>

namespace xp {

constexpr uint8_t POISON = 0xA4;   // destination region before each transfer
constexpr uint8_t GUARD  = 0x5A;   // bytes outside the copied region (oracle only)

inline uint32_t pat_word(uint64_t i, uint32_t salt) {
    uint32_t h = uint32_t(i) * 0x9E3779B1u ^ uint32_t(i >> 32) * 0x7FEB352Du ^ salt * 0x85EBCA6Bu;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h | 0x01010101u;          // all four bytes odd
}

// Byte k of the little-endian word stream.
inline uint8_t pat_byte(uint64_t k, uint32_t salt) {
    return uint8_t(pat_word(k >> 2, salt) >> (8u * uint32_t(k & 3u)));
}

// ---- host helpers ----------------------------------------------------------

inline void host_fill_words(uint32_t* p, size_t words, uint32_t salt) {
    for (size_t i = 0; i < words; ++i) p[i] = pat_word(i, salt);
}

// Returns mismatching words; *first gets the first bad word index (or SIZE_MAX).
inline size_t host_check_words(const uint32_t* p, size_t words, uint32_t salt, size_t* first) {
    size_t bad = 0, f = SIZE_MAX;
    for (size_t i = 0; i < words; ++i) {
        const bool b = p[i] != pat_word(i, salt);
        bad += b;
        if (b && f == SIZE_MAX) f = i;
    }
    if (first) *first = f;
    return bad;
}

inline void host_fill_bytes(uint8_t* p, size_t n, uint32_t salt) {
    for (size_t k = 0; k < n; ++k) p[k] = pat_byte(k, salt);
}

// Exact-byte oracle over a whole buffer of `total` bytes: [0,off) and
// [off+n,total) must be GUARD, [off,off+n) must be the pattern (index from 0).
inline size_t host_check_region(const uint8_t* p, size_t total, size_t off, size_t n,
                                uint32_t salt, size_t* first) {
    size_t bad = 0, f = SIZE_MAX;
    for (size_t k = 0; k < total; ++k) {
        const uint8_t want = (k >= off && k < off + n) ? pat_byte(k - off, salt) : GUARD;
        const bool b = p[k] != want;
        bad += b;
        if (b && f == SIZE_MAX) f = k;
    }
    if (first) *first = f;
    return bad;
}

} // namespace xp
