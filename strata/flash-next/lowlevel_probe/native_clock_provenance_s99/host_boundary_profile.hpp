#pragma once
// Opt-in CPU timestamps only. No GPU events, waits, submissions or per-window output.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace strata::core::host_boundary {
inline bool enabled() {
    static const bool on = [] { const char* p = std::getenv("STRATA_HOST_BOUNDARY"); return p && p[0] == '1'; }();
    return on;
}
struct Record { int64_t ns; uint64_t window; const char* tag; int a, b; };
struct State { std::vector<Record> records; uint64_t next = 0, window = 0; size_t dropped = 0; bool failed = false; };
inline State& state() { thread_local State s; return s; }
inline void mark(const char* tag, int a = 0, int b = 0) noexcept {
    if (!enabled()) return;
    State& s = state();
    if (s.failed || s.records.size() >= 8192) { ++s.dropped; return; }
    try {
        if (s.records.capacity() == 0) s.records.reserve(8192);
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        s.records.push_back({ns, s.window, tag, a, b});
    } catch (...) { s.failed = true; ++s.dropped; }
}
inline void begin() noexcept {
    if (!enabled()) return;
    State& s = state(); s.window = ++s.next; mark("window_enter");
}
inline void flush() noexcept {
    if (!enabled()) return;
    State& s = state();
    std::fprintf(stderr, "strata host boundary: records=%zu dropped=%zu clock=steady_ns\n", s.records.size(), s.dropped);
    for (const auto& r : s.records)
        std::fprintf(stderr, "strata host boundary: w=%llu t=%lld tag=%s a=%d b=%d\n",
                     (unsigned long long) r.window, (long long) r.ns, r.tag, r.a, r.b);
    s.records.clear(); s.window = 0; s.dropped = 0;
}
} // namespace strata::core::host_boundary
