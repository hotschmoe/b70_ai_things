// host_cpu_probe.cpp -- CPU-only companion of xfer_probe (no SYCL, no GPU, no /dev/dri).
//
// 1. Exact-byte oracle self-test: proves pattern.hpp detects poison, stale epoch,
//    fresh zero pages, single-byte corruption at region edges and guard overwrite.
// 2. Host-side cost of the Strata snapshot container: ConversationBuffer grows in
//    16 MiB std::vector segments with reserve()+resize(n, 0), i.e. a zero-fill
//    first touch of every fresh page BEFORE the device copy. This measures that
//    cost and its alternatives with minor-fault counts (page freshness evidence).
//
// Build: g++ -O2 -std=c++20 -o host_cpu_probe host_cpu_probe.cpp
// Run:   taskset -c <cpu> ./host_cpu_probe [total_MiB=1024] [reps=3]
#include "pattern.hpp"

#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static long minflt() { rusage r{}; getrusage(RUSAGE_SELF, &r); return r.ru_minflt; }
static constexpr size_t SEG = 16u << 20;   // strata ConversationBuffer::segment_bytes

static std::string affinity() {
    cpu_set_t s; CPU_ZERO(&s);
    if (sched_getaffinity(0, sizeof s, &s)) return "?";
    std::string out; int lo = -1;
    for (int c = 0; c <= CPU_SETSIZE; ++c) {
        bool in = c < CPU_SETSIZE && CPU_ISSET(c, &s);
        if (in && lo < 0) lo = c;
        if (!in && lo >= 0) {
            if (!out.empty()) out += ",";
            out += std::to_string(lo);
            if (c - 1 > lo) out += "-" + std::to_string(c - 1);
            lo = -1;
        }
    }
    return out;
}
static std::string loadavg() {
    char b[128] = {0};
    if (FILE* f = std::fopen("/proc/loadavg", "r")) { if (!std::fgets(b, sizeof b, f)) b[0] = 0; std::fclose(f); }
    std::string s(b); while (!s.empty() && s.back() == '\n') s.pop_back(); return s;
}

static int fails = 0;
static void expect(bool ok, const char* what) {
    std::printf("ORACLE %-58s %s\n", what, ok ? "PASS" : "FAIL");
    fails += !ok;
}

static void oracle_selftest() {
    // pattern bytes are odd -> never POISON/GUARD/zero
    bool odd = true;
    for (uint64_t k = 0; k < (1u << 20); ++k) odd &= (xp::pat_byte(k, 7u) & 1u) == 1u;
    expect(odd, "pattern bytes odd for 1Mi bytes");
    // word view == byte view (little endian)
    std::vector<uint32_t> w(4096);
    xp::host_fill_words(w.data(), w.size(), 99u);
    std::vector<uint8_t> b(w.size() * 4);
    xp::host_fill_bytes(b.data(), b.size(), 99u);
    expect(std::memcmp(w.data(), b.data(), b.size()) == 0, "word and byte pattern views agree");
    size_t first = 0;
    expect(xp::host_check_words(w.data(), w.size(), 99u, &first) == 0, "fresh words verify");
    expect(xp::host_check_words(w.data(), w.size(), 100u, &first) == w.size(), "stale epoch: every word flagged");
    std::vector<uint32_t> z(4096, 0u);
    expect(xp::host_check_words(z.data(), z.size(), 99u, &first) == z.size(), "zero page: every word flagged");
    std::memset(z.data(), xp::POISON, z.size() * 4);
    expect(xp::host_check_words(z.data(), z.size(), 99u, &first) == z.size(), "poison: every word flagged");

    // exact-byte region oracle with odd sizes/offsets and guards
    const size_t cases[][2] = {{1, 0}, {3, 3}, {4095, 1}, {4097, 5}, {65537, 7}, {(16u << 20) + 3, 2}};
    for (auto& c : cases) {
        const size_t n = c[0], off = c[1], total = n + off + 4096 + 61;
        std::vector<uint8_t> buf(total, xp::GUARD);
        xp::host_fill_bytes(buf.data() + off, n, 1234u);
        char what[96];
        std::snprintf(what, sizeof what, "region n=%zu off=%zu clean", n, off);
        expect(xp::host_check_region(buf.data(), total, off, n, 1234u, &first) == 0, what);
        // corrupt last region byte (tail truncation)
        buf[off + n - 1] = xp::POISON;
        std::snprintf(what, sizeof what, "region n=%zu off=%zu truncated tail caught", n, off);
        expect(xp::host_check_region(buf.data(), total, off, n, 1234u, &first) == 1 && first == off + n - 1, what);
        buf[off + n - 1] = xp::pat_byte(n - 1, 1234u);
        // overwrite first guard byte after region (overrun)
        buf[off + n] = 0x11;
        std::snprintf(what, sizeof what, "region n=%zu off=%zu overrun caught", n, off);
        expect(xp::host_check_region(buf.data(), total, off, n, 1234u, &first) == 1 && first == off + n, what);
        buf[off + n] = xp::GUARD;
        if (off) {
            buf[off - 1] = 0x11;
            std::snprintf(what, sizeof what, "region n=%zu off=%zu underrun caught", n, off);
            expect(xp::host_check_region(buf.data(), total, off, n, 1234u, &first) == 1 && first == off - 1, what);
            buf[off - 1] = xp::GUARD;
        }
    }
}

struct Row { const char* name; double ms; long flt; size_t bytes; };
static void report(const Row& r, int rep) {
    const double gbps = r.bytes / (r.ms * 1e6);
    const double per_316 = r.ms * (3.16e9 / double(r.bytes));
    std::printf("HOST %-26s rep=%d bytes=%zu ms=%.2f GBps=%.2f minflt=%ld flt_per_seg=%.0f extrap_ms_3.16GB=%.0f cpu=%d\n",
                r.name, rep, r.bytes, r.ms, gbps, r.flt, double(r.flt) / double(r.bytes / SEG), per_316, sched_getcpu());
}

int main(int argc, char** argv) {
    const size_t total = size_t(argc > 1 ? std::atol(argv[1]) : 1024) << 20;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    const size_t nseg = total / SEG;
    std::printf("CONFIG total_bytes=%zu segments=%zu seg_bytes=%zu reps=%d affinity=%s pid=%d\n",
                total, nseg, SEG, reps, affinity().c_str(), int(getpid()));
    std::printf("LOAD before=\"%s\"\n", loadavg().c_str());
    oracle_selftest();
    if (fails) { std::printf("RESULT oracle FAIL count=%d\n", fails); return 1; }

    // warm source used for copies
    std::vector<uint8_t> src(SEG);
    xp::host_fill_bytes(src.data(), SEG, 5u);
    volatile uint64_t sink = 0;

    for (int rep = 0; rep < reps; ++rep) {
        // A. Strata today: vector reserve + resize(n, 0) per 16 MiB segment, then
        //    a memcpy into it (stand-in for the device->host write). Kept alive
        //    until the whole image exists, then freed (like a per-layer img).
        {
            std::vector<std::vector<uint8_t>> segs; segs.reserve(nseg);
            long f0 = minflt(); auto t = Clock::now();
            for (size_t i = 0; i < nseg; ++i) { segs.emplace_back(); segs.back().reserve(SEG); segs.back().resize(SEG, 0); }
            report({"A_vec_reserve_resize0", ms_since(t), minflt() - f0, total}, rep);
            f0 = minflt(); t = Clock::now();
            for (auto& s : segs) std::memcpy(s.data(), src.data(), SEG);
            report({"A_copy_into_zeroed", ms_since(t), minflt() - f0, total}, rep);
            for (auto& s : segs) sink += s[SEG - 1];
            f0 = minflt(); t = Clock::now();
            segs.clear(); segs.shrink_to_fit();
            report({"A_free", ms_since(t), minflt() - f0, total}, rep);
        }
        // B. Candidate: default-init allocation (no zero fill); first touch moves into the copy.
        {
            std::vector<std::unique_ptr<uint8_t[]>> segs; segs.reserve(nseg);
            long f0 = minflt(); auto t = Clock::now();
            for (size_t i = 0; i < nseg; ++i) segs.emplace_back(new uint8_t[SEG]);
            report({"B_new_noinit", ms_since(t), minflt() - f0, total}, rep);
            f0 = minflt(); t = Clock::now();
            for (auto& s : segs) std::memcpy(s.get(), src.data(), SEG);
            report({"B_copy_into_untouched", ms_since(t), minflt() - f0, total}, rep);
            for (auto& s : segs) sink += s[SEG - 1];
        }
        // C. Fresh mmap per segment + memset (guaranteed-fresh 4 KiB pages).
        // D. Same with MADV_HUGEPAGE (THP mode here is 'madvise').
        for (int huge = 0; huge < 2; ++huge) {
            std::vector<void*> maps(nseg, nullptr);
            long f0 = minflt(); auto t = Clock::now();
            for (size_t i = 0; i < nseg; ++i) {
                maps[i] = mmap(nullptr, SEG, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (maps[i] == MAP_FAILED) { std::perror("mmap"); return 2; }
                if (huge) madvise(maps[i], SEG, MADV_HUGEPAGE);
                std::memset(maps[i], 0, SEG);
            }
            report({huge ? "D_mmap_thp_memset0" : "C_mmap_4k_memset0", ms_since(t), minflt() - f0, total}, rep);
            for (void* m : maps) { sink += static_cast<uint8_t*>(m)[SEG - 1]; munmap(m, SEG); }
        }
        // E. Warm memcpy into a reused buffer (CPU copy bandwidth floor for a bounce/staging copy).
        {
            std::vector<uint8_t> dst(SEG, 1);
            long f0 = minflt(); auto t = Clock::now();
            for (size_t i = 0; i < nseg; ++i) std::memcpy(dst.data(), src.data(), SEG);
            report({"E_memcpy_warm_reuse", ms_since(t), minflt() - f0, total}, rep);
            sink += dst[SEG - 1];
        }
    }
    std::printf("LOAD after=\"%s\"\n", loadavg().c_str());
    std::printf("RESULT oracle PASS host_rows_done sink=%llu\n", (unsigned long long)(sink & 1));
    return 0;
}
