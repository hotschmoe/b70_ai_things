// xfer_probe.cpp -- single-card SYCL transfer-path probe for the Strata slot
// take/giveback (conversation_kv_save/restore) paths. Standalone: no Strata
// headers or libraries. One visible GPU only (ZE_AFFINITY_MASK=N via run.sh).
//
// Phases (stop at the first failure, correctness first):
//   0  identity: device name, driver, PCI address (gated by --expect-pci),
//      CPU affinity, transfer-relevant env.
//   1  INIT: one 4-byte copy, timed alone (process/context lazy init).
//   2  exact-byte ORACLE: odd sizes/offsets with guard bands, every path.
//   3  TIMED arms, round-robin per iteration, fresh salt + poisoned
//      destination per sample, full-buffer verification after every sample.
//
// Labels (do not mix them up when reading results):
//   submit_us  host wall time inside q.memcpy() (enqueue; for pageable host
//              memory the runtime/UMD may do staging work here)
//   wait_us    host wall time inside event.wait()
//   total_us   submit_us + wait_us, host wall (what the Strata caller pays)
//   queued_us  device profiling: command_start - command_submit
//   service_us device profiling: command_end - command_start (device event
//              service; excludes host-side staging/import work)
//   GBps       decimal GB/s = bytes / time
// Route labels are expectations, not measurements: H2H arms are host->host
// through the SYCL queue (CPU memcpy or copy-engine via system memory); they
// are NOT host<->VRAM PCIe transfers.
#include <sycl/sycl.hpp>
#include "pattern.hpp"

#include <sched.h>
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
static double us_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}
static long minflt() { rusage r{}; getrusage(RUSAGE_SELF, &r); return r.ru_minflt; }
static constexpr size_t SEG = 16u << 20;   // strata ConversationBuffer::segment_bytes

// ---------------------------------------------------------------- device side
class FillWords; class CheckWords; class FillBytes; class CheckRegion;

static void dev_fill_words(sycl::queue& q, uint32_t* p, size_t words, uint32_t salt) {
    q.parallel_for<FillWords>(sycl::range<1>(words), [=](sycl::id<1> i) { p[i] = xp::pat_word(i[0], salt); }).wait();
}
struct DevCounters { unsigned long long bad; unsigned long long first; };
static size_t dev_check_words(sycl::queue& q, const uint32_t* p, size_t words, uint32_t salt,
                              DevCounters* ctr, size_t* first) {
    q.memset(ctr, 0, sizeof(unsigned long long)).wait();
    q.fill(&ctr->first, ~0ull, 1).wait();
    q.parallel_for<CheckWords>(sycl::range<1>(words), [=](sycl::id<1> i) {
        if (p[i] != xp::pat_word(i[0], salt)) {
            sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space> b(ctr->bad), f(ctr->first);
            b.fetch_add(1ull);
            f.fetch_min((unsigned long long) i[0]);
        }
    }).wait();
    DevCounters h{};
    q.memcpy(&h, ctr, sizeof h).wait();
    if (first) *first = h.first == ~0ull ? SIZE_MAX : size_t(h.first);
    return size_t(h.bad);
}
static void dev_fill_region(sycl::queue& q, uint8_t* base, size_t total, size_t off, size_t n, uint32_t salt) {
    q.parallel_for<FillBytes>(sycl::range<1>(total), [=](sycl::id<1> i) {
        const size_t k = i[0];
        base[k] = (k >= off && k < off + n) ? xp::pat_byte(k - off, salt) : xp::GUARD;
    }).wait();
}
static size_t dev_check_region(sycl::queue& q, const uint8_t* base, size_t total, size_t off, size_t n,
                               uint32_t salt, DevCounters* ctr, size_t* first) {
    q.memset(ctr, 0, sizeof(unsigned long long)).wait();
    q.fill(&ctr->first, ~0ull, 1).wait();
    q.parallel_for<CheckRegion>(sycl::range<1>(total), [=](sycl::id<1> i) {
        const size_t k = i[0];
        const uint8_t want = (k >= off && k < off + n) ? xp::pat_byte(k - off, salt) : xp::GUARD;
        if (base[k] != want) {
            sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space> b(ctr->bad), f(ctr->first);
            b.fetch_add(1ull);
            f.fetch_min((unsigned long long) k);
        }
    }).wait();
    DevCounters h{};
    q.memcpy(&h, ctr, sizeof h).wait();
    if (first) *first = h.first == ~0ull ? SIZE_MAX : size_t(h.first);
    return size_t(h.bad);
}

// ---------------------------------------------------------------- timing
struct Sample {
    double submit_us = 0, wait_us = 0, total_us = 0, queued_us = 0, service_us = 0, alloc_us = 0, free_us = 0;
    long flt_alloc = 0, flt_copy = 0;
    int chunks = 0, prof_ok = 1, cpu = -1;
};
static void add_prof(Sample& s, sycl::event& e) {
    try {
        const uint64_t sub = e.get_profiling_info<sycl::info::event_profiling::command_submit>();
        const uint64_t st  = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t en  = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        s.queued_us += st >= sub ? (st - sub) / 1e3 : 0.0;
        s.service_us += en >= st ? (en - st) / 1e3 : 0.0;
    } catch (...) { s.prof_ok = 0; }
}
// One timed q.memcpy(...).wait(), the exact call shape of strata transfer().
static void timed_copy(sycl::queue& q, void* dst, const void* src, size_t n, Sample& s) {
    const auto t0 = Clock::now();
    sycl::event e = q.memcpy(dst, src, n);
    const auto t1 = Clock::now();
    e.wait();
    const auto t2 = Clock::now();
    s.submit_us += us_between(t0, t1);
    s.wait_us += us_between(t1, t2);
    s.total_us += us_between(t0, t2);
    s.chunks += 1;
    add_prof(s, e);
}

// ---------------------------------------------------------------- arms
enum Arm {
    H2D_PAGEABLE, H2D_HOSTUSM, D2H_PAGEABLE, D2H_HOSTUSM, D2D_SAME,
    H2H_HOSTUSM_TO_PAGEABLE, H2H_PAGEABLE_TO_HOSTUSM,
    SEG_SAVE_VECZERO, SEG_RESTORE_VECZERO,     // strata today (kv_mode 0): fresh zeroed vectors
    SEG_SAVE_NOINIT, SEG_RESTORE_NOINIT,       // candidate: no zero-fill
    SEG_SAVE_HOSTUSM, SEG_RESTORE_HOSTUSM,     // candidate: persistent pinned pool
    N_ARMS
};
static const char* arm_name[N_ARMS] = {
    "H2D_PAGEABLE", "H2D_HOSTUSM", "D2H_PAGEABLE", "D2H_HOSTUSM", "D2D_SAME",
    "H2H_HOSTUSM_TO_PAGEABLE", "H2H_PAGEABLE_TO_HOSTUSM",
    "SEG_SAVE_VECZERO", "SEG_RESTORE_VECZERO", "SEG_SAVE_NOINIT", "SEG_RESTORE_NOINIT",
    "SEG_SAVE_HOSTUSM", "SEG_RESTORE_HOSTUSM"};
static const char* arm_route[N_ARMS] = {
    "pageable host -> VRAM (PCIe; UMD may stage/import)", "pinned host USM -> VRAM (PCIe DMA)",
    "VRAM -> pageable host (PCIe; UMD may stage/import)", "VRAM -> pinned host USM (PCIe DMA)",
    "VRAM -> VRAM same card (no PCIe)",
    "host USM -> pageable host via queue (H2H, not a VRAM transfer)",
    "pageable host -> host USM via queue (H2H, not a VRAM transfer)",
    "VRAM -> fresh vector<u8>(16MiB,0) segs, wait per seg (strata kv_mode0 save)",
    "those segs -> VRAM, wait per seg (strata kv_mode0 restore)",
    "VRAM -> fresh new u8[16MiB] untouched segs, wait per seg (candidate)",
    "those segs -> VRAM, wait per seg",
    "VRAM -> persistent host USM in 16MiB chunks, wait per chunk (candidate)",
    "host USM chunks -> VRAM, wait per chunk"};

struct Bufs {
    size_t bytes = 0, words = 0;
    uint32_t *dev_a = nullptr, *dev_b = nullptr, *usm_src = nullptr, *usm_dst = nullptr;
    uint32_t *pg_src = nullptr, *pg_dst = nullptr;
};
static uint32_t* pageable_alloc(size_t bytes) {
    void* p = std::aligned_alloc(4096, (bytes + 4095) & ~size_t(4095));
    if (p) std::memset(p, xp::POISON, bytes);   // prefault: warm pageable arms exclude first-touch faults
    return static_cast<uint32_t*>(p);
}

struct Ctx { sycl::queue* q; DevCounters* ctr; };

static bool fail_report(const char* what, const char* arm, size_t bytes, int iter, size_t bad, size_t first) {
    std::printf("FAIL %s arm=%s bytes=%zu iter=%d mismatches=%zu first=%zd\n", what, arm, bytes, iter, bad,
                first == SIZE_MAX ? (ssize_t) -1 : (ssize_t) first);
    std::fflush(stdout);
    return false;
}

// Runs one paired-or-single arm sample. For SEG pairs, `save` and `restore` are both filled.
static bool run_single(Ctx& c, Arm a, Bufs& b, uint32_t salt, int iter, Sample& s) {
    auto& q = *c.q;
    const size_t n = b.bytes, w = b.words;
    size_t bad = 0, first = SIZE_MAX;
    long f0 = 0;
    switch (a) {
    case H2D_PAGEABLE: case H2D_HOSTUSM: {
        uint32_t* src = a == H2D_PAGEABLE ? b.pg_src : b.usm_src;
        xp::host_fill_words(src, w, salt);
        q.memset(b.dev_a, xp::POISON, n).wait();
        f0 = minflt();
        timed_copy(q, b.dev_a, src, n, s);
        s.flt_copy = minflt() - f0;
        bad = dev_check_words(q, b.dev_a, w, salt, c.ctr, &first);
        break; }
    case D2H_PAGEABLE: case D2H_HOSTUSM: {
        uint32_t* dst = a == D2H_PAGEABLE ? b.pg_dst : b.usm_dst;
        dev_fill_words(q, b.dev_a, w, salt);
        std::memset(dst, xp::POISON, n);
        f0 = minflt();
        timed_copy(q, dst, b.dev_a, n, s);
        s.flt_copy = minflt() - f0;
        bad = xp::host_check_words(dst, w, salt, &first);
        break; }
    case D2D_SAME:
        dev_fill_words(q, b.dev_a, w, salt);
        q.memset(b.dev_b, xp::POISON, n).wait();
        timed_copy(q, b.dev_b, b.dev_a, n, s);
        bad = dev_check_words(q, b.dev_b, w, salt, c.ctr, &first);
        break;
    case H2H_HOSTUSM_TO_PAGEABLE: case H2H_PAGEABLE_TO_HOSTUSM: {
        uint32_t* src = a == H2H_HOSTUSM_TO_PAGEABLE ? b.usm_src : b.pg_src;
        uint32_t* dst = a == H2H_HOSTUSM_TO_PAGEABLE ? b.pg_dst : b.usm_dst;
        xp::host_fill_words(src, w, salt);
        std::memset(dst, xp::POISON, n);
        f0 = minflt();
        timed_copy(q, dst, src, n, s);
        s.flt_copy = minflt() - f0;
        bad = xp::host_check_words(dst, w, salt, &first);
        break; }
    default: return false;
    }
    s.cpu = sched_getcpu();
    return bad == 0 || fail_report("freshness", arm_name[a], n, iter, bad, first);
}

// SEG pair: save (VRAM -> host segments) then restore (segments -> VRAM).
static bool run_seg_pair(Ctx& c, Arm save_arm, Bufs& b, uint32_t salt, int iter, Sample& sv, Sample& rs) {
    auto& q = *c.q;
    const size_t n = b.bytes, w = b.words, nseg = (n + SEG - 1) / SEG;
    dev_fill_words(q, b.dev_a, w, salt);
    std::vector<std::vector<uint8_t>> vsegs;
    std::vector<std::unique_ptr<uint8_t[]>> rsegs;
    std::vector<uint8_t*> seg(nseg);
    // allocation phase (timed separately, part of strata's save cost)
    long f0 = minflt();
    auto t0 = Clock::now();
    for (size_t i = 0; i < nseg; ++i) {
        const size_t len = std::min(SEG, n - i * SEG);
        if (save_arm == SEG_SAVE_VECZERO) {
            vsegs.emplace_back(); vsegs.back().reserve(len); vsegs.back().resize(len, 0);
            seg[i] = vsegs.back().data();
        } else if (save_arm == SEG_SAVE_NOINIT) {
            rsegs.emplace_back(new uint8_t[len]);
            seg[i] = rsegs.back().get();
        } else {
            seg[i] = reinterpret_cast<uint8_t*>(b.usm_dst) + i * SEG;
        }
    }
    sv.alloc_us = us_between(t0, Clock::now());
    sv.flt_alloc = minflt() - f0;
    if (save_arm == SEG_SAVE_HOSTUSM) std::memset(b.usm_dst, xp::POISON, n);   // untimed poison, pinned pool is reused
    f0 = minflt();
    for (size_t i = 0; i < nseg; ++i)
        timed_copy(q, seg[i], reinterpret_cast<uint8_t*>(b.dev_a) + i * SEG, std::min(SEG, n - i * SEG), sv);
    sv.flt_copy = minflt() - f0;
    sv.cpu = sched_getcpu();
    // verify save content segment by segment against the word stream
    size_t bad = 0, first = SIZE_MAX;
    for (size_t i = 0; i < nseg && !bad; ++i) {
        const size_t len = std::min(SEG, n - i * SEG);
        const uint32_t* p = reinterpret_cast<const uint32_t*>(seg[i]);
        for (size_t k = 0; k < len / 4; ++k)
            if (p[k] != xp::pat_word(i * (SEG / 4) + k, salt)) { if (!bad) first = i * (SEG / 4) + k; ++bad; }
    }
    if (bad) return fail_report("freshness", arm_name[save_arm], n, iter, bad, first);
    // restore phase
    const Arm rarm = Arm(save_arm + 1);
    q.memset(b.dev_b, xp::POISON, n).wait();
    f0 = minflt();
    for (size_t i = 0; i < nseg; ++i)
        timed_copy(q, reinterpret_cast<uint8_t*>(b.dev_b) + i * SEG, seg[i], std::min(SEG, n - i * SEG), rs);
    rs.flt_copy = minflt() - f0;
    rs.cpu = sched_getcpu();
    bad = dev_check_words(q, b.dev_b, w, salt, c.ctr, &first);
    if (bad) return fail_report("freshness", arm_name[rarm], n, iter, bad, first);
    // release (strata frees the per-layer image right after restore)
    t0 = Clock::now();
    vsegs.clear(); vsegs.shrink_to_fit(); rsegs.clear();
    rs.free_us = us_between(t0, Clock::now());
    return true;
}

// ---------------------------------------------------------------- oracle
static bool oracle(Ctx& c) {
    auto& q = *c.q;
    const size_t cases[][3] = {   // n, src_off, dst_off
        {1, 0, 0}, {3, 1, 3}, {4095, 3, 1}, {4097, 0, 5}, {65537, 7, 0}, {(16u << 20) + 3, 1, 2}};
    const int paths = 7;          // the seven single-copy arms above
    int pass = 0, total_cases = 0;
    uint32_t salt = 0x0A11CE00u;
    for (auto& cs : cases) {
        const size_t n = cs[0], so = cs[1], dof = cs[2], total = n + 4096 + 64;
        uint8_t* dsrc = sycl::malloc_device<uint8_t>(total, q);
        uint8_t* ddst = sycl::malloc_device<uint8_t>(total, q);
        uint8_t* hsrc = sycl::malloc_host<uint8_t>(total, q);
        uint8_t* hdst = sycl::malloc_host<uint8_t>(total, q);
        std::vector<uint8_t> psrc(total), pdst(total);
        if (!dsrc || !ddst || !hsrc || !hdst) { std::puts("FAIL oracle alloc"); return false; }
        for (int p = 0; p < paths; ++p) {
            ++salt; ++total_cases;
            // source with guards
            const bool src_dev = p == D2H_PAGEABLE || p == D2H_HOSTUSM || p == D2D_SAME;
            const bool src_usm = p == H2D_HOSTUSM || p == H2H_HOSTUSM_TO_PAGEABLE;
            uint8_t* src = src_dev ? dsrc : src_usm ? hsrc : psrc.data();
            if (src_dev) dev_fill_region(q, src, total, so, n, salt);
            else { std::memset(src, xp::GUARD, total); xp::host_fill_bytes(src + so, n, salt); }
            const bool dst_dev = p == H2D_PAGEABLE || p == H2D_HOSTUSM || p == D2D_SAME;
            const bool dst_usm = p == D2H_HOSTUSM || p == H2H_PAGEABLE_TO_HOSTUSM;
            uint8_t* dst = dst_dev ? ddst : dst_usm ? hdst : pdst.data();
            // dst: guards everywhere, POISON in the target region
            if (dst_dev) { q.memset(dst, xp::GUARD, total).wait(); q.memset(dst + dof, xp::POISON, n).wait(); }
            else { std::memset(dst, xp::GUARD, total); std::memset(dst + dof, xp::POISON, n); }
            q.memcpy(dst + dof, src + so, n).wait();
            size_t first = SIZE_MAX;
            const size_t bad = dst_dev ? dev_check_region(q, dst, total, dof, n, salt, c.ctr, &first)
                                       : xp::host_check_region(dst, total, dof, n, salt, &first);
            std::printf("ORACLE arm=%s n=%zu src_off=%zu dst_off=%zu mismatches=%zu first=%zd %s\n", arm_name[p], n,
                        so, dof, bad, first == SIZE_MAX ? (ssize_t) -1 : (ssize_t) first, bad ? "FAIL" : "PASS");
            pass += bad == 0;
        }
        sycl::free(dsrc, q); sycl::free(ddst, q); sycl::free(hsrc, q); sycl::free(hdst, q);
    }
    std::printf("ORACLE_SUMMARY pass=%d of=%d\n", pass, total_cases);
    std::fflush(stdout);
    return pass == total_cases;
}

// ---------------------------------------------------------------- stats/json
static double q_of(std::vector<double> v, double p) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    const double r = p * (v.size() - 1);
    const size_t lo = size_t(r), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (r - lo) * (v[hi] - v[lo]);
}
struct ArmRun { Arm arm; size_t bytes; std::vector<Sample> s; };

static void emit_summary(const ArmRun& r, FILE* js, bool& first_js) {
    if (r.s.empty()) return;
    const Sample& c0 = r.s.front();
    std::vector<double> tot, sub, wt, que, svc, alc, flt;
    for (size_t i = 1; i < r.s.size(); ++i) {
        tot.push_back(r.s[i].total_us); sub.push_back(r.s[i].submit_us); wt.push_back(r.s[i].wait_us);
        que.push_back(r.s[i].queued_us); svc.push_back(r.s[i].service_us); alc.push_back(r.s[i].alloc_us);
        flt.push_back(double(r.s[i].flt_alloc + r.s[i].flt_copy));
    }
    const double p50 = q_of(tot, .5), s50 = q_of(svc, .5);
    std::printf("ARM %-24s bytes=%-10zu first_total_us=%.1f first_alloc_us=%.1f first_flt=%ld warm_n=%zu total_us[p50=%.1f p90=%.1f min=%.1f max=%.1f] "
                "submit_p50=%.1f wait_p50=%.1f queued_p50=%.1f service_p50=%.1f alloc_p50=%.1f flt_p50=%.0f "
                "host_GBps_p50=%.3f service_GBps_p50=%.3f chunks=%d prof=%d\n",
                arm_name[r.arm], r.bytes, c0.total_us, c0.alloc_us, c0.flt_alloc + c0.flt_copy, tot.size(), p50, q_of(tot, .9), q_of(tot, 0),
                q_of(tot, 1), q_of(sub, .5), q_of(wt, .5), q_of(que, .5), s50, q_of(alc, .5), q_of(flt, .5),
                p50 > 0 ? r.bytes / (p50 * 1e3) : -1, s50 > 0 ? r.bytes / (s50 * 1e3) : -1, c0.chunks, c0.prof_ok);
    if (!js) return;
    std::fprintf(js, "%s\n  {\"arm\":\"%s\",\"route\":\"%s\",\"bytes\":%zu,\"chunks\":%d,\"samples\":[", first_js ? "" : ",",
                 arm_name[r.arm], arm_route[r.arm], r.bytes, c0.chunks);
    first_js = false;
    for (size_t i = 0; i < r.s.size(); ++i) {
        const Sample& x = r.s[i];
        std::fprintf(js, "%s{\"i\":%zu,\"cold\":%s,\"submit_us\":%.2f,\"wait_us\":%.2f,\"total_us\":%.2f,"
                         "\"queued_us\":%.2f,\"service_us\":%.2f,\"alloc_us\":%.2f,\"free_us\":%.2f,"
                         "\"flt_alloc\":%ld,\"flt_copy\":%ld,\"prof_ok\":%d,\"cpu\":%d}",
                     i ? "," : "", i, i == 0 ? "true" : "false", x.submit_us, x.wait_us, x.total_us, x.queued_us,
                     x.service_us, x.alloc_us, x.free_us, x.flt_alloc, x.flt_copy, x.prof_ok, x.cpu);
    }
    std::fprintf(js, "]}");
}

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
static std::string env_or(const char* k) { const char* v = std::getenv(k); return v ? v : "(unset)"; }

// size spec: "4K:200,64M:20,256M:12"
static std::vector<std::pair<size_t, int>> parse_sizes(const std::string& spec) {
    std::vector<std::pair<size_t, int>> out;
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t end = spec.find(',', pos); if (end == std::string::npos) end = spec.size();
        std::string item = spec.substr(pos, end - pos);
        size_t colon = item.find(':');
        std::string sz = item.substr(0, colon);
        int it = colon == std::string::npos ? 20 : std::atoi(item.c_str() + colon + 1);
        size_t mult = 1; char u = sz.empty() ? 0 : sz.back();
        if (u == 'K' || u == 'k') mult = 1u << 10; else if (u == 'M' || u == 'm') mult = 1u << 20;
        else if (u == 'G' || u == 'g') mult = size_t(1) << 30;
        size_t v = size_t(std::atoll(sz.c_str())) * mult;
        if (v % 4 || v == 0 || it < 2) { std::printf("FAIL bad size spec '%s' (bytes%%4==0, iters>=2)\n", item.c_str()); std::exit(2); }
        out.push_back({v, it});
        pos = end + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    std::string sizes = "4K:200,64M:20,256M:12", json_path, expect_pci = "any", label = "unlabeled";
    unsigned arm_mask = (1u << N_ARMS) - 1;
    bool oracle_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::printf("FAIL missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--sizes") sizes = next();
        else if (a == "--json") json_path = next();
        else if (a == "--expect-pci") expect_pci = next();
        else if (a == "--label") label = next();
        else if (a == "--arms") arm_mask = unsigned(std::strtoul(next().c_str(), nullptr, 0));
        else if (a == "--oracle-only") oracle_only = true;
        else { std::printf("usage: %s [--sizes 4K:200,64M:20,256M:12] [--json out.json] [--expect-pci 0000:0b:00.0|any]\n"
                           "          [--label text] [--arms 0xMASK] [--oracle-only]\n", argv[0]); return 2; }
    }
    const auto plan = parse_sizes(sizes);
    std::printf("PROBE xfer_probe label=%s pid=%d affinity=%s cpu_at_start=%d\n", label.c_str(), int(getpid()),
                affinity().c_str(), sched_getcpu());
    for (const char* k : {"ZE_AFFINITY_MASK", "ONEAPI_DEVICE_SELECTOR", "SYCL_CACHE_PERSISTENT",
                          "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE", "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
                          "UR_L0_USE_IMMEDIATE_COMMANDLISTS", "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD",
                          "SYCL_UR_USE_LEVEL_ZERO_V2", "ZE_FLAT_DEVICE_HIERARCHY", "NEOReadDebugKeys"})
        std::printf("ENV %s=%s\n", k, env_or(k).c_str());

    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.size() != 1) {
        std::printf("FAIL visible_gpus=%zu (need exactly 1: pin with ZE_AFFINITY_MASK=N and gpu-run --card N)\n", gpus.size());
        return 3;
    }
    sycl::device dev = gpus[0];
    std::string pci = "unknown";
    if (dev.has(sycl::aspect::ext_intel_pci_address)) pci = dev.get_info<sycl::ext::intel::info::device::pci_address>();
    std::printf("DEVICE name=\"%s\" driver=\"%s\" pci=%s backend=%d global_mem=%llu max_alloc=%llu\n",
                dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::info::device::driver_version>().c_str(), pci.c_str(), int(dev.get_backend()),
                (unsigned long long) dev.get_info<sycl::info::device::global_mem_size>(),
                (unsigned long long) dev.get_info<sycl::info::device::max_mem_alloc_size>());
    if (expect_pci != "any" && pci != expect_pci) {
        std::printf("FAIL pci_mismatch expected=%s got=%s (lease/pin mismatch: no transfers run)\n", expect_pci.c_str(), pci.c_str());
        return 4;
    }
    std::fflush(stdout);

    // in-order + profiling: the same queue properties Strata's dpct in-order queue uses
    // (conversation_snapshot.cpp defines DPCT_PROFILING_ENABLED).
    sycl::queue q(dev, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    DevCounters* ctr = sycl::malloc_device<DevCounters>(1, q);
    Ctx c{&q, ctr};

    // INIT: first copy in the process, timed alone
    {
        uint32_t* d = sycl::malloc_device<uint32_t>(1, q);
        uint32_t h = 0x13579BDFu, back = 0;
        Sample s; timed_copy(q, d, &h, 4, s);
        q.memcpy(&back, d, 4).wait();
        std::printf("INIT first_copy_total_us=%.1f submit_us=%.1f wait_us=%.1f service_us=%.2f roundtrip_ok=%d\n",
                    s.total_us, s.submit_us, s.wait_us, s.service_us, back == h);
        sycl::free(d, q);
        if (back != h) { std::puts("FAIL init roundtrip"); return 5; }
    }
    if (!oracle(c)) { std::puts("RESULT FAIL oracle"); return 6; }
    if (oracle_only) { std::puts("RESULT PASS oracle_only"); return 0; }

    FILE* js = json_path.empty() ? nullptr : std::fopen(json_path.c_str(), "w");
    bool first_js = true;
    if (js) std::fprintf(js, "{\"probe\":\"xfer_probe\",\"label\":\"%s\",\"pci\":\"%s\",\"affinity\":\"%s\",\"arms\":[",
                         label.c_str(), pci.c_str(), affinity().c_str());
    uint32_t salt = 0x5EED0000u;
    for (auto [bytes, iters] : plan) {
        Bufs b; b.bytes = bytes; b.words = bytes / 4;
        b.dev_a = sycl::malloc_device<uint32_t>(b.words, q);
        b.dev_b = sycl::malloc_device<uint32_t>(b.words, q);
        b.usm_src = sycl::malloc_host<uint32_t>(b.words, q);
        b.usm_dst = sycl::malloc_host<uint32_t>(b.words, q);
        b.pg_src = pageable_alloc(bytes);
        b.pg_dst = pageable_alloc(bytes);
        if (!b.dev_a || !b.dev_b || !b.usm_src || !b.usm_dst || !b.pg_src || !b.pg_dst) {
            std::printf("FAIL alloc bytes=%zu\n", bytes); return 7;
        }
        std::memset(b.usm_src, xp::POISON, bytes); std::memset(b.usm_dst, xp::POISON, bytes);
        std::vector<ArmRun> runs;
        for (int a = 0; a < N_ARMS; ++a) runs.push_back({Arm(a), bytes, {}});
        std::printf("SIZE bytes=%zu iters=%d (sample 0 per arm = cold: first use of these buffers/addresses)\n", bytes, iters);
        std::fflush(stdout);
        for (int it = 0; it < iters; ++it) {
            for (int a = 0; a < N_ARMS; ++a) {
                if (!(arm_mask >> a & 1u)) continue;
                const Arm arm = Arm(a);
                const bool seg_save = arm == SEG_SAVE_VECZERO || arm == SEG_SAVE_NOINIT || arm == SEG_SAVE_HOSTUSM;
                const bool seg_restore = arm == SEG_RESTORE_VECZERO || arm == SEG_RESTORE_NOINIT || arm == SEG_RESTORE_HOSTUSM;
                if (seg_restore) continue;   // produced together with its save arm
                ++salt;
                bool ok;
                if (seg_save) {
                    Sample sv, rs;
                    ok = run_seg_pair(c, arm, b, salt, it, sv, rs);
                    runs[a].s.push_back(sv); runs[a + 1].s.push_back(rs);
                } else {
                    Sample s;
                    ok = run_single(c, arm, b, salt, it, s);
                    runs[a].s.push_back(s);
                }
                if (!ok) { if (js) std::fclose(js); std::puts("RESULT FAIL freshness"); return 8; }
            }
        }
        for (auto& r : runs) emit_summary(r, js, first_js);
        std::fflush(stdout);
        sycl::free(b.dev_a, q); sycl::free(b.dev_b, q); sycl::free(b.usm_src, q); sycl::free(b.usm_dst, q);
        std::free(b.pg_src); std::free(b.pg_dst);
    }
    if (js) { std::fprintf(js, "\n]}\n"); std::fclose(js); }
    sycl::free(ctr, q);
    std::printf("RESULT PASS all samples fresh cpu_at_end=%d\n", sched_getcpu());
    return 0;
}
