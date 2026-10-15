// readiness_probe.cpp -- single-card SYCL data-readiness / queue-latency probe (S24, claude-opus-5-5).
// Standalone (no Strata headers/libs). Extends the S18 xfer_probe idea: it does NOT repeat the S18 timed
// copy matrix; it times the CPU side BEFORE a transfer can begin (allocation, first touch, pack), the
// enqueue call, the device event (submit/start/end) and the host completion return, plus device-domain
// producer->copy->consumer dependency gaps with finite kernels.
//
// CLOCK DOMAINS (never mixed):
//   host_*  : std::chrono::steady_clock, microseconds relative to the sample's t0 (host domain)
//   dev_*   : SYCL event profiling (command_submit/start/end, ns, device/runtime domain), reported only
//             as differences between timestamps of the SAME domain (queued = start - submit, service =
//             end - start, gap = next.start - prev.end). Host<->device offsets are reported as NA:
//             no calibrated host/device timer pairing is used.
//
// Every timed sample: fresh salt (odd-byte pattern from pattern.hpp), destination poisoned (POISON),
// guard bands of GUARD bytes on both sides of the destination, full exact-byte + guard verification after
// the sample (untimed). Any mismatch aborts with FAIL (exit 5).
//
// One visible GPU only (bin/gpu-run --card N + ZE_AFFINITY_MASK=N); --expect-pci gates the BDF.
#include <sycl/sycl.hpp>
#include "pattern.hpp"

#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;
static double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
static long minflt() { rusage r{}; getrusage(RUSAGE_SELF, &r); return r.ru_minflt; }
static constexpr size_t GB = 64;   // guard bytes on each side of every destination

// ------------------------------------------------------------------ device helpers (finite kernels only)
class KPoison; class KCheck; class KProduce; class KConsume;
struct Ctr { unsigned long long bad, first, sum; };

// Finite busy work per 32-bit work item: a nonlinear recurrence (no closed form) whose result feeds a checksum
// that is read back and verified against a host recomputation, so the loop cannot be eliminated.
inline uint32_t busy_word(uint64_t w, int rounds) {
    uint32_t acc = uint32_t(w) * 0x9E3779B1u + 1u;
    for (int r = 0; r < rounds; ++r) acc = (acc ^ (acc >> 13)) * 0x85EBCA6Bu + uint32_t(r);
    return acc;
}
// Producer: `rounds` iterations of busy_word per work item (summed into *csum by a SYCL reduction), then the fresh
// salted xp pattern for [0,n).  The payload bytes never depend on the busy work.
static sycl::event dev_produce(sycl::queue& q, uint8_t* p, size_t n, uint32_t salt, int rounds, uint64_t* csum,
                               const std::vector<sycl::event>& deps = {}) {
    const size_t items = n / 4 + (n % 4 ? 1 : 0);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        auto red = sycl::reduction(csum, sycl::plus<uint64_t>(), sycl::property::reduction::initialize_to_identity{});
        h.parallel_for<KProduce>(sycl::range<1>(items), red, [=](sycl::id<1> id, auto& sum) {
            const size_t w = id[0];
            sum += uint64_t(busy_word(w, rounds));
            for (int b = 0; b < 4; ++b) {
                const size_t k = w * 4 + b;
                if (k < n) p[k] = xp::pat_byte(k, salt);
            }
        });
    });
}
static std::map<std::pair<size_t, int>, uint64_t> g_expect;   // (items, rounds) -> checksum, precomputed untimed
static uint64_t host_checksum(size_t n, int rounds) {
    const size_t items = n / 4 + (n % 4 ? 1 : 0);
    auto it = g_expect.find({items, rounds});
    if (it != g_expect.end()) return it->second;
    uint64_t s = 0;
    for (size_t w = 0; w < items; ++w) s += busy_word(w, rounds);
    g_expect[{items, rounds}] = s;
    return s;
}
static void verify_checksum(sycl::queue& q, const uint64_t* csum, size_t n, int rounds, const std::string& arm) {
    uint64_t got = 0;
    q.memcpy(&got, csum, sizeof got).wait();
    const uint64_t want = host_checksum(n, rounds);
    if (got != want) {
        std::printf("FAIL producer_checksum arm=%s got=%llu want=%llu\n", arm.c_str(), (unsigned long long) got,
                    (unsigned long long) want);
        std::exit(5);
    }
}
// Consumer: checks the pattern on device and counts mismatches (also the device-side readiness reader).
static sycl::event dev_consume(sycl::queue& q, const uint8_t* p, size_t n, uint32_t salt, Ctr* c,
                               const std::vector<sycl::event>& deps = {}) {
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for<KConsume>(sycl::range<1>(n), [=](sycl::id<1> id) {
            const size_t k = id[0];
            if (p[k] != xp::pat_byte(k, salt)) {
                sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> b(c->bad), f(c->first);
                b.fetch_add(1ull);
                f.fetch_min((unsigned long long) k);
            }
        });
    });
}
// Untimed guard/region check of a device buffer laid out [GB guard][n payload][GB guard].
static size_t dev_check_guarded(sycl::queue& q, const uint8_t* base, size_t n, uint32_t salt, Ctr* c) {
    q.memset(c, 0, sizeof(Ctr)).wait();
    q.fill(&c->first, ~0ull, 1).wait();
    const size_t total = n + 2 * GB;
    q.parallel_for<KCheck>(sycl::range<1>(total), [=](sycl::id<1> id) {
        const size_t k = id[0];
        const uint8_t want = (k >= GB && k < GB + n) ? xp::pat_byte(k - GB, salt) : xp::GUARD;
        if (base[k] != want) {
            sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space> b(c->bad), f(c->first);
            b.fetch_add(1ull);
            f.fetch_min((unsigned long long) k);
        }
    }).wait();
    Ctr h{};
    q.memcpy(&h, c, sizeof h).wait();
    return size_t(h.bad);
}
static void dev_poison_guarded(sycl::queue& q, uint8_t* base, size_t n) {
    q.memset(base, xp::GUARD, GB);
    q.memset(base + GB, xp::POISON, n);
    q.memset(base + GB + n, xp::GUARD, GB).wait();
}
static size_t host_check_guarded(const uint8_t* base, size_t n, uint32_t salt) {
    size_t f = 0;
    return xp::host_check_region(base, n + 2 * GB, GB, n, salt, &f);
}
static void host_poison_guarded(uint8_t* base, size_t n) {
    std::memset(base, xp::GUARD, GB);
    std::memset(base + GB, xp::POISON, n);
    std::memset(base + GB + n, xp::GUARD, GB);
}

// ------------------------------------------------------------------ device event times
struct Ev { double queued = -1, service = -1; uint64_t sub = 0, st = 0, en = 0; bool ok = false; };
static Ev ev_info(sycl::event& e) {
    Ev v;
    try {
        v.sub = e.get_profiling_info<sycl::info::event_profiling::command_submit>();
        v.st = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        v.en = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        v.queued = v.st >= v.sub ? (v.st - v.sub) / 1e3 : -1;
        v.service = v.en >= v.st ? (v.en - v.st) / 1e3 : -1;
        v.ok = true;
    } catch (...) {}
    return v;
}
static double dgap(const Ev& a, const Ev& b) {   // b.start - a.end, same device domain; -1 = NA
    if (!a.ok || !b.ok) return -1;
    return (double) ((int64_t) b.st - (int64_t) a.en) / 1e3;
}

// ------------------------------------------------------------------ arms
enum HostMem { PAGEABLE_FRESH, PAGEABLE_REUSED, PINNED_FRESH, PINNED_PREALLOC, N_HOSTMEM };
static const char* hm_name[N_HOSTMEM] = {"pageable_fresh", "pageable_reused", "pinned_fresh", "pinned_prealloc"};

struct Payload { std::string name; size_t bytes; int reps; };

static FILE* js = nullptr;
static bool js_first = true;
static void js_row(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void js_row(const char* fmt, ...) {
    if (!js) return;
    std::fputs(js_first ? "\n" : ",\n", js);
    js_first = false;
    va_list ap; va_start(ap, fmt); std::vfprintf(js, fmt, ap); va_end(ap);
}
[[noreturn]] static void fail(const char* what, const std::string& arm, size_t bad) {
    std::printf("FAIL %s arm=%s bad=%zu\n", what, arm.c_str(), bad);
    if (js) { std::fputs("\n]}\n", js); std::fclose(js); }
    std::exit(5);
}

// H2D readiness: host buffer [alloc] -> [first touch = poison] -> [pack = pattern fill] -> enqueue -> wait.
// Device destination is preallocated + poisoned + guarded (untimed); checked after (untimed).
static void arm_h2d(sycl::queue& q, const Payload& pl, HostMem hm, uint8_t* hpre_pinned, uint8_t* hpre_page,
                    uint8_t* ddst, Ctr* ctr, uint32_t& salt, int sample) {
    const size_t n = pl.bytes;
    dev_poison_guarded(q, ddst, n);
    const uint32_t s = ++salt;
    const long f0 = minflt();
    const auto t0 = Clock::now();
    uint8_t* h = nullptr;
    if (hm == PAGEABLE_FRESH) h = (uint8_t*) std::malloc(n);
    else if (hm == PINNED_FRESH) h = sycl::malloc_host<uint8_t>(n, q);
    else h = hm == PINNED_PREALLOC ? hpre_pinned : hpre_page;
    const auto t_alloc = Clock::now();
    std::memset(h, xp::POISON, n);   // first touch (fresh) or rewrite (reused): the zero/poison pass a producer pays
    const auto t_touch = Clock::now();
    xp::host_fill_bytes(h, n, s);    // "pack": the producer writes the payload; ready afterwards
    const auto t_ready = Clock::now();
    sycl::event e = q.memcpy(ddst + GB, h, n);
    const auto t_enq = Clock::now();
    e.wait();
    const auto t_done = Clock::now();
    const long f1 = minflt();
    const Ev v = ev_info(e);
    const size_t bad = dev_check_guarded(q, ddst, n, s, ctr);
    if (hm == PAGEABLE_FRESH) std::free(h);
    else if (hm == PINNED_FRESH) sycl::free(h, q);
    const std::string arm = std::string("H2D_") + hm_name[hm] + "_" + pl.name;
    if (bad) fail("h2d_verify", arm, bad);
    std::printf("S %s i=%d alloc=%.1f touch=%.1f pack=%.1f enq=%.1f wait=%.1f ready_to_done=%.1f "
                "dev_queued=%.1f dev_service=%.1f minflt=%ld host_dev_offset=NA\n",
                arm.c_str(), sample, us(t0, t_alloc), us(t_alloc, t_touch), us(t_touch, t_ready),
                us(t_ready, t_enq), us(t_enq, t_done), us(t_ready, t_done), v.queued, v.service, f1 - f0);
    js_row("{\"arm\":\"%s\",\"i\":%d,\"bytes\":%zu,\"host_alloc_us\":%.2f,\"host_first_touch_us\":%.2f,"
           "\"host_pack_us\":%.2f,\"host_enqueue_us\":%.2f,\"host_wait_us\":%.2f,\"host_ready_to_done_us\":%.2f,"
           "\"dev_queued_us\":%.2f,\"dev_service_us\":%.2f,\"minflt\":%ld,\"host_dev_offset\":null}",
           arm.c_str(), sample, n, us(t0, t_alloc), us(t_alloc, t_touch), us(t_touch, t_ready), us(t_ready, t_enq),
           us(t_enq, t_done), us(t_ready, t_done), v.queued, v.service, f1 - f0);
}

// D2H return: finite device producer -> dependent copy into a host buffer (no host wait in between),
// host completion return, then host-side consumer read (verification is the consumer, timed separately).
static void arm_d2h(sycl::queue& q, const Payload& pl, HostMem hm, uint8_t* hpre_pinned, uint8_t* hpre_page,
                    uint8_t* dsrc, uint64_t* csum, int rounds, uint32_t& salt, int sample) {
    const size_t n = pl.bytes;
    const uint32_t s = ++salt;
    uint8_t* h = nullptr;
    const auto t0 = Clock::now();
    if (hm == PAGEABLE_FRESH) h = (uint8_t*) std::malloc(n + 2 * GB);
    else if (hm == PINNED_FRESH) h = sycl::malloc_host<uint8_t>(n + 2 * GB, q);
    else h = hm == PINNED_PREALLOC ? hpre_pinned : hpre_page;
    const auto t_alloc = Clock::now();
    host_poison_guarded(h, n);       // first touch / poison of the landing buffer
    const auto t_touch = Clock::now();
    sycl::event ep = dev_produce(q, dsrc, n, s, rounds, csum);
    sycl::event ec = q.memcpy(h + GB, dsrc, n, ep);
    const auto t_enq = Clock::now();
    ec.wait();
    const auto t_done = Clock::now();
    const size_t bad = host_check_guarded(h, n, s);   // host consumer read of every byte
    const auto t_consumed = Clock::now();
    const Ev vp = ev_info(ep), vc = ev_info(ec);
    if (hm == PAGEABLE_FRESH) std::free(h);
    else if (hm == PINNED_FRESH) sycl::free(h, q);
    const std::string arm = std::string("D2H_") + hm_name[hm] + "_" + pl.name;
    if (bad) fail("d2h_verify", arm, bad);
    verify_checksum(q, csum, n, rounds, arm);
    std::printf("S %s i=%d alloc=%.1f touch=%.1f enq2=%.1f wait=%.1f consume=%.1f prod_service=%.1f "
                "prod_end_to_copy_start=%.1f copy_queued=%.1f copy_service=%.1f host_dev_offset=NA\n",
                arm.c_str(), sample, us(t0, t_alloc), us(t_alloc, t_touch), us(t_touch, t_enq), us(t_enq, t_done),
                us(t_done, t_consumed), vp.service, dgap(vp, vc), vc.queued, vc.service);
    js_row("{\"arm\":\"%s\",\"i\":%d,\"bytes\":%zu,\"host_alloc_us\":%.2f,\"host_first_touch_us\":%.2f,"
           "\"host_enqueue2_us\":%.2f,\"host_wait_us\":%.2f,\"host_consume_us\":%.2f,\"dev_prod_service_us\":%.2f,"
           "\"dev_prod_end_to_copy_start_us\":%.2f,\"dev_copy_queued_us\":%.2f,\"dev_copy_service_us\":%.2f,"
           "\"host_dev_offset\":null}",
           arm.c_str(), sample, n, us(t0, t_alloc), us(t_alloc, t_touch), us(t_touch, t_enq), us(t_enq, t_done),
           us(t_done, t_consumed), vp.service, dgap(vp, vc), vc.queued, vc.service);
}

// Device-only dependency chain: producer(finite) -> D2D copy -> consumer(check), submitted back to back
// with explicit depends_on (works on in-order and out-of-order queues). All gaps in the device domain.
static void arm_chain(sycl::queue& q, const char* qname, const Payload& pl, uint8_t* da, uint8_t* db, Ctr* ctr,
                      uint64_t* csum, int rounds, uint32_t& salt, int sample) {
    const size_t n = pl.bytes;
    dev_poison_guarded(q, db, n);
    q.memset(ctr, 0, sizeof(Ctr)).wait();
    q.fill(&ctr->first, ~0ull, 1).wait();
    const uint32_t s = ++salt;
    const auto t0 = Clock::now();
    sycl::event ep = dev_produce(q, da, n, s, rounds, csum);
    sycl::event ec = q.submit([&](sycl::handler& h) { h.depends_on(ep); h.memcpy(db + GB, da, n); });
    sycl::event en = dev_consume(q, db + GB, n, s, ctr, {ec});
    const auto t_enq = Clock::now();
    en.wait();
    const auto t_done = Clock::now();
    Ctr hc{};
    q.memcpy(&hc, ctr, sizeof hc).wait();
    const size_t bad = dev_check_guarded(q, db, n, s, ctr) + size_t(hc.bad);
    const Ev vp = ev_info(ep), vc = ev_info(ec), vn = ev_info(en);
    const std::string arm = std::string("CHAIN_") + qname + "_" + pl.name;
    if (bad) fail("chain_verify", arm, bad);
    verify_checksum(q, csum, n, rounds, arm);
    std::printf("S %s i=%d enq3=%.1f wait=%.1f prod=%.1f gap_pc=%.1f copy=%.1f gap_cc=%.1f consume=%.1f "
                "first_submit_to_last_end=%.1f host_dev_offset=NA\n",
                arm.c_str(), sample, us(t0, t_enq), us(t_enq, t_done), vp.service, dgap(vp, vc), vc.service,
                dgap(vc, vn), vn.service, (vp.ok && vn.ok) ? ((double) ((int64_t) vn.en - (int64_t) vp.sub)) / 1e3 : -1.0);
    js_row("{\"arm\":\"%s\",\"i\":%d,\"bytes\":%zu,\"host_enqueue3_us\":%.2f,\"host_wait_us\":%.2f,"
           "\"dev_prod_us\":%.2f,\"dev_gap_prod_copy_us\":%.2f,\"dev_copy_us\":%.2f,\"dev_gap_copy_consume_us\":%.2f,"
           "\"dev_consume_us\":%.2f,\"host_dev_offset\":null}",
           arm.c_str(), sample, n, us(t0, t_enq), us(t_enq, t_done), vp.service, dgap(vp, vc), vc.service,
           dgap(vc, vn), vn.service);
}

// ------------------------------------------------------------------ main
static std::string affinity() {
    cpu_set_t s; CPU_ZERO(&s);
    if (sched_getaffinity(0, sizeof s, &s)) return "?";
    std::string r;
    for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &s)) { if (!r.empty()) r += ","; r += std::to_string(c); }
    return r;
}
static size_t parse_bytes(const std::string& v) {
    char* e = nullptr;
    size_t x = std::strtoull(v.c_str(), &e, 10);
    if (*e == 'K') x <<= 10; else if (*e == 'M') x <<= 20; else if (*e == 'G') x <<= 30;
    return x;
}

int main(int argc, char** argv) {
    std::string json_path, expect_pci = "any", label = "unlabeled";
    int rows = 6, n_embd = 2560, hc = 4, k_used = 0, reps_small = 12, reps_big = 4, rounds = 64;
    std::vector<size_t> big = {64u << 20, 256u << 20};
    unsigned mask = 0x7;   // 1 = H2D, 2 = D2H, 4 = CHAIN
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::printf("FAIL missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--json") json_path = next();
        else if (a == "--expect-pci") expect_pci = next();
        else if (a == "--label") label = next();
        else if (a == "--rows") rows = std::atoi(next().c_str());
        else if (a == "--n-embd") n_embd = std::atoi(next().c_str());
        else if (a == "--hc") hc = std::atoi(next().c_str());
        else if (a == "--k-used") k_used = std::atoi(next().c_str());
        else if (a == "--reps") reps_small = std::atoi(next().c_str());
        else if (a == "--reps-big") reps_big = std::atoi(next().c_str());
        else if (a == "--rounds") rounds = std::atoi(next().c_str());
        else if (a == "--big") { big.clear(); std::string v = next(); size_t p = 0;
            while (p < v.size()) { size_t c = v.find(',', p); if (c == std::string::npos) c = v.size(); if (c > p) big.push_back(parse_bytes(v.substr(p, c - p))); p = c + 1; } }
        else if (a == "--mask") mask = unsigned(std::strtoul(next().c_str(), nullptr, 0));
        else { std::printf("usage: %s --expect-pci BDF [--label s] [--json f] [--rows 6] [--n-embd 2560] [--hc 4]\n"
                           "  [--k-used K (experts per token; 0 skips EP payloads)] [--reps 12] [--reps-big 4]\n"
                           "  [--rounds 64] [--big 64M,256M] [--mask 0x7]\n", argv[0]); return 2; }
    }
    if (rows < 1 || rows > 8 || n_embd % 32 || reps_small < 2 || reps_small > 200 || reps_big < 2 || reps_big > 20 ||
        rounds < 0 || rounds > 100000 || k_used < 0 || k_used > 64) { std::printf("FAIL bad arguments (bounded probe)\n"); return 2; }

    // Serving payloads, sizes from runtime source (sycl/src/core/verify.cpp, src-ep 296b59d2 + EP):
    //   q8_1 activation row = (N/32)*36 B (verify.cpp ep_push: nat_xq_ + tb*(N/32)*36)
    //   EP push = n*K int32 ids + rows*(N/32)*36 q8_1 (ep_push(ids_, n*K, xq, xq_bytes, ...))
    //   EP peer return = rows*K*N float32 (ep_send_rows: ep_rows1_ + tb*K*N; parts region MT*K*N*4)
    //   residual rows  = rows*hc*N float32 (final_R stride g.hc*g.n_embd, the batch MTP residual copy)
    //   sampled tokens = rows*int32
    std::vector<Payload> pls;
    const size_t q81 = size_t(rows) * (n_embd / 32) * 36;
    pls.push_back({"tok_" + std::to_string(rows) + "x4B", size_t(rows) * 4, reps_small});
    pls.push_back({"q8_1_" + std::to_string(rows) + "rows", q81, reps_small});
    pls.push_back({"resid_" + std::to_string(rows) + "rows_f32", size_t(rows) * hc * n_embd * 4, reps_small});
    if (k_used > 0) {
        pls.push_back({"ep_push_" + std::to_string(rows) + "rows_k" + std::to_string(k_used), size_t(rows) * k_used * 4 + q81, reps_small});
        pls.push_back({"ep_return_" + std::to_string(rows) + "rows_k" + std::to_string(k_used), size_t(rows) * k_used * n_embd * 4, reps_small});
    }
    for (size_t b : big) pls.push_back({"ctl_" + std::to_string(b >> 20) + "MiB", b, reps_big});
    size_t maxn = 0;
    for (auto& p : pls) maxn = std::max(maxn, p.bytes);

    std::printf("PROBE readiness_probe label=%s pid=%d affinity=%s cpu_at_start=%d rows=%d n_embd=%d hc=%d k_used=%d rounds=%d\n",
                label.c_str(), int(getpid()), affinity().c_str(), sched_getcpu(), rows, n_embd, hc, k_used, rounds);
    for (const char* k : {"ZE_AFFINITY_MASK", "ONEAPI_DEVICE_SELECTOR", "SYCL_UR_USE_LEVEL_ZERO_V2",
                          "UR_L0_USE_IMMEDIATE_COMMANDLISTS", "ZE_FLAT_DEVICE_HIERARCHY"}) {
        const char* v = std::getenv(k);
        std::printf("ENV %s=%s\n", k, v ? v : "(unset)");
    }
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.size() != 1) { std::printf("FAIL visible_gpus=%zu (need exactly 1)\n", gpus.size()); return 3; }
    sycl::device dev = gpus[0];
    std::string pci = "unknown";
    if (dev.has(sycl::aspect::ext_intel_pci_address)) pci = dev.get_info<sycl::ext::intel::info::device::pci_address>();
    std::printf("DEVICE name=\"%s\" driver=\"%s\" pci=%s\n", dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_info<sycl::info::device::driver_version>().c_str(), pci.c_str());
    if (expect_pci != "any" && pci != expect_pci) { std::printf("FAIL pci_mismatch expected=%s got=%s\n", expect_pci.c_str(), pci.c_str()); return 4; }
    std::printf("CLOCKS host=steady_clock(us) device=sycl_event_profiling(ns->us); host_dev_offset=NA (not calibrated); "
                "dev_queued=start-submit ASSUMES submit and start share one epoch (SYCL does not guarantee it; the L0 "
                "adapter derives submit from a host/device timer correlation) -> diagnostic only, -1 when negative; "
                "dev_service and dev gaps use start/end only\n");
    std::fflush(stdout);

    sycl::queue qi(dev, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    sycl::queue qo(qi.get_context(), dev, sycl::property_list{sycl::property::queue::enable_profiling{}});
    uint8_t* hpin = sycl::malloc_host<uint8_t>(maxn + 2 * GB, qi);
    uint8_t* hpage = (uint8_t*) std::malloc(maxn + 2 * GB);
    uint8_t* da = sycl::malloc_device<uint8_t>(maxn + 2 * GB, qi);
    uint8_t* db = sycl::malloc_device<uint8_t>(maxn + 2 * GB, qi);
    Ctr* ctr = sycl::malloc_device<Ctr>(1, qi);
    uint64_t* csum = sycl::malloc_device<uint64_t>(1, qi);
    if (!hpin || !hpage || !da || !db || !ctr || !csum) { std::printf("FAIL allocation\n"); return 6; }
    std::memset(hpage, 0, maxn + 2 * GB);   // prealloc buffers are pre-touched once (untimed)
    std::memset(hpin, 0, maxn + 2 * GB);
    {   // lazy-init sample, reported separately and excluded from arm rows
        uint32_t x = 1;
        const auto t0 = Clock::now();
        qi.memcpy(da, &x, 4).wait();
        std::printf("INIT first_copy_us=%.1f\n", us(t0, Clock::now()));
    }
    if (!json_path.empty()) {
        js = std::fopen(json_path.c_str(), "w");
        if (!js) { std::printf("FAIL cannot open %s\n", json_path.c_str()); return 2; }
        std::fprintf(js, "{\"probe\":\"readiness_probe\",\"label\":\"%s\",\"pci\":\"%s\",\"affinity\":\"%s\","
                         "\"clock_note\":\"host and device domains never subtracted; host_dev_offset null; dev_queued assumes submit/start same epoch (not guaranteed by SYCL), diagnostic only, -1 when negative; service and gaps use start/end\",\"rows\":[",
                     label.c_str(), pci.c_str(), affinity().c_str());
    }
    if (mask & 6) {   // producer checksums for every payload, host-side, before any timed sample
        const auto tc = Clock::now();
        for (const auto& pl : pls) host_checksum(pl.bytes, rounds);
        std::printf("CHECKSUM_PRECOMPUTE payloads=%zu rounds=%d host_us=%.0f (untimed)\n", pls.size(), rounds, us(tc, Clock::now()));
    }
    uint32_t salt = 0x5100u;
    // Round-robin over host-memory kinds inside each sample index so drift hits every arm alike.
    // Sample 0 of every arm is a first-use sample; summarize i>=1 separately.
    for (const auto& pl : pls) {
        for (int i = 0; i < pl.reps; ++i) {
            for (int hm = 0; hm < N_HOSTMEM; ++hm) {
                if (mask & 1) arm_h2d(qi, pl, HostMem(hm), hpin + GB, hpage + GB, da, ctr, salt, i);
                if (mask & 2) arm_d2h(qi, pl, HostMem(hm), hpin, hpage, db, csum, rounds, salt, i);
            }
            if (mask & 4) {
                arm_chain(qi, "inorder", pl, da, db, ctr, csum, rounds, salt, i);
                arm_chain(qo, "ooo", pl, da, db, ctr, csum, rounds, salt, i);
            }
        }
        std::fflush(stdout);
    }
    if (js) { std::fputs("\n]}\n", js); std::fclose(js); }
    sycl::free(hpin, qi); std::free(hpage); sycl::free(da, qi); sycl::free(db, qi); sycl::free(ctr, qi); sycl::free(csum, qi);
    std::printf("DONE readiness_probe label=%s pci=%s\n", label.c_str(), pci.c_str());
    return 0;
}
