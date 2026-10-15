// ep_ready_probe.cpp (S28 fable-barrier: S28 split-publication + probe-local --push scoped|mirror) - bounded EP producer -> IPC message ->
// readiness -> peer consume/return probe at the real decode shape (N2560, K10, 6 rows): push 17520 B (240 B ids +
// 17280 B q8_1), return upper bound 614400 B (60 rows).  rev1 (frozen, original/) found a flag-mode push oracle failure.
// ROOT ONLY under the PAIR lease; agents never execute it.  Reuses ep_mem.hpp IPC mapping unchanged: every region is
// allocated once, stays mapped across all samples, and is freed only after both queues drained (S5 lifetime rule).
//   --mode copy : proven S5 mode-1 pull (consumer q.memcpy from the IPC pointer); readiness is host-mediated.
//   --mode flag : production ep_bump/ep_push/ep_wait/ep_copy_uncached/ep_send_rows, unchanged; consumer chain
//                 submitted first, device spin capped by --spin, host poll capped by --timeout-ms.
//   --poison owner    (rev1 behaviour, default): the OWNER card locally poisons/guards its inbox (card1 inbox1,
//                     card0 parts0) before the PEER writes it remotely -> two GPU writers to the same lines.
//   --poison producer : only the producing card writes poison+guard through the IPC pointer (card0 -> inbox1,
//                     card1 -> parts0), as production's single-writer inbox/return ownership.  Stages stay owner-local.
//   --extra-reads 0|2 (S27): flag mode only.  After the first ep_copy_uncached into stage1 and its end marker, TWO more
//                     immediate same-queue ep_copy_uncached of the same inbox into PRIVATE card-1 buffers stage1b/stage1c
//                     (PB bytes each, their own fresh poison per sample), no host interaction between the three copies;
//                     ep_send_rows still uses the original stage1 ids.  All three are oracled on the host after both
//                     chains drain.  A later read that is clean while the first was stale does NOT by itself distinguish
//                     "writes still in flight" from "a read-side cache was invalidated in between"; it is labelled so.
//   --wait native|debug (S27): card 1's readiness wait.  native = production ep_wait.  debug = a probe-local kernel with
//                     the SAME loop, load, spin cap and error code as ep_wait, plus a device-local side channel written
//                     AFTER the loop: wanted=*ctr as the kernel read it, first and last sys_load(flag), spin count.
//                     Card 0's return wait stays native in both arms.  The side channel changes the kernel's timing.
// On an oracle failure (flag mode): after both queues drain, bounded post-failure re-reads of the producer source,
// the inbox via uncached GPU read, via runtime memcpy, and via the peer's IPC pointer, plus flag/epoch counters.
// Every timed message: fresh salt, destination/stage poisoned, 4 KiB guard after payload, full host oracle.
// Device timing uses only profiling START/END, differenced only within ONE card; inter-card clock join is N/A.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "strata/core/ep_mem.hpp"
#include "strata/kernels/ep_kernels.hpp"
#include "strata/sycl_doorbell.hpp"

namespace {
constexpr int ROWS = 6, K = 10, N = 2560, ENTRIES = ROWS * K;
constexpr size_t IDS_B = size_t(ENTRIES) * 4, XQ_B = size_t(ROWS) * (N / 32) * 36, PUSH_B = IDS_B + XQ_B;
constexpr size_t RET_B = size_t(ENTRIES) * N * 4, GUARD_B = 4096, PB = PUSH_B + GUARD_B, RB = RET_B + GUARD_B;
static_assert(PUSH_B == 17520 && RET_B == 614400, "root shapes");
constexpr uint32_t PUSH = 0, ROWSK = 1, POISON = 2, GUARD = 3, N_TABLE = 256, SALT_STEP = 7919;
// stage poison salts: distinct from the inbox poison (salt) and from each other; all inside the 20-bit poison mask
constexpr uint32_t STAGE1_X = 0x80000u, STAGE1B_X = 0x40000u, STAGE1C_X = 0xC0000u;

// ids < 256 (valid res0 index), q8_1 words nonzero, rows finite nonzero floats in [1,2); poison/guard never collide.
inline uint32_t pattern(uint32_t kind, size_t i, uint32_t salt) {
    const uint32_t h = uint32_t(i) * 2654435761u ^ salt * 0x9E3779B9u;
    if (kind == PUSH) return i < size_t(ENTRIES) ? (uint32_t(i) * 37u + salt) % N_TABLE : (h | 1u);
    if (kind == ROWSK) return 0x3F800000u | ((h ^ 0x5A5A5A5Au) & 0x007FFFFFu);
    if (kind == POISON) return 0xBAD00000u ^ (salt & 0xFFFFFu);
    return 0x6A000000u ^ (salt & 0xFFFFFFu);
}

using clk = std::chrono::steady_clock;
double g_timeout_ms = 5000, g_wait_us = 0;
bool g_host_sync = false;
int g_extra = 0;
std::string g_wait_mode = "native";
std::string g_push_mode = "native";
double us_since(clk::time_point t) { return std::chrono::duration<double, std::micro>(clk::now() - t).count(); }

class MarkK;
sycl::event mark(sycl::queue& q) { return q.single_task<MarkK>([] {}); }
// Bounded completion poll; optional wait on the completed event tests host-visible synchronization.  On the bound exit WITHOUT freeing (a kernel may still use a mapping).
void await(sycl::event e, const char* what) {
    const auto t0 = clk::now();
    while (e.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {
        if (us_since(t0) > g_timeout_ms * 1e3) {
            std::printf("FAIL TIMEOUT %s after %.0f ms; exiting without free, root runs pair health/xe-reset ladder\n",
                        what, g_timeout_ms);
            std::fflush(stdout);
            std::_Exit(7);
        }
        std::this_thread::yield();
    }
    if (g_host_sync) e.wait(); // only after bounded status completion; separates completion polling from wait/coherency
    g_wait_us += us_since(t0);
}
void drain(sycl::queue& q, const char* what) { await(mark(q), what); }
double ts(const sycl::event& e, bool end) {
    return double(end ? e.get_profiling_info<sycl::info::event_profiling::command_end>()
                      : e.get_profiling_info<sycl::info::event_profiling::command_start>());
}
// same-card only: (b at b_end) - (a at a_end), microseconds
double dev_us(const sycl::event& a, bool a_end, const sycl::event& b, bool b_end) { return (ts(b, b_end) - ts(a, a_end)) / 1e3; }
sycl::event fill(sycl::queue& q, uint32_t* p, size_t words, uint32_t kind, uint32_t salt) {
    return q.parallel_for(sycl::range<1>(words), [=](sycl::id<1> i) { p[i] = pattern(kind, i[0], salt); });
}
size_t guard_bad(const std::vector<uint32_t>& h, size_t words, uint32_t salt) {
    size_t b = 0;
    for (size_t j = 0; j < GUARD_B / 4; ++j) b += h[words + j] != pattern(GUARD, j, salt);
    return b;
}
const char* cls(uint32_t got, uint32_t kind, size_t i, uint32_t salt, uint32_t prev) {
    if (got == pattern(POISON, i, salt ^ STAGE1_X)) return "cur_stage_poison";
    if (got == pattern(POISON, i, salt ^ STAGE1B_X)) return "cur_stage1b_poison";
    if (got == pattern(POISON, i, salt ^ STAGE1C_X)) return "cur_stage1c_poison";
    if (got == pattern(POISON, i, salt)) return "cur_poison";
    if (got == pattern(POISON, i, prev)) return "prior_poison";
    if (got == pattern(kind, i, prev)) return "prior_pattern";
    if (got == 0) return "zero";
    if (got == pattern(GUARD, i, salt)) return "cur_guard";
    return "other";
}
// One view of a payload: bad count, class histogram, first 16 mismatches, first 8 contiguous runs (word idx, length).
size_t diag(const char* view, const std::vector<uint32_t>& h, size_t words, uint32_t kind, uint32_t salt, int s) {
    const uint32_t prev = salt - SALT_STEP;
    std::map<std::string, size_t> c;
    std::string first, runs;
    size_t bad = 0, nruns = 0, shown = 0, start = 0;
    bool in = false;
    char buf[96];
    for (size_t i = 0; i <= words; ++i) {
        const bool b = i < words && h[i] != pattern(kind, i, salt);
        if (b) {
            ++bad;
            ++c[cls(h[i], kind, i, salt, prev)];
            if (shown++ < 16) {
                std::snprintf(buf, sizeof buf, " %zu:%08x/%08x:%s", i, h[i], pattern(kind, i, salt), cls(h[i], kind, i, salt, prev));
                first += buf;
            }
            if (!in) { in = true; start = i; }
        } else if (in) {
            in = false;
            if (nruns++ < 8) { std::snprintf(buf, sizeof buf, " [%zu,+%zu,B%zu]", start, i - start, start * 4); runs += buf; }
        }
    }
    std::printf("DIAG s=%d view=%s bad=%zu runs=%zu classes=", s, view, bad, nruns);
    for (auto& [k, v] : c) std::printf("%s:%zu,", k.c_str(), v);
    std::printf(" prior_valid=%d first16=%s runs8=%s\n", s > 0, first.c_str(), runs.c_str());
    return bad;
}
std::string lower(std::string s) { for (auto& c : s) c = char(std::tolower((unsigned char) c)); return s; }

// --push split: probe-local copy of the native ep_push payload body, including its release/system
// fence and workgroup barrier. Only publication moves into a second kernel on this SAME in_order
// producer queue. No host wait or cross-queue event is introduced; q0 return-wait remains AFTER
// these submissions. Payload assignments stay plain (no cache-hint substitution).
// The native branch still calls the frozen archive. A kernel completion boundary is the intended
// discriminator; source-level queue ordering does not prove peer-memory visibility or correctness.
class EpPushProbePayloadK;
class EpPushProbeFlagK;
void ep_push_probe_split(const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes,
                         int32_t* peer_ids, uint8_t* peer_xq, uint32_t* peer_flag,
                         const uint32_t* ctr, sycl::queue& q) {
    constexpr size_t WG = 256;
    const size_t words = xq_bytes / 4;
    q.parallel_for<EpPushProbePayloadK>(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const size_t lid = it.get_local_id(0);
        for (size_t i = lid; i < (size_t) n_ids; i += WG) peer_ids[i] = ids[i];
        const uint32_t* s = (const uint32_t*) xq;
        uint32_t* d = (uint32_t*) peer_xq;
        for (size_t i = lid; i < words; i += WG) d[i] = s[i];
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
    });
    // Same plain epoch load and same hinted sys_store helper as native lid0 publication.
    q.single_task<EpPushProbeFlagK>([=] { strata::sys_store(peer_flag, *ctr); });
}

// --push mirror|scoped (S28 fable-barrier): ONE probe-local WG256 kernel, statement-for-statement the native ep_push
// body (ep_kernels.dp.cpp:93-103): per-WI release/system fence, work-group barrier, then lid0 sys_store(peer_flag,*ctr)
// in the SAME kernel.  The frozen S27 SPIR-V of native ep_push shows no CrossDevice OpMemoryBarrier before its
// OpControlBarrier (mem=Workgroup); the fence next to the barrier was folded into the WG-scope barrier.
//   mirror : group_barrier(it.get_group())                              default fence scope = work_group, as native.
//   scoped : group_barrier(it.get_group(), sycl::memory_scope::system)  the barrier itself carries CrossDevice scope.
// The two arms differ in exactly that one argument (if constexpr below).  mirror vs the archive native differ only by
// TU/kernel name.  Payload stores stay plain; no atomics, cache hints, sleeps or extra work-groups.  Whether the
// barrier's SPIR-V scope survives to the device binary is checked offline (spirv/); it does not prove visibility.
template <bool Scoped> class EpPushProbeOneK;
template <bool Scoped>
void ep_push_probe_one(const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes,
                       int32_t* peer_ids, uint8_t* peer_xq, uint32_t* peer_flag,
                       const uint32_t* ctr, sycl::queue& q) {
    constexpr size_t WG = 256;
    const size_t words = xq_bytes / 4;
    q.parallel_for<EpPushProbeOneK<Scoped>>(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const size_t lid = it.get_local_id(0);
        for (size_t i = lid; i < (size_t) n_ids; i += WG) peer_ids[i] = ids[i];
        const uint32_t* s = (const uint32_t*) xq;
        uint32_t* d = (uint32_t*) peer_xq;
        for (size_t i = lid; i < words; i += WG) d[i] = s[i];
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        if constexpr (Scoped) sycl::group_barrier(it.get_group(), sycl::memory_scope::system);
        else sycl::group_barrier(it.get_group());
        if (lid == 0) strata::sys_store(peer_flag, *ctr);
    });
}

// --wait debug: probe-local copy of ep_wait (ep_kernels.dp.cpp:106-119) with a device-local side channel.  Same
// `want = *ctr` plain load, same sys_load (uncached L1+L3 read + acquire fence, sycl_doorbell.hpp), same wrap-safe
// compare, same spin cap and err code, same trailing acquire fence.  The load sequence is identical to the native
// loop (one load per compare; the cap check happens before the next load).  The side channel is written only after
// the loop and the fence; it is plain device memory on card 1, read by the host after both chains drain.
//   side[0] = wanted epoch as the kernel read it   side[1] = first sys_load(flag)
//   side[2] = last sys_load(flag) (the value that ended the loop, or the value at the cap)   side[3] = spin count
class EpWaitProbeDebugK;
void ep_wait_probe_debug(const uint32_t* flag, const uint32_t* ctr, uint32_t spin_max, uint32_t* err, uint32_t code,
                         uint32_t* side, sycl::queue& q) {
    q.single_task<EpWaitProbeDebugK>([=] {
        const uint32_t want = *ctr;
        uint32_t spin = 0;
        uint32_t v = strata::sys_load(flag);
        const uint32_t first = v;
        while ((int32_t) (v - want) < 0) {
            if (++spin >= spin_max) {
                if (err != nullptr) strata::sys_store(err, code);
                break;
            }
            v = strata::sys_load(flag);
        }
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        side[0] = want; side[1] = first; side[2] = v; side[3] = spin;
    });
}
}  // namespace

int main(int argc, char** argv) {
    std::string mode = "copy", poison = "owner";
    int reps = 20, max_fail = 1;
    uint32_t spin = 1u << 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--mode") mode = v;
        else if (a == "--poison") poison = v;
        else if (a == "--host-sync") g_host_sync = std::atoi(v.c_str()) != 0;
        else if (a == "--extra-reads") g_extra = std::atoi(v.c_str());
        else if (a == "--wait") g_wait_mode = v;
        else if (a == "--push") g_push_mode = v;
        else if (a == "--reps") reps = std::atoi(v.c_str());
        else if (a == "--max-fail") max_fail = std::atoi(v.c_str());
        else if (a == "--spin") spin = uint32_t(std::strtoul(v.c_str(), nullptr, 10));
        else if (a == "--timeout-ms") g_timeout_ms = std::atof(v.c_str());
        else { std::printf("FAIL unknown arg %s\n", a.c_str()); return 2; }
    }
    if ((mode != "copy" && mode != "flag") || (poison != "owner" && poison != "producer") || reps < 1 || reps > 64 ||
        max_fail < 1 || max_fail > 8 || spin == 0 || g_timeout_ms <= 0 || (g_extra != 0 && g_extra != 2) ||
        (g_wait_mode != "native" && g_wait_mode != "debug") ||
        (g_push_mode != "native" && g_push_mode != "split" && g_push_mode != "scoped" && g_push_mode != "mirror")) {
        std::puts("FAIL usage: --mode copy|flag --poison owner|producer --extra-reads 0|2 --wait native|debug "
                  "--push native|split|scoped|mirror "
                  "--host-sync 0|1 --reps 1..64 --max-fail 1..8 --spin N --timeout-ms MS");
        return 2;
    }
    const bool extra = mode == "flag" && g_extra == 2;
    const bool wdebug = mode == "flag" && g_wait_mode == "debug";
    using namespace strata::core;
    namespace sk = strata::kernels;
    if (dpct::device_count() != 2) { std::puts("FAIL need exactly 2 devices"); return 2; }
    const char* want[2] = {"0000:0b:00.0", "0000:44:00.0"};
    sycl::device dev[2] = {dpct::dev_mgr::instance().get_device(0), dpct::dev_mgr::instance().get_device(1)};
    for (int k = 0; k < 2; ++k) {
        if (!dev[k].has(sycl::aspect::ext_intel_pci_address)) { std::printf("FAIL dev%d has no pci_address\n", k); return 2; }
        const std::string pci = lower(dev[k].get_info<sycl::ext::intel::info::device::pci_address>());
        std::printf("BDF dev%d=%s expected=%s\n", k, pci.c_str(), want[k]);
        if (pci != want[k]) { std::puts("FAIL BDF mismatch"); return 2; }
    }
    const sycl::property_list props{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
    sycl::queue q0(dpct::dev_mgr::instance().get_device(0).get_context(), dev[0], props);
    sycl::queue q1(dpct::dev_mgr::instance().get_device(1).get_context(), dev[1], props);

    // owner first; .remote is the other card's address.  Allocated once, freed only at the end (S5 rule).
    EpRegion src0, inbox1, ret1, parts0, flag1, flag0;
    std::string err;
    if (!ep_region_alloc(0, 1, PB, src0, err) || !ep_region_alloc(1, 0, PB, inbox1, err) ||
        !ep_region_alloc(1, 0, RB, ret1, err) || !ep_region_alloc(0, 1, RB, parts0, err) ||
        !ep_region_alloc(1, 0, 4096, flag1, err) || !ep_region_alloc(0, 1, 4096, flag0, err)) {
        std::printf("FAIL %s\n", err.c_str()); return 3;
    }
    auto* push_src0 = sycl::malloc_device<uint32_t>(PUSH_B / 4, q0);   // flag: producer's ids+q8_1 (card 0)
    auto* rows1 = sycl::malloc_device<uint32_t>(RET_B / 4, q1);        // flag: peer's computed rows (card 1)
    auto* stage1 = sycl::malloc_device<uint32_t>(PB / 4, q1);
    auto* stage1b = sycl::malloc_device<uint32_t>(PB / 4, q1);         // --extra-reads: second same-queue read
    auto* stage1c = sycl::malloc_device<uint32_t>(PB / 4, q1);         // --extra-reads: third same-queue read
    auto* stage0 = sycl::malloc_device<uint32_t>(RB / 4, q0);
    auto* diag1 = sycl::malloc_device<uint32_t>(RB / 4, q1);           // post-failure re-read targets only
    auto* diag0 = sycl::malloc_device<uint32_t>(RB / 4, q0);
    auto* ctr0 = sycl::malloc_device<uint32_t>(1, q0);
    auto* ctr1 = sycl::malloc_device<uint32_t>(1, q1);
    auto* done1 = sycl::malloc_device<uint32_t>(1, q1);
    auto* wside = sycl::malloc_device<uint32_t>(4, q1);                // --wait debug side channel (card 1 local)
    auto* res0_1 = sycl::malloc_device<int32_t>(N_TABLE, q1);         // all -1: every entry peer-owned (upper bound)
    auto* err0 = sycl::malloc_host<uint32_t>(1, q0);
    auto* err1 = sycl::malloc_host<uint32_t>(1, q1);
    q0.memset(ctr0, 0, 4);
    q1.memset(ctr1, 0, 4);
    q1.memset(done1, 0, 4);
    q1.memset(wside, 0, 16);
    q1.fill(res0_1, int32_t(-1), N_TABLE);
    drain(q0, "init0");
    drain(q1, "init1");
    std::printf("CONFIG mode=%s poison=%s reps=%d+1first_use max_fail=%d push_B=%zu ret_upper_B=%zu guard_B=%zu spin=%u "
                "timeout_ms=%.0f host_sync=%d extra_reads=%d wait=%s push=%s intercard_clock_join=N/A\n", mode.c_str(),
                mode == "flag" ? poison.c_str() : "n/a", reps, max_fail, PUSH_B, RET_B, GUARD_B, spin, g_timeout_ms,
                (int) g_host_sync, extra ? 2 : 0, mode == "flag" ? g_wait_mode.c_str() : "n/a",
                mode == "flag" ? g_push_mode.c_str() : "n/a");
    std::fflush(stdout);

    std::vector<uint32_t> hp(PB / 4), hb(PB / 4), hc(PB / 4), hr(RB / 4), hd(RB / 4);
    std::map<std::string, std::vector<double>> agg;
    int fails = 0, timed = 0, timed_fail = 0;
    // bounded post-failure read of `bytes` from device pointer src on queue q, through device buffer tmp (or direct)
    auto read = [&](sycl::queue& q, void* tmp, const void* src, size_t bytes, bool uncached, const char* what) {
        if (uncached) { sk::ep_copy_uncached(tmp, src, bytes, &q); q.memcpy(hd.data(), tmp, bytes); }
        else if (tmp) { q.memcpy(tmp, src, bytes); q.memcpy(hd.data(), tmp, bytes); }
        else q.memcpy(hd.data(), src, bytes);
        drain(q, what);
    };
    for (int s = 0; s <= reps && fails < max_fail; ++s) {
        const uint32_t saltP = 0x1000u + uint32_t(s) * SALT_STEP + (mode == "flag" ? 0x400000u : 0u), saltR = saltP + 1u;
        auto* src0w = static_cast<uint32_t*>(src0.local);
        auto* ret1w = static_cast<uint32_t*>(ret1.local);
        const bool prod = poison == "producer";
        // owner: owner-local writes; producer: the single remote writer pokes poison/guard through its IPC pointer
        auto* inboxP = static_cast<uint32_t*>(prod ? inbox1.remote : inbox1.local);
        auto* partsP = static_cast<uint32_t*>(prod ? parts0.remote : parts0.local);
        sycl::queue& qin = prod ? q0 : q1;
        sycl::queue& qpa = prod ? q1 : q0;
        *err0 = 0; *err1 = 0;
        // untimed prep: poison every destination/stage, set guards for this salt, produce flag-mode source data
        fill(q0, stage0, RB / 4, POISON, saltR ^ STAGE1_X);
        fill(q1, stage1, PB / 4, POISON, saltP ^ STAGE1_X);
        if (extra) {
            fill(q1, stage1b, PB / 4, POISON, saltP ^ STAGE1B_X);
            fill(q1, stage1c, PB / 4, POISON, saltP ^ STAGE1C_X);
        }
        if (wdebug) q1.memset(wside, 0xEE, 16);   // 0xEEEEEEEE: "kernel never wrote the side channel"
        if (mode == "copy") {
            fill(q0, src0w, PUSH_B / 4, POISON, saltP); fill(q0, src0w + PUSH_B / 4, GUARD_B / 4, GUARD, saltP);
            fill(q1, ret1w, RET_B / 4, POISON, saltR); fill(q1, ret1w + RET_B / 4, GUARD_B / 4, GUARD, saltR);
        } else {
            fill(q0, push_src0, PUSH_B / 4, PUSH, saltP);
            fill(qpa, partsP, RET_B / 4, POISON, saltR); fill(qpa, partsP + RET_B / 4, GUARD_B / 4, GUARD, saltR);
            fill(qin, inboxP, PUSH_B / 4, POISON, saltP); fill(qin, inboxP + PUSH_B / 4, GUARD_B / 4, GUARD, saltP);
            fill(q1, rows1, RET_B / 4, ROWSK, saltR);
            sk::ep_bump(ctr0, &q0);
            sk::ep_bump(ctr1, &q1);
        }
        drain(q0, "prep0");
        drain(q1, "prep1");

        std::vector<std::pair<std::string, double>> f;
        double enq = 0;
        g_wait_us = 0;
        const auto t0 = clk::now();
        if (mode == "copy") {
            auto sub = [&](auto&& fn) { const auto t = clk::now(); sycl::event e = fn(); enq += us_since(t); return e; };
            sycl::event p0 = sub([&] { return fill(q0, src0w, PUSH_B / 4, PUSH, saltP); });
            await(p0, "copy produce card0");
            sycl::event c1 = sub([&] { return q1.memcpy(stage1, src0.remote, PB); });
            await(c1, "copy consume card1");
            sycl::event p1 = sub([&] { return fill(q1, ret1w, RET_B / 4, ROWSK, saltR); });
            await(p1, "copy return produce card1");
            sycl::event c0 = sub([&] { return q0.memcpy(stage0, ret1.remote, RB); });
            await(c0, "copy return consume card0");
            f = {{"c0_produce_us", dev_us(p0, false, p0, true)}, {"c1_consume_push_us", dev_us(c1, false, c1, true)},
                 {"c1_host_dep_gap_us", dev_us(c1, true, p1, false)}, {"c1_produce_rows_us", dev_us(p1, false, p1, true)},
                 {"c0_consume_ret_us", dev_us(c0, false, c0, true)}, {"c0_roundtrip_us", dev_us(p0, false, c0, true)}};
        } else {
            auto* pushb = reinterpret_cast<const uint8_t*>(push_src0);
            auto* inboxb = static_cast<uint8_t*>(inbox1.remote);
            sycl::event a1 = mark(q1);   // consumer first: card 1 waits for the epoch on its own flag
            if (wdebug) ep_wait_probe_debug(static_cast<const uint32_t*>(flag1.local), ctr1, spin, err1, 11, wside, q1);
            else sk::ep_wait(static_cast<const uint32_t*>(flag1.local), ctr1, spin, err1, 11, &q1);
            sycl::event b1 = mark(q1);
            sk::ep_copy_uncached(stage1, inbox1.local, PB, &q1);
            sycl::event c1 = mark(q1);
            sycl::event c1b = c1, c1c = c1;
            if (extra) {   // two more immediate same-queue reads of the same inbox, no host interaction in between
                sk::ep_copy_uncached(stage1b, inbox1.local, PB, &q1);
                c1b = mark(q1);
                sk::ep_copy_uncached(stage1c, inbox1.local, PB, &q1);
                c1c = mark(q1);
            }
            sk::ep_send_rows(static_cast<float*>(parts0.remote), reinterpret_cast<const float*>(rows1),
                             reinterpret_cast<const int32_t*>(stage1), res0_1, ENTRIES, N, done1,
                             static_cast<uint32_t*>(flag0.remote), ctr1, &q1);
            sycl::event d1 = mark(q1);
            sycl::event a0 = mark(q0);
            if (g_push_mode == "split")
                ep_push_probe_split(reinterpret_cast<const int32_t*>(pushb), ENTRIES, pushb + IDS_B, XQ_B,
                                    reinterpret_cast<int32_t*>(inboxb), inboxb + IDS_B,
                                    static_cast<uint32_t*>(flag1.remote), ctr0, q0);
            else if (g_push_mode == "scoped")
                ep_push_probe_one<true>(reinterpret_cast<const int32_t*>(pushb), ENTRIES, pushb + IDS_B, XQ_B,
                                        reinterpret_cast<int32_t*>(inboxb), inboxb + IDS_B,
                                        static_cast<uint32_t*>(flag1.remote), ctr0, q0);
            else if (g_push_mode == "mirror")
                ep_push_probe_one<false>(reinterpret_cast<const int32_t*>(pushb), ENTRIES, pushb + IDS_B, XQ_B,
                                         reinterpret_cast<int32_t*>(inboxb), inboxb + IDS_B,
                                         static_cast<uint32_t*>(flag1.remote), ctr0, q0);
            else
                sk::ep_push(reinterpret_cast<const int32_t*>(pushb), ENTRIES, pushb + IDS_B, XQ_B,
                            reinterpret_cast<int32_t*>(inboxb), inboxb + IDS_B,
                            static_cast<uint32_t*>(flag1.remote), ctr0, &q0);
            sycl::event b0 = mark(q0);
            sk::ep_wait(static_cast<const uint32_t*>(flag0.local), ctr0, spin, err0, 12, &q0);
            sycl::event c0 = mark(q0);
            sk::ep_copy_uncached(stage0, parts0.local, RB, &q0);
            sycl::event d0 = mark(q0);
            enq = us_since(t0);
            await(d1, "flag card1 chain");
            await(d0, "flag card0 chain");
            // bracketed windows [mark end, next mark start]: include same-card dispatch gaps
            f = {{"c0_push_us", dev_us(a0, true, b0, false)}, {"c0_ret_wait_us", dev_us(b0, true, c0, false)},
                 {"c0_consume_ret_us", dev_us(c0, true, d0, false)}, {"c0_roundtrip_us", dev_us(a0, true, d0, false)},
                 {"c1_flag_wait_us", dev_us(a1, true, b1, false)}, {"c1_consume_push_us", dev_us(b1, true, c1, false)}};
            if (extra) {
                f.push_back({"c1_consume_push2_us", dev_us(c1, true, c1b, false)});
                f.push_back({"c1_consume_push3_us", dev_us(c1b, true, c1c, false)});
            }
            f.push_back({"c1_send_rows_us", dev_us(c1c, true, d1, false)});   // after the last consume marker
            f.push_back({"c1_chain_us", dev_us(a1, true, d1, false)});
        }
        const double wall = us_since(t0);
        f.push_back({"host_wall_us", wall});
        f.push_back({"host_enqueue_us", enq});
        f.push_back({"host_wait_us", g_wait_us});

        q1.memcpy(hp.data(), stage1, PB);
        if (extra) { q1.memcpy(hb.data(), stage1b, PB); q1.memcpy(hc.data(), stage1c, PB); }
        uint32_t ws[4] = {0, 0, 0, 0};
        if (wdebug) q1.memcpy(ws, wside, 16);
        q0.memcpy(hr.data(), stage0, RB);
        drain(q1, "readback1");
        drain(q0, "readback0");
        size_t pb = 0, pg = guard_bad(hp, PUSH_B / 4, saltP), rb = 0, rg = guard_bad(hr, RET_B / 4, saltR);
        size_t pb2 = 0, pg2 = 0, pb3 = 0, pg3 = 0;
        for (size_t i = 0; i < PUSH_B / 4; ++i) pb += hp[i] != pattern(PUSH, i, saltP);
        for (size_t i = 0; i < RET_B / 4; ++i) rb += hr[i] != pattern(ROWSK, i, saltR);
        if (extra) {
            pg2 = guard_bad(hb, PUSH_B / 4, saltP); pg3 = guard_bad(hc, PUSH_B / 4, saltP);
            for (size_t i = 0; i < PUSH_B / 4; ++i) { pb2 += hb[i] != pattern(PUSH, i, saltP); pb3 += hc[i] != pattern(PUSH, i, saltP); }
        }
        std::printf("SAMPLE mode=%s s=%d first_use=%d saltP=%#x push_bad=%zu push_guard_bad=%zu ret_bad=%zu "
                    "ret_guard_bad=%zu err0=%u err1=%u", mode.c_str(), s, s == 0, saltP, pb, pg, rb, rg, *err0, *err1);
        if (extra) std::printf(" push2_bad=%zu push2_guard_bad=%zu push3_bad=%zu push3_guard_bad=%zu", pb2, pg2, pb3, pg3);
        for (auto& [k, v] : f) std::printf(" %s=%.3f", k.c_str(), v);
        std::printf(" intercard=N/A\n");
        if (wdebug) {
            // wanted is what the kernel itself read from ctr1; expect_want is the host's count of ep_bump calls (s+1)
            std::printf("WAITDBG s=%d wanted=%u expect_want=%d first_flag=%u last_flag=%u spins=%u err1=%u want_ok=%d "
                        "ready_at_first=%d\n", s, ws[0], s + 1, ws[1], ws[2], ws[3], *err1, ws[0] == uint32_t(s + 1),
                        (int32_t) (ws[1] - ws[0]) >= 0);
        }
        const bool bad = pb || pg || rb || rg || *err0 || *err1 || pb2 || pg2 || pb3 || pg3;
        if (s > 0) { ++timed; timed_fail += bad; }
        if (bad && mode == "flag" && fails < 2) {
            // Post-failure, both queues drained.  Order: consumer view(s), then producer source, then uncached re-read of
            // the inbox on its owner, runtime memcpy on the owner, and finally the producer's view through IPC.
            uint32_t fl[4];
            q1.memcpy(&fl[0], flag1.local, 4); q1.memcpy(&fl[1], ctr1, 4);
            q0.memcpy(&fl[2], flag0.local, 4); q0.memcpy(&fl[3], ctr0, 4);
            drain(q1, "diag flags1"); drain(q0, "diag flags0");
            std::printf("DIAG s=%d flag1=%u ctr1=%u flag0=%u ctr0=%u (post-drain values; not the wait kernel's view)\n",
                        s, fl[0], fl[1], fl[2], fl[3]);
            if (pb || pg || pb2 || pg2 || pb3 || pg3) {
                diag("push.stage1_consumed", hp, PUSH_B / 4, PUSH, saltP, s);
                if (extra) {
                    diag("push.stage1b_second_read", hb, PUSH_B / 4, PUSH, saltP, s);
                    std::printf("DIAG s=%d view=push.stage1b_second_read guard_bad=%zu\n", s, pg2);
                    diag("push.stage1c_third_read", hc, PUSH_B / 4, PUSH, saltP, s);
                    std::printf("DIAG s=%d view=push.stage1c_third_read guard_bad=%zu\n", s, pg3);
                    std::printf("DIAG s=%d extra_reads_note=%s\n", s,
                                (pb && !pb2 && !pb3) ? "LATE_CLEAN_AMBIGUOUS(in-flight writes OR read-side invalidation; not separable here)"
                                : (pb && pb2 && !pb3) ? "LATE_CLEAN_AMBIGUOUS_after_two(in-flight writes OR read-side invalidation)"
                                : (pb && pb2 && pb3) ? "PERSISTENT_STALE_across_3_reads"
                                : "NONMONOTONIC_or_first_clean(see counts)");
                }
                read(q0, nullptr, push_src0, PUSH_B, false, "diag src0");
                diag("push.src0_producer_gpu0", hd, PUSH_B / 4, PUSH, saltP, s);
                read(q1, diag1, inbox1.local, PB, true, "diag inbox uc");
                diag("push.inbox1_uncached_reread_gpu1", hd, PUSH_B / 4, PUSH, saltP, s);
                std::printf("DIAG s=%d view=push.inbox1_uncached_reread_gpu1 guard_bad=%zu\n", s, guard_bad(hd, PUSH_B / 4, saltP));
                read(q1, diag1, inbox1.local, PB, false, "diag inbox memcpy");
                diag("push.inbox1_memcpy_gpu1", hd, PUSH_B / 4, PUSH, saltP, s);
                read(q0, diag0, inbox1.remote, PB, false, "diag inbox remote");
                diag("push.inbox1_ipc_read_gpu0", hd, PUSH_B / 4, PUSH, saltP, s);
            }
            if (rb || rg) {
                diag("ret.stage0_consumed", hr, RET_B / 4, ROWSK, saltR, s);
                read(q1, nullptr, rows1, RET_B, false, "diag rows1");
                diag("ret.rows1_producer_gpu1", hd, RET_B / 4, ROWSK, saltR, s);
                read(q0, diag0, parts0.local, RB, true, "diag parts uc");
                diag("ret.parts0_uncached_reread_gpu0", hd, RET_B / 4, ROWSK, saltR, s);
                read(q0, diag0, parts0.local, RB, false, "diag parts memcpy");
                diag("ret.parts0_memcpy_gpu0", hd, RET_B / 4, ROWSK, saltR, s);
                read(q1, diag1, parts0.remote, RB, false, "diag parts remote");
                diag("ret.parts0_ipc_read_gpu1", hd, RET_B / 4, ROWSK, saltR, s);
            }
        }
        std::fflush(stdout);
        if (bad) ++fails;
        else if (s > 0) for (auto& [k, v] : f) agg[k].push_back(v);
    }
    drain(q0, "final0");
    drain(q1, "final1");
    for (auto& [k, v] : agg) {
        std::sort(v.begin(), v.end());
        std::printf("SUMMARY mode=%s %s n=%zu min=%.3f median=%.3f p90=%.3f max=%.3f\n", mode.c_str(), k.c_str(), v.size(),
                    v.front(), v[v.size() / 2], v[size_t(0.9 * double(v.size() - 1))], v.back());
    }
    std::printf("FAILRATE mode=%s poison=%s extra_reads=%d wait=%s timed_fail=%d timed_run=%d any_fail_incl_first_use=%d "
                "stopped_early=%d\n", mode.c_str(), mode == "flag" ? poison.c_str() : "n/a", extra ? 2 : 0,
                mode == "flag" ? g_wait_mode.c_str() : "n/a", timed_fail, timed, fails, fails >= max_fail);
    sycl::free(push_src0, q0); sycl::free(rows1, q1); sycl::free(stage1, q1); sycl::free(stage0, q0);
    sycl::free(stage1b, q1); sycl::free(stage1c, q1); sycl::free(wside, q1);
    sycl::free(diag1, q1); sycl::free(diag0, q0);
    sycl::free(ctr0, q0); sycl::free(ctr1, q1); sycl::free(done1, q1); sycl::free(res0_1, q1);
    sycl::free(err0, q0); sycl::free(err1, q1);
    ep_region_free(flag0); ep_region_free(flag1); ep_region_free(parts0);
    ep_region_free(ret1); ep_region_free(inbox1); ep_region_free(src0);
    std::puts(fails ? "FAIL oracle/guard/spin-cap mismatch (see SAMPLE/DIAG)" : "PASS all timed messages fresh, guards intact");
    return fails ? 5 : 0;
}
