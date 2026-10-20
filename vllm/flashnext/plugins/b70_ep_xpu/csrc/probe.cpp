// M0 transport oracle inside torch's XPU context, plus design section 7.5
// probes 1-5. The flag-mode chain matches strata/flash-next/lowlevel_probe/
// ep_ready_probe.cpp (S28, --mode flag --poison producer --push native):
// card 1 waits first, card 0 pushes, both spins are bounded.
#include "doorbell.hpp"
#include "ep_kernels.hpp"
#include "region.hpp"

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <c10/xpu/XPUStream.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace b70ep {
namespace {

constexpr int kRows = 6, kK = 10, kN = 2560, kEntries = kRows * kK;
constexpr size_t kIdsB = (size_t)kEntries * 4;
constexpr size_t kXqB = (size_t)kRows * (kN / 32) * 36;
constexpr size_t kPushB = kIdsB + kXqB;
constexpr size_t kRetB = (size_t)kEntries * (size_t)kN * 4;
constexpr size_t kGuardB = 4096;
constexpr size_t kPB = kPushB + kGuardB;
constexpr size_t kRB = kRetB + kGuardB;
constexpr uint32_t kPush = 0, kRowsk = 1, kPoison = 2, kGuard = 3, kTable = 256, kSaltStep = 7919;
constexpr uint32_t kStageX = 0x80000u;
constexpr size_t kBwB = 256 * 1024;

static_assert(kPushB == 17520 && kRetB == 614400, "S28 shapes");

namespace expg = sycl::ext::oneapi::experimental;
using GraphMod = expg::command_graph<expg::graph_state::modifiable>;
using GraphExec = expg::command_graph<expg::graph_state::executable>;

uint32_t pattern(uint32_t kind, size_t i, uint32_t salt) {
    const uint32_t h = uint32_t(i) * 2654435761u ^ salt * 0x9E3779B9u;
    if (kind == kPush) return i < (size_t)kEntries ? (uint32_t(i) * 37u + salt) % kTable : (h | 1u);
    if (kind == kRowsk) return 0x3F800000u | ((h ^ 0x5A5A5A5Au) & 0x007FFFFFu);
    if (kind == kPoison) return 0xBAD00000u ^ (salt & 0xFFFFFu);
    return 0x6A000000u ^ (salt & 0xFFFFFFu);
}

struct GraphSlot {
    GraphMod mod;
    std::optional<GraphExec> exec;
    explicit GraphSlot(sycl::queue& q) : mod(q) {}
};

struct State {
    bool up = false;
    uint32_t spin = 200000;
    int path = 1;
    std::optional<sycl::queue> q1;
    Region inbox, parts, flag1, flag0, bw1, bw0;
    uint32_t* push_src = nullptr;
    uint32_t* rows1 = nullptr;
    uint32_t* stage1 = nullptr;
    uint32_t* stage0 = nullptr;
    uint32_t* ctr0 = nullptr;
    uint32_t* ctr1 = nullptr;
    uint32_t* done1 = nullptr;
    int32_t* res0 = nullptr;
    uint32_t* err0 = nullptr;
    uint32_t* err1 = nullptr;
    uint32_t* bw_src0 = nullptr;
    uint32_t* bw_src1 = nullptr;
    std::unique_ptr<GraphSlot> graph;
    bool recording = false;
    int sample = -1;
};

State& S() {
    static State s;
    return s;
}

sycl::queue& q0() { return c10::xpu::getCurrentXPUStream(0).queue(); }

void fill(sycl::queue& q, uint32_t* p, size_t words, uint32_t kind, uint32_t salt) {
    q.parallel_for<class b70_m0_fill_k>(sycl::range<1>(words), [=](sycl::id<1> i) { p[i] = pattern(kind, i[0], salt); });
}

std::string bdf_of(const sycl::device& d) {
    ze_pci_ext_properties_t pci{};
    pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    pci.pNext = nullptr;
    ze_result_t z = zeDevicePciGetPropertiesExt(ze_dev(d), &pci);
    char buf[64];
    if (z != ZE_RESULT_SUCCESS) {
        std::snprintf(buf, sizeof buf, "pci_err_%u", (unsigned)z);
        return buf;
    }
    std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.%x", pci.address.domain, pci.address.bus, pci.address.device,
                  pci.address.function);
    return buf;
}

int store4(sycl::queue& writer, void* remote, sycl::queue& owner, void* local) {
    owner.memset(local, 0, 4).wait();
    writer.single_task<class b70_m0_store4_k>([=] {
        *static_cast<uint32_t*>(remote) = 0xA5A5A5A5u;
    }).wait();
    uint32_t got = 0xFFFFFFFFu;
    owner.memcpy(&got, local, 4).wait();
    return got == 0xA5A5A5A5u ? 1 : 0;
}

void free_all() {
    State& s = S();
    auto drop = [](sycl::queue* q, void* p) {
        if (q && p) sycl::free(p, *q);
    };
    if (s.q1) {
        s.q1->wait();
        drop(&*s.q1, s.rows1);
        drop(&*s.q1, s.stage1);
        drop(&*s.q1, s.ctr1);
        drop(&*s.q1, s.done1);
        drop(&*s.q1, s.res0);
        drop(&*s.q1, s.err1);
        drop(&*s.q1, s.bw_src1);
    }
    sycl::queue& a = q0();
    a.wait();
    drop(&a, s.push_src);
    drop(&a, s.stage0);
    drop(&a, s.ctr0);
    drop(&a, s.err0);
    drop(&a, s.bw_src0);
    s.push_src = s.rows1 = s.stage1 = s.stage0 = s.ctr0 = s.ctr1 = s.done1 = nullptr;
    s.res0 = nullptr;
    s.err0 = s.err1 = nullptr;
    s.bw_src0 = s.bw_src1 = nullptr;
    region_free(s.inbox);
    region_free(s.parts);
    region_free(s.flag1);
    region_free(s.flag0);
    region_free(s.bw1);
    region_free(s.bw0);
    s.graph.reset();
    s.recording = false;
    s.up = false;
}

}  // namespace

std::vector<int64_t> m0_setup(int spin) {
    State& s = S();
    if (s.up) free_all();
    if (spin < 1) spin = 200000;
    s.spin = (uint32_t)spin;
    // Touch both devices so torch has created the contexts.
    sycl::queue& stream0 = q0();
    sycl::queue& stream1 = c10::xpu::getCurrentXPUStream(1).queue();
    sycl::property_list props{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
    s.q1.emplace(stream1.get_context(), stream1.get_device(), props);
    ze_init_once();

    const bool same_ctx = stream0.get_context() == stream1.get_context();
    const bool in0 = stream0.is_in_order();
    const bool in1 = s.q1->is_in_order();
    int peer01 = 0, peer10 = 0, atom01 = 0, atom10 = 0;
    try {
        peer01 = stream0.get_device().ext_oneapi_can_access_peer(
            stream1.get_device(), sycl::ext::oneapi::peer_access::access_supported);
        peer10 = stream1.get_device().ext_oneapi_can_access_peer(
            stream0.get_device(), sycl::ext::oneapi::peer_access::access_supported);
        atom01 = stream0.get_device().ext_oneapi_can_access_peer(
            stream1.get_device(), sycl::ext::oneapi::peer_access::atomics_supported);
        atom10 = stream1.get_device().ext_oneapi_can_access_peer(
            stream0.get_device(), sycl::ext::oneapi::peer_access::atomics_supported);
    } catch (const std::exception& ex) {
        std::printf("PROBE5 peer query exception %s\n", ex.what());
    }
    try {
        if (peer01) stream0.get_device().ext_oneapi_enable_peer_access(stream1.get_device());
        if (peer10) stream1.get_device().ext_oneapi_enable_peer_access(stream0.get_device());
    } catch (const std::exception& ex) {
        std::printf("PROBE5 enable_peer exception %s\n", ex.what());
    }

    int ipc_ok = 0, same_ok = -1;
    {
        Region t;
        std::string err;
        if (region_alloc(*s.q1, stream0, 4096, 1, t, err)) {
            ipc_ok = store4(stream0, t.remote, *s.q1, t.local);
            region_free(t);
        } else {
            std::printf("PROBE5 ipc alloc: %s\n", err.c_str());
        }
    }
    if (same_ctx) {
        Region t;
        std::string err;
        if (region_alloc(*s.q1, stream0, 4096, 2, t, err)) {
            same_ok = store4(stream0, t.remote, *s.q1, t.local);
            region_free(t);
        } else {
            std::printf("PROBE5 same alloc: %s\n", err.c_str());
            same_ok = 0;
        }
    }
    if (ipc_ok) s.path = 1;
    else if (same_ok == 1) s.path = 2;
    else throw std::runtime_error("probe5: neither IPC nor same-context pointer stored a word");

    std::string err;
    auto need = [&](const sycl::queue& owner, const sycl::queue& other, size_t bytes, Region& r, const char* name) {
        if (!region_alloc(owner, other, bytes, s.path, r, err))
            throw std::runtime_error(std::string(name) + ": " + err);
    };
    // inbox lives on card 1 (card 0 writes it). parts and the return flag live on card 0.
    need(*s.q1, stream0, kPB, s.inbox, "inbox");
    need(stream0, *s.q1, kRB, s.parts, "parts");
    need(*s.q1, stream0, 4096, s.flag1, "flag1");
    need(stream0, *s.q1, 4096, s.flag0, "flag0");
    need(*s.q1, stream0, kBwB, s.bw1, "bw1");
    need(stream0, *s.q1, kBwB, s.bw0, "bw0");

    s.push_src = sycl::malloc_device<uint32_t>(kPushB / 4, stream0);
    s.stage0 = sycl::malloc_device<uint32_t>(kRB / 4, stream0);
    s.ctr0 = sycl::malloc_device<uint32_t>(1, stream0);
    s.err0 = sycl::malloc_host<uint32_t>(1, stream0);
    s.bw_src0 = sycl::malloc_device<uint32_t>(kBwB / 4, stream0);
    s.rows1 = sycl::malloc_device<uint32_t>(kRetB / 4, *s.q1);
    s.stage1 = sycl::malloc_device<uint32_t>(kPB / 4, *s.q1);
    s.ctr1 = sycl::malloc_device<uint32_t>(1, *s.q1);
    s.done1 = sycl::malloc_device<uint32_t>(1, *s.q1);
    s.res0 = sycl::malloc_device<int32_t>(kTable, *s.q1);
    s.err1 = sycl::malloc_host<uint32_t>(1, *s.q1);
    s.bw_src1 = sycl::malloc_device<uint32_t>(kBwB / 4, *s.q1);
    if (!s.push_src || !s.stage0 || !s.ctr0 || !s.err0 || !s.rows1 || !s.stage1 || !s.ctr1 || !s.done1 || !s.res0 ||
        !s.err1 || !s.bw_src0 || !s.bw_src1)
        throw std::runtime_error("m0 malloc failed");
    stream0.memset(s.ctr0, 0, 4);
    s.q1->memset(s.ctr1, 0, 4);
    s.q1->memset(s.done1, 0, 4);
    s.q1->fill(s.res0, int32_t(-1), kTable);
    *s.err0 = 0;
    *s.err1 = 0;
    stream0.wait();
    s.q1->wait();
    s.up = true;
    const std::string b0 = bdf_of(stream0.get_device());
    const std::string b1 = bdf_of(stream1.get_device());
    std::printf(
        "PROBE5 same_ctx=%d in_order0=%d in_order1=%d peer01=%d peer10=%d atom01=%d atom10=%d ipc_store=%d "
        "same_store=%d path=%d bdf0=%s bdf1=%s spin=%u\n",
        (int)same_ctx, (int)in0, (int)in1, peer01, peer10, atom01, atom10, ipc_ok, same_ok, s.path, b0.c_str(),
        b1.c_str(), s.spin);
    std::fflush(stdout);
    return { (int64_t)same_ctx, (int64_t)in0, (int64_t)in1, peer01, peer10, atom01, atom10, ipc_ok, same_ok, s.path,
             (int64_t)s.spin };
}

void m0_prep(int sample) {
    State& s = S();
    if (!s.up) throw std::runtime_error("m0_prep before setup");
    s.sample = sample;
    const uint32_t saltP = 0x1000u + (uint32_t)sample * kSaltStep + 0x400000u;
    const uint32_t saltR = saltP + 1u;
    *s.err0 = 0;
    *s.err1 = 0;
    sycl::queue& a = q0();
    // Producer poison: the writer is the card that will overwrite the payload.
    fill(a, static_cast<uint32_t*>(s.inbox.remote), kPushB / 4, kPoison, saltP);
    fill(a, static_cast<uint32_t*>(s.inbox.remote) + kPushB / 4, kGuardB / 4, kGuard, saltP);
    fill(*s.q1, static_cast<uint32_t*>(s.parts.remote), kRetB / 4, kPoison, saltR);
    fill(*s.q1, static_cast<uint32_t*>(s.parts.remote) + kRetB / 4, kGuardB / 4, kGuard, saltR);
    fill(a, s.push_src, kPushB / 4, kPush, saltP);
    fill(*s.q1, s.rows1, kRetB / 4, kRowsk, saltR);
    fill(*s.q1, s.stage1, kPB / 4, kPoison, saltP ^ kStageX);
    fill(a, s.stage0, kRB / 4, kPoison, saltR ^ kStageX);
    s.q1->memset(s.done1, 0, 4);
    a.wait();
    s.q1->wait();
}

void m0_submit_card1() {
    State& s = S();
    if (!s.up) throw std::runtime_error("m0 card1 before setup");
    auto* push_ids = reinterpret_cast<const int32_t*>(s.stage1);
    ep_bump(*s.q1, s.ctr1);
    ep_wait(*s.q1, static_cast<uint32_t*>(s.flag1.local), s.ctr1, s.spin, s.err1, 11);
    ep_copy_uncached(*s.q1, s.stage1, s.inbox.local, kPB);
    ep_send_rows(*s.q1, static_cast<float*>(s.parts.remote), reinterpret_cast<const float*>(s.rows1), push_ids, s.res0,
                 kEntries, kN, s.done1, static_cast<uint32_t*>(s.flag0.remote), s.ctr1);
}

void m0_submit_card0(int mirror) {
    State& s = S();
    if (!s.up) throw std::runtime_error("m0 card0 before setup");
    sycl::queue& a = q0();
    auto* src = reinterpret_cast<const uint8_t*>(s.push_src);
    auto* dst = static_cast<uint8_t*>(s.inbox.remote);
    ep_bump(a, s.ctr0);
    if (mirror)
        ep_push_probe<false>(a, reinterpret_cast<const int32_t*>(src), kEntries, src + kIdsB, kXqB,
                             reinterpret_cast<int32_t*>(dst), dst + kIdsB, static_cast<uint32_t*>(s.flag1.remote),
                             s.ctr0);
    else
        ep_push(a, reinterpret_cast<const int32_t*>(src), kEntries, src + kIdsB, kXqB, reinterpret_cast<int32_t*>(dst),
                dst + kIdsB, static_cast<uint32_t*>(s.flag1.remote), s.ctr0);
    ep_wait(a, static_cast<uint32_t*>(s.flag0.local), s.ctr0, s.spin, s.err0, 12);
    ep_copy_uncached(a, s.stage0, s.parts.local, kRB);
}

void m0_card1_capture_begin() {
    State& s = S();
    if (!s.up) throw std::runtime_error("capture before setup");
    if (s.recording) throw std::runtime_error("card1 already recording");
    s.graph = std::make_unique<GraphSlot>(*s.q1);
    s.graph->mod.begin_recording(*s.q1);
    s.recording = true;
}

void m0_card1_capture_end() {
    State& s = S();
    if (!s.recording || !s.graph) return;
    s.graph->mod.end_recording(*s.q1);
    s.graph->exec = s.graph->mod.finalize();
    s.recording = false;
    std::printf("M0 card1 graph finalized\n");
    std::fflush(stdout);
}

void m0_card1_replay() {
    State& s = S();
    if (!s.graph || !s.graph->exec) throw std::runtime_error("card1 replay without a graph");
    // Do not wait. Device flags order this graph against card 0.
    s.q1->ext_oneapi_graph(*s.graph->exec);
}

std::vector<int64_t> m0_check(int sample) {
    State& s = S();
    if (!s.up) throw std::runtime_error("m0_check before setup");
    const uint32_t saltP = 0x1000u + (uint32_t)sample * kSaltStep + 0x400000u;
    const uint32_t saltR = saltP + 1u;
    std::vector<uint32_t> hp(kPB / 4), hr(kRB / 4);
    s.q1->memcpy(hp.data(), s.stage1, kPB).wait();
    q0().memcpy(hr.data(), s.stage0, kRB).wait();
    size_t pb = 0, pg = 0, rb = 0, rg = 0;
    for (size_t i = 0; i < kPushB / 4; ++i) pb += hp[i] != pattern(kPush, i, saltP);
    for (size_t j = 0; j < kGuardB / 4; ++j) pg += hp[kPushB / 4 + j] != pattern(kGuard, j, saltP);
    for (size_t i = 0; i < kRetB / 4; ++i) rb += hr[i] != pattern(kRowsk, i, saltR);
    for (size_t j = 0; j < kGuardB / 4; ++j) rg += hr[kRetB / 4 + j] != pattern(kGuard, j, saltR);
    const uint32_t e0 = s.err0 ? *s.err0 : 999u;
    const uint32_t e1 = s.err1 ? *s.err1 : 999u;
    std::printf("SAMPLE mode=flag s=%d first_use=%d push_bad=%zu push_guard_bad=%zu ret_bad=%zu ret_guard_bad=%zu "
                "err0=%u err1=%u\n",
                sample, sample == 0, pb, pg, rb, rg, e0, e1);
    std::fflush(stdout);
    return {(int64_t)pb, (int64_t)pg, (int64_t)rb, (int64_t)rg, (int64_t)e0, (int64_t)e1};
}

namespace {

double dev_us(const sycl::event& e) {
    const double a = (double)e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const double b = (double)e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return (b - a) / 1e3;
}

double median(std::vector<double> v) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

sycl::event push_words(sycl::queue& q, uint32_t* dst, const uint32_t* src, size_t words, size_t groups) {
    constexpr size_t WG = 256;
    if (groups < 1) groups = 1;
    return q.parallel_for<class b70_m0_push_bw_k>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (size_t i = it.get_global_id(0); i < words; i += groups * WG) dst[i] = src[i];
    });
}

sycl::event copy_plain(sycl::queue& q, uint32_t* dst, const uint32_t* src, size_t words) {
    constexpr size_t WG = 256;
    const size_t groups = std::min<size_t>((words + WG - 1) / WG, 64);
    return q.parallel_for<class b70_m0_plain_k>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (size_t i = it.get_global_id(0); i < words; i += groups * WG) dst[i] = src[i];
    });
}

void flag_ping(sycl::queue& init_q, uint32_t* init_writes, sycl::queue& peer_q, uint32_t* peer_waits,
               uint32_t* peer_writes, uint32_t* init_waits, uint32_t epoch, uint32_t spin,
               std::vector<double>& samples) {
    // Peer spins first on its local flag. Initiator posts that flag, then
    // spins on its own flag. Device time of the initiator kernel is the
    // round trip. Cross-card clocks are not joined.
    sycl::event wp = peer_q.single_task<class b70_m0_ping_peer_k>([=] {
        uint32_t n = 0;
        while ((int32_t)(sys_load(peer_waits) - epoch) < 0) {
            if (++n >= spin) break;
        }
        sys_store(peer_writes, epoch);
    });
    sycl::event wi = init_q.single_task<class b70_m0_ping_init_k>([=] {
        sys_store(init_writes, epoch);
        uint32_t n = 0;
        while ((int32_t)(sys_load(init_waits) - epoch) < 0) {
            if (++n >= spin) break;
        }
    });
    wi.wait();
    wp.wait();
    samples.push_back(dev_us(wi) / 2.0);
}

}  // namespace

void m0_latency() {
    State& s = S();
    if (!s.up) throw std::runtime_error("latency before setup");
    sycl::queue& a = q0();
    // Profiled in-order queue in torch's card-0 context. The torch stream
    // may not have enable_profiling. Peer pointers are context-scoped.
    sycl::property_list props{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
    sycl::queue pq0(a.get_context(), a.get_device(), props);
    auto* f1 = static_cast<uint32_t*>(s.flag1.local);
    auto* f1r = static_cast<uint32_t*>(s.flag1.remote);
    auto* f0 = static_cast<uint32_t*>(s.flag0.local);
    auto* f0r = static_cast<uint32_t*>(s.flag0.remote);
    pq0.memset(f0, 0, 4).wait();
    s.q1->memset(f1, 0, 4).wait();
    std::vector<double> a_to_b, b_to_a;
    a_to_b.reserve(200);
    b_to_a.reserve(200);
    for (int i = 0; i < 200; ++i) {
        const uint32_t epoch = 1000u + (uint32_t)i * 2u;
        flag_ping(pq0, f1r, *s.q1, f1, f0r, f0, epoch, s.spin, a_to_b);
        flag_ping(*s.q1, f0r, pq0, f0, f1r, f1, epoch + 1u, s.spin, b_to_a);
    }
    std::printf("PROBE1 dir=0to1 n=200 one_way_us_min=%.3f p50=%.3f max=%.3f\n",
                *std::min_element(a_to_b.begin(), a_to_b.end()), median(a_to_b),
                *std::max_element(a_to_b.begin(), a_to_b.end()));
    std::printf("PROBE1 dir=1to0 n=200 one_way_us_min=%.3f p50=%.3f max=%.3f\n",
                *std::min_element(b_to_a.begin(), b_to_a.end()), median(b_to_a),
                *std::max_element(b_to_a.begin(), b_to_a.end()));

    const size_t groups_list[] = {1, 4, 16, 64, 256};
    const size_t sizes[] = {2048, 4096, 8192, 16384, 32768, 65536};
    for (size_t bytes : sizes) {
        for (size_t g : groups_list) {
            const size_t words = bytes / 4;
            pq0.memset(s.bw_src0, 1, bytes).wait();
            // Warmup then 7 timed reps. Destination is card 1's bw region.
            push_words(pq0, static_cast<uint32_t*>(s.bw1.remote), s.bw_src0, words, g).wait();
            std::vector<double> us;
            for (int r = 0; r < 7; ++r) {
                sycl::event e = push_words(pq0, static_cast<uint32_t*>(s.bw1.remote), s.bw_src0, words, g);
                e.wait();
                us.push_back(dev_us(e));
            }
            const double med = median(us);
            const double gbps = med > 0 ? (double)bytes / med / 1e3 : 0;
            std::printf("PROBE2 dir=0to1 bytes=%zu groups=%zu us_p50=%.3f GBps=%.3f\n", bytes, g, med, gbps);
        }
    }

    // Concurrent bidirectional 64 KB.
    {
        constexpr size_t bytes = 65536;
        constexpr size_t words = bytes / 4;
        pq0.memset(s.bw_src0, 2, bytes).wait();
        s.q1->memset(s.bw_src1, 3, bytes).wait();
        std::vector<double> u0, u1, wall;
        for (int r = 0; r < 7; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            sycl::event e0 = push_words(pq0, static_cast<uint32_t*>(s.bw1.remote), s.bw_src0, words, 64);
            sycl::event e1 = push_words(*s.q1, static_cast<uint32_t*>(s.bw0.remote), s.bw_src1, words, 64);
            e0.wait();
            e1.wait();
            wall.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            u0.push_back(dev_us(e0));
            u1.push_back(dev_us(e1));
        }
        std::printf("PROBE3 bytes=65536 groups=64 card0_us_p50=%.3f card1_us_p50=%.3f host_wall_us_p50=%.3f\n",
                    median(u0), median(u1), median(wall));
    }

    // Probe 4: 10 KB plain vs uncached after a flag wait.
    {
        constexpr size_t bytes = 10240;
        constexpr size_t words = bytes / 4;
        const uint32_t salt = 0x51u;
        fill(pq0, s.bw_src0, words, kPush, salt);
        pq0.wait();
        auto* dst = static_cast<uint32_t*>(s.bw1.remote);
        push_words(pq0, dst, s.bw_src0, words, 16).wait();
        // Publish a flag after the payload is in the queue (in-order).
        const uint32_t epoch = 900001u;
        pq0.single_task<class b70_m0_p4_flag_k>([=] { sys_store(f1r, epoch); }).wait();
        ep_wait_value(*s.q1, f1, epoch, s.spin, s.err1, 11);
        uint32_t* plain = s.stage1;
        uint32_t* unc = s.stage1 + words;
        auto* src_local = static_cast<uint32_t*>(s.bw1.local);
        sycl::event ep2 = copy_plain(*s.q1, plain, src_local, words);
        ep2.wait();
        sycl::event eu2 = s.q1->parallel_for<class b70_m0_unc_timed_k>(
            sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
                for (size_t i = it.get_global_id(0); i < words; i += 256)
                    unc[i] = load_uncached_u32(src_local + i);
            });
        eu2.wait();
        std::vector<uint32_t> hp(words), hu(words);
        s.q1->memcpy(hp.data(), plain, bytes);
        s.q1->memcpy(hu.data(), unc, bytes);
        s.q1->wait();
        size_t bad_p = 0, bad_u = 0;
        for (size_t i = 0; i < words; ++i) {
            const uint32_t want = pattern(kPush, i, salt);
            bad_p += hp[i] != want;
            bad_u += hu[i] != want;
        }
        std::printf("PROBE4 bytes=%zu plain_bad=%zu uncached_bad=%zu plain_us=%.3f uncached_us=%.3f err1=%u\n", bytes,
                    bad_p, bad_u, dev_us(ep2), dev_us(eu2), s.err1 ? *s.err1 : 0u);
        (void)eu;
    }
    std::fflush(stdout);
}

void m0_teardown() {
    if (!S().up && !S().q1) return;
    free_all();
    std::printf("M0 teardown\n");
    std::fflush(stdout);
}

}  // namespace b70ep
