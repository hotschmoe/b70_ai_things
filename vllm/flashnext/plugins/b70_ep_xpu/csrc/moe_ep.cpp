// Sidecar MoE. Card 0 keeps its expert half and the torch stream. Card 1
// runs the other half on a plugin queue. Transport is the Strata push +
// bounded device flag (ep_kernels.cpp). No oneCCL.
#include "doorbell.hpp"
#include "ep_kernels.hpp"
#include "region.hpp"

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <c10/xpu/XPUStream.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace b70ep {
namespace {

namespace expg = sycl::ext::oneapi::experimental;
using GraphMod = expg::command_graph<expg::graph_state::modifiable>;
using GraphExec = expg::command_graph<expg::graph_state::executable>;

using MoeFn = void (*)(void*, const uint16_t*, const uint8_t*, int, const uint8_t*, int, const uint8_t*, int,
                       const int32_t*, const float*, const uint8_t*, int, const uint8_t*, int, const uint8_t*, int,
                       float*, int, int, int, int, int);
using ReserveFn = void (*)(void*);

struct GraphSlot {
    GraphMod mod;
    std::optional<GraphExec> exec;
    int replays = 0;
    explicit GraphSlot(sycl::queue& q) : mod(q) {}
};

struct Layer {
    const uint8_t* g0 = nullptr;
    const uint8_t* u0 = nullptr;
    const uint8_t* d0 = nullptr;
    const uint8_t* g1 = nullptr;
    const uint8_t* u1 = nullptr;
    const uint8_t* d1 = nullptr;
    int gt = 0, ut = 0, dt = 0;
    int e0 = 0, e1 = 0, I = 0, K = 0;
    bool ready0 = false, ready1 = false;
};

struct State {
    bool up = false;
    int max_t = 0, max_k = 0, max_topk = 0, nlayers = 0;
    uint32_t spin = 200000;
    int path = 1;
    int reclaim = 0;
    std::optional<sycl::queue> q1;
    MoeFn moe = nullptr;
    ReserveFn reserve = nullptr;
    void* so = nullptr;
    Region inbox, psum, flag_in, flag_ret;
    uint32_t* ctr0 = nullptr;
    uint32_t* ctr1 = nullptr;
    uint32_t* done0 = nullptr;
    uint32_t* done1 = nullptr;
    uint32_t* err0 = nullptr;
    uint32_t* err1 = nullptr;
    int32_t* ids0 = nullptr;
    float* y0 = nullptr;
    float* y1 = nullptr;
    int32_t* stage_ids = nullptr;
    float* stage_w = nullptr;
    uint16_t* stage_x = nullptr;
    size_t off_ids = 0, off_w = 0, off_x = 0;
    std::vector<Layer> layers;
    bool cap_on = false;
    bool cap_started = false;
    int cap_key = -1;
    GraphSlot* active = nullptr;
    std::map<int, std::unique_ptr<GraphSlot>> graphs;
    int calls = 0;
};

State& S() {
    static State s;
    return s;
}

sycl::queue& q0() { return c10::xpu::getCurrentXPUStream(0).queue(); }

size_t align256(size_t n) { return (n + 255u) & ~size_t(255); }

int store4(sycl::queue& writer, void* remote, sycl::queue& owner, void* local) {
    owner.memset(local, 0, 4).wait();
    writer.single_task<class b70_ep_store4_k>([=] { *static_cast<uint32_t*>(remote) = 0xA5A5A5A5u; }).wait();
    uint32_t got = 0;
    owner.memcpy(&got, local, 4).wait();
    return got == 0xA5A5A5A5u ? 1 : 0;
}

int choose_path(sycl::queue& a, sycl::queue& b) {
    const bool same = a.get_context() == b.get_context();
    try {
        if (a.get_device().ext_oneapi_can_access_peer(b.get_device(), sycl::ext::oneapi::peer_access::access_supported))
            a.get_device().ext_oneapi_enable_peer_access(b.get_device());
        if (b.get_device().ext_oneapi_can_access_peer(a.get_device(), sycl::ext::oneapi::peer_access::access_supported))
            b.get_device().ext_oneapi_enable_peer_access(a.get_device());
    } catch (const std::exception& ex) {
        std::printf("ep peer enable: %s\n", ex.what());
    }
    int ipc_ok = 0, same_ok = -1;
    {
        Region t;
        std::string err;
        if (region_alloc(b, a, 4096, 1, t, err)) {
            ipc_ok = store4(a, t.remote, b, t.local);
            region_free(t);
        } else {
            std::printf("ep ipc alloc: %s\n", err.c_str());
        }
    }
    if (same) {
        Region t;
        std::string err;
        if (region_alloc(b, a, 4096, 2, t, err)) {
            same_ok = store4(a, t.remote, b, t.local);
            region_free(t);
        }
    }
    std::printf("EP path probe ipc_store=%d same_store=%d same_ctx=%d\n", ipc_ok, same_ok, (int)same);
    std::fflush(stdout);
    if (const char* e = std::getenv("B70_EP_FORCE_PATH")) {
        const int p = std::atoi(e);
        if (p == 1 || p == 2 || p == 3) {
            if (p == 1 && !ipc_ok) throw std::runtime_error("B70_EP_FORCE_PATH=1 but IPC store failed");
            if (p == 2 && same_ok != 1) throw std::runtime_error("B70_EP_FORCE_PATH=2 but same-context store failed");
            return p;
        }
    }
    if (ipc_ok) return 1;
    if (same_ok == 1) return 2;
    if (same) return 3;
    throw std::runtime_error("ep: no working peer pointer");
}

void load_moe() {
    State& s = S();
    const char* env = std::getenv("B70_GGUF_SO");
    const char* fallback =
        "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.ep.so";
    const char* path = (env && env[0]) ? env : fallback;
    s.so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!s.so) throw std::runtime_error(std::string("dlopen ") + path + ": " + dlerror());
    s.moe = reinterpret_cast<MoeFn>(dlsym(s.so, "_Z13b70_moe_layerPvPKtPKhiS3_iS3_iPKiPKfS3_iS3_iS3_iPfiiiii"));
    s.reserve = reinterpret_cast<ReserveFn>(dlsym(s.so, "_Z24b70_reserve_decode_queuePv"));
    if (!s.moe || !s.reserve) throw std::runtime_error("dlsym b70_moe_layer/b70_reserve_decode_queue failed");
    std::printf("EP moe_layer from %s\n", path);
}

void* inbox_at(size_t off) { return static_cast<uint8_t*>(S().inbox.local) + off; }
void* inbox_remote(size_t off) { return static_cast<uint8_t*>(S().inbox.remote) + off; }

}  // namespace

void ep_init(int max_t, int max_k, int max_topk, int nlayers) {
    State& s = S();
    if (s.up) return;
    if (max_t < 1 || max_k < 4 || max_topk < 1 || nlayers < 1)
        throw std::runtime_error("ep_init: bad limits");
    if (const char* e = std::getenv("B70_EP_SPIN")) {
        long v = std::atol(e);
        if (v > 0) s.spin = (uint32_t)v;
    }
    if (const char* e = std::getenv("B70_XPU_CG_RECLAIM")) s.reclaim = std::atoi(e);
    s.max_t = max_t;
    s.max_k = max_k;
    s.max_topk = max_topk;
    s.nlayers = nlayers;
    load_moe();
    sycl::queue& stream0 = q0();
    sycl::queue& stream1 = c10::xpu::getCurrentXPUStream(1).queue();
    sycl::property_list props{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
    s.q1.emplace(stream1.get_context(), stream1.get_device(), props);
    std::printf("EP queues in_order0=%d in_order1=%d\n", (int)stream0.is_in_order(), (int)s.q1->is_in_order());
    s.path = choose_path(stream0, *s.q1);

    s.off_ids = 0;
    s.off_w = align256((size_t)max_t * (size_t)max_topk * 4);
    s.off_x = s.off_w + align256((size_t)max_t * (size_t)max_topk * 4);
    const size_t inbox_b = s.off_x + align256((size_t)max_t * (size_t)max_k * 2);
    const size_t y_b = align256((size_t)max_t * (size_t)max_k * sizeof(float));
    const size_t flag_b = (size_t)nlayers * 4096u;
    std::string err;
    if (!region_alloc(*s.q1, stream0, inbox_b, s.path, s.inbox, err)) throw std::runtime_error("inbox " + err);
    if (!region_alloc(stream0, *s.q1, y_b, s.path, s.psum, err)) throw std::runtime_error("psum " + err);
    if (!region_alloc(*s.q1, stream0, flag_b, s.path, s.flag_in, err)) throw std::runtime_error("flag_in " + err);
    if (!region_alloc(stream0, *s.q1, flag_b, s.path, s.flag_ret, err)) throw std::runtime_error("flag_ret " + err);

    s.ctr0 = sycl::malloc_device<uint32_t>(nlayers, stream0);
    s.done0 = sycl::malloc_device<uint32_t>(1, stream0);
    s.err0 = sycl::malloc_host<uint32_t>(nlayers, stream0);
    s.ids0 = sycl::malloc_device<int32_t>((size_t)max_t * (size_t)max_topk, stream0);
    s.y0 = sycl::malloc_device<float>((size_t)max_t * (size_t)max_k, stream0);
    s.ctr1 = sycl::malloc_device<uint32_t>(nlayers, *s.q1);
    s.done1 = sycl::malloc_device<uint32_t>(1, *s.q1);
    s.err1 = sycl::malloc_host<uint32_t>(nlayers, *s.q1);
    s.y1 = sycl::malloc_device<float>((size_t)max_t * (size_t)max_k, *s.q1);
    s.stage_ids = sycl::malloc_device<int32_t>((size_t)max_t * (size_t)max_topk, *s.q1);
    s.stage_w = sycl::malloc_device<float>((size_t)max_t * (size_t)max_topk, *s.q1);
    s.stage_x = sycl::malloc_device<uint16_t>((size_t)max_t * (size_t)max_k, *s.q1);
    if (!s.ctr0 || !s.done0 || !s.err0 || !s.ids0 || !s.y0 || !s.ctr1 || !s.done1 || !s.err1 || !s.y1 || !s.stage_ids ||
        !s.stage_w || !s.stage_x)
        throw std::runtime_error("ep scratch alloc failed");
    stream0.memset(s.ctr0, 0, (size_t)nlayers * 4);
    stream0.memset(s.done0, 0, 4);
    s.q1->memset(s.ctr1, 0, (size_t)nlayers * 4);
    s.q1->memset(s.done1, 0, 4);
    s.q1->memset(s.flag_in.local, 0, flag_b);
    stream0.memset(s.flag_ret.local, 0, flag_b);
    std::memset(s.err0, 0, (size_t)nlayers * 4);
    std::memset(s.err1, 0, (size_t)nlayers * 4);
    stream0.wait();
    s.q1->wait();
    s.reserve(&stream0);
    s.reserve(&*s.q1);
    stream0.wait();
    s.q1->wait();
    s.layers.assign(nlayers, Layer{});
    s.up = true;
    std::printf("EP init max_t=%d max_k=%d max_topk=%d nlayers=%d spin=%u path=%d reclaim=%d inbox_b=%zu\n", max_t,
                max_k, max_topk, nlayers, s.spin, s.path, s.reclaim, inbox_b);
    std::fflush(stdout);
}

void ep_reserve() {
    State& s = S();
    if (!s.up) throw std::runtime_error("ep_reserve before init");
    sycl::queue& a = q0();
    s.reserve(&a);
    s.reserve(&*s.q1);
}

void ep_register0(int layer, const uint8_t* gate, int gate_type, const uint8_t* up, int up_type, const uint8_t* down,
                  int down_type, int e, int intermediate, int hidden) {
    State& s = S();
    if (!s.up) throw std::runtime_error("ep_register0 before init");
    if (layer < 0 || layer >= s.nlayers) throw std::runtime_error("ep layer out of range");
    Layer& L = s.layers[layer];
    L.g0 = gate;
    L.u0 = up;
    L.d0 = down;
    L.gt = gate_type;
    L.ut = up_type;
    L.dt = down_type;
    L.e0 = e;
    L.I = intermediate;
    L.K = hidden;
    L.ready0 = true;
}

void ep_register1(int layer, const uint8_t* gate, const uint8_t* up, const uint8_t* down, int e) {
    State& s = S();
    if (!s.up) throw std::runtime_error("ep_register1 before init");
    if (layer < 0 || layer >= s.nlayers) throw std::runtime_error("ep layer out of range");
    Layer& L = s.layers[layer];
    L.g1 = gate;
    L.u1 = up;
    L.d1 = down;
    L.e1 = e;
    L.ready1 = true;
}

void ep_capture_begin(int key) {
    State& s = S();
    if (!s.up) throw std::runtime_error("capture before init");
    s.cap_on = true;
    s.cap_started = false;
    s.cap_key = key;
    s.active = nullptr;
}

void ep_capture_end(int key) {
    State& s = S();
    if (!s.cap_on) return;
    if (s.cap_started && s.active) {
        s.active->mod.end_recording(*s.q1);
        s.active->exec = s.active->mod.finalize();
        std::printf("EP card1 graph key=%d finalized\n", key);
    }
    s.cap_on = false;
    s.cap_started = false;
    s.active = nullptr;
    std::fflush(stdout);
}

void ep_replay_prepare(int key) {
    State& s = S();
    if (!s.up) throw std::runtime_error("replay before init");
    auto it = s.graphs.find(key);
    if (it == s.graphs.end() || !it->second->exec) throw std::runtime_error("ep replay missing card1 graph");
    GraphSlot& slot = *it->second;
    slot.replays += 1;
    // Refinalize on the same cadence as d1's XPUGraph::instantiate, before
    // either graph is submitted. The previous step has already synced.
    if (s.reclaim > 0 && (slot.replays % s.reclaim) == 0) {
        slot.exec = slot.mod.finalize();
        std::printf("EP card1 refinalize key=%d replay=%d\n", key, slot.replays);
        std::fflush(stdout);
    }
}

void ep_replay_submit(int key) {
    State& s = S();
    if (!s.up) throw std::runtime_error("replay before init");
    auto it = s.graphs.find(key);
    if (it == s.graphs.end() || !it->second->exec) throw std::runtime_error("ep submit missing card1 graph");
    // Do not wait. Card 0's replay publishes the flag this graph waits on.
    s.q1->ext_oneapi_graph(*it->second->exec);
}

void ep_replay(int key) {
    ep_replay_prepare(key);
    ep_replay_submit(key);
}

std::vector<int64_t> ep_read_errors() {
    State& s = S();
    std::vector<int64_t> out;
    if (!s.up) return out;
    out.reserve((size_t)s.nlayers * 2);
    for (int i = 0; i < s.nlayers; ++i) out.push_back(s.err0[i]);
    for (int i = 0; i < s.nlayers; ++i) out.push_back(s.err1[i]);
    return out;
}

void ep_moe(const uint16_t* x, const float* w, const int32_t* ids, float* y, int T, int topk, int K, int layer) {
    State& s = S();
    if (!s.up) throw std::runtime_error("moe_ep before init");
    if (layer < 0 || layer >= s.nlayers) throw std::runtime_error("moe_ep layer");
    Layer& L = s.layers[layer];
    if (!L.ready0 || !L.ready1) throw std::runtime_error("moe_ep layer not registered");
    if (T < 1 || T > s.max_t || topk < 1 || topk > s.max_topk || K != L.K || (K % 4) != 0)
        throw std::runtime_error("moe_ep shape exceeds the preallocated exchange");
    if (!s.cap_on && (s.err0[layer] || s.err1[layer])) {
        std::printf("EP previous error layer=%d err0=%u err1=%u\n", layer, s.err0[layer], s.err1[layer]);
        std::fflush(stdout);
        throw std::runtime_error("ep flag timeout on a previous call");
    }
    if (s.cap_on && !s.cap_started) {
        auto slot = std::make_unique<GraphSlot>(*s.q1);
        GraphSlot* raw = slot.get();
        s.graphs[s.cap_key] = std::move(slot);
        raw->mod.begin_recording(*s.q1);
        s.active = raw;
        s.cap_started = true;
    }
    const int n_ids = T * topk;
    const size_t ids_b = (size_t)n_ids * 4;
    const size_t w_b = (size_t)n_ids * 4;
    const size_t x_b = (size_t)T * (size_t)K * 2;
    const int64_t n4 = ((int64_t)T * (int64_t)K) / 4;
    uint32_t* ctr0 = s.ctr0 + layer;
    uint32_t* ctr1 = s.ctr1 + layer;
    uint32_t* flag_in = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(s.flag_in.local) + (size_t)layer * 4096);
    uint32_t* flag_in_r = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(s.flag_in.remote) + (size_t)layer * 4096);
    uint32_t* flag_ret = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(s.flag_ret.local) + (size_t)layer * 4096);
    uint32_t* flag_ret_r = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(s.flag_ret.remote) + (size_t)layer * 4096);
    auto* peer_ids = reinterpret_cast<int32_t*>(inbox_remote(s.off_ids));
    auto* peer_w = reinterpret_cast<float*>(inbox_remote(s.off_w));
    auto* peer_x = reinterpret_cast<uint16_t*>(inbox_remote(s.off_x));

    // Card 1 first. Its first node waits, so card 0's push releases it.
    ep_bump(*s.q1, ctr1);
    ep_wait(*s.q1, flag_in, ctr1, s.spin, s.err1 + layer, 11);
    ep_copy_uncached(*s.q1, s.stage_ids, inbox_at(s.off_ids), ids_b);
    ep_copy_uncached(*s.q1, s.stage_w, inbox_at(s.off_w), w_b);
    ep_copy_uncached(*s.q1, s.stage_x, inbox_at(s.off_x), x_b);
    s.q1->memset(s.y1, 0, (size_t)T * (size_t)K * sizeof(float));
    s.moe(&*s.q1, s.stage_x, L.g1, L.gt, L.u1, L.ut, L.d1, L.dt, s.stage_ids, s.stage_w, nullptr, 0, nullptr, 0,
          nullptr, 0, s.y1, T, topk, L.e1, L.I, L.K);
    ep_push_ctr(*s.q1, s.psum.remote, s.y1, n4, s.done1, flag_ret_r, ctr1);

    sycl::queue& a = q0();
    ep_bump(a, ctr0);
    ep_dispatch(a, ids, w, x, s.ids0, peer_ids, peer_w, peer_x, n_ids, x_b, L.e0, L.e1, s.done0, flag_in_r, ctr0);
    a.memset(s.y0, 0, (size_t)T * (size_t)K * sizeof(float));
    s.moe(&a, x, L.g0, L.gt, L.u0, L.ut, L.d0, L.dt, s.ids0, w, nullptr, 0, nullptr, 0, nullptr, 0, s.y0, T, topk,
          L.e0, L.I, L.K);
    ep_wait(a, flag_ret, ctr0, s.spin, s.err0 + layer, 12);
    ep_add_uncached(a, y, s.y0, static_cast<float*>(s.psum.local), n4);
    s.calls += 1;
}

}  // namespace b70ep
