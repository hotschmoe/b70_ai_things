// Sidecar MoE. Card 0 keeps its expert half and the torch stream. Card 1
// runs the other half on a plugin queue. Transport is the Strata push +
// bounded device flag (ep_kernels.cpp). No oneCCL.
#include "doorbell.hpp"
#include "ep_kernels.hpp"
#include "region.hpp"
#include "ep_api.hpp"

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
// O4: b70_moe_layer2 (O3 stage 3): routed + shared expert + sigmoid gate.
using Moe2Fn = void (*)(void*, const uint16_t*, const uint8_t*, int, const uint8_t*, int, const uint8_t*, int,
                        const int32_t*, const float*, const uint8_t*, int, const uint8_t*, int, const uint8_t*, int,
                        void*, int, int, int, int, int, const void*, int, int);

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
    const int32_t* map0 = nullptr;
    const int32_t* map1 = nullptr;
    int nexp = 0;
    bool ready0 = false, ready1 = false;
    // Last eager shape compiled outside the flag spin. 0 means none.
    int eager_T = 0;
    int eager_topk = 0;
    // O4: card-0 shared expert for the fused call (ep_register_shared).
    bool has_shared = false;
    const uint8_t* sg = nullptr;
    const uint8_t* su = nullptr;
    const uint8_t* sd = nullptr;
    int sgt = 0, sut = 0, sdt = 0;
    const void* sgi = nullptr;
    int sgi_dt = 0;
};

struct State {
    bool up = false;
    int max_t = 0, max_k = 0, max_topk = 0, nlayers = 0;
    uint32_t spin = 200000;
    int path = 1;
    int reclaim = 0;
    std::optional<sycl::queue> q1;
    MoeFn moe = nullptr;
    Moe2Fn moe2 = nullptr;
    bool with_shared = false;
    bool out_bf16 = false;
    // O4 route mode (moe_ep_route): bf16 router logits [T, route_E].
    const uint16_t* route_logits = nullptr;
    int route_E = 0;
    int route_renorm = 1;
    float* w0 = nullptr;
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
    bool implicit_cap = false;
    int cap_key = -1;
    int last_ep = -1;
    GraphSlot* active = nullptr;
    std::map<int, std::unique_ptr<GraphSlot>> graphs;
    int calls = 0;
    size_t inbox_b = 0;
    size_t psum_b = 0;
    size_t flag_b = 0;
    bool compute_warm = false;
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
    s.moe2 = reinterpret_cast<Moe2Fn>(
        dlsym(s.so, "_Z14b70_moe_layer2PvPKtPKhiS3_iS3_iPKiPKfS3_iS3_iS3_iS_iiiiiPKvii"));
    std::printf("EP moe_layer2 %s\n", s.moe2 ? "found" : "absent");
    std::printf("EP moe_layer from %s\n", path);
}

void* inbox_at(size_t off) { return static_cast<uint8_t*>(S().inbox.local) + off; }
void* inbox_remote(size_t off) { return static_cast<uint8_t*>(S().inbox.remote) + off; }

void eager_prejit_layer(Layer& L, int layer, const uint16_t* x, const float* w, const int32_t* ids, int T, int topk);

}  // namespace

void ep_transport_warmup();
void ep_compute_warmup(int layer);

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
    s.inbox_b = s.off_x + align256((size_t)max_t * (size_t)max_k * 2);
    s.psum_b = align256((size_t)max_t * (size_t)max_k * sizeof(float));
    s.flag_b = (size_t)nlayers * 4096u;
    const size_t inbox_b = s.inbox_b;
    const size_t y_b = s.psum_b;
    const size_t flag_b = s.flag_b;
    std::string err;
    if (!region_alloc(*s.q1, stream0, inbox_b, s.path, s.inbox, err)) throw std::runtime_error("inbox " + err);
    if (!region_alloc(stream0, *s.q1, y_b, s.path, s.psum, err)) throw std::runtime_error("psum " + err);
    if (!region_alloc(*s.q1, stream0, flag_b, s.path, s.flag_in, err)) throw std::runtime_error("flag_in " + err);
    if (!region_alloc(stream0, *s.q1, flag_b, s.path, s.flag_ret, err)) throw std::runtime_error("flag_ret " + err);

    s.ctr0 = sycl::malloc_device<uint32_t>(nlayers, stream0);
    s.done0 = sycl::malloc_device<uint32_t>(1, stream0);
    s.err0 = sycl::malloc_host<uint32_t>(nlayers, stream0);
    s.ids0 = sycl::malloc_device<int32_t>((size_t)max_t * (size_t)max_topk, stream0);
    s.w0 = sycl::malloc_device<float>((size_t)max_t * (size_t)max_topk, stream0);
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
    ep_transport_warmup();
}

// F1: prefill_ep.cpp allocates its own regions with the path ep_init probed.
int ep_transport_path() { return S().up ? S().path : 0; }

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

void ep_register_shared(int layer, const uint8_t* sg, int sgt, const uint8_t* su, int sut, const uint8_t* sd, int sdt,
                        const void* sgi, int sgi_dt) {
    State& s = S();
    if (!s.up) throw std::runtime_error("ep_register_shared before init");
    if (layer < 0 || layer >= s.nlayers) throw std::runtime_error("ep layer out of range");
    Layer& L = s.layers[layer];
    L.sg = sg;
    L.su = su;
    L.sd = sd;
    L.sgt = sgt;
    L.sut = sut;
    L.sdt = sdt;
    L.sgi = sgi;
    L.sgi_dt = sgi_dt;
    L.has_shared = sg != nullptr && su != nullptr && sd != nullptr;
}

void ep_moe_sh(const uint16_t* x, const float* w, const int32_t* ids, float* y, int T, int topk, int K, int layer,
               bool bf16_out) {
    State& s = S();
    if (!s.moe2) throw std::runtime_error("moe_ep_sh: the MoE library has no b70_moe_layer2");
    s.with_shared = true;
    s.out_bf16 = bf16_out;
    try {
        ep_moe(x, w, ids, y, T, topk, K, layer);
    } catch (...) {
        s.with_shared = false;
        s.out_bf16 = false;
        throw;
    }
    s.with_shared = false;
    s.out_bf16 = false;
}

bool stream_recording();

void ep_moe_route(const uint16_t* x, const uint16_t* logits, uint16_t* y, int T, int E, int topk, int K, int layer,
                  int renorm) {
    State& s = S();
    if (!s.moe2) throw std::runtime_error("moe_ep_route: the MoE library has no b70_moe_layer2");
    if (T < 1 || T > 8 || E < 1 || E > 1024 || topk < 1 || topk > 16) throw std::runtime_error("moe_ep_route shape");
    if (!stream_recording()) {
        // Eager (warmup) prejit reads ids0/w0 before the route kernel runs.
        q0().memset(s.ids0, 0, (size_t)T * (size_t)topk * 4);
        q0().memset(s.w0, 0, (size_t)T * (size_t)topk * 4);
    }
    s.with_shared = true;
    s.out_bf16 = true;
    s.route_logits = logits;
    s.route_E = E;
    s.route_renorm = renorm;
    try {
        ep_moe(x, s.w0, s.ids0, reinterpret_cast<float*>(y), T, topk, K, layer);
    } catch (...) {
        s.with_shared = false;
        s.out_bf16 = false;
        s.route_logits = nullptr;
        throw;
    }
    s.with_shared = false;
    s.out_bf16 = false;
    s.route_logits = nullptr;
}

void ep_register_map(int layer, const int32_t* map0, const int32_t* map1, int nexp) {
    State& s = S();
    if (!s.up) throw std::runtime_error("ep_register_map before init");
    if (layer < 0 || layer >= s.nlayers) throw std::runtime_error("ep layer out of range");
    if (!map0 || !map1 || nexp < 1 || nexp > 4096) throw std::runtime_error("ep_register_map: empty map");
    Layer& L = s.layers[layer];
    if (L.e0 < 1 || L.e1 < 1) throw std::runtime_error("ep_register_map before expert counts");
    // Own the bytes. The caller's tensor can move; the kernel reads this copy.
    sycl::queue& q = q0();
    auto* own0 = sycl::malloc_device<int32_t>((size_t)nexp, q);
    auto* own1 = sycl::malloc_device<int32_t>((size_t)nexp, q);
    if (own0 == nullptr || own1 == nullptr) {
        if (own0) sycl::free(own0, q);
        if (own1) sycl::free(own1, q);
        throw std::runtime_error("ep_register_map: device alloc failed");
    }
    std::vector<int32_t> host0((size_t)nexp), host1((size_t)nexp);
    q.memcpy(host0.data(), map0, (size_t)nexp * sizeof(int32_t)).wait();
    q.memcpy(host1.data(), map1, (size_t)nexp * sizeof(int32_t)).wait();
    std::vector<uint8_t> seen0((size_t)L.e0, 0), seen1((size_t)L.e1, 0);
    int n_on0 = 0, n_on1 = 0;
    for (int i = 0; i < nexp; ++i) {
        const int a = host0[(size_t)i];
        const int b = host1[(size_t)i];
        const bool on0 = a >= 0;
        const bool on1 = b >= 0;
        if ((on0 && a >= L.e0) || (on1 && b >= L.e1) || on0 == on1) {
            sycl::free(own0, q);
            sycl::free(own1, q);
            throw std::runtime_error("ep_register_map: slot out of range or not on one card");
        }
        if (on0) {
            if (seen0[(size_t)a]) {
                sycl::free(own0, q);
                sycl::free(own1, q);
                throw std::runtime_error("ep_register_map: duplicate card 0 slot");
            }
            seen0[(size_t)a] = 1;
            n_on0 += 1;
        } else {
            if (seen1[(size_t)b]) {
                sycl::free(own0, q);
                sycl::free(own1, q);
                throw std::runtime_error("ep_register_map: duplicate card 1 slot");
            }
            seen1[(size_t)b] = 1;
            n_on1 += 1;
        }
    }
    if (n_on0 != L.e0 || n_on1 != L.e1) {
        sycl::free(own0, q);
        sycl::free(own1, q);
        throw std::runtime_error("ep_register_map: slot set does not match the expert counts");
    }
    q.memcpy(own0, host0.data(), (size_t)nexp * sizeof(int32_t)).wait();
    q.memcpy(own1, host1.data(), (size_t)nexp * sizeof(int32_t)).wait();
    if (L.map0 != nullptr) sycl::free(const_cast<int32_t*>(L.map0), q);
    if (L.map1 != nullptr) sycl::free(const_cast<int32_t*>(L.map1), q);
    L.map0 = own0;
    L.map1 = own1;
    L.nexp = nexp;
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
    if (layer > s.last_ep) s.last_ep = layer;
    if (L.ready0 && !s.compute_warm) ep_compute_warmup(layer);
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

void enqueue_moe(const uint16_t* x, const float* w, const int32_t* ids, float* y, int T, int topk, int K, int layer,
                 uint32_t spin) {
    State& s = S();
    Layer& L = s.layers[layer];
    if (s.cap_on && !s.cap_started) {
        // Profile capture and the real capture reuse token-count keys.
        s.graphs.erase(s.cap_key);
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
    ep_wait(*s.q1, flag_in, ctr1, spin, s.err1 + layer, 11);
    ep_copy_uncached(*s.q1, s.stage_ids, inbox_at(s.off_ids), ids_b);
    ep_copy_uncached(*s.q1, s.stage_w, inbox_at(s.off_w), w_b);
    ep_copy_uncached(*s.q1, s.stage_x, inbox_at(s.off_x), x_b);
    s.q1->memset(s.y1, 0, (size_t)T * (size_t)K * sizeof(float));
    s.moe(&*s.q1, s.stage_x, L.g1, L.gt, L.u1, L.ut, L.d1, L.dt, s.stage_ids, s.stage_w, nullptr, 0, nullptr, 0,
          nullptr, 0, s.y1, T, topk, L.e1, L.I, L.K);
    ep_push_ctr(*s.q1, s.psum.remote, s.y1, n4, s.done1, flag_ret_r, ctr1);

    sycl::queue& a = q0();
    if ((L.map0 == nullptr) != (L.map1 == nullptr)) throw std::runtime_error("ep map is half set");
    if (s.route_logits != nullptr) {
        // O4: router top-k, bump and dispatch in one kernel; card 0 reads w0.
        ep_route_dispatch(a, s.route_logits, T, s.route_E, topk, s.route_renorm, s.w0, s.ids0, peer_ids, peer_w, x,
                          peer_x, x_b, L.e0, L.e1, L.map0, L.map1, L.nexp, flag_in_r, ctr0);
        w = s.w0;
    } else {
        ep_bump(a, ctr0);
        ep_dispatch(a, ids, w, x, s.ids0, peer_ids, peer_w, peer_x, n_ids, x_b, L.e0, L.e1, L.map0, L.map1, L.nexp,
                    s.done0, flag_in_r, ctr0);
    }
    const bool fused_sh = s.with_shared && L.has_shared && s.moe2 != nullptr && T <= 8;
    // b70_moe_layer2 runs the fused decode schedule, whose reduce writes
    // every element of y0; the zero fill is only needed by the old path.
    if (!fused_sh) a.memset(s.y0, 0, (size_t)T * (size_t)K * sizeof(float));
    if (fused_sh) {
        // y0 = card-0 routed + sigmoid(x . sgi) * shared, fp32 (out_mode 0).
        s.moe2(&a, x, L.g0, L.gt, L.u0, L.ut, L.d0, L.dt, s.ids0, w, L.sg, L.sgt, L.su, L.sut, L.sd, L.sdt,
               static_cast<void*>(s.y0), T, topk, L.e0, L.I, L.K, L.sgi, L.sgi_dt, 0);
    } else {
        s.moe(&a, x, L.g0, L.gt, L.u0, L.ut, L.d0, L.dt, s.ids0, w, nullptr, 0, nullptr, 0, nullptr, 0, s.y0, T,
              topk, L.e0, L.I, L.K);
    }
    if (y == nullptr) return;  // O4 split: ep_moe_end enqueues the wait and the sum.
    ep_wait(a, flag_ret, ctr0, spin, s.err0 + layer, 12);
    if (s.out_bf16)
        ep_add_uncached_bf16(a, reinterpret_cast<uint16_t*>(y), s.y0, static_cast<float*>(s.psum.local), n4);
    else
        ep_add_uncached(a, y, s.y0, static_cast<float*>(s.psum.local), n4);
    s.calls += 1;
}

// O4: the second half of enqueue_moe. Card-0 work queued between the two
// halves (the shared expert) runs while card 1 computes its experts.
void enqueue_moe_end(float* y, int T, int K, int layer, uint32_t spin) {
    State& s = S();
    const int64_t n4 = ((int64_t)T * (int64_t)K) / 4;
    uint32_t* ctr0 = s.ctr0 + layer;
    uint32_t* flag_ret = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(s.flag_ret.local) + (size_t)layer * 4096);
    sycl::queue& a = q0();
    ep_wait(a, flag_ret, ctr0, spin, s.err0 + layer, 12);
    ep_add_uncached(a, y, s.y0, static_cast<float*>(s.psum.local), n4);
    s.calls += 1;
}

bool stream_recording() {
    return q0().ext_oneapi_get_state() == expg::queue_state::recording;
}

namespace {
void ep_moe_tail(int layer, bool recording) {
    State& s = S();
    if (s.cap_on && s.implicit_cap && layer == s.last_ep && s.cap_started) {
        ep_capture_end(s.cap_key);
        s.implicit_cap = false;
    }
    if (s.cap_on || recording) return;
    s.q1->wait();
    q0().wait();
    if (s.err0[layer] || s.err1[layer]) {
        std::printf("EP flag timeout layer=%d err0=%u err1=%u\n", layer, s.err0[layer], s.err1[layer]);
        std::fflush(stdout);
        throw std::runtime_error("ep flag timeout");
    }
}
}  // namespace

void ep_moe(const uint16_t* x, const float* w, const int32_t* ids, float* y, int T, int topk, int K, int layer) {
    State& s = S();
    if (!s.up) throw std::runtime_error("moe_ep before init");
    if (layer < 0 || layer >= s.nlayers) throw std::runtime_error("moe_ep layer");
    Layer& L = s.layers[layer];
    if (!L.ready0 || !L.ready1) throw std::runtime_error("moe_ep layer not registered");
    if (T < 1 || T > s.max_t || topk < 1 || topk > s.max_topk || K != L.K || (K % 4) != 0)
        throw std::runtime_error("moe_ep shape exceeds the preallocated exchange");
    // FULL capture calls the model with CUDAGraphMode.NONE, so the Python
    // hook never sets cap_on. queue.wait inside that capture is Level Zero
    // error 44 (UR_RESULT_ERROR_UNSUPPORTED_FEATURE).
    const bool recording = stream_recording();
    if (recording && !s.cap_on) {
        s.cap_on = true;
        s.cap_started = false;
        s.cap_key = T;
        s.implicit_cap = true;
        std::printf("EP capture implicit key=%d\n", T);
        std::fflush(stdout);
    }
    if (!s.cap_on && (s.err0[layer] || s.err1[layer])) {
        std::printf("EP previous error layer=%d err0=%u err1=%u\n", layer, s.err0[layer], s.err1[layer]);
        std::fflush(stdout);
        throw std::runtime_error("ep flag timeout on a previous call");
    }
    // Profile runs max_num_batched_tokens with attention metadata skipped.
    // enqueue_moe arms card 1's flag_in wait before card 0 posts. A host
    // JIT inside the card-1 moe launch, or the projections already queued
    // on card 0, burns that spin. Flash-Next's first profile died at layer
    // 2 with err0=0 err1=11 after layers 0 and 1 returned. Compile this
    // shape and drain card 0 before the wait is armed. Capture must not
    // host-wait (Level Zero 44) and records the pipelined chain as-is.
    if (!s.cap_on && !recording) eager_prejit_layer(L, layer, x, w, ids, T, topk);
    // Card 1 has 374 experts against card 0's 138. The flag_ret spin starts
    // only after card 0's own experts finish, so it must cover the rest of
    // card 1. Profile T=2048 topk=10 passed layers 0 and 1 at spin 200000
    // and died on layer 2 with err0=12 err1=0. Eager uses a wider budget.
    // Capture keeps s.spin; the recorded decode shapes are small.
    uint32_t spin = s.spin;
    if (!s.cap_on && !recording && spin < 10000000u) spin = 10000000u;
    enqueue_moe(x, w, ids, y, T, topk, K, layer, spin);
    if (s.cap_on && s.implicit_cap && layer == s.last_ep && s.cap_started) {
        ep_capture_end(s.cap_key);
        s.implicit_cap = false;
    }
    // ep_capture_end clears cap_on. The card-0 stream is still recording,
    // so the last layer must not host-wait either.
    if (s.cap_on || recording) return;
    // Card 1's next-layer wait is queued as soon as its own previous layer
    // push finishes, so it would spin through card 0's expert compute.
    // M2's profile forward left layer 0 clean and layer 1 err1=11. A host
    // wait here makes the next wait cover only the attention already queued
    // ahead of that layer's dispatch. Capture records the pipelined chain
    // and does not execute it.
    s.q1->wait();
    q0().wait();
    if (s.err0[layer] || s.err1[layer]) {
        std::printf("EP flag timeout layer=%d err0=%u err1=%u\n", layer, s.err0[layer], s.err1[layer]);
        std::fflush(stdout);
        throw std::runtime_error("ep flag timeout");
    }
}

// O4 split pair. ep_moe_begin queues everything up to card 0's own experts;
// ep_moe_end queues the flag wait and the sum into y, then does what ep_moe
// does after enqueue (implicit capture end, eager host wait). The caller
// queues independent card-0 work (the shared expert) between them.
void ep_moe_begin(const uint16_t* x, const float* w, const int32_t* ids, int T, int topk, int K, int layer) {
    State& s = S();
    if (!s.up) throw std::runtime_error("moe_ep before init");
    if (layer < 0 || layer >= s.nlayers) throw std::runtime_error("moe_ep layer");
    Layer& L = s.layers[layer];
    if (!L.ready0 || !L.ready1) throw std::runtime_error("moe_ep layer not registered");
    if (T < 1 || T > s.max_t || topk < 1 || topk > s.max_topk || K != L.K || (K % 4) != 0)
        throw std::runtime_error("moe_ep shape exceeds the preallocated exchange");
    const bool recording = stream_recording();
    if (recording && !s.cap_on) {
        s.cap_on = true;
        s.cap_started = false;
        s.cap_key = T;
        s.implicit_cap = true;
        std::printf("EP capture implicit key=%d\n", T);
        std::fflush(stdout);
    }
    if (!s.cap_on && (s.err0[layer] || s.err1[layer])) {
        std::printf("EP previous error layer=%d err0=%u err1=%u\n", layer, s.err0[layer], s.err1[layer]);
        std::fflush(stdout);
        throw std::runtime_error("ep flag timeout on a previous call");
    }
    if (!s.cap_on && !recording) eager_prejit_layer(L, layer, x, w, ids, T, topk);
    uint32_t spin = s.spin;
    if (!s.cap_on && !recording && spin < 10000000u) spin = 10000000u;
    enqueue_moe(x, w, ids, nullptr, T, topk, K, layer, spin);
}

void ep_moe_end(float* y, int T, int K, int layer) {
    State& s = S();
    if (!s.up) throw std::runtime_error("moe_ep_end before init");
    if (layer < 0 || layer >= s.nlayers) throw std::runtime_error("moe_ep_end layer");
    const bool recording = stream_recording();
    uint32_t spin = s.spin;
    if (!s.cap_on && !recording && spin < 10000000u) spin = 10000000u;
    enqueue_moe_end(y, T, K, layer, spin);
    ep_moe_tail(layer, recording);
}

namespace {

int touch_twice(sycl::queue& writer, void* remote, sycl::queue& owner, void* local, size_t bytes) {
    int ok = ep_touch_pages(writer, remote, owner, local, bytes);
    if (!ok) ok = ep_touch_pages(writer, remote, owner, local, bytes);
    return ok;
}

void reset_epochs() {
    State& s = S();
    sycl::queue& a = q0();
    a.memset(s.ctr0, 0, (size_t)s.nlayers * 4);
    a.memset(s.done0, 0, 4);
    a.memset(s.flag_ret.local, 0, s.flag_b);
    s.q1->memset(s.ctr1, 0, (size_t)s.nlayers * 4);
    s.q1->memset(s.done1, 0, 4);
    s.q1->memset(s.flag_in.local, 0, s.flag_b);
    std::memset(s.err0, 0, (size_t)s.nlayers * 4);
    std::memset(s.err1, 0, (size_t)s.nlayers * 4);
    a.wait();
    s.q1->wait();
}

void prejit_moe(sycl::queue& q, const uint16_t* x, const int32_t* ids, const float* w, float* y, const Layer& L,
                const uint8_t* g, const uint8_t* u, const uint8_t* d, int T, int E, int topk) {
    State& s = S();
    q.memset(y, 0, (size_t)T * (size_t)L.K * sizeof(float));
    s.moe(&q, x, g, L.gt, u, L.ut, d, L.dt, ids, w, nullptr, 0, nullptr, 0, nullptr, 0, y, T, topk, E, L.I, L.K);
    q.wait();
}

void eager_prejit_layer(Layer& L, int layer, const uint16_t* x, const float* w, const int32_t* ids, int T, int topk) {
    if (L.eager_T == T && L.eager_topk == topk) return;
    State& s = S();
    sycl::queue& a = q0();
    const size_t n_ids = (size_t)T * (size_t)topk;
    // Card 1 has no real ids yet. Zero its stage so the compile launch
    // does not index off the end of the expert pack.
    s.q1->memset(s.stage_ids, 0, n_ids * 4);
    s.q1->memset(s.stage_w, 0, n_ids * 4);
    s.q1->memset(s.stage_x, 0, (size_t)T * (size_t)L.K * 2);
    s.q1->wait();
    std::printf("EP eager prejit layer=%d T=%d topk=%d\n", layer, T, topk);
    std::fflush(stdout);
    prejit_moe(*s.q1, s.stage_x, s.stage_ids, s.stage_w, s.y1, L, L.g1, L.u1, L.d1, T, L.e1, topk);
    prejit_moe(a, x, ids, w, s.y0, L, L.g0, L.u0, L.d0, T, L.e0, topk);
    L.eager_T = T;
    L.eager_topk = topk;
}

}  // namespace

void ep_transport_warmup() {
    State& s = S();
    if (!s.up) throw std::runtime_error("transport warmup before init");
    sycl::queue& a = q0();
    const int pages = (int)(s.flag_b / 4096u);
    const int ti = touch_twice(a, s.inbox.remote, *s.q1, s.inbox.local, s.inbox_b);
    const int tp = touch_twice(*s.q1, s.psum.remote, a, s.psum.local, s.psum_b);
    const int tf1 = touch_twice(a, s.flag_in.remote, *s.q1, s.flag_in.local, s.flag_b);
    const int tf0 = touch_twice(*s.q1, s.flag_ret.remote, a, s.flag_ret.local, s.flag_b);
    std::printf("EP WARMUP touch inbox=%d psum=%d flag_in=%d flag_ret=%d pages=%d\n", ti, tp, tf1, tf0, pages);
    std::fflush(stdout);
    if (!(ti && tp && tf1 && tf0)) throw std::runtime_error("ep warmup peer-page touch failed");
    ep_store_flag_pages(a, static_cast<uint32_t*>(s.flag_in.remote), pages, 1u);
    ep_store_flag_pages(*s.q1, static_cast<uint32_t*>(s.flag_ret.remote), pages, 1u);
    reset_epochs();

    int32_t* ids = sycl::malloc_device<int32_t>(1, a);
    float* w = sycl::malloc_device<float>(1, a);
    uint16_t* x = sycl::malloc_device<uint16_t>(64, a);
    if (!ids || !w || !x) throw std::runtime_error("ep warmup alloc failed");
    a.memset(ids, 0, 4);
    a.memset(w, 0, 4);
    a.memset(x, 0, 128);
    a.memset(s.y0, 0, 16);
    s.q1->memset(s.y1, 0, 16);
    a.wait();
    s.q1->wait();

    uint32_t* flag_in = static_cast<uint32_t*>(s.flag_in.local);
    uint32_t* flag_in_r = static_cast<uint32_t*>(s.flag_in.remote);
    uint32_t* flag_ret = static_cast<uint32_t*>(s.flag_ret.local);
    uint32_t* flag_ret_r = static_cast<uint32_t*>(s.flag_ret.remote);
    auto* peer_ids = reinterpret_cast<int32_t*>(inbox_remote(s.off_ids));
    auto* peer_w = reinterpret_cast<float*>(inbox_remote(s.off_w));
    auto* peer_x = reinterpret_cast<uint16_t*>(inbox_remote(s.off_x));
    int ok = 0;
    for (int attempt = 1; attempt <= 2; ++attempt) {
        s.err0[0] = 0;
        s.err1[0] = 0;
        a.memset(s.done0, 0, 4);
        s.q1->memset(s.done1, 0, 4);
        a.wait();
        s.q1->wait();
        ep_bump(*s.q1, s.ctr1);
        ep_wait(*s.q1, flag_in, s.ctr1, s.spin, s.err1, 11);
        ep_push_ctr(*s.q1, s.psum.remote, s.y1, 1, s.done1, flag_ret_r, s.ctr1);
        ep_bump(a, s.ctr0);
        ep_dispatch(a, ids, w, x, s.ids0, peer_ids, peer_w, peer_x, 1, 64, 1, 1, nullptr, nullptr, 0, s.done0,
                    flag_in_r, s.ctr0);
        ep_wait(a, flag_ret, s.ctr0, s.spin, s.err0, 12);
        s.q1->wait();
        a.wait();
        std::printf("EP WARMUP exchange attempt=%d err0=%u err1=%u\n", attempt, s.err0[0], s.err1[0]);
        std::fflush(stdout);
        if (s.err0[0] == 0 && s.err1[0] == 0) {
            ok = 1;
            break;
        }
    }
    sycl::free(ids, a);
    sycl::free(w, a);
    sycl::free(x, a);
    reset_epochs();
    std::printf("EP WARMUP reset epochs ok=%d\n", ok);
    std::fflush(stdout);
    if (!ok) throw std::runtime_error("ep warmup exchange failed");
}

void ep_compute_warmup(int layer) {
    State& s = S();
    if (s.compute_warm) return;
    if (!s.up) throw std::runtime_error("compute warmup before init");
    Layer& L = s.layers[layer];
    if (!L.ready0 || !L.ready1 || L.K < 32) throw std::runtime_error("compute warmup layer is not ready");
    sycl::queue& a = q0();
    const int Tpre = s.max_t >= 16 ? 16 : 1;
    const int Talloc = Tpre;
    int32_t* ids = sycl::malloc_device<int32_t>(Talloc, a);
    float* w = sycl::malloc_device<float>(Talloc, a);
    uint16_t* x = sycl::malloc_device<uint16_t>((size_t)Talloc * (size_t)L.K, a);
    float* y = sycl::malloc_device<float>((size_t)Talloc * (size_t)L.K, a);
    if (!ids || !w || !x || !y) throw std::runtime_error("compute warmup alloc failed");
    a.memset(ids, 0, (size_t)Talloc * 4);
    a.memset(w, 0, (size_t)Talloc * 4);
    a.memset(x, 0, (size_t)Talloc * (size_t)L.K * 2);
    s.q1->memset(s.stage_ids, 0, (size_t)Talloc * 4);
    s.q1->memset(s.stage_w, 0, (size_t)Talloc * 4);
    s.q1->memset(s.stage_x, 0, (size_t)Talloc * (size_t)L.K * 2);
    a.wait();
    s.q1->wait();
    // JIT each card's expert kernel on its own queue, outside the flag spin.
    std::printf("EP WARMUP prejit layer=%d T=%d\n", layer, 1);
    std::fflush(stdout);
    prejit_moe(*s.q1, s.stage_x, s.stage_ids, s.stage_w, s.y1, L, L.g1, L.u1, L.d1, 1, L.e1, 1);
    prejit_moe(a, x, ids, w, y, L, L.g0, L.u0, L.d0, 1, L.e0, 1);
    if (Tpre > 1) {
        std::printf("EP WARMUP prejit layer=%d T=%d\n", layer, Tpre);
        std::fflush(stdout);
        prejit_moe(*s.q1, s.stage_x, s.stage_ids, s.stage_w, s.y1, L, L.g1, L.u1, L.d1, Tpre, L.e1, 1);
        prejit_moe(a, x, ids, w, y, L, L.g0, L.u0, L.d0, Tpre, L.e0, 1);
    }
    reset_epochs();
    int ok = 0;
    for (int attempt = 1; attempt <= 2; ++attempt) {
        s.err0[layer] = 0;
        s.err1[layer] = 0;
        a.memset(s.done0, 0, 4);
        s.q1->memset(s.done1, 0, 4);
        a.wait();
        s.q1->wait();
        enqueue_moe(x, w, ids, y, 1, 1, L.K, layer, s.spin);
        s.q1->wait();
        a.wait();
        std::printf("EP WARMUP moe attempt=%d layer=%d err0=%u err1=%u\n", attempt, layer, s.err0[layer],
                    s.err1[layer]);
        std::fflush(stdout);
        if (s.err0[layer] == 0 && s.err1[layer] == 0) {
            ok = 1;
            break;
        }
    }
    sycl::free(ids, a);
    sycl::free(w, a);
    sycl::free(x, a);
    sycl::free(y, a);
    reset_epochs();
    std::printf("EP WARMUP moe reset ok=%d\n", ok);
    std::fflush(stdout);
    if (!ok) throw std::runtime_error("ep compute warmup exchange failed");
    s.compute_warm = true;
}

}  // namespace b70ep
