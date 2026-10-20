// F1: sidecar PREFILL transport. Eager only (T > 8), never captured.
//
// Card 0 pushes the whole chunk (bf16 x, fp32 router weights, int32 ids
// remapped to card 1's local slots) into card 1's inbox with ep_dispatch
// and raises a device flag. Card 1 (torch's current stream on xpu:1)
// waits on that flag, copies the inbox uncached into fresh torch tensors
// and hands them to Python, which runs the fast grouped expert kernels
// on card 1. ep_prefill_return pushes card 1's fp32 (or bf16) partial
// sum into card 0's psum region and card 0 adds it to its own partial.
//
// All queues are torch's current streams (c10::xpu::getCurrentXPUStream(i)),
// so torch ops on either card order naturally against these kernels.
// The decode sidecar (moe_ep.cpp) keeps its own regions, counters and
// plugin queue; nothing here touches them.
#include "doorbell.hpp"
#include "ep_kernels.hpp"
#include "region.hpp"
#include "ep_api.hpp"

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace b70ep {
namespace {

struct PState {
    bool up = false;
    int max_t = 0, K = 0, max_topk = 0, nlayers = 0;
    uint32_t spin = 1000000000u;
    int path = 0;
    Region inbox, psum, flag_in, flag_ret;
    size_t inbox_b = 0, psum_b = 0, flag_b = 0;
    size_t off_ids = 0, off_w = 0, off_x = 0;
    uint32_t* ctr0 = nullptr;
    uint32_t* ctr1 = nullptr;
    uint32_t* done0 = nullptr;
    uint32_t* done1 = nullptr;
    uint32_t* err0 = nullptr;
    uint32_t* err1 = nullptr;
    long calls = 0;
};

PState& P() {
    static PState s;
    return s;
}

sycl::queue& stream_q(int dev) { return c10::xpu::getCurrentXPUStream((c10::DeviceIndex)dev).queue(); }

size_t align256(size_t n) { return (n + 255u) & ~size_t(255); }

uint32_t* flag_at(const Region& r, bool remote, int layer) {
    return reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(remote ? r.remote : r.local) + (size_t)layer * 4096);
}

// y[i] = a[i] + float(uncached bf16 b[i]). n2 = number of bf16 pairs (u32 words).
void add_uncached_bf16(sycl::queue& q, float* y, const float* a, const uint16_t* b, int64_t n2) {
    if (n2 <= 0) return;
    constexpr size_t WG = 256;
    const size_t groups = std::min<size_t>((size_t)((n2 + (int64_t)WG - 1) / (int64_t)WG), 256);
    const uint32_t* bw = reinterpret_cast<const uint32_t*>(b);
    q.parallel_for<class b70_ep_pf_add_bf16_k>(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
        for (int64_t i = (int64_t)it.get_global_id(0); i < n2; i += (int64_t)(groups * WG)) {
            const uint32_t w = load_uncached_u32(bw + i);
            const float lo = sycl::bit_cast<float>(w << 16);
            const float hi = sycl::bit_cast<float>(w & 0xFFFF0000u);
            y[2 * i] = a[2 * i] + lo;
            y[2 * i + 1] = a[2 * i + 1] + hi;
        }
    });
}

int touch_twice(sycl::queue& writer, void* remote, sycl::queue& owner, void* local, size_t bytes) {
    int ok = ep_touch_pages(writer, remote, owner, local, bytes);
    if (!ok) ok = ep_touch_pages(writer, remote, owner, local, bytes);
    return ok;
}

void pf_init(int max_t, int k, int max_topk, int nlayers) {
    PState& s = P();
    if (s.up) {
        if (max_t > s.max_t || k != s.K || max_topk > s.max_topk || nlayers > s.nlayers)
            throw std::runtime_error("ep_prefill_init: already initialized with smaller limits");
        return;
    }
    if (max_t < 1 || k < 4 || (k % 4) != 0 || max_topk < 1 || nlayers < 1) throw std::runtime_error("ep_prefill_init: bad limits");
    s.path = ep_transport_path();
    if (s.path <= 0) throw std::runtime_error("ep_prefill_init: decode sidecar ep_init must run first");
    if (const char* e = std::getenv("B70_EP_PREFILL_SPIN")) {
        long v = std::atol(e);
        if (v > 0) s.spin = (uint32_t)v;
    }
    s.max_t = max_t;
    s.K = k;
    s.max_topk = max_topk;
    s.nlayers = nlayers;
    sycl::queue& q0 = stream_q(0);
    sycl::queue& q1 = stream_q(1);
    s.off_ids = 0;
    s.off_w = align256((size_t)max_t * (size_t)max_topk * 4);
    s.off_x = s.off_w + align256((size_t)max_t * (size_t)max_topk * 4);
    s.inbox_b = s.off_x + align256((size_t)max_t * (size_t)k * 2);
    s.psum_b = align256((size_t)max_t * (size_t)k * sizeof(float));
    s.flag_b = (size_t)nlayers * 4096u;
    std::string err;
    if (!region_alloc(q1, q0, s.inbox_b, s.path, s.inbox, err)) throw std::runtime_error("pf inbox " + err);
    if (!region_alloc(q0, q1, s.psum_b, s.path, s.psum, err)) throw std::runtime_error("pf psum " + err);
    if (!region_alloc(q1, q0, s.flag_b, s.path, s.flag_in, err)) throw std::runtime_error("pf flag_in " + err);
    if (!region_alloc(q0, q1, s.flag_b, s.path, s.flag_ret, err)) throw std::runtime_error("pf flag_ret " + err);
    s.ctr0 = sycl::malloc_device<uint32_t>(nlayers, q0);
    s.done0 = sycl::malloc_device<uint32_t>(1, q0);
    s.err0 = sycl::malloc_host<uint32_t>(nlayers, q0);
    s.ctr1 = sycl::malloc_device<uint32_t>(nlayers, q1);
    s.done1 = sycl::malloc_device<uint32_t>(1, q1);
    s.err1 = sycl::malloc_host<uint32_t>(nlayers, q1);
    if (!s.ctr0 || !s.done0 || !s.err0 || !s.ctr1 || !s.done1 || !s.err1) throw std::runtime_error("pf scratch alloc failed");
    q0.memset(s.ctr0, 0, (size_t)nlayers * 4);
    q0.memset(s.done0, 0, 4);
    q0.memset(s.flag_ret.local, 0, s.flag_b);
    q1.memset(s.ctr1, 0, (size_t)nlayers * 4);
    q1.memset(s.done1, 0, 4);
    q1.memset(s.flag_in.local, 0, s.flag_b);
    std::memset(s.err0, 0, (size_t)nlayers * 4);
    std::memset(s.err1, 0, (size_t)nlayers * 4);
    q0.wait();
    q1.wait();
    const int pages = (int)(s.flag_b / 4096u);
    const int ti = touch_twice(q0, s.inbox.remote, q1, s.inbox.local, s.inbox_b);
    const int tp = touch_twice(q1, s.psum.remote, q0, s.psum.local, s.psum_b);
    const int tf1 = touch_twice(q0, s.flag_in.remote, q1, s.flag_in.local, s.flag_b);
    const int tf0 = touch_twice(q1, s.flag_ret.remote, q0, s.flag_ret.local, s.flag_b);
    std::printf("EP-PF init max_t=%d K=%d topk=%d nlayers=%d spin=%u path=%d inbox_b=%zu psum_b=%zu touch=%d%d%d%d\n",
                max_t, k, max_topk, nlayers, s.spin, s.path, s.inbox_b, s.psum_b, ti, tp, tf1, tf0);
    std::fflush(stdout);
    if (!(ti && tp && tf1 && tf0)) throw std::runtime_error("pf warmup peer-page touch failed");
    ep_store_flag_pages(q0, static_cast<uint32_t*>(s.flag_in.remote), pages, 0u);
    ep_store_flag_pages(q1, static_cast<uint32_t*>(s.flag_ret.remote), pages, 0u);
    q0.memset(s.flag_ret.local, 0, s.flag_b);
    q1.memset(s.flag_in.local, 0, s.flag_b);
    q0.wait();
    q1.wait();
    s.up = true;
}

void check_errors(int layer) {
    PState& s = P();
    if (s.err0[layer] || s.err1[layer]) {
        std::printf("EP-PF flag timeout layer=%d err0=%u err1=%u calls=%ld\n", layer, s.err0[layer], s.err1[layer], s.calls);
        std::fflush(stdout);
        throw std::runtime_error("ep prefill flag timeout");
    }
}

void need_xpu(const at::Tensor& t, const char* name, int dev) {
    TORCH_CHECK(t.is_xpu(), name, " must be an XPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK((int)t.device().index() == dev, name, " must be on xpu:", dev);
}

at::Tensor pf_init_op(int64_t max_t, int64_t k, int64_t max_topk, int64_t nlayers) {
    pf_init((int)max_t, (int)k, (int)max_topk, (int)nlayers);
    return at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
}

// Returns (ids0 [T,topk] int32 xpu:0, x1 [T,K] bf16 xpu:1, w1 [T,topk] fp32 xpu:1, ids1 [T,topk] int32 xpu:1).
std::vector<at::Tensor> pf_dispatch_op(at::Tensor x, at::Tensor w, at::Tensor ids, at::Tensor map0, at::Tensor map1,
                                       int64_t n0, int64_t e1, int64_t layer) {
    PState& s = P();
    TORCH_CHECK(s.up, "ep_prefill_dispatch before ep_prefill_init");
    need_xpu(x, "x", 0);
    need_xpu(w, "w", 0);
    need_xpu(ids, "ids", 0);
    need_xpu(map0, "map0", 0);
    need_xpu(map1, "map1", 0);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == s.K, "x must be bf16 [T, K]");
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.dim() == 2, "w must be fp32 [T, topk]");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.sizes() == w.sizes() && ids.size(0) == x.size(0), "ids int32 [T, topk]");
    TORCH_CHECK(map0.scalar_type() == at::kInt && map1.scalar_type() == at::kInt && map0.numel() == map1.numel(), "maps int32");
    TORCH_CHECK(layer >= 0 && layer < s.nlayers, "layer");
    const int T = (int)x.size(0);
    const int topk = (int)ids.size(1);
    TORCH_CHECK(T >= 1 && T <= s.max_t && topk >= 1 && topk <= s.max_topk, "ep_prefill_dispatch shape exceeds the exchange");
    check_errors((int)layer);
    const int n_ids = T * topk;
    const size_t ids_b = (size_t)n_ids * 4;
    const size_t x_b = (size_t)T * (size_t)s.K * 2;
    auto opt1 = x.options().device(at::Device(at::kXPU, 1));
    at::Tensor ids0 = at::empty({T, topk}, ids.options());
    at::Tensor x1 = at::empty({T, (int64_t)s.K}, opt1);
    at::Tensor w1 = at::empty({T, topk}, opt1.dtype(at::kFloat));
    at::Tensor ids1 = at::empty({T, topk}, opt1.dtype(at::kInt));
    uint32_t* ctr0 = s.ctr0 + layer;
    uint32_t* ctr1 = s.ctr1 + layer;
    auto* peer_ids = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(s.inbox.remote) + s.off_ids);
    auto* peer_w = reinterpret_cast<float*>(static_cast<uint8_t*>(s.inbox.remote) + s.off_w);
    auto* peer_x = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(s.inbox.remote) + s.off_x);
    sycl::queue& q0 = stream_q(0);
    sycl::queue& q1 = stream_q(1);
    // Card 1 first: its wait is released by card 0's push.
    ep_bump(q1, ctr1);
    ep_wait(q1, flag_at(s.flag_in, false, (int)layer), ctr1, s.spin, s.err1 + layer, 11);
    ep_copy_uncached(q1, ids1.data_ptr(), static_cast<uint8_t*>(s.inbox.local) + s.off_ids, ids_b);
    ep_copy_uncached(q1, w1.data_ptr(), static_cast<uint8_t*>(s.inbox.local) + s.off_w, ids_b);
    ep_copy_uncached(q1, x1.data_ptr(), static_cast<uint8_t*>(s.inbox.local) + s.off_x, x_b);
    ep_bump(q0, ctr0);
    ep_dispatch(q0, ids.data_ptr<int32_t>(), w.data_ptr<float>(), static_cast<const uint16_t*>(x.data_ptr()),
                ids0.data_ptr<int32_t>(), peer_ids, peer_w, peer_x, n_ids, x_b, (int)n0, (int)e1,
                map0.data_ptr<int32_t>(), map1.data_ptr<int32_t>(), (int)map0.numel(), s.done0,
                flag_at(s.flag_in, true, (int)layer), ctr0);
    s.calls += 1;
    return {ids0, x1, w1, ids1};
}

// y1 on xpu:1 (fp32 or bf16, [T,K]), y0 on xpu:0 fp32 [T,K]. Returns y = y0 + y1 on xpu:0 (fp32).
at::Tensor pf_return_op(at::Tensor y1, at::Tensor y0, int64_t layer) {
    PState& s = P();
    TORCH_CHECK(s.up, "ep_prefill_return before ep_prefill_init");
    need_xpu(y1, "y1", 1);
    need_xpu(y0, "y0", 0);
    TORCH_CHECK(y0.scalar_type() == at::kFloat && y0.dim() == 2 && y0.size(1) == s.K, "y0 fp32 [T, K]");
    TORCH_CHECK(y1.dim() == 2 && y1.sizes() == y0.sizes(), "y1 must match y0");
    TORCH_CHECK(y1.scalar_type() == at::kFloat || y1.scalar_type() == at::kBFloat16, "y1 fp32 or bf16");
    TORCH_CHECK(layer >= 0 && layer < s.nlayers, "layer");
    const int T = (int)y0.size(0);
    TORCH_CHECK(T <= s.max_t, "T");
    const bool half = y1.scalar_type() == at::kBFloat16;
    const int64_t n4 = ((int64_t)T * (int64_t)s.K) / 4;   // fp32 float4 count
    const int64_t n4h = half ? n4 / 2 : n4;                // float4 count of the pushed bytes
    uint32_t* ctr0 = s.ctr0 + layer;
    uint32_t* ctr1 = s.ctr1 + layer;
    at::Tensor y = at::empty_like(y0);
    sycl::queue& q0 = stream_q(0);
    sycl::queue& q1 = stream_q(1);
    ep_push_ctr(q1, s.psum.remote, y1.data_ptr(), n4h, s.done1, flag_at(s.flag_ret, true, (int)layer), ctr1);
    ep_wait(q0, flag_at(s.flag_ret, false, (int)layer), ctr0, s.spin, s.err0 + layer, 12);
    if (half)
        add_uncached_bf16(q0, y.data_ptr<float>(), y0.data_ptr<float>(), static_cast<const uint16_t*>(s.psum.local),
                          (int64_t)T * (int64_t)s.K / 2);
    else
        ep_add_uncached(q0, y.data_ptr<float>(), y0.data_ptr<float>(), static_cast<const float*>(s.psum.local), n4);
    return y;
}

at::Tensor pf_sync_op() {
    PState& s = P();
    if (!s.up) return at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    stream_q(1).wait();
    stream_q(0).wait();
    for (int l = 0; l < s.nlayers; ++l) check_errors(l);
    return at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
}

at::Tensor pf_errors_op() {
    PState& s = P();
    std::vector<int64_t> v;
    if (s.up) {
        for (int i = 0; i < s.nlayers; ++i) v.push_back(s.err0[i]);
        for (int i = 0; i < s.nlayers; ++i) v.push_back(s.err1[i]);
    }
    auto t = at::empty({(int64_t)v.size()}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    if (!v.empty()) std::memcpy(t.data_ptr<int64_t>(), v.data(), v.size() * sizeof(int64_t));
    return t;
}

}  // namespace
}  // namespace b70ep

TORCH_LIBRARY_FRAGMENT(b70_ep, m) {
    m.def("ep_prefill_init(int max_t, int k, int max_topk, int nlayers) -> Tensor");
    m.def("ep_prefill_dispatch(Tensor x, Tensor w, Tensor ids, Tensor map0, Tensor map1, int n0, int e1, int layer) -> Tensor[]");
    m.def("ep_prefill_return(Tensor y1, Tensor y0, int layer) -> Tensor");
    m.def("ep_prefill_sync() -> Tensor");
    m.def("ep_prefill_errors() -> Tensor");
}

TORCH_LIBRARY_IMPL(b70_ep, XPU, m) {
    m.impl("ep_prefill_dispatch", b70ep::pf_dispatch_op);
    m.impl("ep_prefill_return", b70ep::pf_return_op);
}

TORCH_LIBRARY_IMPL(b70_ep, CompositeExplicitAutograd, m) {
    m.impl("ep_prefill_init", b70ep::pf_init_op);
    m.impl("ep_prefill_sync", b70ep::pf_sync_op);
    m.impl("ep_prefill_errors", b70ep::pf_errors_op);
}
