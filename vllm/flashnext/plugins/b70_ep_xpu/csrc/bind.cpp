// torch.ops.b70_ep. Ops with no tensor arguments use
// CompositeExplicitAutograd. moe_ep's arguments are all on card 0.
#include "ep_api.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstring>
#include <vector>

namespace {

at::Tensor longs(const std::vector<int64_t>& v) {
    auto t = at::empty({(int64_t)v.size()}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    if (!v.empty()) std::memcpy(t.data_ptr<int64_t>(), v.data(), v.size() * sizeof(int64_t));
    return t;
}

at::Tensor ok_i() { return at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(at::kCPU)); }

void need_xpu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_xpu(), name, " must be an XPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor m0_setup_op(int64_t spin) { return longs(b70ep::m0_setup((int)spin)); }
at::Tensor m0_prep_op(int64_t sample) {
    b70ep::m0_prep((int)sample);
    return ok_i();
}
at::Tensor m0_submit_card1_op() {
    b70ep::m0_submit_card1();
    return ok_i();
}
at::Tensor m0_submit_card0_op(int64_t mirror) {
    b70ep::m0_submit_card0((int)mirror);
    return ok_i();
}
at::Tensor m0_check_op(int64_t sample) { return longs(b70ep::m0_check((int)sample)); }
at::Tensor m0_card1_capture_begin_op() {
    b70ep::m0_card1_capture_begin();
    return ok_i();
}
at::Tensor m0_card1_capture_end_op() {
    b70ep::m0_card1_capture_end();
    return ok_i();
}
at::Tensor m0_card1_replay_op() {
    b70ep::m0_card1_replay();
    return ok_i();
}
at::Tensor m0_latency_op() {
    b70ep::m0_latency();
    return ok_i();
}
at::Tensor m0_teardown_op() {
    b70ep::m0_teardown();
    return ok_i();
}

at::Tensor ep_init_op(int64_t max_t, int64_t max_k, int64_t max_topk, int64_t nlayers) {
    b70ep::ep_init((int)max_t, (int)max_k, (int)max_topk, (int)nlayers);
    return ok_i();
}
at::Tensor ep_reserve_op() {
    b70ep::ep_reserve();
    return ok_i();
}
at::Tensor ep_register0_op(int64_t layer, at::Tensor gate, int64_t gate_type, at::Tensor up, int64_t up_type,
                           at::Tensor down, int64_t down_type, int64_t e, int64_t intermediate, int64_t hidden) {
    need_xpu(gate, "gate0");
    need_xpu(up, "up0");
    need_xpu(down, "down0");
    TORCH_CHECK(gate.scalar_type() == at::kByte, "gate0 must be uint8");
    b70ep::ep_register0((int)layer, gate.data_ptr<uint8_t>(), (int)gate_type, up.data_ptr<uint8_t>(), (int)up_type,
                        down.data_ptr<uint8_t>(), (int)down_type, (int)e, (int)intermediate, (int)hidden);
    return ok_i();
}
at::Tensor ep_register1_op(int64_t layer, at::Tensor gate, at::Tensor up, at::Tensor down, int64_t e) {
    need_xpu(gate, "gate1");
    need_xpu(up, "up1");
    need_xpu(down, "down1");
    b70ep::ep_register1((int)layer, gate.data_ptr<uint8_t>(), up.data_ptr<uint8_t>(), down.data_ptr<uint8_t>(), (int)e);
    return ok_i();
}
at::Tensor ep_register_map_op(int64_t layer, at::Tensor map0, at::Tensor map1) {
    need_xpu(map0, "map0");
    need_xpu(map1, "map1");
    TORCH_CHECK(map0.scalar_type() == at::kInt && map1.scalar_type() == at::kInt, "maps must be int32");
    TORCH_CHECK(map0.is_contiguous() && map1.is_contiguous(), "maps must be contiguous");
    TORCH_CHECK(map0.numel() == map1.numel() && map0.numel() > 0, "maps must be the same non-empty length");
    TORCH_CHECK(map0.device() == map1.device(), "maps must share a device");
    b70ep::ep_register_map((int)layer, map0.data_ptr<int32_t>(), map1.data_ptr<int32_t>(), (int)map0.numel());
    return ok_i();
}
at::Tensor moe_ep_op(at::Tensor x, at::Tensor w, at::Tensor ids, int64_t layer) {
    need_xpu(x, "x");
    need_xpu(w, "w");
    need_xpu(ids, "ids");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x must be bf16 [T, K]");
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.dim() == 2, "w must be fp32 [T, topk]");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.sizes() == w.sizes(), "ids must be int32 [T, topk]");
    TORCH_CHECK(ids.size(0) == x.size(0), "ids rows must match tokens");
    auto y = at::empty({x.size(0), x.size(1)}, x.options().dtype(at::kFloat));
    b70ep::ep_moe(static_cast<const uint16_t*>(x.data_ptr()), w.data_ptr<float>(), ids.data_ptr<int32_t>(),
                  y.data_ptr<float>(), (int)x.size(0), (int)ids.size(1), (int)x.size(1), (int)layer);
    return y;
}
// O4: card-0 shared expert fused into the card-0 expert call.
at::Tensor ep_register_shared_op(int64_t layer, at::Tensor sg, int64_t sgt, at::Tensor su, int64_t sut, at::Tensor sd,
                                 int64_t sdt, at::Tensor sgi) {
    need_xpu(sg, "sg");
    need_xpu(su, "su");
    need_xpu(sd, "sd");
    const void* gi = nullptr;
    int gdt = 0;
    if (sgi.numel() > 0) {
        need_xpu(sgi, "sgi");
        TORCH_CHECK(sgi.scalar_type() == at::kBFloat16 || sgi.scalar_type() == at::kFloat, "sgi bf16 or fp32");
        gi = sgi.data_ptr();
        gdt = sgi.scalar_type() == at::kBFloat16 ? 1 : 2;
    }
    b70ep::ep_register_shared((int)layer, sg.data_ptr<uint8_t>(), (int)sgt, su.data_ptr<uint8_t>(), (int)sut,
                              sd.data_ptr<uint8_t>(), (int)sdt, gi, gdt);
    return ok_i();
}
at::Tensor moe_ep_sh_op(at::Tensor x, at::Tensor w, at::Tensor ids, int64_t layer) {
    need_xpu(x, "x");
    need_xpu(w, "w");
    need_xpu(ids, "ids");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x must be bf16 [T, K]");
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.dim() == 2, "w must be fp32 [T, topk]");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.sizes() == w.sizes(), "ids must be int32 [T, topk]");
    auto y = at::empty({x.size(0), x.size(1)}, x.options().dtype(at::kFloat));
    b70ep::ep_moe_sh(static_cast<const uint16_t*>(x.data_ptr()), w.data_ptr<float>(), ids.data_ptr<int32_t>(),
                     y.data_ptr<float>(), (int)x.size(0), (int)ids.size(1), (int)x.size(1), (int)layer);
    return y;
}
at::Tensor moe_ep_shb_op(at::Tensor x, at::Tensor w, at::Tensor ids, int64_t layer) {
    need_xpu(x, "x");
    need_xpu(w, "w");
    need_xpu(ids, "ids");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x must be bf16 [T, K]");
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.dim() == 2, "w must be fp32 [T, topk]");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.sizes() == w.sizes(), "ids must be int32 [T, topk]");
    auto y = at::empty({x.size(0), x.size(1)}, x.options());
    b70ep::ep_moe_sh(static_cast<const uint16_t*>(x.data_ptr()), w.data_ptr<float>(), ids.data_ptr<int32_t>(),
                     static_cast<float*>(y.data_ptr()), (int)x.size(0), (int)ids.size(1), (int)x.size(1), (int)layer,
                     true);
    return y;
}
at::Tensor moe_ep_route_op(at::Tensor x, at::Tensor logits, int64_t layer, int64_t topk, int64_t renorm) {
    need_xpu(x, "x");
    need_xpu(logits, "logits");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x must be bf16 [T, K]");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 && logits.dim() == 2 && logits.is_contiguous() &&
                    logits.size(0) == x.size(0), "logits must be bf16 [T, E]");
    auto y = at::empty({x.size(0), x.size(1)}, x.options());
    b70ep::ep_moe_route(static_cast<const uint16_t*>(x.data_ptr()), static_cast<const uint16_t*>(logits.data_ptr()),
                        static_cast<uint16_t*>(y.data_ptr()), (int)x.size(0), (int)logits.size(1), (int)topk,
                        (int)x.size(1), (int)layer, (int)renorm);
    return y;
}
// O4 split: begin returns the fp32 output buffer; end fills it in place.
at::Tensor moe_ep_begin_op(at::Tensor x, at::Tensor w, at::Tensor ids, int64_t layer) {
    need_xpu(x, "x");
    need_xpu(w, "w");
    need_xpu(ids, "ids");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x must be bf16 [T, K]");
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.dim() == 2, "w must be fp32 [T, topk]");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.sizes() == w.sizes(), "ids must be int32 [T, topk]");
    TORCH_CHECK(ids.size(0) == x.size(0), "ids rows must match tokens");
    auto y = at::empty({x.size(0), x.size(1)}, x.options().dtype(at::kFloat));
    b70ep::ep_moe_begin(static_cast<const uint16_t*>(x.data_ptr()), w.data_ptr<float>(), ids.data_ptr<int32_t>(),
                        (int)x.size(0), (int)ids.size(1), (int)x.size(1), (int)layer);
    return y;
}
void moe_ep_end_op(at::Tensor y, int64_t layer) {
    need_xpu(y, "y");
    TORCH_CHECK(y.scalar_type() == at::kFloat && y.dim() == 2 && y.is_contiguous(), "y must be fp32 [T, K]");
    b70ep::ep_moe_end(y.data_ptr<float>(), (int)y.size(0), (int)y.size(1), (int)layer);
}
at::Tensor ep_capture_begin_op(int64_t key) {
    b70ep::ep_capture_begin((int)key);
    return ok_i();
}
at::Tensor ep_capture_end_op(int64_t key) {
    b70ep::ep_capture_end((int)key);
    return ok_i();
}
at::Tensor ep_replay_prepare_op(int64_t key) {
    b70ep::ep_replay_prepare((int)key);
    return ok_i();
}
at::Tensor ep_replay_submit_op(int64_t key) {
    b70ep::ep_replay_submit((int)key);
    return ok_i();
}
at::Tensor ep_replay_op(int64_t key) {
    b70ep::ep_replay((int)key);
    return ok_i();
}
at::Tensor ep_read_errors_op() { return longs(b70ep::ep_read_errors()); }

}  // namespace

TORCH_LIBRARY(b70_ep, m) {
    m.def("m0_setup(int spin) -> Tensor");
    m.def("m0_prep(int sample) -> Tensor");
    m.def("m0_submit_card1() -> Tensor");
    m.def("m0_submit_card0(int mirror) -> Tensor");
    m.def("m0_check(int sample) -> Tensor");
    m.def("m0_card1_capture_begin() -> Tensor");
    m.def("m0_card1_capture_end() -> Tensor");
    m.def("m0_card1_replay() -> Tensor");
    m.def("m0_latency() -> Tensor");
    m.def("m0_teardown() -> Tensor");
    m.def("ep_init(int max_t, int max_k, int max_topk, int nlayers) -> Tensor");
    m.def("ep_reserve() -> Tensor");
    m.def("ep_register0(int layer, Tensor gate, int gate_type, Tensor up, int up_type, Tensor down, int down_type, "
          "int e, int intermediate, int hidden) -> Tensor");
    m.def("ep_register1(int layer, Tensor gate, Tensor up, Tensor down, int e) -> Tensor");
    m.def("ep_register_map(int layer, Tensor map0, Tensor map1) -> Tensor");
    m.def("moe_ep(Tensor x, Tensor w, Tensor ids, int layer) -> Tensor");
    m.def("ep_register_shared(int layer, Tensor sg, int sgt, Tensor su, int sut, Tensor sd, int sdt, Tensor sgi) -> Tensor");
    m.def("moe_ep_sh(Tensor x, Tensor w, Tensor ids, int layer) -> Tensor");
    m.def("moe_ep_shb(Tensor x, Tensor w, Tensor ids, int layer) -> Tensor");
    m.def("moe_ep_route(Tensor x, Tensor logits, int layer, int topk, int renorm) -> Tensor");
    m.def("moe_ep_begin(Tensor x, Tensor w, Tensor ids, int layer) -> Tensor");
    m.def("moe_ep_end(Tensor(a!) y, int layer) -> ()");
    m.def("ep_capture_begin(int key) -> Tensor");
    m.def("ep_capture_end(int key) -> Tensor");
    m.def("ep_replay_prepare(int key) -> Tensor");
    m.def("ep_replay_submit(int key) -> Tensor");
    m.def("ep_replay(int key) -> Tensor");
    m.def("ep_read_errors() -> Tensor");
}

TORCH_LIBRARY_IMPL(b70_ep, XPU, m) {
    m.impl("ep_register0", ep_register0_op);
    m.impl("ep_register1", ep_register1_op);
    m.impl("ep_register_map", ep_register_map_op);
    m.impl("moe_ep", moe_ep_op);
    m.impl("moe_ep_begin", moe_ep_begin_op);
    m.impl("ep_register_shared", ep_register_shared_op);
    m.impl("moe_ep_sh", moe_ep_sh_op);
    m.impl("moe_ep_shb", moe_ep_shb_op);
    m.impl("moe_ep_route", moe_ep_route_op);
    m.impl("moe_ep_end", moe_ep_end_op);
}

TORCH_LIBRARY_IMPL(b70_ep, CompositeExplicitAutograd, m) {
    m.impl("m0_setup", m0_setup_op);
    m.impl("m0_prep", m0_prep_op);
    m.impl("m0_submit_card1", m0_submit_card1_op);
    m.impl("m0_submit_card0", m0_submit_card0_op);
    m.impl("m0_check", m0_check_op);
    m.impl("m0_card1_capture_begin", m0_card1_capture_begin_op);
    m.impl("m0_card1_capture_end", m0_card1_capture_end_op);
    m.impl("m0_card1_replay", m0_card1_replay_op);
    m.impl("m0_latency", m0_latency_op);
    m.impl("m0_teardown", m0_teardown_op);
    m.impl("ep_init", ep_init_op);
    m.impl("ep_reserve", ep_reserve_op);
    m.impl("ep_capture_begin", ep_capture_begin_op);
    m.impl("ep_capture_end", ep_capture_end_op);
    m.impl("ep_replay_prepare", ep_replay_prepare_op);
    m.impl("ep_replay_submit", ep_replay_submit_op);
    m.impl("ep_replay", ep_replay_op);
    m.impl("ep_read_errors", ep_read_errors_op);
}
