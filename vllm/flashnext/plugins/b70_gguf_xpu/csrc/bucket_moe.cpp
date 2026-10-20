// F1: one native op for a bucketed dequant+bmm expert layer (card 1 of the
// EP sidecar). Replaces serve_prefill._bucketed (13 Python calls x ~22 torch
// ops per layer) with: device histogram/scan/scatter, ONE D2H of the expert
// counts, a host bucket plan, then per chunk: index_select of the packed
// rows, C13 wide dequant into bf16 scratch, pad + inverse index, two oneDNN
// bmm, swiglu, and a deterministic per-token combine (no atomics).
//
//   prefill_moe_bucketed(x bf16 [T,H], ids i32 [T,topk] (local slots, -1 =
//   not here), w fp32 [T,topk], gate/up/down packed u8, type_gu, type_down,
//   H, F, tile, maxpad) -> fp32 [T,H]
//
// tile caps the experts per chunk, maxpad caps the padded rows (experts x
// bucket width) per chunk. Buckets are power-of-two row counts (>= 8), as in
// serve_prefill._bucketed. Runs on the current device's stream.
#include "dequant_wide.hpp"
#include "decode.hpp"

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

namespace {

class b70_bk_hist;
class b70_bk_scan;
class b70_bk_scatter;
class b70_bk_pad;
class b70_bk_swiglu;
class b70_bk_combine;

sycl::queue& q_cur() { return c10::xpu::getCurrentXPUStream().queue(); }

void check_xpu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_xpu(), name, " must be an XPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// counts[e] = #slots with ids==e; starts = exclusive scan; order = slots sorted by expert.
void group_ids(sycl::queue& q, const int32_t* ids, int M, int E, int32_t* counts, int32_t* starts, int32_t* cursor,
               int32_t* order) {
    q.memset(counts, 0, sizeof(int32_t) * (size_t)E);
    const int wg = 256;
    const int groups = std::min(1024, std::max(1, (M + wg - 1) / wg));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> lc(E, h);
        h.parallel_for<b70_bk_hist>(sycl::nd_range<1>((size_t)groups * wg, wg), [=](sycl::nd_item<1> it) {
            const int tid = (int)it.get_local_id(0);
            int* bin = lc.get_multi_ptr<sycl::access::decorated::no>().get();
            for (int e = tid; e < E; e += wg) bin[e] = 0;
            it.barrier(sycl::access::fence_space::local_space);
            const int gs = (int)it.get_global_range(0);
            for (int i = (int)it.get_global_id(0); i < M; i += gs) {
                const int e = ids[i];
                if ((unsigned)e < (unsigned)E) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        ar(bin[e]);
                    ar.fetch_add(1);
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int e = tid; e < E; e += wg) {
                const int c = bin[e];
                if (c) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        ag(counts[e]);
                    ag.fetch_add(c);
                }
            }
        });
    });
    q.single_task<b70_bk_scan>([=]() {
        int sum = 0;
        starts[0] = 0;
        for (int e = 0; e < E; ++e) {
            cursor[e] = sum;
            sum += counts[e];
            starts[e + 1] = sum;
        }
    });
    q.parallel_for<b70_bk_scatter>(sycl::range<1>((size_t)((M + 255) & ~255)), [=](sycl::item<1> it) {
        const int i = (int)it.get_id(0);
        if (i >= M) return;
        const int e = ids[i];
        if ((unsigned)e >= (unsigned)E) return;
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ag(cursor[e]);
        order[ag.fetch_add(1)] = i;
    });
}

// xpad[j, r, :] = x[order[starts[sel[j]] + r] / topk, :]; inv[slot] = j*cap + r.
// One work-group per (j, r) row; H bf16 as u32 words.
void pad_rows(sycl::queue& q, const uint16_t* x, uint16_t* xpad, int32_t* inv, const int32_t* order,
              const int32_t* starts, const int32_t* sel, int n, int cap, int H, int topk) {
    const int wg = 256;
    const int words = H / 2;
    q.parallel_for<b70_bk_pad>(sycl::nd_range<2>(sycl::range<2>((size_t)n, (size_t)cap * wg), sycl::range<2>(1, wg)),
                               [=](sycl::nd_item<2> it) {
                                   const int j = (int)it.get_group(0);
                                   const int r = (int)it.get_group(1);
                                   const int e = sel[j];
                                   const int lo = starts[e];
                                   const int cnt = starts[e + 1] - lo;
                                   uint32_t* d = reinterpret_cast<uint32_t*>(xpad + ((size_t)j * cap + r) * H);
                                   if (r >= cnt) {
                                       for (int c = (int)it.get_local_id(1); c < words; c += wg) d[c] = 0u;
                                       return;
                                   }
                                   const int slot = order[lo + r];
                                   const uint32_t* s = reinterpret_cast<const uint32_t*>(x + (size_t)(slot / topk) * H);
                                   for (int c = (int)it.get_local_id(1); c < words; c += wg) d[c] = s[c];
                                   if (it.get_local_id(1) == 0) inv[slot] = j * cap + r;
                               });
}

inline float bf2f(uint16_t b) { return sycl::bit_cast<float>((uint32_t)b << 16); }
inline uint16_t f2bf(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// hid[row, :F] = silu(gu[row, :F]) * gu[row, F:2F]
void swiglu(sycl::queue& q, const uint16_t* gu, uint16_t* hid, int64_t rows, int F) {
    const int64_t total = rows * (int64_t)F;
    q.parallel_for<b70_bk_swiglu>(sycl::range<1>((size_t)((total + 255) & ~255)), [=](sycl::item<1> it) {
        const int64_t i = (int64_t)it.get_id(0);
        if (i >= total) return;
        const int64_t row = i / F;
        const int c = (int)(i - row * F);
        const float g = bf2f(gu[row * 2 * F + c]);
        const float u = bf2f(gu[row * 2 * F + F + c]);
        const float s = g / (1.0f + sycl::exp(-g));
        hid[i] = f2bf(s * u);
    });
}

// out[t, c] += sum_k w[t,k] * yd[inv[t*topk+k], c] over slots placed in this chunk.
void combine(sycl::queue& q, const uint16_t* yd, const int32_t* inv, const float* w, float* out, int T, int topk,
             int H) {
    const int wg = 256;
    const int cols = (H + wg - 1) / wg;
    q.parallel_for<b70_bk_combine>(sycl::nd_range<2>(sycl::range<2>((size_t)T, (size_t)cols * wg), sycl::range<2>(1, wg)),
                                   [=](sycl::nd_item<2> it) {
                                       const int t = (int)it.get_group(0);
                                       const int c = (int)it.get_global_id(1);
                                       if (c >= H) return;
                                       float acc = 0.0f;
                                       bool any = false;
                                       for (int k = 0; k < topk; ++k) {
                                           const int pos = inv[t * topk + k];
                                           if (pos < 0) continue;
                                           any = true;
                                           acc += w[t * topk + k] * bf2f(yd[(size_t)pos * H + c]);
                                       }
                                       if (any) out[(size_t)t * H + c] += acc;
                                   });
}

at::Tensor prefill_moe_bucketed_op(at::Tensor x, at::Tensor ids, at::Tensor w, at::Tensor gate, at::Tensor up,
                                   at::Tensor down, int64_t type_gu, int64_t type_down, int64_t H, int64_t F,
                                   int64_t tile, int64_t maxpad) {
    check_xpu(x, "x");
    check_xpu(ids, "ids");
    check_xpu(w, "w");
    check_xpu(gate, "gate");
    check_xpu(up, "up");
    check_xpu(down, "down");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == H, "x bf16 [T,H]");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 2 && ids.size(0) == x.size(0), "ids i32 [T,topk]");
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.sizes() == ids.sizes(), "w fp32 [T,topk]");
    TORCH_CHECK(gate.scalar_type() == at::kByte && up.scalar_type() == at::kByte && down.scalar_type() == at::kByte, "u8 packs");
    TORCH_CHECK(H % 2 == 0, "H even");
    const int T = (int)x.size(0);
    const int topk = (int)ids.size(1);
    const int M = T * topk;
    const int rb_gu = b70_row_bytes((int)type_gu, (int)H);
    const int rb_dn = b70_row_bytes((int)type_down, (int)F);
    TORCH_CHECK(rb_gu > 0 && rb_dn > 0, "bad expert type");
    TORCH_CHECK(gate.numel() % ((int64_t)F * rb_gu) == 0 && up.numel() == gate.numel(), "gate/up pack");
    const int E = (int)(gate.numel() / ((int64_t)F * rb_gu));
    TORCH_CHECK(down.numel() == (int64_t)E * H * rb_dn, "down pack");
    TORCH_CHECK(E > 0 && E <= 4096 && M > 0, "E/M");
    sycl::queue& q = q_cur();
    auto opt_i = ids.options();
    at::Tensor counts = at::empty({E}, opt_i);
    at::Tensor starts = at::empty({E + 1}, opt_i);
    at::Tensor cursor = at::empty({E}, opt_i);
    at::Tensor order = at::empty({M}, opt_i);
    at::Tensor inv = at::empty({M}, opt_i);
    at::Tensor out = at::zeros({T, H}, x.options().dtype(at::kFloat));
    group_ids(q, ids.data_ptr<int32_t>(), M, E, counts.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
              cursor.data_ptr<int32_t>(), order.data_ptr<int32_t>());
    // The one host sync of the layer: expert counts.
    std::vector<int32_t> hc((size_t)E);
    q.memcpy(hc.data(), counts.data_ptr<int32_t>(), sizeof(int32_t) * (size_t)E).wait();
    // Bucket plan: power-of-two widths >= 8, tile experts, maxpad padded rows.
    std::map<int, std::vector<int>> buckets;
    for (int e = 0; e < E; ++e) {
        const int n = hc[(size_t)e];
        if (n <= 0) continue;
        int width = 8;
        while (width < n) width <<= 1;
        buckets[width].push_back(e);
    }
    struct Chunk {
        int off, n, cap;
    };
    std::vector<Chunk> chunks;
    std::vector<int32_t> sel_all;
    for (auto& kv : buckets) {
        const int width = kv.first;
        const std::vector<int>& ex = kv.second;
        int step = tile > 0 ? (int)tile : (int)ex.size();
        if (maxpad > 0) step = std::min(step, std::max(1, (int)(maxpad / width)));
        for (size_t a = 0; a < ex.size(); a += (size_t)step) {
            const int n = (int)std::min((size_t)step, ex.size() - a);
            chunks.push_back(Chunk{(int)sel_all.size(), n, width});
            sel_all.insert(sel_all.end(), ex.begin() + (long)a, ex.begin() + (long)a + n);
        }
    }
    if (chunks.empty()) return out;
    at::Tensor sel_dev = at::from_blob(sel_all.data(), {(int64_t)sel_all.size()},
                                       at::TensorOptions().dtype(at::kInt).device(at::kCPU))
                             .to(x.device());
    at::Tensor gate2 = gate.view({E, (int64_t)F * rb_gu});
    at::Tensor up2 = up.view({E, (int64_t)F * rb_gu});
    at::Tensor down2 = down.view({E, (int64_t)H * rb_dn});
    for (const Chunk& ch : chunks) {
        at::Tensor sel = sel_dev.narrow(0, ch.off, ch.n);
        const int n = ch.n;
        const int cap = ch.cap;
        at::Tensor g_rows = at::index_select(gate2, 0, sel);
        at::Tensor u_rows = at::index_select(up2, 0, sel);
        at::Tensor d_rows = at::index_select(down2, 0, sel);
        at::Tensor w_gu = at::empty({n, 2 * F, H}, x.options());
        at::Tensor w_dn = at::empty({n, H, F}, x.options());
        at::Tensor gu_flat = w_gu.view({(int64_t)n * 2 * F, H});
        at::Tensor dn_flat = w_dn.view({(int64_t)n * H, F});
        b70_prefill_dequant_wide_op(g_rows.view({-1}), type_gu, (int64_t)n * F, H, gu_flat, 1, F);
        b70_prefill_dequant_wide_op(u_rows.view({-1}), type_gu, (int64_t)n * F, H, gu_flat, 2, F);
        b70_prefill_dequant_wide_op(d_rows.view({-1}), type_down, (int64_t)n * H, F, dn_flat, 0, 0);
        at::Tensor xpad = at::empty({n, cap, H}, x.options());
        q.memset(inv.data_ptr<int32_t>(), 0xFF, sizeof(int32_t) * (size_t)M);
        pad_rows(q, static_cast<const uint16_t*>(x.data_ptr()), static_cast<uint16_t*>(xpad.data_ptr()),
                 inv.data_ptr<int32_t>(), order.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
                 sel.data_ptr<int32_t>(), n, cap, (int)H, topk);
        at::Tensor gu = at::bmm(xpad, w_gu.transpose(1, 2));  // [n, cap, 2F] bf16
        at::Tensor hid = at::empty({n, cap, F}, x.options());
        swiglu(q, static_cast<const uint16_t*>(gu.data_ptr()), static_cast<uint16_t*>(hid.data_ptr()),
               (int64_t)n * cap, (int)F);
        at::Tensor yd = at::bmm(hid, w_dn.transpose(1, 2));  // [n, cap, H] bf16
        combine(q, static_cast<const uint16_t*>(yd.data_ptr()), inv.data_ptr<int32_t>(), w.data_ptr<float>(),
                out.data_ptr<float>(), T, topk, (int)H);
    }
    return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(b70_gguf, m) {
    m.def("prefill_dequant_wide(Tensor q, int ggml_type, int rows, int cols, Tensor out, int dst_mode, int pair) -> Tensor");
    m.def("prefill_moe_bucketed(Tensor x, Tensor ids, Tensor w, Tensor gate, Tensor up, Tensor down, int type_gu, "
          "int type_down, int H, int F, int tile, int maxpad) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_gguf, XPU, m) {
    m.impl("prefill_dequant_wide", b70_prefill_dequant_wide_op);
    m.impl("prefill_moe_bucketed", prefill_moe_bucketed_op);
}
