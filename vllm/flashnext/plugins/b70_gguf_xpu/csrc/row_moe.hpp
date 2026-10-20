// Subgroup-per-row Q8 GEMV, and one host call that queues a MoE layer.
// Cooperative launch and SYCL graphs are not used: night lab measured grid
// barriers above kernel boundaries, and reduce_over_group is not a graph op.
// The layer is four or five in-order submits with depends_on and no wait:
// Q8(x), fused gate+up+SiLU (routed top-k and the shared expert), Q8(hidden),
// down, routing accumulate.
#pragma once

template <int TYPE>
class b70_row_k;

template <int TYPE>
class b70_dense_k;

class b70_q8_f32_k;
class b70_fused_gu_k;
class b70_fused_down_k;
class b70_fused_red_k;

// Two output rows share one Q8 block in registers (Strata RPL=2,
// iq_kernels.dp.cpp:5013: IQ gate/up are faster only at 2 rows).
template <int TYPE>
inline void b70_acc_pair(const uint8_t* r0, const uint8_t* r1, bool l0, bool l1, const B70Q8* xr, int ng, int lane,
                         float& a0, float& a1) {
    for (int g = lane; g < ng; g += 16) {
        const B70Q8 xv = xr[g];
        int u[8];
        b70_q8_i8(xv, u);
        if constexpr (b70_dp4a_type<TYPE>()) {
            if (l0) {
                B70Part p;
                b70_prep<TYPE>(r0, g, p);
                a0 += b70_apply_u<TYPE>(p, u, xv.d);
            }
            if (l1) {
                B70Part p;
                b70_prep<TYPE>(r1, g, p);
                a1 += b70_apply_u<TYPE>(p, u, xv.d);
            }
        } else {
            float v0[32], v1[32];
            if (l0) b70_decode32<TYPE>(r0, g, v0);
            if (l1) b70_decode32<TYPE>(r1, g, v1);
            if (l0) {
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < 32; ++j) s = sycl::fma(v0[j], (float)xv.qs[j], s);
                a0 += s * xv.d;
            }
            if (l1) {
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < 32; ++j) s = sycl::fma(v1[j], (float)xv.qs[j], s);
                a1 += s * xv.d;
            }
        }
    }
}

inline void b70_acc_pair_sw(int type, const uint8_t* r0, const uint8_t* r1, bool l0, bool l1, const B70Q8* xr, int ng,
                            int lane, float& a0, float& a1) {
    switch (type) {
    case B70_IQ2_XXS: b70_acc_pair<B70_IQ2_XXS>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_IQ2_XS: b70_acc_pair<B70_IQ2_XS>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_IQ2_S: b70_acc_pair<B70_IQ2_S>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_IQ3_XXS: b70_acc_pair<B70_IQ3_XXS>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_IQ3_S: b70_acc_pair<B70_IQ3_S>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_IQ4_NL: b70_acc_pair<B70_IQ4_NL>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_Q2_0: b70_acc_pair<B70_Q2_0>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_Q4_K: b70_acc_pair<B70_Q4_K>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_Q5_K: b70_acc_pair<B70_Q5_K>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_Q6_K: b70_acc_pair<B70_Q6_K>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_IQ4_XS: b70_acc_pair<B70_IQ4_XS>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_Q2_K: b70_acc_pair<B70_Q2_K>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    case B70_Q3_K: b70_acc_pair<B70_Q3_K>(r0, r1, l0, l1, xr, ng, lane, a0, a1); break;
    default: break;
    }
}

template <int TYPE>
inline sycl::event b70_launch_row(sycl::queue& q, const B70Q8* xq, const uint8_t* w, const int32_t* expert_ids,
                                  float* y, int T, int topk, int E, int N, int K, int row_bytes, sycl::event dep) {
    const int nout = T * topk * N;
    constexpr int R = 2;
    constexpr int NSG = 4;
    constexpr int WG = NSG * 16;
    const int npairs = (nout + R - 1) / R;
    const int ngroups = (npairs + NSG - 1) / NSG;
    const int ng = K / 32;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for<b70_row_k<TYPE>>(
            sycl::nd_range<1>((size_t)ngroups * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_linear_id();
                const int sgid = (int)it.get_local_id(0) >> 4;
                const int pair = (int)it.get_group(0) * NSG + sgid;
                const int o0 = pair * R;
                const int o1 = o0 + 1;
                const bool in0 = o0 < nout;
                const bool in1 = o1 < nout;
                const int which0 = in0 ? o0 / N : 0;
                const int n0 = in0 ? o0 - which0 * N : 0;
                const int which1 = in1 ? o1 / N : 0;
                const int n1 = in1 ? o1 - which1 * N : 0;
                const int token0 = which0 / topk;
                const int slot0 = which0 - token0 * topk;
                const int token1 = which1 / topk;
                const int slot1 = which1 - token1 * topk;
                int expert0 = 0, expert1 = 0;
                bool live0 = in0, live1 = in1;
                if (expert_ids != nullptr) {
                    expert0 = live0 ? expert_ids[(size_t)token0 * (size_t)topk + slot0] : -1;
                    expert1 = live1 ? expert_ids[(size_t)token1 * (size_t)topk + slot1] : -1;
                    live0 = live0 && expert0 >= 0 && expert0 < E;
                    live1 = live1 && expert1 >= 0 && expert1 < E;
                }
                const uint8_t* src0 = w + ((size_t)expert0 * (size_t)N + (size_t)n0) * (size_t)row_bytes;
                const uint8_t* src1 = w + ((size_t)expert1 * (size_t)N + (size_t)n1) * (size_t)row_bytes;
                const B70Q8* xr0 = xq + (size_t)token0 * (size_t)ng;
                float acc0 = 0.f, acc1 = 0.f;
                if (live1 && token0 == token1) {
                    b70_acc_pair<TYPE>(src0, src1, live0, live1, xr0, ng, lane, acc0, acc1);
                } else {
                    if (live0) b70_acc_pair<TYPE>(src0, src0, true, false, xr0, ng, lane, acc0, acc1);
                    if (live1) {
                        const B70Q8* xr1 = xq + (size_t)token1 * (size_t)ng;
                        float dump = 0.f;
                        b70_acc_pair<TYPE>(src1, src1, true, false, xr1, ng, lane, acc1, dump);
                    }
                }
                const float red0 = sycl::reduce_over_group(sg, acc0, sycl::plus<float>());
                const float red1 = sycl::reduce_over_group(sg, acc1, sycl::plus<float>());
                if (lane == 0 && in0) y[(size_t)which0 * (size_t)N + (size_t)n0] = live0 ? red0 : 0.f;
                if (lane == 0 && in1) y[(size_t)which1 * (size_t)N + (size_t)n1] = live1 ? red1 : 0.f;
            });
    });
}

// Dense GEMV: one subgroup owns two N columns and reuses each weight decode
// across a chunk of M activations. Rows are staged in SLM so the decode
// reads coalesced bytes (the M=1 IQ3_S path got slower without that copy).
template <int TYPE>
inline sycl::event b70_launch_dense(sycl::queue& q, const B70Q8* xq, const uint8_t* w, float* y, int M, int N, int K,
                                    int row_bytes, sycl::event dep) {
    constexpr int R = 2;
    constexpr int MC = 4;
    int nsg = 4;
    while (nsg < 16 && (nsg * R) * row_bytes <= 32768) nsg *= 2;
    const int wg = nsg * 16;
    const int npairs = (N + R - 1) / R;
    const int ngroups = (npairs + nsg - 1) / nsg;
    const int ng = K / 32;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> slm((size_t)nsg * R * (size_t)row_bytes, h);
        h.depends_on(dep);
        h.parallel_for<b70_dense_k<TYPE>>(
            sycl::nd_range<1>((size_t)ngroups * (size_t)wg, (size_t)wg),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_linear_id();
                const int sgid = (int)it.get_local_id(0) >> 4;
                const int pair = (int)it.get_group(0) * nsg + sgid;
                const int n0 = pair * R;
                const int n1 = n0 + 1;
                const bool l0 = n0 < N;
                const bool l1 = n1 < N;
                uint8_t* base = slm.get_multi_ptr<sycl::access::decorated::no>().get() + (size_t)sgid * R * (size_t)row_bytes;
                auto copy_row = [&](const uint8_t* src, uint8_t* dst, bool live) {
                    if (!live) return;
                    const bool align4 = ((uintptr_t)src & 3u) == 0 && ((uintptr_t)dst & 3u) == 0;
                    const int n4 = row_bytes & ~3;
                    if (align4) {
                        for (int off = lane * 4; off < n4; off += 64) {
                            *reinterpret_cast<uint32_t*>(dst + off) = *reinterpret_cast<const uint32_t*>(src + off);
                        }
                    } else {
                        for (int off = lane; off < row_bytes; off += 16) dst[off] = src[off];
                    }
                    if (align4)
                        for (int off = n4 + lane; off < row_bytes; off += 16) dst[off] = src[off];
                };
                copy_row(w + (size_t)n0 * (size_t)row_bytes, base, l0);
                copy_row(w + (size_t)n1 * (size_t)row_bytes, base + row_bytes, l1);
                sycl::group_barrier(sg);
                const uint8_t* s0 = base;
                const uint8_t* s1 = base + row_bytes;
                for (int m0 = 0; m0 < M; m0 += MC) {
                    const int mc = (M - m0) < MC ? (M - m0) : MC;
                    float a0[MC] = {};
                    float a1[MC] = {};
                    for (int g = lane; g < ng; g += 16) {
                        if constexpr (b70_dp4a_type<TYPE>()) {
                            B70Part p0, p1;
                            if (l0) b70_prep<TYPE>(s0, g, p0);
                            if (l1) b70_prep<TYPE>(s1, g, p1);
#pragma unroll
                            for (int m = 0; m < MC; ++m) {
                                if (m >= mc) break;
                                const B70Q8& xv = xq[(size_t)(m0 + m) * (size_t)ng + g];
                                int u[8];
                                b70_q8_i8(xv, u);
                                if (l0) a0[m] += b70_apply_u<TYPE>(p0, u, xv.d);
                                if (l1) a1[m] += b70_apply_u<TYPE>(p1, u, xv.d);
                            }
                        } else {
                            float v0[32], v1[32];
                            if (l0) b70_decode32<TYPE>(s0, g, v0);
                            if (l1) b70_decode32<TYPE>(s1, g, v1);
#pragma unroll
                            for (int m = 0; m < MC; ++m) {
                                if (m >= mc) break;
                                const B70Q8& xv = xq[(size_t)(m0 + m) * (size_t)ng + g];
                                if (l0) {
                                    float s = 0.f;
#pragma unroll
                                    for (int j = 0; j < 32; ++j) s = sycl::fma(v0[j], (float)xv.qs[j], s);
                                    a0[m] += s * xv.d;
                                }
                                if (l1) {
                                    float s = 0.f;
#pragma unroll
                                    for (int j = 0; j < 32; ++j) s = sycl::fma(v1[j], (float)xv.qs[j], s);
                                    a1[m] += s * xv.d;
                                }
                            }
                        }
                    }
#pragma unroll
                    for (int m = 0; m < MC; ++m) {
                        const float r0 = sycl::reduce_over_group(sg, m < mc ? a0[m] : 0.f, sycl::plus<float>());
                        const float r1 = sycl::reduce_over_group(sg, m < mc ? a1[m] : 0.f, sycl::plus<float>());
                        if (lane == 0 && m < mc) {
                            if (l0) y[(size_t)(m0 + m) * (size_t)N + n0] = r0;
                            if (l1) y[(size_t)(m0 + m) * (size_t)N + n1] = r1;
                        }
                    }
                }
            });
    });
}

inline sycl::event b70_q8_quant_f32(sycl::queue& q, const float* x, B70Q8* y, int M, int K, sycl::event dep) {
    const int nb = K / 32;
    const int n = M * nb;
    constexpr int L = 64;
    const size_t global = (size_t)((n + L - 1) / L) * (size_t)L;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for<b70_q8_f32_k>(sycl::nd_range<1>(global, (size_t)L), [=](sycl::nd_item<1> it) {
            const int i = (int)it.get_global_id(0);
            if (i >= n) return;
            const float* p = x + (size_t)i * 32;
            float m = 0.f;
#pragma unroll
            for (int t = 0; t < 32; ++t) m = sycl::fmax(m, sycl::fabs(p[t]));
            const float id = m > 0.f ? 127.f / m : 0.f;
            B70Q8 b{};
            b.d = m * (1.f / 127.f);
#pragma unroll
            for (int t = 0; t < 32; ++t) {
                int r = (int)sycl::rint(p[t] * id);
                if (r > 127) r = 127;
                if (r < -127) r = -127;
                b.qs[t] = (int8_t)r;
            }
            y[i] = b;
        });
    });
}

struct B70Ws {
    float* p = nullptr;
    size_t cap = 0;
};

inline float* b70_ws(sycl::queue& q, B70Ws& ws, size_t nfloat) {
    if (nfloat <= ws.cap) return ws.p;
    q.wait();
    if (ws.p) sycl::free(ws.p, q);
    ws.cap = nfloat < 1024 ? 1024 : nfloat;
    void* raw = sycl::aligned_alloc_device(16, ws.cap * sizeof(float), q);
    if (!raw) throw std::runtime_error("b70_gguf: moe workspace alloc failed");
    ws.p = static_cast<float*>(raw);
    return ws.p;
}

inline sycl::event b70_launch_gu(sycl::queue& q, const B70Q8* xq, const uint8_t* gate, int gate_type, int gate_rb,
                                 const uint8_t* up, int up_type, int up_rb, const uint8_t* sgate, int sg_type, int sg_rb,
                                 const uint8_t* sup, int su_type, int su_rb, const int32_t* ids, float* act, int T,
                                 int topk, int slots, int E, int I, int K, sycl::event dep) {
    const int nout = T * slots * I;
    constexpr int R = 2;
    constexpr int NSG = 4;
    constexpr int WG = NSG * 16;
    const int npairs = (nout + R - 1) / R;
    const int ngroups = (npairs + NSG - 1) / NSG;
    const int ng = K / 32;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for<b70_fused_gu_k>(
            sycl::nd_range<1>((size_t)ngroups * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_linear_id();
                const int sgid = (int)it.get_local_id(0) >> 4;
                const int pair = (int)it.get_group(0) * NSG + sgid;
                const int o0 = pair * R;
                const int o1 = o0 + 1;
                auto one = [&](int out_i, bool& live, int& which, int& n, int& tg, int& tu, const uint8_t*& src_g,
                               const uint8_t*& src_u, const B70Q8*& xr) {
                    const bool in = out_i < nout;
                    which = in ? out_i / I : 0;
                    n = in ? out_i - which * I : 0;
                    const int token = which / slots;
                    const int slot = which - token * slots;
                    const bool shared = slot >= topk;
                    int expert = 0;
                    live = in;
                    if (!shared) {
                        expert = live ? ids[(size_t)token * (size_t)topk + slot] : -1;
                        live = live && expert >= 0 && expert < E;
                    }
                    tg = shared ? sg_type : gate_type;
                    tu = shared ? su_type : up_type;
                    const uint8_t* bg = shared ? sgate : gate;
                    const uint8_t* bu = shared ? sup : up;
                    const int rbg = shared ? sg_rb : gate_rb;
                    const int rbu = shared ? su_rb : up_rb;
                    const int use_e = shared ? 0 : expert;
                    src_g = bg + ((size_t)use_e * (size_t)I + (size_t)n) * (size_t)rbg;
                    src_u = bu + ((size_t)use_e * (size_t)I + (size_t)n) * (size_t)rbu;
                    xr = xq + (size_t)token * (size_t)(K / 32);
                };
                bool live0 = false, live1 = false;
                int which0 = 0, which1 = 0, n0 = 0, n1 = 0, tg0 = 0, tg1 = 0, tu0 = 0, tu1 = 0;
                const uint8_t* g0 = nullptr;
                const uint8_t* g1 = nullptr;
                const uint8_t* u0 = nullptr;
                const uint8_t* u1 = nullptr;
                const B70Q8* xr0 = xq;
                const B70Q8* xr1 = xq;
                one(o0, live0, which0, n0, tg0, tu0, g0, u0, xr0);
                one(o1, live1, which1, n1, tg1, tu1, g1, u1, xr1);
                float ag0 = 0.f, au0 = 0.f, ag1 = 0.f, au1 = 0.f;
                const bool same = live0 && live1 && xr0 == xr1 && tg0 == tg1 && tu0 == tu1;
                if (same) {
                    b70_acc_pair_sw(tg0, g0, g1, true, true, xr0, ng, lane, ag0, ag1);
                    b70_acc_pair_sw(tu0, u0, u1, true, true, xr0, ng, lane, au0, au1);
                } else {
                    float dump = 0.f;
                    if (live0) {
                        b70_acc_pair_sw(tg0, g0, g0, true, false, xr0, ng, lane, ag0, dump);
                        b70_acc_pair_sw(tu0, u0, u0, true, false, xr0, ng, lane, au0, dump);
                    }
                    if (live1) {
                        b70_acc_pair_sw(tg1, g1, g1, true, false, xr1, ng, lane, ag1, dump);
                        b70_acc_pair_sw(tu1, u1, u1, true, false, xr1, ng, lane, au1, dump);
                    }
                }
                const float rg0 = sycl::reduce_over_group(sg, ag0, sycl::plus<float>());
                const float ru0 = sycl::reduce_over_group(sg, au0, sycl::plus<float>());
                const float rg1 = sycl::reduce_over_group(sg, ag1, sycl::plus<float>());
                const float ru1 = sycl::reduce_over_group(sg, au1, sycl::plus<float>());
                if (lane == 0 && o0 < nout) {
                    float h = 0.f;
                    if (live0) h = rg0 * (1.f / (1.f + sycl::exp(-rg0))) * ru0;
                    act[(size_t)which0 * (size_t)I + (size_t)n0] = h;
                }
                if (lane == 0 && o1 < nout) {
                    float h = 0.f;
                    if (live1) h = rg1 * (1.f / (1.f + sycl::exp(-rg1))) * ru1;
                    act[(size_t)which1 * (size_t)I + (size_t)n1] = h;
                }
            });
    });
}

inline sycl::event b70_launch_down(sycl::queue& q, const B70Q8* hq, const uint8_t* down, int down_type, int down_rb,
                                   const uint8_t* sdown, int sd_type, int sd_rb, const int32_t* ids, float* y, int T,
                                   int topk, int slots, int E, int I, int K, sycl::event dep) {
    const int nout = T * slots * K;
    constexpr int R = 2;
    constexpr int NSG = 4;
    constexpr int WG = NSG * 16;
    const int npairs = (nout + R - 1) / R;
    const int ngroups = (npairs + NSG - 1) / NSG;
    const int ng = I / 32;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for<b70_fused_down_k>(
            sycl::nd_range<1>((size_t)ngroups * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_linear_id();
                const int sgid = (int)it.get_local_id(0) >> 4;
                const int pair = (int)it.get_group(0) * NSG + sgid;
                const int o0 = pair * R;
                const int o1 = o0 + 1;
                auto one = [&](int out_i, bool& live, int& which, int& n, int& ty, const uint8_t*& src, const B70Q8*& xr) {
                    const bool in = out_i < nout;
                    which = in ? out_i / K : 0;
                    n = in ? out_i - which * K : 0;
                    const int token = which / slots;
                    const int slot = which - token * slots;
                    const bool shared = slot >= topk;
                    int expert = 0;
                    live = in;
                    if (!shared) {
                        expert = live ? ids[(size_t)token * (size_t)topk + slot] : -1;
                        live = live && expert >= 0 && expert < E;
                    }
                    ty = shared ? sd_type : down_type;
                    const uint8_t* base = shared ? sdown : down;
                    const int rb = shared ? sd_rb : down_rb;
                    const int use_e = shared ? 0 : expert;
                    src = base + ((size_t)use_e * (size_t)K + (size_t)n) * (size_t)rb;
                    xr = hq + (size_t)which * (size_t)ng;
                };
                bool live0 = false, live1 = false;
                int which0 = 0, which1 = 0, n0 = 0, n1 = 0, ty0 = 0, ty1 = 0;
                const uint8_t* s0 = nullptr;
                const uint8_t* s1 = nullptr;
                const B70Q8* xr0 = hq;
                const B70Q8* xr1 = hq;
                one(o0, live0, which0, n0, ty0, s0, xr0);
                one(o1, live1, which1, n1, ty1, s1, xr1);
                float a0 = 0.f, a1 = 0.f;
                if (live0 && live1 && xr0 == xr1 && ty0 == ty1) {
                    b70_acc_pair_sw(ty0, s0, s1, true, true, xr0, ng, lane, a0, a1);
                } else {
                    float dump = 0.f;
                    if (live0) b70_acc_pair_sw(ty0, s0, s0, true, false, xr0, ng, lane, a0, dump);
                    if (live1) b70_acc_pair_sw(ty1, s1, s1, true, false, xr1, ng, lane, a1, dump);
                }
                const float r0 = sycl::reduce_over_group(sg, a0, sycl::plus<float>());
                const float r1 = sycl::reduce_over_group(sg, a1, sycl::plus<float>());
                if (lane == 0 && o0 < nout) y[(size_t)which0 * (size_t)K + (size_t)n0] = live0 ? r0 : 0.f;
                if (lane == 0 && o1 < nout) y[(size_t)which1 * (size_t)K + (size_t)n1] = live1 ? r1 : 0.f;
            });
    });
}

inline sycl::event b70_launch_red(sycl::queue& q, const float* down, const float* router, float* y, int T, int topk,
                                  int slots, int K, sycl::event dep) {
    const int n = T * K;
    constexpr int L = 128;
    const size_t global = (size_t)((n + L - 1) / L) * (size_t)L;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for<b70_fused_red_k>(sycl::nd_range<1>(global, (size_t)L), [=](sycl::nd_item<1> it) {
            const int i = (int)it.get_global_id(0);
            if (i >= n) return;
            const int t = i / K;
            const int k = i - t * K;
            float s = 0.f;
            for (int slot = 0; slot < slots; ++slot) {
                const float w = slot < topk ? router[(size_t)t * (size_t)topk + slot] : 1.f;
                s = sycl::fma(down[((size_t)t * slots + slot) * (size_t)K + k], w, s);
            }
            y[i] = s;
        });
    });
}

inline void b70_moe_layer_launch(sycl::queue& q, const uint16_t* x, const uint8_t* gate, int gate_type, int gate_rb,
                                 const uint8_t* up, int up_type, int up_rb, const uint8_t* down, int down_type, int down_rb,
                                 const int32_t* ids, const float* router, const uint8_t* sgate, int sg_type, int sg_rb,
                                 const uint8_t* sup, int su_type, int su_rb, const uint8_t* sdown, int sd_type, int sd_rb,
                                 float* y, int T, int topk, int E, int I, int K) {
    const bool shared = sgate != nullptr;
    const int slots = topk + (shared ? 1 : 0);
    const int nb_x = K / 32;
    const int nb_h = I / 32;
    const size_t q8_x = (size_t)T * (size_t)nb_x;
    const size_t q8_h = (size_t)T * (size_t)slots * (size_t)nb_h;
    const size_t q8_n = q8_x > q8_h ? q8_x : q8_h;
    B70Q8* xq = b70_q8_scratch(q, q8_n);
    static B70Ws act_ws;
    static B70Ws down_ws;
    float* act = b70_ws(q, act_ws, (size_t)T * (size_t)slots * (size_t)I);
    float* partial = b70_ws(q, down_ws, (size_t)T * (size_t)slots * (size_t)K);
    const sycl::event eq = b70_q8_quant(q, x, xq, T, K);
    const sycl::event egu = b70_launch_gu(q, xq, gate, gate_type, gate_rb, up, up_type, up_rb, sgate, sg_type, sg_rb, sup,
                                          su_type, su_rb, ids, act, T, topk, slots, E, I, K, eq);
    const sycl::event eh = b70_q8_quant_f32(q, act, xq, T * slots, I, egu);
    const sycl::event ed = b70_launch_down(q, xq, down, down_type, down_rb, sdown, sd_type, sd_rb, ids, partial, T, topk,
                                           slots, E, I, K, eh);
    b70_launch_red(q, partial, router, y, T, topk, slots, K, ed);
}
