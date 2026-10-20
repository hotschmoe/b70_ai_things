// Grouped decode MoE schedule (C9). Include once, after the anonymous
// namespace in kernels.cpp, so b70_prep / b70_apply_u / b70_device_grow
// are visible and this header can include the C library at global scope.
//
// Attribution: Strata night lab, iq_kernels.dp.cpp (worktree 1c360526).
//   rows_dot_multi_iq        ~2263  activation once, reused across rows and entries
//   native_gu_kernel_rows    ~2302  RPL=2, gate then up, chunks of 4/2/1
//   STRATA_EXPERT_LANES      ~799   default 8; lanes_sum xor offsets 4,2,1
//   launch_gu_rows_k         ~4970  the measured R2 kernel is reqd_sub_group_size(16)
// The 1-row fallback at launch_gu_port ~5130 is subgroup 32 and is the slower one.
// This file implements the R2 winner: work-group 256, subgroup 16, 8 lanes,
// 2 rows per lane (32 apart), gate and up in one launch.
//
// Q8 block is 36 bytes with a float scale (not Strata's fp16 d and d*sum).
// The dot only reads d. Keeping fp32 matches the existing quantizer so the
// 8e-2 gate stays on reduction order. KREG's 48-byte B70Q8 is untouched.
//
// B70_MOE_SCHED=0 keeps b70_launch_gu / b70_launch_down. Default is on.
// Fast path: T<=8, topk<=16, I and K in (0, 8192], multiples of 32, and every
// live type is DP4A (IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL, Q2_0). Otherwise the
// old fused kernels run. The plan is a device kernel so expert ids can change
// under a captured graph. The grid is the host ceiling (T*topk [+ shared]);
// extra groups exit. Scratch is reserved at that ceiling before capture.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

struct B70Q8S {
    float d;
    int8_t qs[32];
};
static_assert(sizeof(B70Q8S) == 36, "q8s block");
static_assert(offsetof(B70Q8S, qs) == 4, "q8s qs");

constexpr int kSchedMaxT = 8;
constexpr int kSchedMaxTopk = 16;
constexpr int kSchedMaxEnt = kSchedMaxT * kSchedMaxTopk + kSchedMaxT; // 136
constexpr int kSchedMaxGrp = kSchedMaxEnt;
constexpr int kSchedMaxDim = 8192;
constexpr int kSchedWg = 256;
constexpr int kSchedLn = 8;
constexpr int kSchedSg = 16;
constexpr int kSchedRb = kSchedWg / kSchedLn; // 32
constexpr int kSchedRows = kSchedRb * 2;      // 64 virtual rows per work-group

struct B70SchedPlan {
    int ng;
    int nent;
    int start[kSchedMaxGrp + 1];
    int tok[kSchedMaxEnt];
    int slot[kSchedMaxEnt];
    int expert[kSchedMaxGrp];
    int gtype[kSchedMaxGrp];
    int utype[kSchedMaxGrp];
    int dtype[kSchedMaxGrp];
    int grb[kSchedMaxGrp];
    int urb[kSchedMaxGrp];
    int drb[kSchedMaxGrp];
};

inline bool b70_sched_dp4a(int type) {
    return type == B70_IQ2_XXS || type == B70_IQ2_XS || type == B70_IQ2_S || type == B70_IQ3_XXS ||
           type == B70_IQ3_S || type == B70_IQ4_NL || type == B70_Q2_0;
}

inline bool b70_sched_enabled() {
    static const int on = [] {
        const char* e = std::getenv("B70_MOE_SCHED");
        if (e != nullptr && e[0] == '0' && e[1] == '\0') return 0;
        return 1;
    }();
    return on != 0;
}

inline size_t b70_sched_align(size_t n) { return (n + 255ull) & ~255ull; }

// Byte offsets into the one ceiling allocation. elem size 1 so a 36-byte
// block is not rounded down by b70_device_grow's 16-byte pad.
inline size_t b70_sched_bytes() {
    const size_t q8x = (size_t)kSchedMaxT * (size_t)(kSchedMaxDim / 32) * 36ull;
    const size_t q8h = (size_t)kSchedMaxEnt * (size_t)(kSchedMaxDim / 32) * 36ull;
    const size_t ff = (size_t)kSchedMaxEnt * (size_t)kSchedMaxDim * sizeof(float);
    size_t o = 0;
    o = b70_sched_align(o + sizeof(B70SchedPlan));
    o = b70_sched_align(o + q8x);
    o = b70_sched_align(o + q8h);
    o = b70_sched_align(o + ff);
    o = b70_sched_align(o + ff);
    o = b70_sched_align(o + ff);
    return o;
}

struct B70SchedMem {
    B70SchedPlan* plan = nullptr;
    uint8_t* q8x = nullptr;
    uint8_t* q8h = nullptr;
    float* gate = nullptr;
    float* up = nullptr;
    float* partial = nullptr;
};

inline B70SchedMem b70_sched_bind(uint8_t* base) {
    const size_t q8x = (size_t)kSchedMaxT * (size_t)(kSchedMaxDim / 32) * 36ull;
    const size_t q8h = (size_t)kSchedMaxEnt * (size_t)(kSchedMaxDim / 32) * 36ull;
    const size_t ff = (size_t)kSchedMaxEnt * (size_t)kSchedMaxDim * sizeof(float);
    B70SchedMem m;
    size_t o = 0;
    m.plan = reinterpret_cast<B70SchedPlan*>(base + o);
    o = b70_sched_align(o + sizeof(B70SchedPlan));
    m.q8x = base + o;
    o = b70_sched_align(o + q8x);
    m.q8h = base + o;
    o = b70_sched_align(o + q8h);
    m.gate = reinterpret_cast<float*>(base + o);
    o = b70_sched_align(o + ff);
    m.up = reinterpret_cast<float*>(base + o);
    o = b70_sched_align(o + ff);
    m.partial = reinterpret_cast<float*>(base + o);
    return m;
}

struct B70SchedArena {
    uint8_t* p = nullptr;
    size_t cap = 0;
};

inline B70SchedArena& b70_sched_arena() {
    static B70SchedArena a;
    return a;
}

inline B70SchedMem b70_sched_scratch(sycl::queue& q) {
    B70SchedArena& a = b70_sched_arena();
    const size_t n = b70_sched_bytes();
    void* raw = b70_device_grow(q, a.p, a.cap, n, 1, n);
    a.p = static_cast<uint8_t*>(raw);
    return b70_sched_bind(a.p);
}

inline void b70_sched_reserve(sycl::queue& q) { (void)b70_sched_scratch(q); }

inline B70Q8S* b70_q8s_ptr(uint8_t* base, size_t block) {
    return reinterpret_cast<B70Q8S*>(base + block * 36ull);
}
inline const B70Q8S* b70_q8s_cptr(const uint8_t* base, size_t block) {
    return reinterpret_cast<const B70Q8S*>(base + block * 36ull);
}

// xor 4,2,1 stays inside an aligned 8-lane rib of a 16-lane subgroup.
inline float b70_sched_lanes(sycl::sub_group sg, float v) {
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}

// One k-pass. Activation words are loaded once and applied to both rows and
// every entry in the chunk. Weight prep is once per live row.
template <int TYPE, int EC>
inline void b70_sched_pair(sycl::sub_group sg, const uint8_t* row0, const uint8_t* row1, bool l0, bool l1,
                           const B70Q8S* const xs[4], int nb, int sub, float a0[4], float a1[4]) {
    float s0[4];
    float s1[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        s0[e] = 0.f;
        s1[e] = 0.f;
    }
    if (l0 || l1) {
        for (int blk = sub; blk < nb; blk += kSchedLn) {
            int u[4][8];
            float d[4];
#pragma unroll
            for (int e = 0; e < EC; ++e) {
                const B70Q8S* a = xs[e] + blk;
                d[e] = a->d;
                __builtin_memcpy(u[e], a->qs, 32);
            }
            if (l0) {
                B70Part m;
                b70_prep<TYPE>(row0, blk, m);
#pragma unroll
                for (int e = 0; e < EC; ++e) s0[e] += b70_apply_u<TYPE>(m, u[e], d[e]);
            }
            if (l1) {
                B70Part m;
                b70_prep<TYPE>(row1, blk, m);
#pragma unroll
                for (int e = 0; e < EC; ++e) s1[e] += b70_apply_u<TYPE>(m, u[e], d[e]);
            }
        }
    }
#pragma unroll
    for (int e = 0; e < EC; ++e) {
        a0[e] = b70_sched_lanes(sg, s0[e]);
        a1[e] = b70_sched_lanes(sg, s1[e]);
    }
}

template <int EC>
inline void b70_sched_dot(sycl::sub_group sg, int type, const uint8_t* row0, const uint8_t* row1, bool l0, bool l1,
                          const B70Q8S* const xs[4], int nb, int sub, float a0[4], float a1[4]) {
    switch (type) {
    case B70_IQ2_XXS: b70_sched_pair<B70_IQ2_XXS, EC>(sg, row0, row1, l0, l1, xs, nb, sub, a0, a1); break;
    case B70_IQ2_XS: b70_sched_pair<B70_IQ2_XS, EC>(sg, row0, row1, l0, l1, xs, nb, sub, a0, a1); break;
    case B70_IQ2_S: b70_sched_pair<B70_IQ2_S, EC>(sg, row0, row1, l0, l1, xs, nb, sub, a0, a1); break;
    case B70_IQ3_XXS: b70_sched_pair<B70_IQ3_XXS, EC>(sg, row0, row1, l0, l1, xs, nb, sub, a0, a1); break;
    case B70_IQ3_S: b70_sched_pair<B70_IQ3_S, EC>(sg, row0, row1, l0, l1, xs, nb, sub, a0, a1); break;
    case B70_IQ4_NL: b70_sched_pair<B70_IQ4_NL, EC>(sg, row0, row1, l0, l1, xs, nb, sub, a0, a1); break;
    case B70_Q2_0: b70_sched_pair<B70_Q2_0, EC>(sg, row0, row1, l0, l1, xs, nb, sub, a0, a1); break;
    default: break;
    }
}

class b70_sched_plan_k;
class b70_sched_q8_k;
class b70_sched_q8f_k;
class b70_sched_gu_k;
class b70_sched_silu_k;
class b70_sched_down_k;
class b70_sched_red_k;

inline sycl::event b70_sched_plan(sycl::queue& q, B70SchedPlan* plan, const int32_t* ids, int T, int topk, int E,
                                  int gate_type, int up_type, int down_type, int grb, int urb, int drb, int sg_type,
                                  int su_type, int sd_type, int sgrb, int surb, int sdrb, int shared) {
    return q.submit([&](sycl::handler& h) {
        h.single_task<b70_sched_plan_k>([=]() {
            const int nslot = T * topk;
            int taken[kSchedMaxT * kSchedMaxTopk];
            for (int i = 0; i < nslot; ++i) taken[i] = 0;
            int cursor = 0;
            int g = 0;
            for (int i = 0; i < nslot; ++i) {
                if (taken[i]) continue;
                taken[i] = 1;
                const int id = ids[i];
                if (id < 0 || id >= E) continue;
                plan->start[g] = cursor;
                plan->expert[g] = id;
                plan->gtype[g] = gate_type;
                plan->utype[g] = up_type;
                plan->dtype[g] = down_type;
                plan->grb[g] = grb;
                plan->urb[g] = urb;
                plan->drb[g] = drb;
                plan->tok[cursor] = i / topk;
                plan->slot[cursor] = i - plan->tok[cursor] * topk;
                cursor += 1;
                for (int j = i + 1; j < nslot; ++j) {
                    if (taken[j]) continue;
                    if (ids[j] != id) continue;
                    taken[j] = 1;
                    plan->tok[cursor] = j / topk;
                    plan->slot[cursor] = j - plan->tok[cursor] * topk;
                    cursor += 1;
                }
                g += 1;
            }
            if (shared) {
                plan->start[g] = cursor;
                plan->expert[g] = -1;
                plan->gtype[g] = sg_type;
                plan->utype[g] = su_type;
                plan->dtype[g] = sd_type;
                plan->grb[g] = sgrb;
                plan->urb[g] = surb;
                plan->drb[g] = sdrb;
                for (int t = 0; t < T; ++t) {
                    plan->tok[cursor] = t;
                    plan->slot[cursor] = topk;
                    cursor += 1;
                }
                g += 1;
            }
            plan->start[g] = cursor;
            plan->ng = g;
            plan->nent = cursor;
        });
    });
}

inline sycl::event b70_sched_q8_bf16(sycl::queue& q, const uint16_t* x, uint8_t* yb, int M, int K) {
    const int nb = K / 32;
    const int n = M * nb;
    constexpr int L = 64;
    const size_t global = (size_t)((n + L - 1) / L) * (size_t)L;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for<b70_sched_q8_k>(sycl::nd_range<1>(global, (size_t)L), [=](sycl::nd_item<1> it) {
            const int i = (int)it.get_global_id(0);
            if (i >= n) return;
            const uint16_t* p = x + (size_t)i * 32;
            float xv[32];
            float m = 0.f;
#pragma unroll
            for (int t = 0; t < 32; ++t) {
                xv[t] = b70_bf16_to_fp32(p[t]);
                m = sycl::fmax(m, sycl::fabs(xv[t]));
            }
            const float id = m > 0.f ? 127.f / m : 0.f;
            B70Q8S* b = b70_q8s_ptr(yb, (size_t)i);
            b->d = m * (1.f / 127.f);
#pragma unroll
            for (int t = 0; t < 32; ++t) {
                int r = (int)sycl::rint(xv[t] * id);
                if (r > 127) r = 127;
                if (r < -127) r = -127;
                b->qs[t] = (int8_t)r;
            }
        });
    });
}

inline sycl::event b70_sched_q8_f32(sycl::queue& q, const float* x, uint8_t* yb, const B70SchedPlan* plan, int rows,
                                    int K, sycl::event dep) {
    const int nb = K / 32;
    const int n = rows * nb;
    constexpr int L = 64;
    const size_t global = (size_t)((n + L - 1) / L) * (size_t)L;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for<b70_sched_q8f_k>(sycl::nd_range<1>(global, (size_t)L), [=](sycl::nd_item<1> it) {
            const int i = (int)it.get_global_id(0);
            if (i >= n) return;
            const int row = nb > 0 ? i / nb : 0;
            if (row >= plan->nent) return;
            const float* p = x + (size_t)i * 32;
            float m = 0.f;
#pragma unroll
            for (int t = 0; t < 32; ++t) m = sycl::fmax(m, sycl::fabs(p[t]));
            const float id = m > 0.f ? 127.f / m : 0.f;
            B70Q8S* b = b70_q8s_ptr(yb, (size_t)i);
            b->d = m * (1.f / 127.f);
#pragma unroll
            for (int t = 0; t < 32; ++t) {
                int r = (int)sycl::rint(p[t] * id);
                if (r > 127) r = 127;
                if (r < -127) r = -127;
                b->qs[t] = (int8_t)r;
            }
        });
    });
}

inline const uint8_t* b70_sched_gu_ptr(int v, int I, int expert, int grb, int urb, const uint8_t* gate, const uint8_t* up,
                                       const uint8_t* sgate, const uint8_t* sup) {
    const int vv = v < 2 * I ? v : 0;
    const bool is_up = vv >= I;
    const int r = is_up ? vv - I : vv;
    const bool sh = expert < 0;
    const uint8_t* base = is_up ? (sh ? sup : up) : (sh ? sgate : gate);
    const int rb = is_up ? urb : grb;
    const int ex = sh ? 0 : expert;
    return base + ((size_t)ex * (size_t)I + (size_t)r) * (size_t)rb;
}

inline sycl::event b70_sched_gu(sycl::queue& q, const B70SchedPlan* plan, const uint8_t* q8x, const uint8_t* gate,
                                const uint8_t* up, const uint8_t* sgate, const uint8_t* sup, float* gout, float* uout,
                                int host_groups, int I, int K, sycl::event dep_plan, sycl::event dep_q8) {
    const int gx = (2 * I) / kSchedRows;
    const size_t global = (size_t)host_groups * (size_t)gx * (size_t)kSchedWg;
    const int nb = K / 32;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(std::vector<sycl::event>{dep_plan, dep_q8});
        h.parallel_for<b70_sched_gu_k>(sycl::nd_range<1>(global, (size_t)kSchedWg),
                                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSchedSg)]] {
                                           const int wg = (int)it.get_group(0);
                                           const int g = gx > 0 ? wg / gx : 0;
                                           const int row_wg = gx > 0 ? wg - g * gx : 0;
                                           if (g >= plan->ng) return;
                                           if (row_wg * kSchedRows >= 2 * I) return;
                                           const int lid = (int)it.get_local_id(0);
                                           const int rib = lid / kSchedLn;
                                           const int sub = lid - rib * kSchedLn;
                                           auto sg = it.get_sub_group();
                                           const int va = row_wg * kSchedRows + rib;
                                           const int vb = va + kSchedRb;
                                           const bool l0 = va < 2 * I;
                                           const bool l1 = vb < 2 * I;
                                           const int expert = plan->expert[g];
                                           const int grb = plan->grb[g];
                                           const int urb = plan->urb[g];
                                           const int gtype = plan->gtype[g];
                                           const int utype = plan->utype[g];
                                           const uint8_t* row0 =
                                               b70_sched_gu_ptr(va, I, expert, grb, urb, gate, up, sgate, sup);
                                           const uint8_t* row1 =
                                               b70_sched_gu_ptr(vb, I, expert, grb, urb, gate, up, sgate, sup);
                                           const bool up0 = l0 && va >= I;
                                           const bool up1 = l1 && vb >= I;
                                           const int ty0 = up0 ? utype : gtype;
                                           const int ty1 = up1 ? utype : gtype;
                                           const int r0 = up0 ? va - I : va;
                                           const int r1 = up1 ? vb - I : vb;
                                           const bool same = (!l0 || !l1) || ty0 == ty1;
                                           const int e0 = plan->start[g];
                                           const int e1 = plan->start[g + 1];
                                           const int xb = nb;
                                           auto store = [&](int e, int ec, const float* a0, const float* a1) {
                                               if (sub != 0) return;
#pragma unroll
                                               for (int j = 0; j < 4; ++j) {
                                                   if (j >= ec) break;
                                                   const size_t slot = (size_t)(e + j) * (size_t)I;
                                                   if (l0) {
                                                       float* d = up0 ? uout : gout;
                                                       d[slot + (size_t)r0] = a0[j];
                                                   }
                                                   if (l1) {
                                                       float* d = up1 ? uout : gout;
                                                       d[slot + (size_t)r1] = a1[j];
                                                   }
                                               }
                                           };
                                           auto chunk = [&](int e, int ec) {
                                               const B70Q8S* xs[4];
                                               const int t0 = plan->tok[e];
                                               const B70Q8S* x0 = b70_q8s_cptr(q8x, (size_t)t0 * (size_t)xb);
#pragma unroll
                                               for (int j = 0; j < 4; ++j) xs[j] = x0;
                                               for (int j = 0; j < ec; ++j) {
                                                   const int t = plan->tok[e + j];
                                                   xs[j] = b70_q8s_cptr(q8x, (size_t)t * (size_t)xb);
                                               }
                                               float a0[4], a1[4];
                                               if (same) {
                                                   const int ty = l0 ? ty0 : ty1;
                                                   if (ec == 4)
                                                       b70_sched_dot<4>(sg, ty, row0, row1, l0, l1, xs, nb, sub, a0, a1);
                                                   else if (ec == 2)
                                                       b70_sched_dot<2>(sg, ty, row0, row1, l0, l1, xs, nb, sub, a0, a1);
                                                   else
                                                       b70_sched_dot<1>(sg, ty, row0, row1, l0, l1, xs, nb, sub, a0, a1);
                                                   store(e, ec, a0, a1);
                                               } else {
                                                   float z0[4], z1[4];
                                                   if (ec == 4) {
                                                       b70_sched_dot<4>(sg, ty0, row0, row0, l0, false, xs, nb, sub, a0, z0);
                                                       b70_sched_dot<4>(sg, ty1, row1, row1, l1, false, xs, nb, sub, a1, z1);
                                                   } else if (ec == 2) {
                                                       b70_sched_dot<2>(sg, ty0, row0, row0, l0, false, xs, nb, sub, a0, z0);
                                                       b70_sched_dot<2>(sg, ty1, row1, row1, l1, false, xs, nb, sub, a1, z1);
                                                   } else {
                                                       b70_sched_dot<1>(sg, ty0, row0, row0, l0, false, xs, nb, sub, a0, z0);
                                                       b70_sched_dot<1>(sg, ty1, row1, row1, l1, false, xs, nb, sub, a1, z1);
                                                   }
                                                   store(e, ec, a0, a1);
                                               }
                                           };
                                           int e = e0;
                                           for (; e + 4 <= e1; e += 4) chunk(e, 4);
                                           if (e + 2 <= e1) {
                                               chunk(e, 2);
                                               e += 2;
                                           }
                                           if (e < e1) chunk(e, 1);
                                       });
    });
}

inline sycl::event b70_sched_silu(sycl::queue& q, const B70SchedPlan* plan, const float* gate, const float* up, float* h,
                                  int host_ent, int I, sycl::event dep) {
    const int n = host_ent * I;
    constexpr int L = 128;
    const size_t global = (size_t)((n + L - 1) / L) * (size_t)L;
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dep);
        cgh.parallel_for<b70_sched_silu_k>(sycl::nd_range<1>(global, (size_t)L), [=](sycl::nd_item<1> it) {
            const int i = (int)it.get_global_id(0);
            if (i >= n) return;
            const int e = i / I;
            if (e >= plan->nent) return;
            const int r = i - e * I;
            const float rg = gate[(size_t)e * (size_t)I + r];
            const float ru = up[(size_t)e * (size_t)I + r];
            h[(size_t)e * (size_t)I + r] = rg * (1.f / (1.f + sycl::exp(-rg))) * ru;
        });
    });
}

inline sycl::event b70_sched_down(sycl::queue& q, const B70SchedPlan* plan, const uint8_t* q8h, const uint8_t* down,
                                  const uint8_t* sdown, float* partial, int host_groups, int I, int K, sycl::event dep) {
    const int gx = (K + kSchedRows - 1) / kSchedRows;
    const size_t global = (size_t)host_groups * (size_t)gx * (size_t)kSchedWg;
    const int nb = I / 32;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for<b70_sched_down_k>(
            sycl::nd_range<1>(global, (size_t)kSchedWg),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSchedSg)]] {
                const int wg = (int)it.get_group(0);
                const int g = gx > 0 ? wg / gx : 0;
                const int row_wg = gx > 0 ? wg - g * gx : 0;
                if (g >= plan->ng) return;
                if (row_wg * kSchedRows >= K) return;
                const int lid = (int)it.get_local_id(0);
                const int rib = lid / kSchedLn;
                const int sub = lid - rib * kSchedLn;
                auto sg = it.get_sub_group();
                const int ra = row_wg * kSchedRows + rib;
                const int rb = ra + kSchedRb;
                const bool l0 = ra < K;
                const bool l1 = rb < K;
                const int expert = plan->expert[g];
                const bool sh = expert < 0;
                const uint8_t* base = sh ? sdown : down;
                const int ex = sh ? 0 : expert;
                const int drb = plan->drb[g];
                const int ty = plan->dtype[g];
                const int r0 = l0 ? ra : 0;
                const int r1 = l1 ? rb : 0;
                const uint8_t* row0 = base + ((size_t)ex * (size_t)K + (size_t)r0) * (size_t)drb;
                const uint8_t* row1 = base + ((size_t)ex * (size_t)K + (size_t)r1) * (size_t)drb;
                const int e0 = plan->start[g];
                const int e1 = plan->start[g + 1];
                const int hb = nb;
                auto store = [&](int e, int ec, const float* a0, const float* a1) {
                    if (sub != 0) return;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (j >= ec) break;
                        if (l0) partial[(size_t)(e + j) * (size_t)K + r0] = a0[j];
                        if (l1) partial[(size_t)(e + j) * (size_t)K + r1] = a1[j];
                    }
                };
                auto chunk = [&](int e, int ec) {
                    const B70Q8S* xs[4];
                    const B70Q8S* x0 = b70_q8s_cptr(q8h, (size_t)e * (size_t)hb);
#pragma unroll
                    for (int j = 0; j < 4; ++j) xs[j] = x0;
                    for (int j = 0; j < ec; ++j) xs[j] = b70_q8s_cptr(q8h, (size_t)(e + j) * (size_t)hb);
                    float a0[4], a1[4];
                    if (ec == 4)
                        b70_sched_dot<4>(sg, ty, row0, row1, l0, l1, xs, nb, sub, a0, a1);
                    else if (ec == 2)
                        b70_sched_dot<2>(sg, ty, row0, row1, l0, l1, xs, nb, sub, a0, a1);
                    else
                        b70_sched_dot<1>(sg, ty, row0, row1, l0, l1, xs, nb, sub, a0, a1);
                    store(e, ec, a0, a1);
                };
                int e = e0;
                for (; e + 4 <= e1; e += 4) chunk(e, 4);
                if (e + 2 <= e1) {
                    chunk(e, 2);
                    e += 2;
                }
                if (e < e1) chunk(e, 1);
            });
    });
}

inline sycl::event b70_sched_red(sycl::queue& q, const B70SchedPlan* plan, const float* partial, const float* router,
                                 float* y, int T, int topk, int K, sycl::event dep) {
    const int n = T * K;
    constexpr int L = 128;
    const size_t global = (size_t)((n + L - 1) / L) * (size_t)L;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for<b70_sched_red_k>(sycl::nd_range<1>(global, (size_t)L), [=](sycl::nd_item<1> it) {
            const int i = (int)it.get_global_id(0);
            if (i >= n) return;
            const int t = i / K;
            const int k = i - t * K;
            const int nent = plan->nent;
            float s = 0.f;
            for (int e = 0; e < nent; ++e) {
                if (plan->tok[e] != t) continue;
                const int slot = plan->slot[e];
                const float w = slot < topk ? router[(size_t)t * (size_t)topk + slot] : 1.f;
                s = sycl::fma(partial[(size_t)e * (size_t)K + k], w, s);
            }
            y[i] = s;
        });
    });
}

// Submits the whole layer, or returns false so the caller keeps the old kernels.
inline bool b70_sched_try(sycl::queue& q, const uint16_t* x, const uint8_t* gate, int gate_type, int gate_rb,
                          const uint8_t* up, int up_type, int up_rb, const uint8_t* down, int down_type, int down_rb,
                          const int32_t* ids, const float* router, const uint8_t* sgate, int sg_type, int sg_rb,
                          const uint8_t* sup, int su_type, int su_rb, const uint8_t* sdown, int sd_type, int sd_rb,
                          float* y, int T, int topk, int E, int I, int K) {
    if (!b70_sched_enabled()) return false;
    if (T <= 0 || topk <= 0 || T > kSchedMaxT || topk > kSchedMaxTopk) return false;
    if (I <= 0 || K <= 0 || I > kSchedMaxDim || K > kSchedMaxDim) return false;
    if ((I % 32) != 0 || (K % 32) != 0) return false;
    if (!b70_sched_dp4a(gate_type) || !b70_sched_dp4a(up_type) || !b70_sched_dp4a(down_type)) return false;
    const int shared = sgate != nullptr;
    if (shared && (!b70_sched_dp4a(sg_type) || !b70_sched_dp4a(su_type) || !b70_sched_dp4a(sd_type))) return false;
    const B70SchedMem mem = b70_sched_scratch(q);
    const int host_groups = T * topk + shared;
    const int host_ent = T * topk + (shared ? T : 0);
    const sycl::event ep = b70_sched_plan(q, mem.plan, ids, T, topk, E, gate_type, up_type, down_type, gate_rb, up_rb,
                                          down_rb, sg_type, su_type, sd_type, sg_rb, su_rb, sd_rb, shared);
    const sycl::event eq = b70_sched_q8_bf16(q, x, mem.q8x, T, K);
    const sycl::event egu = b70_sched_gu(q, mem.plan, mem.q8x, gate, up, sgate, sup, mem.gate, mem.up, host_groups, I, K,
                                         ep, eq);
    // Hidden floats reuse the gate buffer after silu. Up is still live until silu reads it.
    const sycl::event es = b70_sched_silu(q, mem.plan, mem.gate, mem.up, mem.up, host_ent, I, egu);
    const sycl::event eh = b70_sched_q8_f32(q, mem.up, mem.q8h, mem.plan, host_ent, I, es);
    const sycl::event ed = b70_sched_down(q, mem.plan, mem.q8h, down, sdown, mem.partial, host_groups, I, K, eh);
    b70_sched_red(q, mem.plan, mem.partial, router, y, T, topk, K, ed);
    return true;
}
