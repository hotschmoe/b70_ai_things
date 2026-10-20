// Grouped decode MoE schedule (C9 geometry, C12 per-type launches).
// Include once, after the anonymous namespace in kernels.cpp, so b70_prep /
// b70_apply_u / b70_decode32 / b70_device_grow are visible. This header stays
// at global scope so the SYCL kernel-name templates can too.
//
// Attribution: Strata night lab, iq_kernels.dp.cpp (worktree 1c360526).
//   rows_dot_multi_iq        ~2264  activation once, reused across rows and entries
//   native_gu_kernel_rows    ~2302  templated on ONE gate type, RPL=2, chunks 4/2/1
//   native_down_kernel_rows  ~2359  templated on ONE down type
//   STRATA_EXPERT_LANES      ~799   default 8; lanes_sum xor offsets 4,2,1
//   launch_gu_rows_k         ~4962  reqd_sub_group_size(16); 3D nd_range
// R2 does not use SLM. The AMD LDS path is STRATA_HIP_GFX906 only.
// The reduce is a separate routing-sum kernel; down writes partials.
// C12 keeps the R2 geometry (WG 256, SG16, 8 lanes, 2 rows 32 apart) and
// launches one GU kernel per distinct gate/up type and one DOWN kernel per
// distinct down type. The grid stays 1D. Empty groups return.
//
// Q8 block is 36 bytes with a float scale (not Strata's fp16 d and d*sum).
// The dot only reads d. fp32 matches the existing quantizer so the 8e-2 gate
// stays on reduction order. KREG's 48-byte B70Q8 is untouched.
// qs sits at offset 4, so the k-loop loads eight aligned ints.
//
// B70_MOE_SCHED=0 keeps b70_launch_gu / b70_launch_down. Default is on.
// Fast path: T<=8, topk<=16, I and K in (0, 8192], multiples of 32, and every
// live type is Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS/XS/S, IQ3_XXS/S,
// IQ4_NL, IQ4_XS, or Q2_0. Otherwise the old fused kernels run.
// All 14 types use b70_prep + DP4A (O3 stage 2 added the K-quants and IQ4_XS;
// -DB70_KQ_DP4A=0 sends those back to b70_decode32 + fma).
// The plan is a device kernel so expert ids can change under a captured graph.
// The grid is the host ceiling (T*topk [+ shared]); extra groups exit.
// Scratch is reserved at that ceiling before capture.
//
// O3 (2026-10-20): the 2x gap to Strata R2 was the weight loads, not the
// schedule. b70_prep read 8/4-byte fields at 2-aligned block addresses through
// memcpy, which IGC lowered to d64/d32 LSC loads at misaligned addresses
// (gate/up 123 us -> 44 us, down 38 -> 16.5 us at T=1 once those became the
// aligned-dword loads of Strata load8_a2; see iq_dot.hpp). The single-thread
// plan kernel (8.5 us T=1, 74 us T=4) is now one parallel work-group fused
// with Q8(x), and SiLU is fused with Q8(hidden): 5 launches per layer.
//
// B70_MOE_PHASE=1 prints C12_PHASE on stderr (host submit+wait, serialized).
// It drains the queue. Do not set it during graph capture.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

struct B70Q8S {
    float d;
    int8_t qs[32];
};
static_assert(sizeof(B70Q8S) == 36, "q8s block");
static_assert(offsetof(B70Q8S, qs) == 4, "q8s qs");

// B70_SCHED_FUSE=0 at compile time restores the separate plan, Q8(x), SiLU and
// Q8(hidden) launches (O3 A/B). Default fused.
#ifndef B70_SCHED_FUSE
#define B70_SCHED_FUSE 1
#endif

constexpr int kSchedMaxT = 8;
constexpr int kSchedMaxTopk = 16;
constexpr int kSchedMaxEnt = kSchedMaxT * kSchedMaxTopk + kSchedMaxT; // 136
constexpr int kSchedMaxGrp = kSchedMaxEnt;
constexpr int kSchedMaxDim = 8192;
constexpr int kSchedWg = 256;
#ifndef B70_PROBE_LOADONLY
#define B70_PROBE_LOADONLY 0
#endif
#ifndef B70_PTR_WALK
#define B70_PTR_WALK 1
#endif
constexpr int b70_c_elems(int t) {
    return (t == B70_Q8_0 || t == B70_IQ4_NL) ? 32 : t == B70_Q2_0 ? 64 : 256;
}
constexpr int b70_c_bytes(int t) {
    switch (t) {
    case B70_Q8_0: return 34;
    case B70_Q4_K: return 144;
    case B70_Q5_K: return 176;
    case B70_Q6_K: return 210;
    case B70_IQ2_XXS: return 66;
    case B70_IQ2_XS: return 74;
    case B70_IQ3_XXS: return 98;
    case B70_IQ4_NL: return 18;
    case B70_IQ3_S: return 110;
    case B70_IQ2_S: return 82;
    case B70_IQ4_XS: return 136;
    case B70_Q2_K: return 84;
    case B70_Q3_K: return 110;
    case B70_Q2_0: return 18;
    default: return 0;
    }
}
#ifndef B70_GRID_SLM
#define B70_GRID_SLM 1
#endif
constexpr int kSchedLn = 8;
constexpr int kSchedSg = 16;
constexpr int kSchedRb = kSchedWg / kSchedLn; // 32
// Rows per 8-lane rib (Strata RPL). -DB70_SCHED_RPL=4 for the bandwidth A/B.
#ifndef B70_SCHED_RPL
#define B70_SCHED_RPL 2
#endif
constexpr int kSchedRpl = B70_SCHED_RPL;
constexpr int kSchedRows = kSchedRb * kSchedRpl; // virtual rows per work-group
// Down kernel lanes per row (I is short: 20 parts per row at I=640). -DB70_DOWN_LN=4 for the A/B.
#ifndef B70_DOWN_LN
#define B70_DOWN_LN 8
#endif
constexpr int kDownLn = B70_DOWN_LN;
constexpr int kDownRb = kSchedWg / kDownLn;
constexpr int kDownRows = kDownRb * kSchedRpl;

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
    float sw[kSchedMaxT]; // O3 stage 3: per-token shared weight sigmoid(x . w_sg)
};

#define B70_SCHED_TYPES(M)                                                                                             \
    M(B70_Q8_0)                                                                                                        \
    M(B70_Q2_K)                                                                                                        \
    M(B70_Q3_K)                                                                                                        \
    M(B70_Q4_K)                                                                                                        \
    M(B70_Q5_K)                                                                                                        \
    M(B70_Q6_K)                                                                                                        \
    M(B70_IQ2_XXS)                                                                                                     \
    M(B70_IQ2_XS)                                                                                                      \
    M(B70_IQ2_S)                                                                                                       \
    M(B70_IQ3_XXS)                                                                                                     \
    M(B70_IQ3_S)                                                                                                       \
    M(B70_IQ4_NL)                                                                                                      \
    M(B70_IQ4_XS)                                                                                                      \
    M(B70_Q2_0)

inline bool b70_sched_type(int type) {
    switch (type) {
#define B70_SCHED_CASE(T)                                                                                              \
    case T:                                                                                                            \
        return true;
        B70_SCHED_TYPES(B70_SCHED_CASE)
#undef B70_SCHED_CASE
    default:
        return false;
    }
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

inline B70SchedArena& b70_sched_arena(sycl::queue& q) {
    static B70SchedArena a[2];
    return a[b70_device_slot(q)];
}

inline B70SchedMem b70_sched_scratch(sycl::queue& q) {
    B70SchedArena& a = b70_sched_arena(q);
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
template <int LN = kSchedLn>
inline float b70_sched_lanes(sycl::sub_group sg, float v) {
#pragma unroll
    for (int o = LN / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}

// qs is 4-aligned (block base is 4-aligned, offsetof(qs) == 4). Eight ints,
// same bytes the old 32-byte memcpy moved, without the memcpy call in the k-loop.
inline void b70_q8s_i8(const B70Q8S* a, int u[8]) {
    const int* p = reinterpret_cast<const int*>(a->qs);
#pragma unroll
    for (int i = 0; i < 8; ++i) u[i] = p[i];
}

// One k-pass over R rows (Strata rows_dot_multi_iq, RPL=R). Dead rows repeat a
// live pointer and are not stored. Activation words load once per block and
// are reused across the R rows and EC entries.
template <int TYPE, int EC, int R, class G = B70GridG, int LN = kSchedLn>
inline void b70_sched_rows(sycl::sub_group sg, const uint8_t* const (&rows)[R], const B70Q8S* const (&xs)[EC], int nb,
                           int sub, float (&out)[R][EC], const G& grid = G{}) {
    float s[R][EC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < EC; ++e) s[r][e] = 0.f;
#if B70_PTR_WALK
    if constexpr (b70_dp4a_type<TYPE>() && LN % (b70_c_elems(TYPE) / 32) == 0) {
        // O3 stage 3: walk block pointers. With P parts per block and P | 8,
        // lane sub always takes part (sub % P) of blocks sub/P, +8/P, ... so the
        // per-iteration 64-bit block multiply becomes one pointer add.
        constexpr int P = b70_c_elems(TYPE) / 32;
        constexpr int BB = b70_c_bytes(TYPE);
        const int gl = sub & (P - 1);
        const uint8_t* rp[R];
#pragma unroll
        for (int r = 0; r < R; ++r) rp[r] = rows[r] + (size_t)(sub / P) * BB;
        const B70Q8S* xp[EC];
#pragma unroll
        for (int e = 0; e < EC; ++e) xp[e] = xs[e] + sub;
        for (int blk = sub; blk < nb; blk += LN) {
            int u[EC][8];
            float d[EC];
#pragma unroll
            for (int e = 0; e < EC; ++e) {
                d[e] = xp[e]->d;
                b70_q8s_i8(xp[e], u[e]);
                xp[e] += LN;
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#if B70_PROBE_LOADONLY
                if constexpr (TYPE == B70_IQ3_S) {
                    // Probe only: the weight loads of b70_prep<IQ3_S>, almost no ALU.
                    const uint8_t* b = rp[r];
                    const uint64_t qq = b70_ld8_a2(b + 2 + 8 * gl);
                    const uint32_t sb4 = b70_ld4_a2(b + 74 + 4 * gl);
                    const uint32_t x = (uint32_t)qq ^ (uint32_t)(qq >> 32) ^ sb4 ^ b[66 + gl] ^ b[106 + (gl >> 1)];
                    rp[r] += (LN / P) * BB;
#pragma unroll
                    for (int e = 0; e < EC; ++e) s[r][e] += (float)(x & 0xffu) * b70_h2f(b) * d[e] + (float)u[e][0];
                    continue;
                }
#endif
                B70Part m;
                b70_prep<TYPE>(rp[r], gl, m, grid);
                rp[r] += (LN / P) * BB;
#pragma unroll
                for (int e = 0; e < EC; ++e) s[r][e] += b70_apply_u<TYPE>(m, u[e], d[e]);
            }
        }
    } else
#endif
    for (int blk = sub; blk < nb; blk += LN) {
        if constexpr (b70_dp4a_type<TYPE>()) {
            int u[EC][8];
            float d[EC];
#pragma unroll
            for (int e = 0; e < EC; ++e) {
                const B70Q8S* a = xs[e] + blk;
                d[e] = a->d;
                b70_q8s_i8(a, u[e]);
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                B70Part m;
                b70_prep<TYPE>(rows[r], blk, m, grid);
#pragma unroll
                for (int e = 0; e < EC; ++e) s[r][e] += b70_apply_u<TYPE>(m, u[e], d[e]);
            }
        } else {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float w[32];
                b70_decode32<TYPE>(rows[r], blk, w);
#pragma unroll
                for (int e = 0; e < EC; ++e) {
                    const B70Q8S* a = xs[e] + blk;
                    float acc = 0.f;
#pragma unroll
                    for (int t = 0; t < 32; ++t) acc = sycl::fma(w[t], (float)a->qs[t], acc);
                    s[r][e] += acc * a->d;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int e = 0; e < EC; ++e) out[r][e] = b70_sched_lanes<LN>(sg, s[r][e]);
}

class b70_sched_plan_k;
class b70_sched_planq8_k;
class b70_sched_siluq8_k;
class b70_sched_q8_k;
class b70_sched_q8f_k;
class b70_sched_silu_k;
class b70_sched_red_k;
template <int TYPE>
class b70_sched_gu_k;
template <int TYPE>
class b70_sched_down_k;

inline sycl::nd_range<3> b70_sched_grid(int groups, int gx) {
    return sycl::nd_range<3>(sycl::range<3>(1, (size_t)groups, (size_t)gx * (size_t)kSchedWg),
                             sycl::range<3>(1, 1, (size_t)kSchedWg));
}

inline void b70_sched_deps(sycl::handler& h, const sycl::event* deps, int ndeps) {
    if (ndeps <= 0) return;
    if (ndeps == 1) {
        h.depends_on(deps[0]);
        return;
    }
    h.depends_on(std::vector<sycl::event>(deps, deps + ndeps));
}

inline bool b70_sched_phase() {
    static const int on = [] {
        const char* e = std::getenv("B70_MOE_PHASE");
        return (e != nullptr && e[0] == '1' && e[1] == '\0') ? 1 : 0;
    }();
    return on != 0;
}

inline void b70_sched_push(int* types, int& n, int v) {
    for (int i = 0; i < n; ++i) {
        if (types[i] == v) return;
    }
    types[n++] = v;
}

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

// O3 (reports/2O3_moe_decode.md): plan + Q8(x) in one launch. Work-group 0 builds the plan in parallel
// (one work-item per slot, ids staged in SLM). The other work-groups quantize
// x, eight lanes per 32-value block. Same plan bytes as the old single_task
// (groups in first-occurrence slot order, entries in slot order, shared group
// last) and the same Q8 bytes as b70_sched_q8_bf16 (amax is order-free).
constexpr int kSchedPlanWg = 256;
static_assert(kSchedPlanWg >= kSchedMaxT * kSchedMaxTopk, "plan work-group covers every slot");

inline sycl::event b70_sched_plan_q8(sycl::queue& q, B70SchedPlan* plan, const int32_t* ids, int T, int topk, int E,
                                     int gate_type, int up_type, int down_type, int grb, int urb, int drb, int sg_type,
                                     int su_type, int sd_type, int sgrb, int surb, int sdrb, int shared,
                                     const uint16_t* x, uint8_t* yb, int K, const void* sgi = nullptr,
                                     int sgi_dtype = 0) {
    const int nb = K / 32;
    const int nq = T * nb * 8;
    // O3 stage 3: work-groups 1..T compute the shared-expert gate
    // sigmoid(x[t] . w_sg) when sgi is given (bf16 = 1, fp32 = 2).
    const int nsg = sgi != nullptr ? T : 0;
    const int nwg = 1 + nsg + (nq + kSchedPlanWg - 1) / kSchedPlanWg;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> sid(sycl::range<1>(kSchedMaxT * kSchedMaxTopk), h);
        sycl::local_accessor<int, 1> scnt(sycl::range<1>(kSchedMaxT * kSchedMaxTopk), h);
        h.parallel_for<b70_sched_planq8_k>(
            sycl::nd_range<1>((size_t)nwg * kSchedPlanWg, (size_t)kSchedPlanWg),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSchedSg)]] {
                const int wg = (int)it.get_group(0);
                const int lid = (int)it.get_local_id(0);
                if (wg == 0) {
                    const int nslot = T * topk;
                    const int i = lid;
                    const int id = i < nslot ? ids[i] : -1;
                    const bool valid = i < nslot && id >= 0 && id < E;
                    if (i < nslot) sid[i] = id;
                    sycl::group_barrier(it.get_group());
                    int first = i, rank = 0, cnt = 0;
                    if (valid) {
                        first = nslot;
                        for (int j = 0; j < nslot; ++j) {
                            const bool same = sid[j] == id;
                            if (same && j < first) first = j;
                            if (same && j < i) rank += 1;
                            if (same) cnt += 1;
                        }
                    }
                    const bool lead = valid && first == i;
                    if (i < nslot) scnt[i] = lead ? cnt : 0;
                    sycl::group_barrier(it.get_group());
                    if (valid) {
                        int g = 0, st = 0;
                        for (int j = 0; j < first; ++j) {
                            const int c = scnt[j];
                            g += c > 0 ? 1 : 0;
                            st += c;
                        }
                        const int pos = st + rank;
                        plan->tok[pos] = i / topk;
                        plan->slot[pos] = i - (i / topk) * topk;
                        if (lead) {
                            plan->start[g] = st;
                            plan->expert[g] = id;
                            plan->gtype[g] = gate_type;
                            plan->utype[g] = up_type;
                            plan->dtype[g] = down_type;
                            plan->grb[g] = grb;
                            plan->urb[g] = urb;
                            plan->drb[g] = drb;
                        }
                    }
                    const int ngr = sycl::reduce_over_group(it.get_group(), lead ? 1 : 0, sycl::plus<int>());
                    const int ner = sycl::reduce_over_group(it.get_group(), valid ? 1 : 0, sycl::plus<int>());
                    if (shared) {
                        if (lid < T) {
                            plan->tok[ner + lid] = lid;
                            plan->slot[ner + lid] = topk;
                        }
                        if (lid == 0) {
                            plan->start[ngr] = ner;
                            plan->expert[ngr] = -1;
                            plan->gtype[ngr] = sg_type;
                            plan->utype[ngr] = su_type;
                            plan->dtype[ngr] = sd_type;
                            plan->grb[ngr] = sgrb;
                            plan->urb[ngr] = surb;
                            plan->drb[ngr] = sdrb;
                        }
                    }
                    if (lid == 0) {
                        const int ng = ngr + (shared ? 1 : 0);
                        const int ne = ner + (shared ? T : 0);
                        plan->start[ng] = ne;
                        plan->ng = ng;
                        plan->nent = ne;
                    }
                    return;
                }
                if (wg <= nsg) {
                    const int t = wg - 1;
                    const uint16_t* xr = x + (size_t)t * (size_t)K;
                    float acc = 0.f;
                    if (sgi_dtype == 1) {
                        const uint16_t* w = static_cast<const uint16_t*>(sgi);
                        for (int k = lid; k < K; k += kSchedPlanWg)
                            acc = sycl::fma(b70_bf16_to_fp32(xr[k]), b70_bf16_to_fp32(w[k]), acc);
                    } else {
                        const float* w = static_cast<const float*>(sgi);
                        for (int k = lid; k < K; k += kSchedPlanWg) acc = sycl::fma(b70_bf16_to_fp32(xr[k]), w[k], acc);
                    }
                    const float dot = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
                    if (lid == 0) plan->sw[t] = 1.f / (1.f + sycl::exp(-dot));
                    return;
                }
                // Q8(x): lane l of an 8-lane rib takes values 4l..4l+3 of block blk.
                const int gid = (wg - 1 - nsg) * kSchedPlanWg + lid;
                const int blk0 = gid >> 3;
                const int lane = gid & 7;
                const bool live = gid < nq;
                const int blk = live ? blk0 : 0;
                const uint16_t* p = x + (size_t)blk * 32 + lane * 4;
                const uint64_t raw = *reinterpret_cast<const uint64_t*>(p);
                float v[4];
                float m = 0.f;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    v[t] = b70_bf16_to_fp32((uint16_t)(raw >> (16 * t)));
                    m = sycl::fmax(m, sycl::fabs(v[t]));
                }
                auto sg = it.get_sub_group();
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) m = sycl::fmax(m, sycl::permute_group_by_xor(sg, m, o));
                if (!live) return;
                const float idv = m > 0.f ? 127.f / m : 0.f;
                uint32_t packed = 0;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int r = (int)sycl::rint(v[t] * idv);
                    if (r > 127) r = 127;
                    if (r < -127) r = -127;
                    packed |= (uint32_t)(uint8_t)(int8_t)r << (8 * t);
                }
                uint8_t* b = yb + (size_t)blk * 36ull;
                if (lane == 0) *reinterpret_cast<float*>(b) = m * (1.f / 127.f);
                *reinterpret_cast<uint32_t*>(b + 4 + 4 * lane) = packed;
            });
    });
}

// O3 (reports/2O3_moe_decode.md): SiLU(gate) * up, then Q8 of the hidden row, in one launch. Eight lanes
// per 32-value block. Same expression and rounding as b70_sched_silu followed
// by b70_sched_q8_f32, so the Q8 bytes are identical; the float hidden row is
// never written.
inline sycl::event b70_sched_silu_q8(sycl::queue& q, const B70SchedPlan* plan, const float* gate, const float* up,
                                     uint8_t* yb, int host_ent, int I, sycl::event dep) {
    const int nb = I / 32;
    const int n = host_ent * nb * 8;
    constexpr int L = 256;
    const size_t global = (size_t)((n + L - 1) / L) * (size_t)L;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        h.parallel_for<b70_sched_siluq8_k>(
            sycl::nd_range<1>(global, (size_t)L), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSchedSg)]] {
                const int gid = (int)it.get_global_id(0);
                const int blk0 = gid >> 3;
                const int lane = gid & 7;
                const int nent = plan->nent;
                const bool live = gid < n && blk0 / nb < nent;
                const int blk = live ? blk0 : 0;
                const size_t off = (size_t)blk * 32 + (size_t)lane * 4;
                const sycl::float4 g4 = *reinterpret_cast<const sycl::float4*>(gate + off);
                const sycl::float4 u4 = *reinterpret_cast<const sycl::float4*>(up + off);
                float hv[4];
                float m = 0.f;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const float rg = g4[t];
                    const float ru = u4[t];
                    hv[t] = rg * (1.f / (1.f + sycl::exp(-rg))) * ru;
                    m = sycl::fmax(m, sycl::fabs(hv[t]));
                }
                auto sg = it.get_sub_group();
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) m = sycl::fmax(m, sycl::permute_group_by_xor(sg, m, o));
                if (!live) return;
                const float idv = m > 0.f ? 127.f / m : 0.f;
                uint32_t packed = 0;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int r = (int)sycl::rint(hv[t] * idv);
                    if (r > 127) r = 127;
                    if (r < -127) r = -127;
                    packed |= (uint32_t)(uint8_t)(int8_t)r << (8 * t);
                }
                uint8_t* b = yb + (size_t)blk * 36ull;
                if (lane == 0) *reinterpret_cast<float*>(b) = m * (1.f / 127.f);
                *reinterpret_cast<uint32_t*>(b + 4 + 4 * lane) = packed;
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

// One chunk width. Called from the single gate/up kernel for EC 4, then 2, then 1.
// The slices are disjoint. nchunk is work-group uniform (it depends only on the group).
template <int TYPE, int EC, class G>
inline void b70_sched_gu_chunks(const G& grid, sycl::sub_group sg, const B70SchedPlan* plan, const uint8_t* q8x,
                                const uint8_t* const (&rows)[kSchedRpl], float* gout, float* uout, int base, int nchunk,
                                int nb, int sub, int I, const bool (&live)[kSchedRpl], const bool (&isup)[kSchedRpl],
                                const int (&rr)[kSchedRpl]) {
    for (int c = 0; c < nchunk; ++c) {
        const int e = base + c * EC;
        const B70Q8S* xs[EC];
#pragma unroll
        for (int j = 0; j < EC; ++j) {
            const int t = plan->tok[e + j];
            xs[j] = b70_q8s_cptr(q8x, (size_t)t * (size_t)nb);
        }
        float o[kSchedRpl][EC];
        b70_sched_rows<TYPE, EC, kSchedRpl>(sg, rows, xs, nb, sub, o, grid);
        if (sub == 0) {
#pragma unroll
            for (int j = 0; j < EC; ++j) {
                const size_t slot = (size_t)(e + j) * (size_t)I;
#pragma unroll
                for (int i = 0; i < kSchedRpl; ++i) {
                    if (!live[i]) continue;
                    float* d = isup[i] ? uout : gout;
                    d[slot + (size_t)rr[i]] = o[i][j];
                }
            }
        }
    }
}

// 3D grid, no use_root_sync. EC 4/2/1 are constexpr passes inside this kernel.
// Each 8-lane rib takes kSchedRpl virtual rows kSchedRb apart (Strata RPL).
template <int TYPE>
inline sycl::event b70_sched_gu_one(sycl::queue& q, const B70SchedPlan* plan, const uint8_t* q8x, const uint8_t* gate,
                                    const uint8_t* up, const uint8_t* sgate, const uint8_t* sup, float* gout, float* uout,
                                    int host_groups, int I, int K, const sycl::event* deps, int ndeps) {
    const int gx = (2 * I + kSchedRows - 1) / kSchedRows;
    if (gx <= 0 || host_groups <= 0) return sycl::event();
    const int nb = K / 32;
    return q.submit([&](sycl::handler& h) {
        b70_sched_deps(h, deps, ndeps);
        constexpr int GW = (B70_GRID_SLM && b70_dp4a_type<TYPE>()) ? b70_grid_words<TYPE>() : 0;
        constexpr int SW = (GW > 0 && b70_sign_tab_type<TYPE>()) ? 512 : 0;  // 256 x u64 sign masks
        sycl::local_accessor<uint64_t, 1> gslm(sycl::range<1>(GW > 0 ? (GW + SW) / 2 : 1), h);
        h.parallel_for<b70_sched_gu_k<TYPE>>(
            b70_sched_grid(host_groups, gx),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(kSchedSg)]] {
                const int g = (int)it.get_group(1);
                const int row_wg = (int)it.get_group(2);
                if (g >= plan->ng) return;
                if (row_wg >= gx) return;
                const int e0 = plan->start[g];
                const int span = plan->start[g + 1] - e0;
                if (span <= 0) return;
                // O3 stage 3: this type's codebook grid is staged in SLM after the work-group-uniform
                // exits and before any per-lane exit (32-bit addresses for the 8 gathers per part).
                auto body = [&](const auto& grid) {
                const int lid = (int)it.get_local_id(2);
                const int rib = lid / kSchedLn;
                const int sub = lid - rib * kSchedLn;
                auto sg = it.get_sub_group();
                const int gtype = plan->gtype[g];
                const int utype = plan->utype[g];
                const int expert = plan->expert[g];
                const int grb = plan->grb[g];
                const int urb = plan->urb[g];
                int vv[kSchedRpl];
                bool live[kSchedRpl];
                bool isup[kSchedRpl];
                int rr[kSchedRpl];
                int first = -1;
#pragma unroll
                for (int i = 0; i < kSchedRpl; ++i) {
                    const int v = row_wg * kSchedRows + rib + i * kSchedRb;
                    const bool in = v < 2 * I;
                    isup[i] = in && v >= I;
                    const int ty = isup[i] ? utype : gtype;
                    live[i] = in && ty == TYPE;
                    rr[i] = isup[i] ? v - I : v;
                    vv[i] = v;
                    if (live[i] && first < 0) first = v;
                }
                if (first < 0) return;
                const uint8_t* rows[kSchedRpl];
#pragma unroll
                for (int i = 0; i < kSchedRpl; ++i)
                    rows[i] = b70_sched_gu_ptr(live[i] ? vv[i] : first, I, expert, grb, urb, gate, up, sgate, sup);
                const int n4 = span / 4;
                const int n2 = (span % 4) / 2;
                const int n1 = span % 2;
                int e = e0;
                if (n4 > 0) {
                    b70_sched_gu_chunks<TYPE, 4>(grid, sg, plan, q8x, rows, gout, uout, e, n4, nb, sub, I, live, isup, rr);
                    e += n4 * 4;
                }
                if (n2 > 0) {
                    b70_sched_gu_chunks<TYPE, 2>(grid, sg, plan, q8x, rows, gout, uout, e, n2, nb, sub, I, live, isup, rr);
                    e += n2 * 2;
                }
                if (n1 > 0) {
                    b70_sched_gu_chunks<TYPE, 1>(grid, sg, plan, q8x, rows, gout, uout, e, n1, nb, sub, I, live, isup, rr);
                }
                };
                if constexpr (GW > 0) {
                    uint64_t* raw = gslm.template get_multi_ptr<sycl::access::decorated::no>().get();
                    uint32_t* w32 = reinterpret_cast<uint32_t*>(raw);
                    const uint32_t* src = b70_grid_src<TYPE>();
                    for (int i = (int)it.get_local_id(2); i < GW; i += kSchedWg) w32[i] = src[i];
                    if constexpr (SW > 0) {
                        // entry v: byte j = 0xFF if bit j of v; ksign7 types fold parity into bit 7
                        for (int v = (int)it.get_local_id(2); v < 256; v += kSchedWg) {
                            uint32_t b = (uint32_t)v;
                            if constexpr (TYPE == B70_IQ2_XXS || TYPE == B70_IQ2_XS || TYPE == B70_IQ3_XXS)
                                b = (b & 127u) ^ (((uint32_t)sycl::popcount(b & 127u) & 1u) << 7);
                            const uint32_t l0 = ((b & 0xFu) * 0x00204081u) & 0x01010101u;
                            const uint32_t l1 = (((b >> 4) & 0xFu) * 0x00204081u) & 0x01010101u;
                            raw[GW / 2 + v] = (uint64_t)(l0 * 0xFFu) | ((uint64_t)(l1 * 0xFFu) << 32);
                        }
                    }
                    sycl::group_barrier(it.get_group());
                    B70GridL gl{sycl::address_space_cast<sycl::access::address_space::local_space,
                                                         sycl::access::decorated::yes>(
                                    static_cast<const uint32_t*>(w32)),
                                sycl::address_space_cast<sycl::access::address_space::local_space,
                                                         sycl::access::decorated::yes>(
                                    static_cast<const uint64_t*>(raw)),
                                sycl::address_space_cast<sycl::access::address_space::local_space,
                                                         sycl::access::decorated::yes>(
                                    static_cast<const uint64_t*>(raw + GW / 2))};
                    body(gl);
                } else {
                    body(B70GridG{});
                }
            });
    });
}

template <int TYPE>
inline sycl::event b70_sched_gu_type(sycl::queue& q, const B70SchedPlan* plan, const uint8_t* q8x, const uint8_t* gate,
                                     const uint8_t* up, const uint8_t* sgate, const uint8_t* sup, float* gout, float* uout,
                                     int host_groups, int I, int K, const sycl::event* deps, int ndeps) {
    return b70_sched_gu_one<TYPE>(q, plan, q8x, gate, up, sgate, sup, gout, uout, host_groups, I, K, deps, ndeps);
}

inline sycl::event b70_sched_gu_dispatch(sycl::queue& q, int type, const B70SchedPlan* plan, const uint8_t* q8x,
                                         const uint8_t* gate, const uint8_t* up, const uint8_t* sgate,
                                         const uint8_t* sup, float* gout, float* uout, int host_groups, int I, int K,
                                         const sycl::event* deps, int ndeps) {
    switch (type) {
#define B70_SCHED_GU(T)                                                                                                \
    case T:                                                                                                            \
        return b70_sched_gu_type<T>(q, plan, q8x, gate, up, sgate, sup, gout, uout, host_groups, I, K, deps, ndeps);
        B70_SCHED_TYPES(B70_SCHED_GU)
#undef B70_SCHED_GU
    default:
        throw std::runtime_error("b70_sched: gate/up type escaped the fast-path check");
    }
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

template <int TYPE, int EC, class G>
inline void b70_sched_down_chunks(const G& grid, sycl::sub_group sg, const uint8_t* q8h, const uint8_t* const (&rows)[kSchedRpl],
                                  float* partial, int base, int nchunk, int nb, int sub, int K,
                                  const bool (&live)[kSchedRpl], const int (&rr)[kSchedRpl]) {
    for (int c = 0; c < nchunk; ++c) {
        const int e = base + c * EC;
        const B70Q8S* xs[EC];
#pragma unroll
        for (int j = 0; j < EC; ++j) xs[j] = b70_q8s_cptr(q8h, (size_t)(e + j) * (size_t)nb);
        float o[kSchedRpl][EC];
        b70_sched_rows<TYPE, EC, kSchedRpl, G, kDownLn>(sg, rows, xs, nb, sub, o, grid);
        if (sub == 0) {
#pragma unroll
            for (int j = 0; j < EC; ++j)
#pragma unroll
                for (int i = 0; i < kSchedRpl; ++i)
                    if (live[i]) partial[(size_t)(e + j) * (size_t)K + rr[i]] = o[i][j];
        }
    }
}

// Same 3D grid as gate/up. EC passes stay inside this launch. No use_root_sync.
template <int TYPE>
inline sycl::event b70_sched_down_one(sycl::queue& q, const B70SchedPlan* plan, const uint8_t* q8h, const uint8_t* down,
                                      const uint8_t* sdown, float* partial, int host_groups, int I, int K,
                                      sycl::event dep) {
    const int gx = (K + kDownRows - 1) / kDownRows;
    if (gx <= 0 || host_groups <= 0) return dep;
    const int nb = I / 32;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep);
        constexpr int GW = (B70_GRID_SLM && b70_dp4a_type<TYPE>()) ? b70_grid_words<TYPE>() : 0;
        constexpr int SW = (GW > 0 && b70_sign_tab_type<TYPE>()) ? 512 : 0;  // 256 x u64 sign masks
        sycl::local_accessor<uint64_t, 1> gslm(sycl::range<1>(GW > 0 ? (GW + SW) / 2 : 1), h);
        h.parallel_for<b70_sched_down_k<TYPE>>(
            b70_sched_grid(host_groups, gx),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(kSchedSg)]] {
                const int g = (int)it.get_group(1);
                const int row_wg = (int)it.get_group(2);
                if (g >= plan->ng) return;
                if (plan->dtype[g] != TYPE) return;
                if (row_wg >= gx) return;
                const int e0 = plan->start[g];
                const int span = plan->start[g + 1] - e0;
                if (span <= 0) return;
                // O3 stage 3: this type's codebook grid is staged in SLM after the work-group-uniform
                // exits and before any per-lane exit (32-bit addresses for the 8 gathers per part).
                auto body = [&](const auto& grid) {
                const int lid = (int)it.get_local_id(2);
                const int rib = lid / kDownLn;
                const int sub = lid - rib * kDownLn;
                auto sg = it.get_sub_group();
                const int r0 = row_wg * kDownRows + rib;
                if (r0 >= K) return;
                const int expert = plan->expert[g];
                const bool sh = expert < 0;
                const uint8_t* basep = sh ? sdown : down;
                const int ex = sh ? 0 : expert;
                const int drb = plan->drb[g];
                bool live[kSchedRpl];
                int rr[kSchedRpl];
                const uint8_t* rows[kSchedRpl];
#pragma unroll
                for (int i = 0; i < kSchedRpl; ++i) {
                    const int r = r0 + i * kDownRb;
                    live[i] = r < K;
                    rr[i] = live[i] ? r : r0;
                    rows[i] = basep + ((size_t)ex * (size_t)K + (size_t)rr[i]) * (size_t)drb;
                }
                const int n4 = span / 4;
                const int n2 = (span % 4) / 2;
                const int n1 = span % 2;
                int e = e0;
                if (n4 > 0) {
                    b70_sched_down_chunks<TYPE, 4>(grid, sg, q8h, rows, partial, e, n4, nb, sub, K, live, rr);
                    e += n4 * 4;
                }
                if (n2 > 0) {
                    b70_sched_down_chunks<TYPE, 2>(grid, sg, q8h, rows, partial, e, n2, nb, sub, K, live, rr);
                    e += n2 * 2;
                }
                if (n1 > 0) {
                    b70_sched_down_chunks<TYPE, 1>(grid, sg, q8h, rows, partial, e, n1, nb, sub, K, live, rr);
                }
                };
                if constexpr (GW > 0) {
                    uint64_t* raw = gslm.template get_multi_ptr<sycl::access::decorated::no>().get();
                    uint32_t* w32 = reinterpret_cast<uint32_t*>(raw);
                    const uint32_t* src = b70_grid_src<TYPE>();
                    for (int i = (int)it.get_local_id(2); i < GW; i += kSchedWg) w32[i] = src[i];
                    if constexpr (SW > 0) {
                        // entry v: byte j = 0xFF if bit j of v; ksign7 types fold parity into bit 7
                        for (int v = (int)it.get_local_id(2); v < 256; v += kSchedWg) {
                            uint32_t b = (uint32_t)v;
                            if constexpr (TYPE == B70_IQ2_XXS || TYPE == B70_IQ2_XS || TYPE == B70_IQ3_XXS)
                                b = (b & 127u) ^ (((uint32_t)sycl::popcount(b & 127u) & 1u) << 7);
                            const uint32_t l0 = ((b & 0xFu) * 0x00204081u) & 0x01010101u;
                            const uint32_t l1 = (((b >> 4) & 0xFu) * 0x00204081u) & 0x01010101u;
                            raw[GW / 2 + v] = (uint64_t)(l0 * 0xFFu) | ((uint64_t)(l1 * 0xFFu) << 32);
                        }
                    }
                    sycl::group_barrier(it.get_group());
                    B70GridL gl{sycl::address_space_cast<sycl::access::address_space::local_space,
                                                         sycl::access::decorated::yes>(
                                    static_cast<const uint32_t*>(w32)),
                                sycl::address_space_cast<sycl::access::address_space::local_space,
                                                         sycl::access::decorated::yes>(
                                    static_cast<const uint64_t*>(raw)),
                                sycl::address_space_cast<sycl::access::address_space::local_space,
                                                         sycl::access::decorated::yes>(
                                    static_cast<const uint64_t*>(raw + GW / 2))};
                    body(gl);
                } else {
                    body(B70GridG{});
                }
            });
    });
}

template <int TYPE>
inline sycl::event b70_sched_down_type(sycl::queue& q, const B70SchedPlan* plan, const uint8_t* q8h, const uint8_t* down,
                                       const uint8_t* sdown, float* partial, int host_groups, int I, int K,
                                       sycl::event dep) {
    return b70_sched_down_one<TYPE>(q, plan, q8h, down, sdown, partial, host_groups, I, K, dep);
}

inline sycl::event b70_sched_down_dispatch(sycl::queue& q, int type, const B70SchedPlan* plan, const uint8_t* q8h,
                                           const uint8_t* down, const uint8_t* sdown, float* partial, int host_groups,
                                           int I, int K, sycl::event dep) {
    switch (type) {
#define B70_SCHED_DN(T)                                                                                                \
    case T:                                                                                                            \
        return b70_sched_down_type<T>(q, plan, q8h, down, sdown, partial, host_groups, I, K, dep);
        B70_SCHED_TYPES(B70_SCHED_DN)
#undef B70_SCHED_DN
    default:
        throw std::runtime_error("b70_sched: down type escaped the fast-path check");
    }
}

// out_mode 0: fp32 y [T, K] = routed + shared (shared weight sw[t] if has_sg,
// else 1). out_mode 1: bf16 [2, T, K], routed in [0], weighted shared in [1]
// (the MoE runner adds them, as it did with its own shared output).
inline sycl::event b70_sched_red(sycl::queue& q, const B70SchedPlan* plan, const float* partial, const float* router,
                                 void* yv, int T, int topk, int K, sycl::event dep, int has_sg = 0, int out_mode = 0) {
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
            const float shw = has_sg ? plan->sw[t] : 1.f;
            if (out_mode == 0) {
                float s = 0.f;
                for (int e = 0; e < nent; ++e) {
                    if (plan->tok[e] != t) continue;
                    const int slot = plan->slot[e];
                    const float w = slot < topk ? router[(size_t)t * (size_t)topk + slot] : shw;
                    s = sycl::fma(partial[(size_t)e * (size_t)K + k], w, s);
                }
                static_cast<float*>(yv)[i] = s;
            } else {
                float s = 0.f;
                float sh = 0.f;
                for (int e = 0; e < nent; ++e) {
                    if (plan->tok[e] != t) continue;
                    const int slot = plan->slot[e];
                    const float pv = partial[(size_t)e * (size_t)K + k];
                    if (slot < topk) s = sycl::fma(pv, router[(size_t)t * (size_t)topk + slot], s);
                    else sh = sycl::fma(pv, shw, sh);
                }
                uint16_t* yb = static_cast<uint16_t*>(yv);
                yb[i] = b70_fp32_to_bf16(s);
                yb[(size_t)n + i] = b70_fp32_to_bf16(sh);
            }
        });
    });
}

inline double b70_sched_stamp(bool phase, sycl::queue& q, sycl::event ev, std::chrono::steady_clock::time_point t0) {
    if (!phase) return 0.0;
    ev.wait();
    (void)q;
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

// Submits the whole layer, or returns false so the caller keeps the old kernels.
// One gate/up kernel per distinct type, then one down kernel per distinct type.
// Entry widths 4, 2, and 1 are constexpr passes inside that kernel.
// use_root_sync is off. Types stay chained so two types do not fight the EUs.
inline bool b70_sched_try(sycl::queue& q, const uint16_t* x, const uint8_t* gate, int gate_type, int gate_rb,
                          const uint8_t* up, int up_type, int up_rb, const uint8_t* down, int down_type, int down_rb,
                          const int32_t* ids, const float* router, const uint8_t* sgate, int sg_type, int sg_rb,
                          const uint8_t* sup, int su_type, int su_rb, const uint8_t* sdown, int sd_type, int sd_rb,
                          void* y, int T, int topk, int E, int I, int K, const void* sgi = nullptr,
                          int sgi_dtype = 0, int out_mode = 0) {
    if (!b70_sched_enabled()) return false;
    if (!B70_SCHED_FUSE && (sgi != nullptr || out_mode != 0)) return false;
    if (T <= 0 || topk <= 0 || T > kSchedMaxT || topk > kSchedMaxTopk) return false;
    if (I <= 0 || K <= 0 || I > kSchedMaxDim || K > kSchedMaxDim) return false;
    if ((I % 32) != 0 || (K % 32) != 0) return false;
    if (!b70_sched_type(gate_type) || !b70_sched_type(up_type) || !b70_sched_type(down_type)) return false;
    const int shared = sgate != nullptr;
    if (shared && (!b70_sched_type(sg_type) || !b70_sched_type(su_type) || !b70_sched_type(sd_type))) return false;
    const bool phase = b70_sched_phase();
    const B70SchedMem mem = b70_sched_scratch(q);
    const int host_groups = T * topk + shared;
    const int host_ent = T * topk + (shared ? T : 0);

    if (phase) q.wait();
    auto t0 = std::chrono::steady_clock::now();
#if B70_SCHED_FUSE
    const sycl::event ep = b70_sched_plan_q8(q, mem.plan, ids, T, topk, E, gate_type, up_type, down_type, gate_rb, up_rb,
                                             down_rb, sg_type, su_type, sd_type, sg_rb, su_rb, sd_rb, shared, x, mem.q8x,
                                             K, shared ? sgi : nullptr, sgi_dtype);
    const double tp = b70_sched_stamp(phase, q, ep, t0);
    const sycl::event eq = ep;
    const double tq = 0.0;
#else
    const sycl::event ep = b70_sched_plan(q, mem.plan, ids, T, topk, E, gate_type, up_type, down_type, gate_rb, up_rb,
                                          down_rb, sg_type, su_type, sd_type, sg_rb, su_rb, sd_rb, shared);
    const double tp = b70_sched_stamp(phase, q, ep, t0);

    if (phase) q.wait();
    t0 = std::chrono::steady_clock::now();
    const sycl::event eq = b70_sched_q8_bf16(q, x, mem.q8x, T, K);
    const double tq = b70_sched_stamp(phase, q, eq, t0);
#endif

    int gu_types[8];
    int ngu = 0;
    b70_sched_push(gu_types, ngu, gate_type);
    b70_sched_push(gu_types, ngu, up_type);
    if (shared) {
        b70_sched_push(gu_types, ngu, sg_type);
        b70_sched_push(gu_types, ngu, su_type);
    }
    sycl::event egu = eq;
    double tgu = 0.0;
    for (int i = 0; i < ngu; ++i) {
        sycl::event deps[3];
        int nd = 0;
        deps[nd++] = ep;
        deps[nd++] = eq;
        if (i > 0) deps[nd++] = egu;
        if (phase) q.wait();
        t0 = std::chrono::steady_clock::now();
        egu = b70_sched_gu_dispatch(q, gu_types[i], mem.plan, mem.q8x, gate, up, sgate, sup, mem.gate, mem.up,
                                    host_groups, I, K, deps, nd);
        tgu += b70_sched_stamp(phase, q, egu, t0);
    }
    // Hidden floats reuse the up buffer after silu. Gate is still live until silu reads it.
    if (phase) q.wait();
    t0 = std::chrono::steady_clock::now();
#if B70_SCHED_FUSE
    const sycl::event eh = b70_sched_silu_q8(q, mem.plan, mem.gate, mem.up, mem.q8h, host_ent, I, egu);
    const double ts = b70_sched_stamp(phase, q, eh, t0);
    const double th = 0.0;
#else
    const sycl::event es = b70_sched_silu(q, mem.plan, mem.gate, mem.up, mem.up, host_ent, I, egu);
    const double ts = b70_sched_stamp(phase, q, es, t0);

    if (phase) q.wait();
    t0 = std::chrono::steady_clock::now();
    const sycl::event eh = b70_sched_q8_f32(q, mem.up, mem.q8h, mem.plan, host_ent, I, es);
    const double th = b70_sched_stamp(phase, q, eh, t0);
#endif

    int dn_types[4];
    int ndn = 0;
    b70_sched_push(dn_types, ndn, down_type);
    if (shared) b70_sched_push(dn_types, ndn, sd_type);
    sycl::event ed = eh;
    double td = 0.0;
    for (int i = 0; i < ndn; ++i) {
        if (phase) q.wait();
        t0 = std::chrono::steady_clock::now();
        ed = b70_sched_down_dispatch(q, dn_types[i], mem.plan, mem.q8h, down, sdown, mem.partial, host_groups, I, K, ed);
        td += b70_sched_stamp(phase, q, ed, t0);
    }

    if (phase) q.wait();
    t0 = std::chrono::steady_clock::now();
    const sycl::event er = b70_sched_red(q, mem.plan, mem.partial, router, y, T, topk, K, ed,
                                           (shared && sgi != nullptr) ? 1 : 0, out_mode);
    const double tr = b70_sched_stamp(phase, q, er, t0);
    if (phase) {
        std::fprintf(stderr,
                     "C12_PHASE T=%d plan=%.2f q8=%.2f gu=%.2f silu=%.2f q8h=%.2f down=%.2f red=%.2f\n",
                     T, tp, tq, tgu, ts, th, td, tr);
        std::fflush(stderr);
    }
    return true;
}
