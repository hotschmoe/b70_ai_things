// C6 mode-0 schedules. Included inside namespace b70pf.
//
// reuse: 2D grid over M tiles and N tiles of 16 columns. The work-group
// dequants one BN x BK weight panel into SLM once, then every subgroup
// walks a different strip of the M tile with fp16 DPAS 8x16x16. X stays
// in global memory. E==1 (dense, or a single expert) uses the 2D grid so
// row tiles run in parallel. E>1 loops row tiles inside the expert group;
// Flash-Next prefill has <512 rows per expert, so that loop is one pass.
//
// s64: Strata night_grouped_expert.inc:235 with SGN=4 (64 columns). Each
// subgroup owns 16 columns, preloads the next 32-wide block (inc:275),
// and loads X from global. One group per expert; row tiles above MT*8
// walk serially inside that group.
//
// B70_PREFILL_SCHED=reuse|s64 overrides the auto pick (avg rows >= 48
// selects reuse). B70_PREFILL_MT=4|16 and B70_PREFILL_RM=32|64 too.

constexpr int kReuseNsg = 8;
constexpr int kReuseBk = 256;
constexpr int kReuseBn = 16;

int sched_kind(int rows, int experts) {
    const char* e = std::getenv("B70_PREFILL_SCHED");
    if (e && std::strcmp(e, "s64") == 0) return 1;
    if (e && std::strcmp(e, "reuse") == 0) return 2;
    const int avg = (rows + experts - 1) / experts;
    return avg >= 48 ? 2 : 1;
}

int sched_mt(int rows, int experts) {
    const char* e = std::getenv("B70_PREFILL_MT");
    if (e && e[0]) return std::atoi(e) == 4 ? 4 : 16;
    const int avg = (rows + experts - 1) / experts;
    return avg <= 32 ? 4 : 16;
}

int sched_rm() {
    const char* e = std::getenv("B70_PREFILL_RM");
    if (e && std::strcmp(e, "32") == 0) return 32;
    return 64;
}

template <int TYPE>
inline void weight_at(const uint8_t* gate, const uint8_t* up, int expert, int n_wrow, int rb, int gu, int col,
                      const uint8_t*& base, int& wrow) {
    const uint8_t* gate_e = gate + (size_t)expert * (size_t)n_wrow * (size_t)rb;
    if (gu) {
        const uint8_t* up_e = up + (size_t)expert * (size_t)n_wrow * (size_t)rb;
        base = (col & 1) ? up_e : gate_e;
        wrow = col >> 1;
    } else {
        base = gate_e;
        wrow = col;
    }
}

inline void store_h32(sycl::half* dst, const float* v) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        sycl::vec<sycl::half, 8> hv;
#pragma unroll
        for (int j = 0; j < 8; ++j) hv[j] = sycl::half(v[8 * c + j]);
        hv.store(c, dst);
    }
}

template <int TYPE, int MT>
class b70_pf_s64;

// Strata SGN=4. MT is the number of 8-row DPAS accumulators (4 or 16).
template <int TYPE, int MT>
void launch_s64(sycl::queue& q, const GemmArgs& a) {
    constexpr int SGN = 4;
    constexpr int WG = SGN * kSg;
    constexpr int LDB = 32 + 8;
    constexpr int TILE = MT * 8;
    const int ntiles = a.N / WG;
    const int E = a.E;
    const int N = a.N, K = a.K, rb = a.rb, n_wrow = a.n_wrow, gu = a.gu;
    const uint16_t* X = a.X;
    const uint8_t* gate = a.gate;
    const uint8_t* up = a.up;
    const int32_t* starts = a.starts;
    float* Y = a.Y;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<(MT > 4 ? 256 : 128)>};
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> slm(SGN * kSg * LDB * sizeof(sycl::half) + 128, h);
        h.parallel_for<b70_pf_s64<TYPE, MT>>(
            sycl::nd_range<2>({(size_t)E, (size_t)ntiles * WG}, {1, (size_t)WG}), props,
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSg)]] {
                auto sg = it.get_sub_group();
                const int lane = (int)sg.get_local_linear_id();
                const int sgid = (int)it.get_local_id(1) / kSg;
                const int expert = (int)it.get_group(0);
                const int row0 = starts[expert];
                const int rows_all = starts[expert + 1] - row0;
                if (rows_all <= 0) return;
                const int n0 = (int)it.get_group(1) * WG + sgid * kSg;
                const int col = n0 + lane;
                const uint8_t* base = nullptr;
                int wrow = 0;
                weight_at<TYPE>(gate, up, expert, n_wrow, rb, gu, col, base, wrow);
                uint8_t* raw = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                const uintptr_t mis = reinterpret_cast<uintptr_t>(raw) & 63u;
                sycl::half* hs = reinterpret_cast<sycl::half*>(raw + (mis ? (64u - mis) : 0u));
                sycl::half* sp = hs + sgid * kSg * LDB;
                sycl::half* my = sp + lane * LDB;
                const int64_t flat0 = (int64_t)wrow * K;
                for (int rbase = 0; rbase < rows_all; rbase += TILE) {
                    const int rows_here = (rows_all - rbase) < TILE ? (rows_all - rbase) : TILE;
                    using Acc = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, kSg>;
                    Acc acc[MT];
#pragma unroll
                    for (int i = 0; i < MT; ++i) jm::joint_matrix_fill(sg, acc[i], 0.f);
                    B70Raw cur = b70_load_raw<TYPE>(base, flat0 >> 8, (int)((flat0 & 255) >> 5));
                    for (int k0 = 0; k0 < K; k0 += 32) {
                        const int64_t flat = flat0 + k0;
                        const int ib = (int)((flat & 255) >> 5);
                        B70Raw nxt{};
                        if (k0 + 32 < K) {
                            const int64_t fn = flat + 32;
                            nxt = b70_load_raw<TYPE>(base, fn >> 8, (int)((fn & 255) >> 5));
                        }
                        float v[32];
                        b70_decode_raw<TYPE>(cur, ib, v);
                        cur = nxt;
                        store_h32(my, v);
                        sycl::group_barrier(sg);
                        auto xp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                            reinterpret_cast<sycl::half*>(const_cast<uint16_t*>(X + (size_t)(row0 + rbase) * K)));
#pragma unroll
                        for (int kk = 0; kk < 32; kk += 16) {
                            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, kSg, jm::layout::col_major> bm;
                            auto bp = as_local<sycl::half>(reinterpret_cast<uint8_t*>(sp + kk));
                            jm::joint_matrix_load(sg, bm, bp, LDB);
#pragma unroll
                            for (int i = 0; i < MT; ++i) {
                                if (i * 8 < rows_here) {
                                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> am;
                                    const int valid = rows_here - i * 8;
                                    if (valid >= 8) {
                                        auto ap = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                            reinterpret_cast<sycl::half*>(const_cast<uint16_t*>(X + ((size_t)(row0 + rbase + i * 8) * K) + (k0 + kk))));
                                        jm::joint_matrix_load(sg, am, ap, (size_t)K);
                                    } else {
                                        jmi::joint_matrix_load_checked(sg, am, xp, (size_t)K, (size_t)rows_here, (size_t)K, (size_t)(i * 8),
                                                                       (size_t)(k0 + kk));
                                    }
                                    jm::joint_matrix_mad(sg, acc[i], am, bm, acc[i]);
                                }
                            }
                        }
                        sycl::group_barrier(sg);
                    }
                    auto yp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        Y + (size_t)(row0 + rbase) * N);
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (i * 8 < rows_here) {
                            const int valid = rows_here - i * 8;
                            if (valid >= 8) {
                                auto spn = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                    Y + (size_t)(row0 + rbase + i * 8) * N + n0);
                                jm::joint_matrix_store(sg, acc[i], spn, (size_t)N, jm::layout::row_major);
                            } else {
                                jmi::joint_matrix_store_checked(sg, acc[i], yp, (size_t)N, jm::layout::row_major, (size_t)rows_here, (size_t)N,
                                                                (size_t)(i * 8), (size_t)n0);
                            }
                        }
                    }
                }
            });
    });
}

template <int TYPE, int RM>
class b70_pf_reuse;

// RM rows per subgroup. Eight subgroups share one 16-column weight panel.
template <int TYPE, int RM>
void launch_reuse(sycl::queue& q, const GemmArgs& a) {
    constexpr int NSG = kReuseNsg;
    constexpr int BK = kReuseBk;
    constexpr int BN = kReuseBn;
    constexpr int WG = NSG * kSg;
    constexpr int LD = BK;
    constexpr int MTILE = NSG * RM;
    constexpr int NACC = RM / 8;
    const bool dense2d = a.E == 1;
    const int ntiles = a.N / BN;
    const int mtiles = dense2d ? (a.rows_hint + MTILE - 1) / MTILE : a.E;
    if (mtiles <= 0 || ntiles <= 0) return;
    const int slow = dense2d ? ntiles : a.E;
    const int fast = dense2d ? mtiles : ntiles;
    const int N = a.N, K = a.K, rb = a.rb, n_wrow = a.n_wrow, gu = a.gu;
    const int rows_hint = a.rows_hint;
    const int dense_flag = dense2d ? 1 : 0;
    const uint16_t* X = a.X;
    const uint8_t* gate = a.gate;
    const uint8_t* up = a.up;
    const int32_t* starts = a.starts;
    float* Y = a.Y;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> slm((size_t)BN * LD * sizeof(sycl::half) + 128, h);
        h.parallel_for<b70_pf_reuse<TYPE, RM>>(
            sycl::nd_range<2>({(size_t)slow, (size_t)fast * WG}, {1, (size_t)WG}), props,
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSg)]] {
                auto sg = it.get_sub_group();
                const int tid = (int)it.get_local_id(1);
                const int sgid = tid / kSg;
                const int ntile = dense_flag ? (int)it.get_group(0) : (int)it.get_group(1);
                const int n0 = ntile * BN;
                int expert, row_origin, rows_all;
                if (dense_flag) {
                    expert = 0;
                    const int mtile = (int)it.get_group(1);
                    row_origin = mtile * MTILE;
                    rows_all = rows_hint - row_origin;
                    if (rows_all > MTILE) rows_all = MTILE;
                } else {
                    expert = (int)it.get_group(0);
                    row_origin = starts[expert];
                    rows_all = starts[expert + 1] - row_origin;
                }
                if (rows_all <= 0) return;
                uint8_t* raw = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                const uintptr_t mis = reinterpret_cast<uintptr_t>(raw) & 63u;
                sycl::half* hs = reinterpret_cast<sycl::half*>(raw + (mis ? (64u - mis) : 0u));
                for (int m0 = 0; m0 < rows_all; m0 += MTILE) {
                    int rows = rows_all - m0;
                    if (rows > MTILE) rows = MTILE;
                    int my_rows = rows - sgid * RM;
                    if (my_rows < 0) my_rows = 0;
                    if (my_rows > RM) my_rows = RM;
                    using Acc = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, kSg>;
                    Acc acc[NACC];
#pragma unroll
                    for (int i = 0; i < NACC; ++i) jm::joint_matrix_fill(sg, acc[i], 0.f);
                    for (int k0 = 0; k0 < K; k0 += BK) {
                        int bk = K - k0;
                        if (bk > BK) bk = BK;
                        const int nwork = BN * (bk >> 5);
                        if (tid < nwork) {
                            const int n_local = tid % BN;
                            const int k = (tid / BN) * 32;
                            const int col = n0 + n_local;
                            const uint8_t* base = nullptr;
                            int wrow = 0;
                            weight_at<TYPE>(gate, up, expert, n_wrow, rb, gu, col, base, wrow);
                            const int64_t flat = (int64_t)wrow * K + (k0 + k);
                            const int ib = (int)((flat & 255) >> 5);
                            B70Raw cur = b70_load_raw<TYPE>(base, flat >> 8, ib);
                            float v[32];
                            b70_decode_raw<TYPE>(cur, ib, v);
                            store_h32(hs + n_local * LD + k, v);
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                        if (my_rows > 0) {
                            const int xrow0 = row_origin + m0 + sgid * RM;
                            auto xp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                reinterpret_cast<sycl::half*>(const_cast<uint16_t*>(X + (size_t)xrow0 * K)));
                            for (int kk = 0; kk < bk; kk += 16) {
                                jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, kSg, jm::layout::col_major> bm;
                                auto bp = as_local<sycl::half>(reinterpret_cast<uint8_t*>(hs + kk));
                                jm::joint_matrix_load(sg, bm, bp, LD);
#pragma unroll
                                for (int i = 0; i < NACC; ++i) {
                                    if (i * 8 >= my_rows) break;
                                    jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> am;
                                    const int valid = my_rows - i * 8;
                                    if (valid >= 8) {
                                        auto ap = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                            reinterpret_cast<sycl::half*>(
                                                const_cast<uint16_t*>(X + ((size_t)(xrow0 + i * 8) * K) + (k0 + kk))));
                                        jm::joint_matrix_load(sg, am, ap, (size_t)K);
                                    } else {
                                        jmi::joint_matrix_load_checked(sg, am, xp, (size_t)K, (size_t)my_rows, (size_t)K, (size_t)(i * 8),
                                                                       (size_t)(k0 + kk));
                                    }
                                    jm::joint_matrix_mad(sg, acc[i], am, bm, acc[i]);
                                }
                            }
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    if (my_rows > 0) {
                        const int yrow0 = row_origin + m0 + sgid * RM;
                        auto yp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                            Y + (size_t)yrow0 * N + n0);
#pragma unroll
                        for (int i = 0; i < NACC; ++i) {
                            if (i * 8 >= my_rows) break;
                            const int valid = my_rows - i * 8;
                            if (valid >= 8) {
                                auto spn = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                    Y + (size_t)(yrow0 + i * 8) * N + n0);
                                jm::joint_matrix_store(sg, acc[i], spn, (size_t)N, jm::layout::row_major);
                            } else {
                                jmi::joint_matrix_store_checked(sg, acc[i], yp, (size_t)N, jm::layout::row_major, (size_t)my_rows, (size_t)BN,
                                                                (size_t)(i * 8), 0);
                            }
                        }
                    }
                }
            });
    });
}

template <int TYPE>
void launch_mode0(sycl::queue& q, const GemmArgs& a, int kind, int mt, int rm) {
    if (kind == 1) {
        if (mt == 4) launch_s64<TYPE, 4>(q, a);
        else launch_s64<TYPE, 16>(q, a);
    } else if (rm == 32) {
        launch_reuse<TYPE, 32>(q, a);
    } else {
        launch_reuse<TYPE, 64>(q, a);
    }
}

void launch_sched(sycl::queue& q, int type, const GemmArgs& a) {
    const int kind = sched_kind(a.rows_hint, a.E);
    const int mt = sched_mt(a.rows_hint, a.E);
    const int rm = sched_rm();
    g_split.cols = kind == 1 ? 64 : kReuseBn;
    g_split.mt = kind == 1 ? mt : (kReuseNsg * rm);
    if (split_on()) {
        std::fprintf(stderr, "PREFILL_SCHED %s cols=%d mt=%d E=%d rows=%d N=%d K=%d\n", kind == 1 ? "s64" : "reuse", g_split.cols,
                     g_split.mt, a.E, a.rows_hint, a.N, a.K);
    }
#ifdef B70_PREFILL_SMOKE
    switch (type) {
    case B70_Q4_K: launch_mode0<B70_Q4_K>(q, a, kind, mt, rm); break;
    case B70_IQ3_XXS: launch_mode0<B70_IQ3_XXS>(q, a, kind, mt, rm); break;
    case B70_Q2_0: launch_mode0<B70_Q2_0>(q, a, kind, mt, rm); break;
    default: throw std::runtime_error("b70 prefill smoke build: fp16 Q4_K, IQ3_XXS, Q2_0 only");
    }
#else
    switch (type) {
    case B70_Q4_K: launch_mode0<B70_Q4_K>(q, a, kind, mt, rm); break;
    case B70_Q5_K: launch_mode0<B70_Q5_K>(q, a, kind, mt, rm); break;
    case B70_Q6_K: launch_mode0<B70_Q6_K>(q, a, kind, mt, rm); break;
    case B70_IQ2_XXS: launch_mode0<B70_IQ2_XXS>(q, a, kind, mt, rm); break;
    case B70_IQ2_XS: launch_mode0<B70_IQ2_XS>(q, a, kind, mt, rm); break;
    case B70_IQ2_S: launch_mode0<B70_IQ2_S>(q, a, kind, mt, rm); break;
    case B70_IQ3_XXS: launch_mode0<B70_IQ3_XXS>(q, a, kind, mt, rm); break;
    case B70_IQ3_S: launch_mode0<B70_IQ3_S>(q, a, kind, mt, rm); break;
    case B70_IQ4_NL: launch_mode0<B70_IQ4_NL>(q, a, kind, mt, rm); break;
    case B70_IQ4_XS: launch_mode0<B70_IQ4_XS>(q, a, kind, mt, rm); break;
    case B70_Q2_0: launch_mode0<B70_Q2_0>(q, a, kind, mt, rm); break;
    default: throw std::runtime_error("b70 prefill: unsupported ggml type");
    }
#endif
}
