"""SOURCE EXTRACTION ONLY; writes this private header/record, no main edit/build/GPU."""
from pathlib import Path
import hashlib, json
P = Path(__file__).resolve().parent
SOURCE = Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/kernels/cuda/iq_kernels.dp.cpp')
text = SOURCE.read_text()
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(SOURCE.read_bytes()) == 'acaa7da63733af9f3831b3a74b99e6176bf46a06e498e0a33c0591f7f4a29851'
def between(start, stop):
    a = text.index(start)
    return text[a:text.index(stop, a)].rstrip() + '\n'
helpers = between('__dpct_inline__ int get_int_b2', '__dpct_inline__ sycl::int2 get_int_from_table_16')
swar = between('#ifndef STRATA_SWAR', '__dpct_inline__ float vec_dot_iq2_xxs')
original = between('__dpct_inline__ float vec_dot_iq2_xs_q8_1', '__dpct_inline__ float vec_dot_iq2_s_q8_1')
fmt = 'template<int TY> struct Fmt;\n' + between('template<> struct Fmt<17>', 'template<> struct Fmt<18>')
lanes = between('template <int LANES>\n__dpct_inline__ float lanes_sum', '// One row against one q8_1 activation, the whole warp')

# Extract original weight-only expressions. Do not introduce Split17's
# alternative byte emulations or its old aliasing pattern.
body = original[original.index('    const block_iq2_xs*'):original.index('    int sumi0 = 0, sumi1 = 0;')]
weight_loop = original[original.index('#pragma unroll'):original.index('    const int sumi =')]
for line in ['        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);\n',
             '        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);\n']:
    assert weight_loop.count(line) == 1
    weight_loop = weight_loop.replace(line, '')
old_dots = '''        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
'''
assert weight_loop.count(old_dots) == 1
weight_loop = weight_loop.replace(old_dots, '        m.w[l0 + 0] = grid_l;\n        m.w[l0 + 1] = grid_h;\n')
multi = '''struct MultiW { int w[8]; int a = 0, b = 0; float d = 0.f; };
template<int TY> struct Multi;
template<> struct Multi<17> {
    static constexpr int NW = 8;
    __dpct_inline__ static void prep(const void* vbq, int kbx, int iqs, MultiW& m, const void* = nullptr) {
''' + body + weight_loop + '''        m.a = ls0;
        m.b = ls1;
        m.d = sycl::vec<sycl::half, 1>(bq2->d)
                  .convert<float, sycl::rounding_mode::automatic>()[0];
    }
    __dpct_inline__ static void acts(const block_q8_1* y, int iqs, int* u, float& ds) {
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = get_int_b4(y[iqs / 2].qs, j);
        ds = y[iqs / 2].ds[0];
    }
    __dpct_inline__ static float finish(int s0, int s1, const MultiW& m, float ds) {
        const int sumi = (s0 * m.a + s1 * m.b + (s0 + s1) / 2) / 4;
        const float d = m.d * ds;
        return d * sumi;
    }
};
'''
row_multi = between('template <int TY, int LANES, int E>\n__dpct_inline__ void row_dot_multi_iq', 'template <int TY, int LANES, int E>\n__dpct_inline__ void row_dot_multi(')
row_dispatch = '''template<int TY, int LANES, int E>
__dpct_inline__ void row_dot_multi(const uint8_t* row, const block_q8_1* const* xs,
                                  int nb, int sub, float* out, const void* grid = nullptr) {
    static_assert(TY == 17 && LANES == 8 && (E == 2 || E == 4));
    row_dot_multi_iq<TY, LANES, E>(row, xs, nb, sub, out, grid);
}
'''
entries = between('template <int TY, int LANES, bool KM = true>\n__dpct_inline__ void row_entries', 'constexpr int GU_ROWS')
entries = entries.replace('bool KM = true', 'bool USE_MULTI = false').replace('if constexpr (kEntryMulti<TY, KM>)', 'if constexpr (USE_MULTI)')
assert 'KM' not in entries and entries.count('if constexpr (USE_MULTI)') == 1
kernel = between('template <int TG, int LN = kExpertLanes, bool KM = true>\n__dpct_inline__ void native_gu_kernel', '// native_gu_kernel with the group')
kernel = kernel.replace('bool KM = true', 'bool USE_MULTI = false').replace('void native_gu_kernel(', 'void private_gu_kernel(').replace('row_entries<TG, LN, KM>', 'row_entries<TG, LN, USE_MULTI>')
assert 'KM' not in kernel and 'native_gu_kernel(' not in kernel
header = '''// S98 PRIVATE IQ2_XS routed GU original/multi oracle, SOURCE ONLY.
// Exact extracted source and transformations are pinned in extraction.json.
#pragma once
#include "strata/sycl_math.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dp4a.hpp"
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include <type_traits>
namespace s98 {
using strata::kernels::NativeExpertLayout;
static_assert(sizeof(block_iq2_xs) == 74 && sizeof(block_q8_1) == 36);
constexpr int kExpertLanes = 8;
#define ggml_cuda_dp4a(a, b, c) strata::dp4a((a), (b), (c))
#ifndef STRATA_GRID_SLM
#define STRATA_GRID_SLM 0
#endif
'''
launch = '''template<bool USE_MULTI> class PrivateGu17;
template<bool USE_MULTI>
inline void launch(const unsigned long long* grp_ptr, const int32_t* grp_start,
                   const int32_t* n_groups, const int32_t* ent_tok,
                   const block_q8_1* xq, NativeExpertLayout L, float* gate,
                   float* up, unsigned grid_groups, void* stream) {
    constexpr int ROWS = 256 / 8;
    const unsigned gx = (unsigned)((2 * L.n_ff + ROWS - 1) / ROWS);
    auto* q = strata::q_of(stream);
    auto props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    q->parallel_for<PrivateGu17<USE_MULTI>>(
        sycl::nd_range<3>(sycl::range<3>(1, grid_groups, (size_t)gx * 256),
                          sycl::range<3>(1, 1, 256)), props,
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            private_gu_kernel<17, 8, USE_MULTI>(grp_ptr, grp_start, n_groups,
                                               ent_tok, xq, L, gate, up);
        });
}
inline void launch_original(const unsigned long long* p, const int32_t* s,
                            const int32_t* n, const int32_t* t,
                            const block_q8_1* x, NativeExpertLayout L,
                            float* g, float* u, unsigned grids, void* stream) {
    launch<false>(p, s, n, t, x, L, g, u, grids, stream);
}
inline void launch_multi(const unsigned long long* p, const int32_t* s,
                         const int32_t* n, const int32_t* t,
                         const block_q8_1* x, NativeExpertLayout L,
                         float* g, float* u, unsigned grids, void* stream) {
    launch<true>(p, s, n, t, x, L, g, u, grids, stream);
}
#undef ggml_cuda_dp4a
} // namespace s98
'''
result = header + helpers + swar + original + fmt + lanes + multi + row_multi + row_dispatch + entries + kernel + launch
result.encode('ascii')
(P / 'private_iq2_gu.hpp').write_text(result, encoding='ascii')
record = {'status': 'SOURCE_ONLY_NO_BUILD_GPU', 'source': str(SOURCE),
    'source_sha256': sha(SOURCE.read_bytes()), 'header_sha256': sha(result.encode()),
    'Fmt17_dot_byte_exact_source': True, 'original_dot_sha256': sha(original.encode()),
    'helpers_swar_lanes_and_row_dot_multi_iq_exact': True,
    'row_entries_changes': ['Only compile-time reuse predicate FALSE(current17) or TRUE(candidate17); exact E4/E2/singleton bodies'],
    'kernel_changes': ['Private name/compile-time USE_MULTI forwarded; same TG17/LN8/row/group/entry/output ownership'],
    'candidate_prep': ['Original q2 uint32 shifts, grid lookup/sign/SWAR expressions; activation reads moved to acts; only weight results cached'],
    'finish': 'Original nested signed integer divisions, then const float d=m.d*ds; return d*sumi;',
    'launch': 'Same public port SG32/WG256/root_sync/gx/groups; unique s98::PrivateGu17<false/true>',
    'public_anchor': 'native_expert_grouped phase1 scratch gate/up first, full strict bytes incl inactive',
    'root_binding_required': 'Actual final public/private Info/cache/guard/registered-name/device image before GPU; private symbol names alone insufficient',
    'not_changed': ['Main TU/object/archive', 'Q8', 'K/XOR order', 'integer truncations', 'FP conversion/scales', 'group grid/lanes', 'routing', 'metadata/output planes']}
(P / 'extraction.json').write_text(json.dumps(record, indent=2) + '\n', encoding='ascii')
print(json.dumps({'source_generation_only': True, 'header_bytes': len(result.encode()), 'header_sha256': record['header_sha256']}, indent=2))
