#!/usr/bin/env python3
"""Extract one fixed C6 specialization; no compilation or main source edits."""
from pathlib import Path
import difflib,hashlib,json,re
p=Path(__file__).parent
s=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/kernels/cuda/iq_kernels.dp.cpp')
t=s.read_text()
assert hashlib.sha256(s.read_bytes()).hexdigest()=='acaa7da63733af9f3831b3a74b99e6176bf46a06e498e0a33c0591f7f4a29851'
def between(a,b):return t[t.index(a):t.index(b,t.index(a))].rstrip()+'\n'
loadhelpers=between('__dpct_inline__ int get_int_b2','__dpct_inline__ uint32_t unpack_ksigns')
swar=between('#ifndef STRATA_SWAR','__dpct_inline__ float vec_dot_iq2_xxs')
assert '__dpct_inline__ int swar_ne4' in swar and '__dpct_inline__ int swar_sub4' in swar
orig=between('template<> struct Split<21>','template<> inline constexpr bool kSplit<29>')
orig=orig.replace('template<> struct Split<21>','struct ControlSplit')
load=orig[orig.index('    template<bool STAGE_GRID'):orig.index('    static float apply')]
fast=load
fast=re.sub(r'dpct::vectorized_binary<sycl::uchar4>\(\s*(\(\(signs_packed_8\[l0 / 2\].*?),\s*0x00000000, std::not_equal_to<>\(\)\)',r'swar_ne4(\1)',fast,flags=re.S)
fast=re.sub(r'dpct::vectorized_binary<sycl::uchar4>\(\s*(grid_pos\.[xy]\(\) \^ signs[01]), (signs[01]), std::minus<>\(\)\)',r'swar_sub4(\1, \2)',fast)
assert fast.count('swar_ne4(')==2 and fast.count('swar_sub4(')==2
assert 'vectorized_binary' not in fast and fast.count('get_int_b2')==load.count('get_int_b2')==3
candidate='struct CandidateSplit : ControlSplit {\n'+fast+'};\n'
warp=between('__dpct_inline__ float warp_sum','// SYCL port: lanes per row')
row=between('template <int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>\n__dpct_inline__ void\nrow_dot_multi','// 8-lane sub-warp row dot')
row=row.replace('bool STAGE_GRID = false>','bool STAGE_GRID = false, bool FAST = false>')
row=row.replace('using S = Split<TY>;','using S = std::conditional_t<FAST, CandidateSplit, ControlSplit>;\n    static_assert(TY == 21 && NC == 6 && EXACT_N && !STAGE_GRID);')
stage=between('template <int TY, bool STAGE_GRID = kStageIqGrid<TY>>\n__dpct_inline__ const uint32_t *\nstage_iq_grid','// mmvq_kernel with the columns taken NC')
kernel=between('template <int TY, int NC, bool STAGE_GRID = kStageIqGrid<TY>,\n          bool EXACT_N = false>\n__dpct_inline__ void\nmmvq_multi_kernel','// ---------------------------------------------------------------- grouped native experts')
kernel=kernel.replace('bool EXACT_N = false>','bool EXACT_N = false, bool FAST = false>')
kernel=kernel.replace('row_dot_multi<TY, NC, true, STAGE_GRID>','row_dot_multi<TY, NC, true, STAGE_GRID, FAST>')
kernel=kernel.replace('row_dot_multi<TY, NC, false, STAGE_GRID>','row_dot_multi<TY, NC, false, STAGE_GRID, FAST>')
header='''// S65 private fixed IQ3_S C6 oracle; source pin and transformations in extraction.json.
// Main source and global Split21/expert dispatch remain untouched.
#pragma once
#include "strata/sycl_math.hpp"
#include "strata/kernels/dp4a.hpp"
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"
#include <type_traits>
namespace s65 {
#define ggml_cuda_dp4a(a, b, c) strata::dp4a((a), (b), (c))
'''
fmt='''template<int TY> struct Fmt;
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2; };
template<int TY> inline constexpr bool kStageIqGrid = (TY == 16 || TY == 17 || TY == 22 || TY == 29);
'''
launch='''template<bool FAST> class PrivateC6;
template<bool FAST> void launch(const void* W, const void* X, float* Y, int K, int M, sycl::queue& q) {
    const dpct::dim3 grid((unsigned)((M + 3) / 4)), block(32, 4);
    const size_t rb = size_t(K / 256) * 110;
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<PrivateC6<FAST>>(
            sycl::nd_range<3>(grid * block, block), exp_props,
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                mmvq_multi_kernel<21, 6, false, true, FAST>(
                    static_cast<const uint8_t*>(W), rb,
                    static_cast<const block_q8_1*>(X), Y, K, M, 6);
            });
    });
}
#undef ggml_cuda_dp4a
} // namespace s65
'''
result=header+loadhelpers+swar+orig+candidate+warp+fmt+row+stage+kernel+launch
(p/'private_iq3s_c6.hpp').write_text(result)
(p/'decode-only.diff').write_text(''.join(difflib.unified_diff(load.splitlines(True),fast.splitlines(True),fromfile='ControlSplit::load',tofile='CandidateSplit::load')))
e={'source':str(s),'source_sha256':hashlib.sha256(s.read_bytes()).hexdigest(),
 'control_split_exact_source_except_struct_name':True,'candidate_inherits_exact_control_apply':True,
 'original_load_sha256':hashlib.sha256(load.encode()).hexdigest(),
 'candidate_load_sha256':hashlib.sha256(fast.encode()).hexdigest(),
 'copied_row_dot_changes':['add compile-time FAST parameter and fixed-specialization assertion','choose private control/candidate load policy; identical Fmt21 K schedule/apply/reduction'],
 'copied_kernel_changes':['add FAST parameter and forward to row_dot_multi; same local staging false branch and exact-C6 specialization'],
 'private_launch':'same C6 geometry, reqd subgroup32, use_root_sync; new private kernel type names',
 'numerical_gate':'private control full bytes against public production native_mmvq before any candidate timing',
 'candidate_changes':['ONLY two byte compare and two byte subtract expressions -> existing SWAR helpers'],
 'retained_loads':'Exact original two qs get_int_b2 and one signs get_int_b2; S66 actual ISA already folds to three d32 messages',
 'not_changed':['main IQ TU','global Split21','expert paths','all weight loads','Split21::apply','float grouping','lane K order','XOR reduction','canonical Q8 operands']}
(p/'extraction.json').write_text(json.dumps(e,indent=2)+'\n')
print('SOURCE_EXTRACTION_ONLY',len(result),'bytes')
