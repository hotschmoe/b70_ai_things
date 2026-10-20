#pragma once
// C13 wide dequant. Declarations only. Decode TUs do not include this header.
#include <ATen/ATen.h>

// out is bf16 or fp16.
// dst_mode 0: packed [rows, cols]. out has `rows` rows.
// dst_mode 1: source row r = e*pair+f is written at row e*(2*pair)+f
//             (gate half of [E, 2*pair, cols]). out has E*2*pair rows.
// dst_mode 2: same, written at e*(2*pair)+pair+f (up half).
// Bits match prefill_dequant / b70_decode32. Tuned types use one subgroup
// per 512 outputs: vector block loads, one lane per 32-wide group, one
// 64-byte store.
at::Tensor b70_prefill_dequant_wide_op(at::Tensor q, int64_t ggml_type, int64_t rows, int64_t cols, at::Tensor out,
                                       int64_t dst_mode, int64_t pair);
