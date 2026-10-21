#pragma once
#include <ATen/ATen.h>
namespace b70_p1_dequant {
at::Tensor b70_prefill_dequant_wide_op(at::Tensor q, int64_t ggml_type, int64_t rows, int64_t cols, at::Tensor out, int64_t dst_mode, int64_t pair);
at::Tensor b70_prefill_dequant_selected_op(at::Tensor q, int64_t ggml_type, int64_t rows, int64_t cols, at::Tensor out, int64_t dst_mode, int64_t pair, at::Tensor sel, int64_t expert_rows);
}
