#pragma once
#include <ATen/ATen.h>

// C2 prefill entry points. Registered from bind.cpp in a separate block.
at::Tensor b70_prefill_grouped_op(at::Tensor x, at::Tensor q, at::Tensor expert_ids, int64_t ggml_type, int64_t N,
                                  int64_t K, int64_t mode);
at::Tensor b70_prefill_dense_op(at::Tensor x, at::Tensor q, int64_t ggml_type, int64_t N, int64_t K, int64_t mode);
at::Tensor b70_prefill_moe_layer_op(at::Tensor x, at::Tensor ids, at::Tensor q_gate, at::Tensor q_up, at::Tensor q_down,
                                    at::Tensor router, int64_t type_gu, int64_t type_down, int64_t H, int64_t F,
                                    int64_t mode);
at::Tensor b70_prefill_split_us_op();
// 1-element int32 CPU tensor. 40 when the wide SGN kernels are in this binary.
at::Tensor b70_prefill_wide_sgn_op();
