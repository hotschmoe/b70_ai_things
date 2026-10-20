#pragma once
// C10 prefill helpers. Declarations only; bind.cpp registers the ops.
// Decode TUs do not include this header.
#include <ATen/ATen.h>

// out is bf16 or fp16 [rows, cols]. Writes out, returns out.
at::Tensor b70_prefill_dequant_op(at::Tensor q, int64_t ggml_type, int64_t rows, int64_t cols, at::Tensor out);

// ids int32 [M]. Invalid ids (<0 or >=E) are dropped.
// order [M], starts [E+1], counts [E], cursor [E] are device int32 scratch.
// order[0:starts[E]] is grouped by expert. Scratch must outlive the queued kernels.
at::Tensor b70_prefill_group_ids_op(at::Tensor ids, int64_t E, at::Tensor order, at::Tensor starts, at::Tensor counts,
                                    at::Tensor cursor);

// dst[p] = src[index[p]] for p in 0..mv. src [M,K], dst [mv,K], index [M] int32. 2-byte elements, K%8==0.
at::Tensor b70_prefill_row_gather_op(at::Tensor src, at::Tensor dst, at::Tensor index, int64_t mv);

// dst[index[p]] = src[p] for p in 0..mv. src [mv,K], dst [M,K].
at::Tensor b70_prefill_row_scatter_op(at::Tensor src, at::Tensor dst, at::Tensor index, int64_t mv);

// xpad[e, local] = xflat[order[p]], p in 0..mv, local = p - starts[e].
// xflat [M,K], xpad [E, mmax, K], order [M], starts [E+1].
at::Tensor b70_prefill_pad_grouped_op(at::Tensor xflat, at::Tensor xpad, at::Tensor order, at::Tensor starts, int64_t mv);

// slots[order[p]] = ypad[e, local]. ypad [E, mmax, K], slots [M, K].
at::Tensor b70_prefill_unpad_scatter_op(at::Tensor ypad, at::Tensor slots, at::Tensor order, at::Tensor starts,
                                        int64_t mv);
