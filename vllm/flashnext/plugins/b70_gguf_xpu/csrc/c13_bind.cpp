// C13 registration. Linked into _C.c13.so only. Same ops as c11_bind.cpp
// plus prefill_dequant_wide. Does not replace bind.cpp.
#include "dequant_fast.hpp"
#include "dequant_wide.hpp"
#include "grouped_api.hpp"

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(b70_gguf, m) {
    m.def("prefill_grouped(Tensor x, Tensor q, Tensor expert_ids, int type, int N, int K, int mode) -> Tensor");
    m.def("prefill_dense(Tensor x, Tensor q, int type, int N, int K, int mode) -> Tensor");
    m.def(
        "prefill_moe_layer(Tensor x, Tensor ids, Tensor q_gate, Tensor q_up, Tensor q_down, Tensor router, "
        "int type_gu, int type_down, int H, int F, int mode) -> Tensor");
    m.def("prefill_split_us() -> Tensor");
    m.def("prefill_wide_sgn() -> Tensor");
}

TORCH_LIBRARY_IMPL(b70_gguf, XPU, m) {
    m.impl("prefill_grouped", b70_prefill_grouped_op);
    m.impl("prefill_dense", b70_prefill_dense_op);
    m.impl("prefill_moe_layer", b70_prefill_moe_layer_op);
}

TORCH_LIBRARY_IMPL(b70_gguf, CompositeExplicitAutograd, m) {
    m.impl("prefill_split_us", b70_prefill_split_us_op);
    m.impl("prefill_wide_sgn", b70_prefill_wide_sgn_op);
}

TORCH_LIBRARY_FRAGMENT(b70_gguf, m) {
    m.def("prefill_dequant(Tensor q, int ggml_type, int rows, int cols, Tensor out) -> Tensor");
    m.def("prefill_group_ids(Tensor ids, int E, Tensor order, Tensor starts, Tensor counts, Tensor cursor) -> Tensor");
    m.def("prefill_row_gather(Tensor src, Tensor dst, Tensor index, int mv) -> Tensor");
    m.def("prefill_row_scatter(Tensor src, Tensor dst, Tensor index, int mv) -> Tensor");
    m.def("prefill_pad_grouped(Tensor xflat, Tensor xpad, Tensor order, Tensor starts, int mv) -> Tensor");
    m.def("prefill_unpad_scatter(Tensor ypad, Tensor slots, Tensor order, Tensor starts, int mv) -> Tensor");
    m.def(
        "prefill_dequant_wide(Tensor q, int ggml_type, int rows, int cols, Tensor out, int dst_mode, int pair) -> Tensor");
}

TORCH_LIBRARY_IMPL(b70_gguf, XPU, m) {
    m.impl("prefill_dequant", b70_prefill_dequant_op);
    m.impl("prefill_group_ids", b70_prefill_group_ids_op);
    m.impl("prefill_row_gather", b70_prefill_row_gather_op);
    m.impl("prefill_row_scatter", b70_prefill_row_scatter_op);
    m.impl("prefill_pad_grouped", b70_prefill_pad_grouped_op);
    m.impl("prefill_unpad_scatter", b70_prefill_unpad_scatter_op);
    m.impl("prefill_dequant_wide", b70_prefill_dequant_wide_op);
}
