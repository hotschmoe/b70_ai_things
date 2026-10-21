#include "p1_dequant.hpp"
#include <torch/library.h>
TORCH_LIBRARY(b70_p1, m) {
    m.def("prefill_dequant_selected(Tensor q, int ggml_type, int rows, int cols, Tensor out, int dst_mode, int pair, Tensor sel, int expert_rows) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_p1, XPU, m) {
    m.impl("prefill_dequant_selected", b70_p1_dequant::b70_prefill_dequant_selected_op);
}
