// Additive C13 dequant extension; load after the full decode library.
#include "dequant_wide.hpp"
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(b70_gguf, m) {
    m.def("prefill_dequant_wide(Tensor q, int ggml_type, int rows, int cols, Tensor out, int dst_mode, int pair) -> Tensor");
}
TORCH_LIBRARY_IMPL(b70_gguf, XPU, m) {
    m.impl("prefill_dequant_wide", b70_prefill_dequant_wide_op);
}
