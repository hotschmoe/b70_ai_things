#include "decode.hpp"
#include "kernels.h"

#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>

#include <stdexcept>

namespace {

void check_xpu(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_xpu(), name, " must be an XPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

at::Tensor dequant_bf16(at::Tensor q, int64_t ggml_type, int64_t rows, int64_t cols) {
    check_xpu(q, "q");
    TORCH_CHECK(q.scalar_type() == at::kByte, "q must be uint8");
    TORCH_CHECK(rows > 0 && cols > 0, "dequant shape");
    const int64_t need = rows * b70_row_bytes((int)ggml_type, (int)cols);
    TORCH_CHECK(need > 0 && q.numel() >= need, "q is shorter than rows*row_bytes");
    auto out = at::empty({rows, cols}, q.options().dtype(at::kBFloat16));
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    b70_dequant_bf16(&queue, (int)ggml_type, q.data_ptr<uint8_t>(), rows, cols,
                     static_cast<uint16_t*>(out.data_ptr()));
    return out;
}

at::Tensor gemv(at::Tensor x, at::Tensor q, int64_t ggml_type, int64_t N, int64_t K) {
    check_xpu(x, "x");
    check_xpu(q, "q");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16 [M, K]");
    TORCH_CHECK(q.scalar_type() == at::kByte, "q must be uint8");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K, "x must be [M, K]");
    const int64_t M = x.size(0);
    const int rb = b70_row_bytes((int)ggml_type, (int)K);
    TORCH_CHECK(rb > 0, "K is not a multiple of the block, or type is unknown");
    TORCH_CHECK(q.numel() >= N * rb, "q is shorter than N*row_bytes");
    auto y = at::empty({M, N}, x.options().dtype(at::kFloat));
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    b70_gemv(&queue, (int)ggml_type, static_cast<const uint16_t*>(x.data_ptr()), q.data_ptr<uint8_t>(),
             y.data_ptr<float>(), (int)M, (int)N, (int)K);
    return y;
}

at::Tensor moe_gemv(at::Tensor x, at::Tensor q, at::Tensor expert_ids, int64_t ggml_type, int64_t N, int64_t K) {
    check_xpu(x, "x");
    check_xpu(q, "q");
    check_xpu(expert_ids, "expert_ids");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == K, "x must be bf16 [T, K]");
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt && expert_ids.dim() == 2, "expert_ids must be int32 [T, topk]");
    TORCH_CHECK(expert_ids.size(0) == x.size(0), "expert_ids rows must match tokens");
    const int64_t T = x.size(0);
    const int64_t topk = expert_ids.size(1);
    const int rb = b70_row_bytes((int)ggml_type, (int)K);
    TORCH_CHECK(rb > 0, "bad type or K");
    TORCH_CHECK(q.numel() % (N * rb) == 0, "q numel is not E*N*row_bytes");
    const int64_t E = q.numel() / (N * rb);
    auto y = at::empty({T, topk, N}, x.options().dtype(at::kFloat));
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    b70_moe_gemv(&queue, (int)ggml_type, static_cast<const uint16_t*>(x.data_ptr()), q.data_ptr<uint8_t>(),
                 expert_ids.data_ptr<int32_t>(), y.data_ptr<float>(), (int)T, (int)topk, (int)E, (int)N, (int)K);
    return y;
}

at::Tensor grouped_gemm(at::Tensor x, at::Tensor q, at::Tensor entries, int64_t ggml_type, int64_t N, int64_t K) {
    check_xpu(x, "x");
    check_xpu(q, "q");
    check_xpu(entries, "entries");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K, "x must be [M, K]");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "x must be fp16 or bf16");
    TORCH_CHECK(entries.scalar_type() == at::kInt && entries.dim() == 2 && entries.size(1) == 3,
                "entries must be int32 [n_ent, 3] (expert, row0, nrows)");
    at::Tensor xf = x.scalar_type() == at::kHalf ? x : x.to(at::kHalf);
    const int64_t M = xf.size(0);
    const int rb = b70_row_bytes((int)ggml_type, (int)K);
    TORCH_CHECK(rb > 0, "bad type or K");
    TORCH_CHECK(q.numel() % (N * rb) == 0, "q numel is not E*N*row_bytes");
    const int64_t E = q.numel() / (N * rb);
    auto y = at::zeros({M, N}, xf.options().dtype(at::kFloat));
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    const int rc = b70_grouped_gemm(&queue, (int)ggml_type, static_cast<const uint16_t*>(xf.data_ptr()),
                                    q.data_ptr<uint8_t>(), entries.data_ptr<int32_t>(), y.data_ptr<float>(),
                                    (int)entries.size(0), (int)E, (int)M, (int)N, (int)K);
    TORCH_CHECK(rc == 0, "grouped_gemm refused the shape (need N%64==0, K%32==0, known type); rc=", rc);
    return y;
}

}  // namespace

TORCH_LIBRARY(b70_gguf, m) {
    m.def("dequant_bf16(Tensor q, int ggml_type, int rows, int cols) -> Tensor");
    m.def("gemv(Tensor x, Tensor q, int ggml_type, int N, int K) -> Tensor");
    m.def("moe_gemv(Tensor x, Tensor q, Tensor expert_ids, int ggml_type, int N, int K) -> Tensor");
    m.def("grouped_gemm(Tensor x, Tensor q, Tensor entries, int ggml_type, int N, int K) -> Tensor");
}

TORCH_LIBRARY_IMPL(b70_gguf, XPU, m) {
    m.impl("dequant_bf16", dequant_bf16);
    m.impl("gemv", gemv);
    m.impl("moe_gemv", moe_gemv);
    m.impl("grouped_gemm", grouped_gemm);
}
