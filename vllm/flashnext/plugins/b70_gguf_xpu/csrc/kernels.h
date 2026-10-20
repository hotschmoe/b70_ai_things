#pragma once
#include <cstdint>

// queue is a sycl::queue* (the torch XPU stream's queue). Pointers are device USM.
void b70_dequant_bf16(void* queue, int type, const uint8_t* q, int64_t rows, int64_t cols, uint16_t* out);
void b70_gemv(void* queue, int type, const uint16_t* x, const uint8_t* w, float* y,
              int M, int N, int K);
// expert_ids: int32 [T, topk]. y: float [T, topk, N]. x: bf16 [T, K]. w packed [E, N, row].
void b70_moe_gemv(void* queue, int type, const uint16_t* x, const uint8_t* w, const int32_t* expert_ids,
                  float* y, int T, int topk, int E, int N, int K);
// x: fp16 [M, K]. entries: int32 [n_ent, 3] = expert, row0, nrows. y: float [M, N].
// Returns 0 when the XMX kernel is launched. Nonzero when the shape is refused.
int b70_grouped_gemm(void* queue, int type, const uint16_t* x, const uint8_t* w, const int32_t* entries,
                     float* y, int n_ent, int E, int M, int N, int K);
