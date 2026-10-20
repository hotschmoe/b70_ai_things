#pragma once
#include <cstdint>

// queue is a sycl::queue* (the torch XPU stream's queue). Pointers are device USM.
void b70_dequant_bf16(void* queue, int type, const uint8_t* q, int64_t rows, int64_t cols, uint16_t* out);
void b70_gemv(void* queue, int type, const uint16_t* x, const uint8_t* w, float* y,
              int M, int N, int K);
// expert_ids: int32 [T, topk]. y: float [T, topk, N]. x: bf16 [T, K]. w packed [E, N, row].
void b70_moe_gemv(void* queue, int type, const uint16_t* x, const uint8_t* w, const int32_t* expert_ids,
                  float* y, int T, int topk, int E, int N, int K);
// One host call. Queues Q8(x), gate+up+SiLU, Q8(hidden), down, route-accumulate.
// Shared pointers are null when the layer has no shared expert. y: float [T, K].
// I is the intermediate width (gate N). K is the hidden width (down N).
void b70_moe_layer(void* queue, const uint16_t* x, const uint8_t* gate, int gate_type, const uint8_t* up, int up_type,
                   const uint8_t* down, int down_type, const int32_t* expert_ids, const float* router,
                   const uint8_t* sgate, int sg_type, const uint8_t* sup, int su_type, const uint8_t* sdown, int sd_type,
                   float* y, int T, int topk, int E, int I, int K);
// x: fp16 [M, K]. entries: int32 [n_ent, 3] = expert, row0, nrows. y: float [M, N].
// Returns 0 when the XMX kernel is launched. Nonzero when the shape is refused.
int b70_grouped_gemm(void* queue, int type, const uint16_t* x, const uint8_t* w, const int32_t* entries,
                     float* y, int n_ent, int E, int M, int N, int K);
