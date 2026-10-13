// Experimental primitives only. No model path calls these until separately qualified.
#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::kernels {
namespace hc_projection_detail {
// CPU-testable shape contract; bounded products avoid overflow in the launchers.
inline bool shape_ok(int k, int m, int tokens, size_t inputs, size_t outputs) {
    return k > 0 && k <= 10240 && k % 32 == 0 && m > 0 && m <= 10240 &&
           tokens >= 1 && tokens <= 8 && inputs >= size_t(k)*tokens && outputs >= size_t(m)*tokens;
}
}
// Row-major weights [m,k], activations [tokens,k], output [tokens,m].
// Q8_0 uses exact GGUF blocks: little-endian FP16 scale + 32 signed int8 codes.
// F32 is for source F32 injection matrices; it never passes through BF16/Q8.
// Caller owns device-accessible, non-overlapping storage and an in-order stream.
// No allocation, synchronization, activation quantization or weight mutation.
// False means invalid descriptor/queue and NO kernel was submitted.
bool hc_q8_0_project_f32(const uint8_t * weights, size_t weight_bytes,
        const float * input, size_t input_floats, float * output, size_t output_floats,
        int k, int m, int tokens, void * stream);
// Explicit experimental token reuse; tokens1..512/tile2 or4 only. Same F32 lane arithmetic.
// STRATA_HC_Q8_TOKEN_TILE absent/0 preserves old production launch;2/4 uses this on tokens>=2.
// Production composition descriptors/caller still split prefill into batches<=8.
bool hc_q8_0_project_f32_token_reuse(const uint8_t* weights,size_t weight_bytes,
        const float* input,size_t input_floats,float* output,size_t output_floats,
        int k,int m,int tokens,int tile,void* stream);
bool hc_f32_project_f32(const float * weights, size_t weight_floats,
        const float * input, size_t input_floats, float * output, size_t output_floats,
        int k, int m, int tokens, void * stream);
// Explicit contiguous prefill F32 injection; same launch/reduction, old bound unchanged.
bool hc_f32_project_f32_prefill(const float* weights,size_t weight_floats,
        const float* input,size_t input_floats,float* output,size_t output_floats,
        int k,int m,int tokens,void* stream);

}
