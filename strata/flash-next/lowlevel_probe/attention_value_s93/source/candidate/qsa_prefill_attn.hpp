#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

namespace strata::kernels {

// SYCL staged-prefill-only entrypoint. Caller guarantees a full 32-query batch,
// all selection widths2051, and complete identity-mapped staged KV publication.
// Exact flag1/INT8/page4/cap2051/24x256 with2KV heads selects the value pipeline;
// every other domain delegates to the original qsa_decode_attn_batch.
void qsa_prefill_attn_value_batch(const float* q, const QsaAttnPools& pools,
                                const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, float* scratch,
                                float* attn, int64_t n_q, void* stream);

}  // namespace strata::kernels
