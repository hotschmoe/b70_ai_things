// Experimental composition only: callers must qualify and wire every model route.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cmath>
namespace strata::kernels {
// Exact source types: Q8_0 down/up, original F32 norm and injection.
// Row-major GGUF [M,K]; contiguous token-major activations. No reinterpretation
// of WeightRef::hc_q8, BF16 compatibility storage or Q8 injection conversion.
struct HcNativeWeights {
    const float *norm = nullptr; size_t norm_floats = 0;
    const uint8_t *down = nullptr; size_t down_bytes = 0;
    const uint8_t *up = nullptr; size_t up_bytes = 0;
    const float *inject = nullptr; size_t inject_floats = 0;
};
struct HcNativeArgs {
    HcNativeWeights w;
    int n_embd = 2560, hc = 4, lr = 320, tokens = 1;
    float eps = 1e-6f;
    const float *R = nullptr; size_t R_floats = 0;
    bool apply = false;
    const float *bo_prev = nullptr; size_t bo_floats = 0;
    const float *inj_prev = nullptr; size_t prev_floats = 0;
    float *R_out = nullptr; size_t R_out_floats = 0;
    // All workspace/output arrays are owned by the active stage/session.
    float *xn = nullptr; size_t xn_floats = 0;
    float *lo = nullptr; size_t lo_floats = 0;
    float *gate = nullptr; size_t gate_floats = 0;
    float *rs = nullptr; size_t rs_floats = 0;
    float *mixed = nullptr; size_t mixed_floats = 0;
    float *inject_out = nullptr; size_t inject_out_floats = 0;
};
// Validate the complete descriptor before any submission. Only this selected
// model geometry and 1..8 rows are supported; callers must chunk larger prompts.
// USM accessibility, context ownership and nonoverlap are caller contracts.
// Only R_out==R is permitted to overlap. No hidden allocations or global state.
inline bool hc_native_descriptor_ok(const HcNativeArgs& a) {
    if (a.n_embd != 2560 || a.hc != 4 || a.lr != 320 || a.tokens < 1 || a.tokens > 8 ||
        !(a.eps > 0) || !std::isfinite(a.eps)) return false;
    const size_t D = 10240, T = size_t(a.tokens), N = 2560, H = 4, L = 320;
    if (!a.w.norm || a.w.norm_floats < D || !a.w.down || a.w.down_bytes < L*(D/32)*34 ||
        !a.w.up || a.w.up_bytes < D*(L/32)*34 ||
        !a.R || a.R_floats < T*D || !a.xn || a.xn_floats < T*D ||
        !a.lo || a.lo_floats < T*L || !a.gate || a.gate_floats < T*D ||
        !a.rs || a.rs_floats < T*H || !a.mixed || a.mixed_floats < T*N) return false;
    if (a.w.inject) {
        if (a.w.inject_floats < H*D || !a.inject_out || a.inject_out_floats < T*H) return false;
    } else if (a.w.inject_floats || a.inject_out || a.inject_out_floats) return false;
    if (a.apply) {
        if (!a.bo_prev || a.bo_floats < T*N || !a.inj_prev || a.prev_floats < T*H ||
            !a.R_out || a.R_out_floats < T*D) return false;
    } else if (a.bo_prev || a.bo_floats || a.inj_prev || a.prev_floats || a.R_out || a.R_out_floats) return false;
    return true;
}
// Explicit contiguous prefill entry; old descriptor remains bounded at eight.
inline bool hc_native_prefill_descriptor_ok(const HcNativeArgs& a) {
    if (a.n_embd != 2560 || a.hc != 4 || a.lr != 320 || a.tokens < 1 || a.tokens > 512 ||
        !(a.eps > 0) || !std::isfinite(a.eps)) return false;
    const size_t D = 10240, T = size_t(a.tokens), N = 2560, H = 4, L = 320;
    if (!a.w.norm || a.w.norm_floats < D || !a.w.down || a.w.down_bytes < L*(D/32)*34 ||
        !a.w.up || a.w.up_bytes < D*(L/32)*34 ||
        !a.R || a.R_floats < T*D || !a.xn || a.xn_floats < T*D ||
        !a.lo || a.lo_floats < T*L || !a.gate || a.gate_floats < T*D ||
        !a.rs || a.rs_floats < T*H || !a.mixed || a.mixed_floats < T*N) return false;
    if (a.w.inject) {
        if (a.w.inject_floats < H*D || !a.inject_out || a.inject_out_floats < T*H) return false;
    } else if (a.w.inject_floats || a.inject_out || a.inject_out_floats) return false;
    if (a.apply) {
        if (!a.bo_prev || a.bo_floats < T*N || !a.inj_prev || a.prev_floats < T*H ||
            !a.R_out || a.R_out_floats < T*D) return false;
    } else if (a.bo_prev || a.bo_floats || a.inj_prev || a.prev_floats || a.R_out || a.R_out_floats) return false;
    return true;
}
// False: descriptor or queue rejected, zero kernels submitted.
// SYCL submission/execution exceptions propagate: caller must abort the request;
// no compatibility fallback or retry on partly written state is permitted.
bool hc_native_read_f32(const HcNativeArgs& a, void *stream);
bool hc_native_prefill_read_f32(const HcNativeArgs& a, void *stream);
// Contiguous token-major residual write, same FMA/sigmoid math as pending read.
// Invalid pointer/capacity/row count or queue returns false before submission.
inline bool hc_native_write_shape_ok(size_t R,size_t bo,size_t inject,int tokens) {
    return tokens>=1 && tokens<=8 && R>=size_t(tokens)*10240 &&
        bo>=size_t(tokens)*2560 && inject>=size_t(tokens)*4;
}
bool hc_native_write_f32(float* R,size_t R_floats,const float* bo,size_t bo_floats,
                        const float* inject,size_t inject_floats,int tokens,void* stream);
bool hc_native_prefill_write_f32(float* R,size_t R_floats,const float* bo,size_t bo_floats,
        const float* inject,size_t inject_floats,int tokens,void* stream);

}
