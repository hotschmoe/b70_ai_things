#pragma once
// int8_gemm_w8a16_reorder.h -- EXPERIMENTAL VNNI16-blocked weight-only int8 W8A16
// small-M GEMV for the B70 (Xe2). COMPILE-ONLY artifact for the coordinator.
//
// STATUS: NO-GO for production (see decode_gemv/FINDINGS.md). The oneDNN
// int8_gemm_w8a16 op ALREADY streams the s8 weight at 92-98% of the 581 GB/s read
// roofline at M=1 (gate_up 567 GB/s = 97.6%, down 533 = 91.7%, qkv 538 = 92.5%).
// The only headroom is the ~8% tail on down_proj / qkv. This header targets that
// residual by giving the weight a VNNI16 layout ([K/16, N, 16]) so each work-item
// streams 16 contiguous s8 (one 128-bit load) per step -- the same layout trick
// that gave llama.cpp #21527 3.1x, but that win was measured from a SUB-roofline
// baseline; here the baseline is already at roofline, so the expected uplift is
// <=1.05-1.08x on down/qkv and ~1.0x on gate_up. Ship ONLY if the microbench shows
// the reorder GEMV beats oneDNN int8_gemm_w8a16(graph) by a MEASURED, coherent margin.
//
// Numerics: y[n] = wscale[n] * sum_k ( x_f32[k] * (float)Wq[n,k] ), symmetric int8
// weight (no zero point), f16/f32 activation. Matches the oneDNN op (relerr ~9e-3
// vs bf16). Accumulate in fp32.
//
// Standalone SYCL (no torch): the coordinator can icpx -fsycl compile-check this
// with bench_reorder_standalone.cpp (COMPILE ONLY -- running dispatches to the GPU).

#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>

namespace w8a16_reorder {

// ---------------------------------------------------------------------------
// OFFLINE weight repack (host side, run ONCE at process_weights_after_loading).
// Input : Wq  [N, K] s8 row-major (per-channel-symmetric int8 weight, the same
//              tensor the oneDNN op consumes as B_nt = Wq.t()).
// Output: Wp  [K/16, N, 16] s8 -- 16 consecutive K for a given output n are
//              contiguous, so the GEMV loads them as one vec<int8_t,16>.
// K must be a multiple of 16 (qwen3.6-27b: 5120, 6144, 17408 all %16==0). N free.
// ---------------------------------------------------------------------------
static inline void repack_vnni16(
    const int8_t* Wq, int64_t N, int64_t K, std::vector<int8_t>& Wp) {
  const int64_t Kb = K / 16;
  Wp.assign(static_cast<size_t>(Kb) * N * 16, 0);
  for (int64_t n = 0; n < N; ++n)
    for (int64_t kb = 0; kb < Kb; ++kb)
      for (int64_t j = 0; j < 16; ++j)
        // Wp[kb, n, j] = Wq[n, kb*16 + j]
        Wp[(kb * N + n) * 16 + j] = Wq[n * K + kb * 16 + j];
}

// ---------------------------------------------------------------------------
// Device GEMV. One sub-group (SG_SIZE lanes) cooperatively reduces one output n.
// Each lane owns a stripe of the K/16 blocks; per block it does a 16-wide vector
// load of the packed weight and a 16-wide load of x, multiply-accumulates in
// fp32, then a sub-group reduce -> y[n] = wscale[n] * acc.
// grid: N sub-groups. Small M loop is outer (M<=8; the weight is read once per M
// only if you hoist the M loop inside -- for true MTP amortization prefer the
// oneDNN op which reads the weight once for all M. This kernel is the M=1 probe).
// ---------------------------------------------------------------------------
template <int SG_SIZE = 16>
struct W8A16GemvVNNI16 {
  const int8_t* __restrict Wp;    // [K/16, N, 16] s8
  const float* __restrict x;      // [K] f32 (dequantized activation row)
  const float* __restrict wscale; // [N] f32
  float* __restrict y;            // [N] f32
  int64_t N, K;

  void operator()[[sycl::reqd_sub_group_size(SG_SIZE)]](
      sycl::nd_item<1> it) const {
    const int64_t n = it.get_group(0);
    if (n >= N) return;
    auto sg = it.get_sub_group();
    const int lane = sg.get_local_id()[0];
    const int64_t Kb = K / 16;

    float acc = 0.0f;
    // stripe the K-blocks across the SG lanes; contiguous 16-wide loads
    for (int64_t kb = lane; kb < Kb; kb += SG_SIZE) {
      const int8_t* wp = Wp + (kb * N + n) * 16;
      const float* xp = x + kb * 16;
#pragma unroll
      for (int j = 0; j < 16; ++j)
        acc += static_cast<float>(wp[j]) * xp[j];
    }
    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = wscale[n] * acc;
  }
};

// Launch helper (host): global = N * SG_SIZE, local = SG_SIZE.
template <int SG_SIZE = 16>
static inline sycl::event launch_w8a16_vnni16(
    sycl::queue& q, const int8_t* Wp, const float* x, const float* wscale,
    float* y, int64_t N, int64_t K) {
  return q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(N) * SG_SIZE),
                        sycl::range<1>(SG_SIZE)),
      W8A16GemvVNNI16<SG_SIZE>{Wp, x, wscale, y, N, K});
}

}  // namespace w8a16_reorder
