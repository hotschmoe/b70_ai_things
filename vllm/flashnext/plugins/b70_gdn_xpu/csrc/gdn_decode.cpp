// Ornith-35B GDN decode + paged full-attention decode for Intel B70 (BMG G31).
//
// Gated-delta schedule follows strata::kernels::fused_gdn_step_norm
// (sycl/src/kernels/cuda/fused_gdn.dp.cpp, Flash-Next / Strata
// night-20261016, /mnt/vm_8tb/b70/build/strata-ab/src-ep). State stays in
// registers across the token loop. That is an attribution of the schedule,
// not a copy of Strata's math or layout:
//   * vLLM state is [head, V, K] with K contiguous, fp32, 32 x 128 x 128.
//   * Conv, beta/gate, L2, and RMSNorm-gated-silu are the vLLM XPU
//     gdn_attention math (vllm-xpu-kernels csrc/xpu/gdn_attn,
//     causal_conv1d.hpp + gated_delta_rule.hpp + RMSNormGated).
//   * Strata's sigmoid(z), native::exp, and the extra rsqrt(S) on the
//     output are not used. Those do not match Ornith.
//
// One work-group owns one (sequence, value head): conv window, recurrent
// update, and the group RMS. q/k conv channels are shared by the two value
// heads of a key head; each head keeps a private SLM window and only
// local_v == 0 writes q/k state back.
//
// Non-spec (C1, T=1, one SSM slot) and spec (MTP verify, T=spec+1, a slot
// per token, conv line of length Width-1+(T-1)) are separate launches.
// Prefill is not this kernel.
//
// paged_decode is a short-grid GQA decode: one subgroup per (sequence,
// KV head, split) so the 8 query heads that share a KV head read KV once.
// splits==1 writes fp16. splits>1 writes partial max/sum/acc and a second
// launch combines them. The grid is fixed at capture; empty splits exit.

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/library.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace {

constexpr int kSg = 32;
constexpr int kWg = 1024;  // 32 subgroups x 4 V-rows, 16 state floats / thread
constexpr int kHeads = 32;
constexpr int kKHeads = 16;
constexpr int kDim = 128;
constexpr int kConv = 8192;
constexpr int kQkvz = 12288;
constexpr int kBa = 64;
constexpr int kAttnQ = 16;
constexpr int kAttnKv = 2;
constexpr int kAttnD = 256;
constexpr int kAttnWg = 32;  // one subgroup covers 256 dims, 8 elems / lane

constexpr int kSlmHq = 0;
constexpr int kSlmHk = 3 * kDim;
constexpr int kSlmHv = 6 * kDim;
constexpr int kSlmSq = 9 * kDim;
constexpr int kSlmSk = 10 * kDim;
constexpr int kSlmSv = 11 * kDim;
constexpr int kSlmSz = 12 * kDim;
constexpr int kSlmOut = 13 * kDim;
constexpr int kSlmGamma = 14 * kDim;
constexpr int kSlmParts = 15 * kDim;
constexpr int kSlmFloats = kSlmParts + kWg / kSg;

void record_use(const torch::Tensor& t) {
  if (!t.defined() || t.numel() == 0 || !t.is_xpu()) {
    return;
  }
  c10::xpu::XPUCachingAllocator::recordStream(
      t.storage().data_ptr(), c10::xpu::getCurrentXPUStream());
}

void record_all(std::initializer_list<torch::Tensor> ts) {
  for (const torch::Tensor& t : ts) {
    record_use(t);
  }
}

inline float h2f(sycl::half h) { return static_cast<float>(h); }

inline sycl::half f2h(float x) { return static_cast<sycl::half>(x); }

inline float softplus(float x) {
  if (x < 20.0f) {
    return sycl::log(1.0f + sycl::exp(x));
  }
  return x;
}

inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }

// Sum of 128 values held by threads 0..127. Other threads pass 0.
// Every thread in the group must call this. Returns the sum to every thread.
inline float reduce_128(sycl::nd_item<3> item, float value, float* parts) {
  auto sg = item.get_sub_group();
  float part = sycl::reduce_over_group(sg, value, sycl::plus<float>());
  if (sg.get_local_id() == 0) {
    parts[sg.get_group_id()] = part;
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (item.get_local_id(2) == 0) {
    float sum = 0.0f;
    // Threads 0..127 are the first four subgroups.
    for (int i = 0; i < 4; ++i) {
      sum += parts[i];
    }
    parts[0] = sum;
  }
  item.barrier(sycl::access::fence_space::local_space);
  return parts[0];
}

template <bool IsSpec>
void launch_decode(
    sycl::queue& queue,
    sycl::half* core,
    sycl::half* out,
    const sycl::half* qkvz,
    const sycl::half* ba,
    sycl::half* conv,
    float* ssm,
    const sycl::half* conv_w,
    const float* A_log,
    const sycl::half* dt_bias,
    const float* norm_w,
    const int* cu,
    const int* slot,
    const bool* has_initial,
    const int* token_indx,
    const int* num_accepted,
    int nseq,
    int nspec,
    int64_t conv_stride_slot,
    int64_t conv_stride_row,
    int64_t ssm_stride_slot,
    int64_t qkvz_stride,
    int64_t ba_stride,
    int64_t core_stride,
    int64_t out_stride,
    int slot_stride,
    float eps) {
  sycl::range<3> local(1, 1, kWg);
  sycl::range<3> global(static_cast<size_t>(nseq), kHeads, kWg);
  sycl::nd_range<3> nd(global, local);
  const float k_scale = 1.0f / sycl::sqrt(128.0f);

  queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> slm_acc(sycl::range<1>(kSlmFloats), h);
    h.parallel_for(nd, [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(kSg)]] {
      const int seq = static_cast<int>(item.get_group(0));
      const int head = static_cast<int>(item.get_group(1));
      const int tid = static_cast<int>(item.get_local_id(2));
      auto sg = item.get_sub_group();
      const int sg_id = static_cast<int>(sg.get_group_id());
      const int lane = static_cast<int>(sg.get_local_id());
      float* sm = slm_acc.get_pointer();

      const int hk = head >> 1;
      const int lv = head & 1;
      const bool active = IsSpec ? (slot[seq * slot_stride] >= 0) : (slot[seq] >= 0);

      int ntok = 0;
      int base = 0;
      if constexpr (IsSpec) {
        ntok = nspec;
      } else {
        base = cu[seq];
        ntok = cu[seq + 1] - base;
      }
      // Pad slots skip the loop. Barriers inside it are uniform per group.
      if (!active) {
        ntok = 0;
      }

      // 16 registers: 4 V-rows x 4 coalesced K-groups.
      float st[4][4];
      const int v0 = sg_id * 4;
      if (active && (IsSpec || (has_initial == nullptr || has_initial[seq]))) {
        int init_slot;
        if constexpr (IsSpec) {
          int col = num_accepted[seq] - 1;
          if (col < 0) {
            col = 0;
          }
          init_slot = slot[seq * slot_stride + col];
        } else {
          init_slot = slot[seq];
        }
        const float* sp = ssm + static_cast<int64_t>(init_slot) * ssm_stride_slot +
                          static_cast<int64_t>(head) * (kDim * kDim);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int k = i * kSg + lane;
            st[j][i] = sp[(v0 + j) * kDim + k];
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            st[j][i] = 0.0f;
          }
        }
      }

      // Private conv window. q/k are duplicated per value head on purpose.
      if (tid < kDim) {
        const int cq = hk * kDim + tid;
        const int ck = 2048 + hk * kDim + tid;
        const int cv = 4096 + head * kDim + tid;
        int init_row = 0;
        bool load_hist = active;
        if constexpr (IsSpec) {
          init_row = num_accepted[seq] - 1;
          if (init_row < 0) {
            init_row = 0;
          }
        } else {
          load_hist = load_hist && (has_initial == nullptr || has_initial[seq]);
        }
        if (load_hist) {
          int conv_slot;
          if constexpr (IsSpec) {
            conv_slot = slot[seq * slot_stride];
          } else {
            conv_slot = slot[seq];
          }
          const sycl::half* cp =
              conv + static_cast<int64_t>(conv_slot) * conv_stride_slot;
#pragma unroll
          for (int r = 0; r < 3; ++r) {
            const sycl::half* row =
                cp + static_cast<int64_t>(init_row + r) * conv_stride_row;
            sm[kSlmHq + r * kDim + tid] = h2f(row[cq]);
            sm[kSlmHk + r * kDim + tid] = h2f(row[ck]);
            sm[kSlmHv + r * kDim + tid] = h2f(row[cv]);
          }
        } else {
#pragma unroll
          for (int r = 0; r < 3; ++r) {
            sm[kSlmHq + r * kDim + tid] = 0.0f;
            sm[kSlmHk + r * kDim + tid] = 0.0f;
            sm[kSlmHv + r * kDim + tid] = 0.0f;
          }
        }
        if (norm_w != nullptr) {
          sm[kSlmGamma + tid] = norm_w[tid];
        } else {
          sm[kSlmGamma + tid] = 1.0f;
        }
      }
      item.barrier(sycl::access::fence_space::local_space);

      const float A_neg = -sycl::exp(A_log[head]);
      const float dt = h2f(dt_bias[head]);

      for (int t = 0; t < ntok; ++t) {
        int global_t = base + t;
        if constexpr (IsSpec) {
          global_t = token_indx[seq * nspec + t];
        }

        if (tid < kDim && active) {
          const int cq = hk * kDim + tid;
          const int ck = 2048 + hk * kDim + tid;
          const int cv = 4096 + head * kDim + tid;
          const int cz = 8192 + head * kDim + tid;
          const sycl::half* src = qkvz + static_cast<int64_t>(global_t) * qkvz_stride;

          auto conv_ch = [&](int hist_base, int channel, int src_off) {
            float h0 = sm[hist_base + tid];
            float h1 = sm[hist_base + kDim + tid];
            float h2 = sm[hist_base + 2 * kDim + tid];
            float x = h2f(src[src_off]);
            const sycl::half* w = conv_w + channel * 4;
            float acc = h0 * h2f(w[0]) + h1 * h2f(w[1]) + h2 * h2f(w[2]) + x * h2f(w[3]);
            sm[hist_base + tid] = h1;
            sm[hist_base + kDim + tid] = h2;
            sm[hist_base + 2 * kDim + tid] = x;
            return h2f(f2h(silu(acc)));
          };
          sm[kSlmSq + tid] = conv_ch(kSlmHq, cq, cq);
          sm[kSlmSk + tid] = conv_ch(kSlmHk, ck, ck);
          sm[kSlmSv + tid] = conv_ch(kSlmHv, cv, cv);
          sm[kSlmSz + tid] = h2f(src[cz]);
        }
        item.barrier(sycl::access::fence_space::local_space);

        float qv = (tid < kDim) ? sm[kSlmSq + tid] : 0.0f;
        float kvv = (tid < kDim) ? sm[kSlmSk + tid] : 0.0f;
        float qsum = reduce_128(item, qv * qv, sm + kSlmParts);
        float ksum = reduce_128(item, kvv * kvv, sm + kSlmParts);
        qsum += 1.0e-6f;
        ksum += 1.0e-6f;
        const float q_inv = sycl::rsqrt(qsum) * k_scale;
        const float k_inv = sycl::rsqrt(ksum);
        if (tid < kDim) {
          sm[kSlmSq + tid] *= q_inv;
          sm[kSlmSk + tid] *= k_inv;
        }
        item.barrier(sycl::access::fence_space::local_space);

        const sycl::half* bap = ba + static_cast<int64_t>(global_t) * ba_stride;
        const float beta = 1.0f / (1.0f + sycl::exp(-h2f(bap[head])));
        const float g = sycl::exp(A_neg * softplus(h2f(bap[32 + head]) + dt));

        float res[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float dot = 0.0f;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            st[j][i] *= g;
            const int k = i * kSg + lane;
            dot += st[j][i] * sm[kSlmSk + k];
          }
          dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
          const float vv = sm[kSlmSv + v0 + j];
          const float delta = (vv - dot) * beta;
          float acc = 0.0f;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int k = i * kSg + lane;
            st[j][i] += sm[kSlmSk + k] * delta;
            acc += st[j][i] * sm[kSlmSq + k];
          }
          res[j] = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        }
        if (lane == 0) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            sm[kSlmOut + v0 + j] = res[j];
          }
        }
        item.barrier(sycl::access::fence_space::local_space);

        float rounded = 0.0f;
        if (tid < kDim) {
          rounded = h2f(f2h(sm[kSlmOut + tid]));
        }
        float sumsq = reduce_128(item, rounded * rounded, sm + kSlmParts);
        const float inv = sycl::rsqrt(sumsq / 128.0f + eps);
        if (tid < kDim && active) {
          sycl::half* cp = core + static_cast<int64_t>(global_t) * core_stride +
                           static_cast<int64_t>(head) * kDim;
          sycl::half* op = out + static_cast<int64_t>(global_t) * out_stride +
                           static_cast<int64_t>(head) * kDim;
          const float x = h2f(f2h(sm[kSlmOut + tid]));
          cp[tid] = f2h(x);
          const float z = sm[kSlmSz + tid];
          const float y = x * inv * sm[kSlmGamma + tid] * silu(z);
          op[tid] = f2h(y);
        }
        item.barrier(sycl::access::fence_space::local_space);

        if constexpr (IsSpec) {
          if (active) {
            const int dst_slot = slot[seq * slot_stride + t];
            if (dst_slot >= 0) {
              float* dp = ssm + static_cast<int64_t>(dst_slot) * ssm_stride_slot +
                          static_cast<int64_t>(head) * (kDim * kDim);
#pragma unroll
              for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                  dp[(v0 + j) * kDim + (i * kSg + lane)] = st[j][i];
                }
              }
            }
          }
        }
      }

      if constexpr (!IsSpec) {
        if (active) {
          float* dp = ssm + static_cast<int64_t>(slot[seq]) * ssm_stride_slot +
                      static_cast<int64_t>(head) * (kDim * kDim);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              dp[(v0 + j) * kDim + (i * kSg + lane)] = st[j][i];
            }
          }
        }
      }

      // Conv state writeback.
      if (tid < kDim && active) {
        if constexpr (IsSpec) {
          const int conv_slot = slot[seq * slot_stride];
          sycl::half* cp = conv + static_cast<int64_t>(conv_slot) * conv_stride_slot;
          int init_row = num_accepted[seq] - 1;
          if (init_row < 0) {
            init_row = 0;
          }
          const int state_len = 3 + (nspec - 1);
          const int hist_rows = state_len - nspec;
          const int cq = hk * kDim + tid;
          const int ck = 2048 + hk * kDim + tid;
          const int cv = 4096 + head * kDim + tid;
          for (int j = 0; j < state_len; ++j) {
            sycl::half* dst = cp + static_cast<int64_t>(j) * conv_stride_row;
            if (j < hist_rows) {
              const sycl::half* src =
                  cp + static_cast<int64_t>(init_row + 1 + j) * conv_stride_row;
              if (lv == 0) {
                dst[cq] = src[cq];
                dst[ck] = src[ck];
              }
              dst[cv] = src[cv];
            } else {
              const int draft = j - hist_rows;
              const int gt = token_indx[seq * nspec + draft];
              const sycl::half* src = qkvz + static_cast<int64_t>(gt) * qkvz_stride;
              if (lv == 0) {
                dst[cq] = src[cq];
                dst[ck] = src[ck];
              }
              dst[cv] = src[cv];
            }
          }
        } else {
          sycl::half* cp =
              conv + static_cast<int64_t>(slot[seq]) * conv_stride_slot;
          const int cq = hk * kDim + tid;
          const int ck = 2048 + hk * kDim + tid;
          const int cv = 4096 + head * kDim + tid;
#pragma unroll
          for (int r = 0; r < 3; ++r) {
            sycl::half* row = cp + static_cast<int64_t>(r) * conv_stride_row;
            if (lv == 0) {
              row[cq] = f2h(sm[kSlmHq + r * kDim + tid]);
              row[ck] = f2h(sm[kSlmHk + r * kDim + tid]);
            }
            row[cv] = f2h(sm[kSlmHv + r * kDim + tid]);
          }
        }
      }
    });
  });
}

void decode_fused(
    torch::Tensor core,
    torch::Tensor out,
    torch::Tensor qkvz,
    torch::Tensor ba,
    torch::Tensor conv_state,
    torch::Tensor ssm_state,
    torch::Tensor conv_w,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor norm_w,
    torch::Tensor cu_seqlens,
    torch::Tensor slot,
    torch::Tensor has_initial,
    std::optional<torch::Tensor> token_indx,
    std::optional<torch::Tensor> num_accepted,
    double eps) {
  TORCH_CHECK(core.is_xpu() && out.is_xpu() && qkvz.is_xpu(), "xpu tensors");
  TORCH_CHECK(qkvz.scalar_type() == at::kHalf, "qkvz fp16");
  TORCH_CHECK(core.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf, "out fp16");
  TORCH_CHECK(qkvz.size(1) == kQkvz, "qkvz width");
  TORCH_CHECK(ba.size(1) == kBa, "ba width");
  TORCH_CHECK(conv_w.size(0) == kConv && conv_w.size(1) == 4, "conv_w [8192,4]");
  TORCH_CHECK(A_log.numel() == kHeads && A_log.scalar_type() == at::kFloat, "A_log");
  TORCH_CHECK(dt_bias.numel() == kHeads && dt_bias.scalar_type() == at::kHalf, "dt_bias fp16");
  TORCH_CHECK(norm_w.numel() == kDim, "norm_w");
  TORCH_CHECK(conv_state.size(2) == kConv, "conv channels");
  TORCH_CHECK(ssm_state.size(-3) == kHeads && ssm_state.size(-2) == kDim &&
                  ssm_state.size(-1) == kDim,
              "ssm [slot,32,128,128]");
  TORCH_CHECK(ssm_state.scalar_type() == at::kFloat, "ssm fp32");
  TORCH_CHECK(conv_state.scalar_type() == at::kHalf, "conv fp16");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu int32");
  TORCH_CHECK(slot.scalar_type() == at::kInt, "slot int32");
  TORCH_CHECK(core.is_contiguous() && out.is_contiguous() && qkvz.is_contiguous(), "contig");
  TORCH_CHECK(ba.is_contiguous() && conv_w.is_contiguous() && A_log.is_contiguous(), "contig w");
  // Mamba pages pack conv then ssm into one int8 page, then view them
  // back. Each state is a non-contiguous slot stride with a packed
  // inner tail. stride(0) is the page gap; the kernel already uses it.
  TORCH_CHECK(dt_bias.is_contiguous(), "contig dt");
  TORCH_CHECK(conv_state.stride(-1) == 1, "conv channel stride");
  TORCH_CHECK(conv_state.stride(-2) == conv_state.size(-1), "conv row packed");
  TORCH_CHECK(ssm_state.stride(-1) == 1, "ssm K stride");
  TORCH_CHECK(ssm_state.stride(-2) == ssm_state.size(-1), "ssm V packed");
  TORCH_CHECK(ssm_state.stride(-3) == ssm_state.size(-1) * ssm_state.size(-2),
              "ssm head packed");

  const bool is_spec = token_indx.has_value();
  if (is_spec) {
    TORCH_CHECK(num_accepted.has_value(), "spec needs num_accepted");
    TORCH_CHECK(slot.dim() == 2, "spec slot [B, T]");
    TORCH_CHECK(token_indx->scalar_type() == at::kInt, "token_indx int32");
    TORCH_CHECK(num_accepted->scalar_type() == at::kInt, "num_accepted int32");
    const int state_len = 3 + (static_cast<int>(slot.size(1)) - 1);
    TORCH_CHECK(conv_state.size(1) >= state_len, "conv rows ", conv_state.size(1),
                " < spec state_len ", state_len);
  } else {
    TORCH_CHECK(slot.dim() == 1, "non-spec slot [B]");
    TORCH_CHECK(conv_state.size(1) >= 3, "conv rows");
    TORCH_CHECK(cu_seqlens.size(0) == slot.size(0) + 1, "cu vs slot");
  }

  torch::Tensor norm_f = norm_w;
  if (norm_w.scalar_type() != at::kFloat) {
    // Graph replay must not allocate. Callers pass fp32, or a cached cast
    // made before capture. Refuse a cast here so a captured call cannot
    // hide a malloc.
    TORCH_CHECK(false, "norm_w must be fp32 (cast outside capture)");
  }

  const at::DeviceGuard guard(core.device());
  record_all({core, out, qkvz, ba, conv_state, ssm_state, conv_w, A_log, dt_bias, norm_f,
              cu_seqlens, slot, has_initial});
  if (is_spec) {
    record_use(*token_indx);
    record_use(*num_accepted);
  }

  const int nseq = static_cast<int>(slot.size(0));
  if (nseq == 0) {
    return;
  }
  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
  const int nspec = is_spec ? static_cast<int>(slot.size(1)) : 0;
  const int slot_stride = is_spec ? static_cast<int>(slot.stride(0)) : 1;

  auto go = [&](auto spec_tag) {
    launch_decode<decltype(spec_tag)::value>(
        queue,
        reinterpret_cast<sycl::half*>(core.data_ptr()),
        reinterpret_cast<sycl::half*>(out.data_ptr()),
        reinterpret_cast<const sycl::half*>(qkvz.data_ptr()),
        reinterpret_cast<const sycl::half*>(ba.data_ptr()),
        reinterpret_cast<sycl::half*>(conv_state.data_ptr()),
        reinterpret_cast<float*>(ssm_state.data_ptr()),
        reinterpret_cast<const sycl::half*>(conv_w.data_ptr()),
        reinterpret_cast<const float*>(A_log.data_ptr()),
        reinterpret_cast<const sycl::half*>(dt_bias.data_ptr()),
        reinterpret_cast<const float*>(norm_f.data_ptr()),
        reinterpret_cast<const int*>(cu_seqlens.data_ptr()),
        reinterpret_cast<const int*>(slot.data_ptr()),
        has_initial.defined() ? reinterpret_cast<const bool*>(has_initial.data_ptr()) : nullptr,
        is_spec ? reinterpret_cast<const int*>(token_indx->data_ptr()) : nullptr,
        is_spec ? reinterpret_cast<const int*>(num_accepted->data_ptr()) : nullptr,
        nseq,
        nspec,
        conv_state.stride(0),
        conv_state.stride(1),
        ssm_state.stride(0),
        qkvz.stride(0),
        ba.stride(0),
        core.stride(0),
        out.stride(0),
        slot_stride,
        static_cast<float>(eps));
  };
  if (is_spec) {
    go(std::true_type{});
  } else {
    go(std::false_type{});
  }
}

// ---------------------------------------------------------------------------
// Paged GQA decode. One subgroup per (sequence, KV head, split).
// ---------------------------------------------------------------------------

inline void load8(const sycl::half* base, int64_t stride_dim, int d0, float* dst) {
  if (stride_dim == 1) {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(base + d0);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      uint32_t bits = p[u];
      uint16_t lo = static_cast<uint16_t>(bits & 0xffffu);
      uint16_t hi = static_cast<uint16_t>(bits >> 16);
      sycl::half a;
      sycl::half b;
      __builtin_memcpy(&a, &lo, 2);
      __builtin_memcpy(&b, &hi, 2);
      dst[2 * u] = h2f(a);
      dst[2 * u + 1] = h2f(b);
    }
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      dst[e] = h2f(base[(d0 + e) * stride_dim]);
    }
  }
}


void launch_paged2(
    sycl::queue& queue,
    sycl::half* out,
    const sycl::half* q,
    const sycl::half* k_cache,
    const sycl::half* v_cache,
    const int* block_table,
    const int* seqused,
    const int* cu,
    float* partial_o,
    float* partial_m,
    float* partial_l,
    int nseq,
    int splits,
    int ntokens,
    int block_size,
    int64_t ksb,
    int64_t kst,
    int64_t ksh,
    int64_t ksd,
    int64_t vsb,
    int64_t vst,
    int64_t vsh,
    int64_t vsd,
    int64_t q_stride,
    int64_t out_stride,
    int64_t bt_stride,
    float scale) {
  sycl::range<3> local(1, 1, kAttnWg);
  sycl::range<3> global(static_cast<size_t>(nseq), kAttnKv, static_cast<size_t>(splits) * kAttnWg);
  sycl::nd_range<3> nd(global, local);
  const int64_t partial_split_stride = static_cast<int64_t>(ntokens) * kAttnQ * kAttnD;
  const int64_t ml_split_stride = static_cast<int64_t>(ntokens) * kAttnQ;

  queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> slm(sycl::range<1>(2 * 8 * kAttnD), h);
    h.parallel_for(nd, [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(kSg)]] {
      const int seq = static_cast<int>(item.get_group(0));
      const int kv = static_cast<int>(item.get_group(1));
      const int split = static_cast<int>(item.get_group(2));
      const int lane = static_cast<int>(item.get_local_id(2));
      const int d0 = lane * 8;
      float* qsl = slm.get_pointer();
      float* asl = qsl + 8 * kAttnD;
      auto sg = item.get_sub_group();

      const int S = seqused[seq];
      const int q0 = cu[seq];
      const int T = cu[seq + 1] - q0;
      const int chunk = S > 0 ? (S + splits - 1) / splits : 0;
      const int begin = split * chunk;

      for (int qi = 0; qi < T; ++qi) {
        const int nkeys = S - T + qi + 1;
        int end = begin + chunk;
        if (end > nkeys) {
          end = nkeys;
        }
        const bool live = begin < end;
        const int global_q = q0 + qi;
        const sycl::half* qp = q + static_cast<int64_t>(global_q) * q_stride +
                               static_cast<int64_t>(kv) * 8 * kAttnD;
        for (int hh = 0; hh < 8; ++hh) {
          load8(qp + static_cast<int64_t>(hh) * kAttnD, 1, d0, qsl + hh * kAttnD + d0);
        }
        item.barrier(sycl::access::fence_space::local_space);

        float mhead[8];
        float lhead[8];
#pragma unroll
        for (int hh = 0; hh < 8; ++hh) {
          mhead[hh] = -1.0e30f;
          lhead[hh] = 0.0f;
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            asl[hh * kAttnD + d0 + e] = 0.0f;
          }
        }

        if (live) {
          for (int pos = begin; pos < end; ++pos) {
            const int blk = pos / block_size;
            const int off = pos - blk * block_size;
            const int page = block_table[seq * bt_stride + blk];
            float kk[8];
            float vv[8];
            if (page >= 0) {
              const sycl::half* kp = k_cache + static_cast<int64_t>(page) * ksb +
                                     static_cast<int64_t>(off) * kst +
                                     static_cast<int64_t>(kv) * ksh;
              const sycl::half* vp = v_cache + static_cast<int64_t>(page) * vsb +
                                     static_cast<int64_t>(off) * vst +
                                     static_cast<int64_t>(kv) * vsh;
              load8(kp, ksd, d0, kk);
              load8(vp, vsd, d0, vv);
            } else {
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                kk[e] = 0.0f;
                vv[e] = 0.0f;
              }
            }
            for (int hh = 0; hh < 8; ++hh) {
              float partial = 0.0f;
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                partial += qsl[hh * kAttnD + d0 + e] * kk[e];
              }
              partial = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
              const float score = partial * scale;
              const float m_new = sycl::fmax(mhead[hh], score);
              const float alpha = sycl::exp(mhead[hh] - m_new);
              const float prob = sycl::exp(score - m_new);
              mhead[hh] = m_new;
              lhead[hh] = lhead[hh] * alpha + prob;
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                float* ap = asl + hh * kAttnD + d0 + e;
                *ap = (*ap) * alpha + prob * vv[e];
              }
            }
          }
        }

        if (splits == 1) {
          sycl::half* op = out + static_cast<int64_t>(global_q) * out_stride +
                           static_cast<int64_t>(kv) * 8 * kAttnD;
          for (int hh = 0; hh < 8; ++hh) {
            const float inv_l = (lhead[hh] > 0.0f) ? (1.0f / lhead[hh]) : 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              op[hh * kAttnD + d0 + e] = f2h(asl[hh * kAttnD + d0 + e] * inv_l);
            }
          }
        } else {
          for (int hh = 0; hh < 8; ++hh) {
            const int qh = kv * 8 + hh;
            float* dst = partial_o + static_cast<int64_t>(split) * partial_split_stride +
                         static_cast<int64_t>(global_q) * (kAttnQ * kAttnD) +
                         static_cast<int64_t>(qh) * kAttnD;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              dst[d0 + e] = asl[hh * kAttnD + d0 + e];
            }
            if (lane == 0) {
              partial_m[static_cast<int64_t>(split) * ml_split_stride +
                        static_cast<int64_t>(global_q) * kAttnQ + qh] = mhead[hh];
              partial_l[static_cast<int64_t>(split) * ml_split_stride +
                        static_cast<int64_t>(global_q) * kAttnQ + qh] = lhead[hh];
            }
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
    });
  });

  if (splits == 1 || partial_o == nullptr) {
    return;
  }

  // Combine splits. One subgroup per (token, query head).
  const int groups = ntokens * kAttnQ;
  if (groups <= 0) {
    return;
  }
  sycl::range<1> local1(kAttnWg);
  sycl::range<1> global1(static_cast<size_t>(groups) * kAttnWg);
  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(global1, local1), [=](sycl::nd_item<1> item)
                                                           [[sycl::reqd_sub_group_size(kSg)]] {
      const int group = static_cast<int>(item.get_group(0));
      const int lane = static_cast<int>(item.get_local_id(0));
      const int token = group / kAttnQ;
      const int qh = group - token * kAttnQ;
      const int d0 = lane * 8;
      float m = -1.0e30f;
      for (int s = 0; s < splits; ++s) {
        float ls = partial_l[static_cast<int64_t>(s) * ml_split_stride +
                             static_cast<int64_t>(token) * kAttnQ + qh];
        if (ls <= 0.0f) {
          continue;
        }
        float ms = partial_m[static_cast<int64_t>(s) * ml_split_stride +
                             static_cast<int64_t>(token) * kAttnQ + qh];
        m = sycl::fmax(m, ms);
      }
      float l = 0.0f;
      float acc[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        acc[e] = 0.0f;
      }
      for (int s = 0; s < splits; ++s) {
        float ls = partial_l[static_cast<int64_t>(s) * ml_split_stride +
                             static_cast<int64_t>(token) * kAttnQ + qh];
        if (ls <= 0.0f) {
          continue;
        }
        float ms = partial_m[static_cast<int64_t>(s) * ml_split_stride +
                             static_cast<int64_t>(token) * kAttnQ + qh];
        const float w = sycl::exp(ms - m);
        l += ls * w;
        const float* src = partial_o + static_cast<int64_t>(s) * partial_split_stride +
                           static_cast<int64_t>(token) * (kAttnQ * kAttnD) +
                           static_cast<int64_t>(qh) * kAttnD;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          acc[e] += src[d0 + e] * w;
        }
      }
      const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
      sycl::half* op = out + static_cast<int64_t>(token) * out_stride +
                       static_cast<int64_t>(qh) * kAttnD;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        op[d0 + e] = f2h(acc[e] * inv_l);
      }
    });
  });
}

void paged_decode(
    torch::Tensor out,
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seqused,
    torch::Tensor cu_seqlens,
    double scale,
    int64_t splits,
    std::optional<torch::Tensor> partial_o,
    std::optional<torch::Tensor> partial_m,
    std::optional<torch::Tensor> partial_l) {
  TORCH_CHECK(q.is_xpu() && out.is_xpu() && k_cache.is_xpu(), "xpu");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k_cache.scalar_type() == at::kHalf, "fp16");
  TORCH_CHECK(q.size(-1) == kAttnD && q.size(-2) == kAttnQ, "q [T,16,256]");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(2) == kAttnKv && k_cache.size(3) == kAttnD,
              "k [blocks, block, 2, 256]");
  TORCH_CHECK(v_cache.dim() == 4 && v_cache.size(0) == k_cache.size(0) &&
                  v_cache.size(1) == k_cache.size(1) && v_cache.size(2) == k_cache.size(2) &&
                  v_cache.size(3) == k_cache.size(3),
              "v shape");
  TORCH_CHECK(out.sizes() == q.sizes(), "out shape");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && seqused.scalar_type() == at::kInt, "int32");
  TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table int32");
  TORCH_CHECK(splits >= 1 && splits <= 128, "splits");
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous(), "q/out contig");
  const int nseq = static_cast<int>(seqused.size(0));
  TORCH_CHECK(cu_seqlens.size(0) == nseq + 1, "cu");
  TORCH_CHECK(block_table.size(0) == nseq, "block_table batch");
  const int ntokens = static_cast<int>(q.size(0));
  if (splits > 1) {
    TORCH_CHECK(partial_o.has_value() && partial_m.has_value() && partial_l.has_value(),
                "partials");
    TORCH_CHECK(partial_o->scalar_type() == at::kFloat, "partial fp32");
    TORCH_CHECK(partial_o->size(0) == splits && partial_o->size(1) == ntokens, "partial shape");
    TORCH_CHECK(partial_o->is_contiguous() && partial_m->is_contiguous() &&
                    partial_l->is_contiguous(),
                "partial contig");
  }
  if (nseq == 0 || ntokens == 0) {
    return;
  }
  const at::DeviceGuard guard(q.device());
  record_all({out, q, k_cache, v_cache, block_table, seqused, cu_seqlens});
  float* po = nullptr;
  float* pm = nullptr;
  float* pl = nullptr;
  if (splits > 1) {
    record_use(*partial_o);
    record_use(*partial_m);
    record_use(*partial_l);
    po = reinterpret_cast<float*>(partial_o->data_ptr());
    pm = reinterpret_cast<float*>(partial_m->data_ptr());
    pl = reinterpret_cast<float*>(partial_l->data_ptr());
  }
  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
  launch_paged2(
      queue,
      reinterpret_cast<sycl::half*>(out.data_ptr()),
      reinterpret_cast<const sycl::half*>(q.data_ptr()),
      reinterpret_cast<const sycl::half*>(k_cache.data_ptr()),
      reinterpret_cast<const sycl::half*>(v_cache.data_ptr()),
      reinterpret_cast<const int*>(block_table.data_ptr()),
      reinterpret_cast<const int*>(seqused.data_ptr()),
      reinterpret_cast<const int*>(cu_seqlens.data_ptr()),
      po,
      pm,
      pl,
      nseq,
      static_cast<int>(splits),
      ntokens,
      static_cast<int>(k_cache.size(1)),
      k_cache.stride(0),
      k_cache.stride(1),
      k_cache.stride(2),
      k_cache.stride(3),
      v_cache.stride(0),
      v_cache.stride(1),
      v_cache.stride(2),
      v_cache.stride(3),
      q.stride(0),
      out.stride(0),
      block_table.stride(0),
      static_cast<float>(scale));
}

}  // namespace

TORCH_LIBRARY(b70_gdn, m) {
  m.def(
      "decode_fused(Tensor core, Tensor out, Tensor qkvz, Tensor ba, "
      "Tensor conv_state, Tensor ssm_state, Tensor conv_w, Tensor A_log, "
      "Tensor dt_bias, Tensor norm_w, Tensor cu_seqlens, Tensor slot, "
      "Tensor has_initial, Tensor? token_indx, Tensor? num_accepted, "
      "float eps) -> ()");
  m.def(
      "paged_decode(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, "
      "Tensor block_table, Tensor seqused, Tensor cu_seqlens, float scale, "
      "int splits, Tensor? partial_o, Tensor? partial_m, Tensor? partial_l) -> ()");
}

TORCH_LIBRARY_IMPL(b70_gdn, XPU, m) {
  m.impl("decode_fused", &decode_fused);
  m.impl("paged_decode", &paged_decode);
}
