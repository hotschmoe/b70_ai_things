// QSA decode kernels for Intel B70 (BMG G31). SIMT on purpose.
//
// The formulas follow Strata night-20261016 (src-ep). This is an attribution,
// not a copy of that source:
//   include/strata/kernels/qsa.hpp
//       QsaStep, indexer state, index, top-k, attend, gate
//   src/kernels/cuda/qsa.cu
//       indexer_key_append_kernel (double RMS, spare at angle 0, in-kernel
//       RoPE), qsa_index_kernel (+1e9 on the open block), qsa_attend_kernel,
//       qsa_gate_apply_kernel (sigmoid in double, one fp16 store)
//   src/kernels/cuda/kv_q8.cu
//       group-64 int8, quantize against the stored fp16 scale, clamp
//       [-127, 127], zero group stores scale 0
//   include/strata/kernels/qsa_select.hpp
//       the open block scores `dead`, not whatever pooled[n_bid] holds
//   include/strata/kernels/mrope.hpp
//       pair % 3; a null table is 1D
//   include/strata/kernels/rope.hpp
//       rope_neox_pair: oa = a*c - b*s, ob = a*s + b*c
//
// The CPU reference these kernels must match is
// vllm/flashnext/flashnext_ref/qsa_layer.py.
//
// Launch grids are capacities (max blocks, max ids, head count). Every
// per-token count is read from the device step[4] int32
// (pos, n_kv, n_bid, width). Surplus groups return before any barrier.
// Host scalars are geometry and pos_base, which do not change per token.
//
// Not in this library: sycl/.../qsa_prompt_attn_union.dp.cpp (joint_matrix
// 8x16x16, fixed head dim 256). That is the fast prefill path.

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/library.h>
#include <torch/torch.h>

#include <bit>
#include <cstdint>

namespace {

constexpr int kGroup = 64;
constexpr int kTopkThreads = 256;
// Attend keeps one logit per selected cell in SLM. Flash-Next width is 2051.
constexpr int kMaxAttendIds = 4096;

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

int div_up_int(int n, int d) {
  return (n + d - 1) / d;
}

uint32_t order_f32(float score) {
  // qsa_layer._ordered_keys: sign bit selects the xor mask. Not Strata's
  // extra +0 (which folds -0 into +0) and not its NaN-to-zero map.
  uint32_t bits = std::bit_cast<uint32_t>(score);
  uint32_t mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
  return bits ^ mask;
}

uint64_t pack_key(float score, uint32_t index) {
  uint32_t inv = 0xFFFFFFFFu ^ index;
  return (static_cast<uint64_t>(order_f32(score)) << 32) | static_cast<uint64_t>(inv);
}

// fp32 -> fp16 -> fp32, round to nearest even. sycl::half(float) then
// float(...) was compiled back into the original float, so the stored
// scale skipped the round trip Strata quantizes against (kv_q8.cu
// f16_from_f32 / f32_from_f16). numpy float16 matches these bits.
uint16_t f32_to_f16(float value) {
  uint32_t x = std::bit_cast<uint32_t>(value);
  uint32_t xs = x & 0x80000000u;
  uint32_t xe = x & 0x7f800000u;
  uint32_t xm = x & 0x007fffffu;
  uint16_t hs = static_cast<uint16_t>(xs >> 16);
  if (xe == 0x7f800000u) {
    uint16_t hm = (xm == 0) ? 0 : static_cast<uint16_t>((xm >> 13) | 0x200u);
    return static_cast<uint16_t>(hs | 0x7c00u | hm);
  }
  int exp = static_cast<int>(xe >> 23) - 127 + 15;
  if (exp <= 0) {
    if (exp < -10) {
      return hs;
    }
    xm |= 0x00800000u;
    int shift = 1 - exp;
    uint32_t m = xm >> (shift + 13);
    uint32_t rem = xm & ((1u << (shift + 13)) - 1u);
    uint32_t halfway = 1u << (shift + 12);
    if (rem > halfway || (rem == halfway && (m & 1u))) {
      m += 1u;
    }
    return static_cast<uint16_t>(hs | static_cast<uint16_t>(m));
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(hs | 0x7c00u);
  }
  uint32_t m = xm >> 13;
  uint32_t rem = xm & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
    m += 1u;
    if (m == 0x400u) {
      m = 0;
      exp += 1;
      if (exp >= 31) {
        return static_cast<uint16_t>(hs | 0x7c00u);
      }
    }
  }
  return static_cast<uint16_t>(hs | static_cast<uint16_t>(exp << 10) | static_cast<uint16_t>(m));
}

float f16_to_f32(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp -= 1;
      }
      mant &= 0x3ffu;
      out = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
  }
  return std::bit_cast<float>(out);
}

sycl::queue& stream_queue(const torch::Tensor& t) {
  const at::DeviceGuard guard(t.device());
  return c10::xpu::getCurrentXPUStream().queue();
}

// ---------------------------------------------------------------------------
// write_step. One thread. qsa.hpp QsaStep: pos, n_kv=pos+1, n_bid, width.
// ---------------------------------------------------------------------------

torch::Tensor write_step(
    const torch::Tensor& pos,
    const torch::Tensor& step,
    int64_t top_k,
    int64_t block) {
  TORCH_CHECK(pos.is_xpu() && pos.is_contiguous(), "pos");
  TORCH_CHECK(pos.scalar_type() == torch::kInt32 && pos.numel() >= 1, "pos i32[1]");
  TORCH_CHECK(step.is_xpu() && step.is_contiguous(), "step");
  TORCH_CHECK(step.scalar_type() == torch::kInt32 && step.numel() >= 4, "step i32[4]");
  TORCH_CHECK(top_k >= 1 && block >= 1, "top_k and block");
  const at::DeviceGuard guard(pos.device());
  record_all({pos, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  int32_t* step_p = step.data_ptr<int32_t>();
  const int32_t* pos_p = pos.data_ptr<int32_t>();
  const int topk = static_cast<int>(top_k);
  const int r = static_cast<int>(block);
  q.single_task([=]() {
    int p = pos_p[0];
    if (p < 0) {
      p = 0;
    }
    int n_kv = p + 1;
    int n_bid = n_kv / r;
    int cap = topk + r - 1;
    int width = n_kv < cap ? n_kv : cap;
    step_p[0] = p;
    step_p[1] = n_kv;
    step_p[2] = n_bid;
    step_p[3] = width;
  });
  return step;
}

// ---------------------------------------------------------------------------
// kv_append. kv_q8.cu: one group of 64 per (K or V, head, group of 64).
// scale = fp16(amax/127) stored back as f32. code = rint(x/scale).
// ---------------------------------------------------------------------------

torch::Tensor kv_append(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& k_codes,
    const torch::Tensor& v_codes,
    const torch::Tensor& k_scale,
    const torch::Tensor& v_scale,
    const torch::Tensor& page_table,
    const torch::Tensor& step) {
  TORCH_CHECK(k.is_xpu() && k.is_contiguous() && k.scalar_type() == torch::kFloat, "k");
  TORCH_CHECK(v.is_xpu() && v.is_contiguous() && v.scalar_type() == torch::kFloat, "v");
  TORCH_CHECK(k.dim() == 2 && v.sizes() == k.sizes(), "k, v [kv, dim]");
  const int n_kv = static_cast<int>(k.size(0));
  const int head_dim = static_cast<int>(k.size(1));
  TORCH_CHECK(head_dim % kGroup == 0 && n_kv >= 1, "head_dim multiple of 64");
  const int groups = head_dim / kGroup;
  auto codes_ok = [&](const torch::Tensor& c, const char* name) {
    TORCH_CHECK(
        c.is_xpu() && c.is_contiguous() && c.scalar_type() == torch::kInt8 && c.dim() == 4,
        name);
    TORCH_CHECK(c.size(1) == n_kv && c.size(3) == head_dim, name);
  };
  codes_ok(k_codes, "k_codes");
  codes_ok(v_codes, "v_codes");
  TORCH_CHECK(k_codes.sizes() == v_codes.sizes(), "code shapes");
  const int page_size = static_cast<int>(k_codes.size(2));
  const int pages = static_cast<int>(k_codes.size(0));
  auto scale_ok = [&](const torch::Tensor& s, const char* name) {
    TORCH_CHECK(
        s.is_xpu() && s.is_contiguous() && s.scalar_type() == torch::kFloat && s.dim() == 4,
        name);
    TORCH_CHECK(
        s.size(0) == pages && s.size(1) == n_kv && s.size(2) == page_size && s.size(3) == groups,
        name);
  };
  scale_ok(k_scale, "k_scale");
  scale_ok(v_scale, "v_scale");
  TORCH_CHECK(
      page_table.is_xpu() && page_table.is_contiguous() && page_table.scalar_type() == torch::kInt32,
      "page_table");
  TORCH_CHECK(page_table.numel() >= pages, "page_table");
  TORCH_CHECK(step.is_xpu() && step.is_contiguous() && step.scalar_type() == torch::kInt32, "step");
  const at::DeviceGuard guard(k.device());
  record_all({k, v, k_codes, v_codes, k_scale, v_scale, page_table, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();

  const float* kp = k.data_ptr<float>();
  const float* vp = v.data_ptr<float>();
  int8_t* kc = k_codes.data_ptr<int8_t>();
  int8_t* vc = v_codes.data_ptr<int8_t>();
  float* ks = k_scale.data_ptr<float>();
  float* vs = v_scale.data_ptr<float>();
  const int32_t* table = page_table.data_ptr<int32_t>();
  const int32_t* step_p = step.data_ptr<int32_t>();

  // Grid is the head/group geometry, not the token. pos comes from step.
  auto nd = sycl::nd_range<3>(
      sycl::range<3>(2, static_cast<size_t>(n_kv), static_cast<size_t>(groups) * kGroup),
      sycl::range<3>(1, 1, kGroup));
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> red(sycl::range<1>(kGroup), h);
    h.parallel_for(nd, [=](sycl::nd_item<3> it) {
      const int is_v = static_cast<int>(it.get_group(0));
      const int head = static_cast<int>(it.get_group(1));
      const int g = static_cast<int>(it.get_group(2));
      const int t = static_cast<int>(it.get_local_id(2));
      const int pos = step_p[0];
      const float* src = is_v ? vp : kp;
      const float x = src[static_cast<long long>(head) * head_dim + g * kGroup + t];
      red[t] = sycl::fabs(x);
      it.barrier(sycl::access::fence_space::local_space);
      for (int s = kGroup / 2; s > 0; s >>= 1) {
        if (t < s) {
          red[t] = sycl::fmax(red[t], red[t + s]);
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      const float amax = red[0];
      // Quantize against the stored fp16 scale (kv_q8.cu). The half
      // round trip is explicit so the compiler cannot drop it.
      const float sf = f16_to_f32(f32_to_f16(amax / 127.0f));
      int qv = 0;
      if (sf > 0.0f) {
        float r = sycl::rint(x / sf);
        if (r > 127.0f) {
          r = 127.0f;
        }
        if (r < -127.0f) {
          r = -127.0f;
        }
        qv = static_cast<int>(r);
      }
      const int page_i = pos / page_size;
      const int phys = table[page_i];
      const int slot = pos - page_i * page_size;
      const long long row =
          (static_cast<long long>(phys) * n_kv + head) * page_size + slot;
      int8_t* dst = is_v ? vc : kc;
      dst[row * head_dim + g * kGroup + t] = static_cast<int8_t>(qv);
      if (t == 0) {
        float* sp = is_v ? vs : ks;
        sp[row * groups + g] = sf;
      }
    });
  });
  return k_codes;
}

// ---------------------------------------------------------------------------
// indexer_append. qsa.cu indexer_key_append_kernel.
// One work-group, one thread per indexer dim. Barriers are uniform: pos and
// slot are the same on every thread.
// ---------------------------------------------------------------------------

torch::Tensor indexer_append(
    const torch::Tensor& raw,
    const torch::Tensor& k_norm,
    const torch::Tensor& tail,
    const torch::Tensor& dead,
    const torch::Tensor& pooled,
    const torch::Tensor& block_pos,
    const torch::Tensor& cos_tab,
    const torch::Tensor& sin_tab,
    const torch::Tensor& step,
    const torch::Tensor& mrope,
    int64_t pos_base,
    double eps) {
  TORCH_CHECK(raw.is_xpu() && raw.is_contiguous() && raw.scalar_type() == torch::kFloat, "raw");
  const int dim = static_cast<int>(raw.numel());
  TORCH_CHECK(dim >= 1 && dim <= 256, "idx_dim 1..256");
  TORCH_CHECK(
      k_norm.is_xpu() && k_norm.is_contiguous() && k_norm.scalar_type() == torch::kFloat &&
          k_norm.numel() == dim,
      "k_norm");
  TORCH_CHECK(tail.is_xpu() && tail.is_contiguous() && tail.scalar_type() == torch::kFloat, "tail");
  TORCH_CHECK(tail.dim() == 2 && tail.size(1) == dim, "tail [block-1, dim]");
  const int r = static_cast<int>(tail.size(0)) + 1;
  TORCH_CHECK(r >= 2, "idx_block >= 2");
  TORCH_CHECK(
      dead.is_xpu() && dead.is_contiguous() && dead.scalar_type() == torch::kFloat && dead.numel() == dim,
      "dead");
  TORCH_CHECK(
      pooled.is_xpu() && pooled.is_contiguous() && pooled.scalar_type() == torch::kFloat &&
          pooled.dim() == 2 && pooled.size(1) == dim,
      "pooled");
  TORCH_CHECK(
      block_pos.is_xpu() && block_pos.is_contiguous() && block_pos.scalar_type() == torch::kInt32 &&
          block_pos.numel() >= 1,
      "block_pos");
  TORCH_CHECK(cos_tab.is_xpu() && cos_tab.is_contiguous() && cos_tab.scalar_type() == torch::kFloat, "cos");
  TORCH_CHECK(sin_tab.is_xpu() && sin_tab.is_contiguous() && sin_tab.sizes() == cos_tab.sizes(), "sin");
  TORCH_CHECK(cos_tab.dim() == 2, "cos [pos, half]");
  const int half = static_cast<int>(cos_tab.size(1));
  TORCH_CHECK(half == 0 || (half * 2 <= dim), "rotary dim fits the indexer dim");
  TORCH_CHECK(step.is_xpu() && step.is_contiguous() && step.scalar_type() == torch::kInt32, "step");
  TORCH_CHECK(mrope.is_xpu() && mrope.is_contiguous() && mrope.scalar_type() == torch::kInt32, "mrope");
  const bool have_mrope = mrope.numel() > 0;
  if (have_mrope) {
    TORCH_CHECK(mrope.dim() == 2 && mrope.size(1) == 3, "mrope [cells, 3]");
  }
  const at::DeviceGuard guard(raw.device());
  record_all({raw, k_norm, tail, dead, pooled, block_pos, cos_tab, sin_tab, step, mrope});
  auto& q = c10::xpu::getCurrentXPUStream().queue();

  const float* raw_p = raw.data_ptr<float>();
  const float* norm_p = k_norm.data_ptr<float>();
  float* tail_p = tail.data_ptr<float>();
  float* dead_p = dead.data_ptr<float>();
  float* pooled_p = pooled.data_ptr<float>();
  int32_t* bpos_p = block_pos.data_ptr<int32_t>();
  const float* cos_p = cos_tab.data_ptr<float>();
  const float* sin_p = sin_tab.data_ptr<float>();
  const int32_t* step_p = step.data_ptr<int32_t>();
  const int32_t* mtab = have_mrope ? mrope.data_ptr<int32_t>() : nullptr;
  const int base = static_cast<int>(pos_base);
  const double eps_d = eps;

  auto nd = sycl::nd_range<1>(static_cast<size_t>(dim), static_cast<size_t>(dim));
  q.submit([&](sycl::handler& h) {
    // Double mean, one element per dim. qsa.cu keeps this in shared memory
    // so the sum of squares sees the unrounded mean.
    sycl::local_accessor<double, 1> mean(sycl::range<1>(static_cast<size_t>(dim)), h);
    h.parallel_for(nd, [=](sycl::nd_item<1> it) {
      const int d = static_cast<int>(it.get_local_id(0));
      const int pos = step_p[0];
      const int slot = pos % r;

      // Completing cell is not stored. The r-1 tail rows are the open block.
      if (slot < r - 1) {
        tail_p[static_cast<long long>(slot) * dim + d] = raw_p[d];
      }

      // Spare key: rms(raw of cell 0), rotated at angle 0. qsa.cu multiplies
      // by cos row 0 because sin(0) is 0, so there is no partner to read.
      if (pos == 0) {
        const double p = static_cast<double>(raw_p[d]);
        double ss = 0.0;
        for (int i = 0; i < dim; ++i) {
          const double v = static_cast<double>(raw_p[i]);
          ss = ss + v * v;
        }
        const double inv = 1.0 / sycl::sqrt(ss / static_cast<double>(dim) + eps_d);
        float y = static_cast<float>(p * inv * static_cast<double>(norm_p[d]));
        if (half > 0 && d < half * 2) {
          y = y * cos_p[d % half];
        }
        dead_p[d] = y;
        pooled_p[d] = y;
      }

      // slot == r-1 is uniform across the work-group. The return skips the
      // barriers below, and every thread takes it or none does.
      if (slot != r - 1) {
        return;
      }

      double m = 0.0;
      for (int j = 0; j < r - 1; ++j) {
        m = m + static_cast<double>(tail_p[static_cast<long long>(j) * dim + d]);
      }
      m = m + static_cast<double>(raw_p[d]);
      m = m / static_cast<double>(r);
      mean[d] = m;
      it.barrier(sycl::access::fence_space::local_space);

      double ss = 0.0;
      for (int i = 0; i < dim; ++i) {
        const double v = mean[i];
        ss = ss + v * v;
      }
      const double inv = 1.0 / sycl::sqrt(ss / static_cast<double>(dim) + eps_d);
      const int b = pos / r;
      float* row = pooled_p + static_cast<long long>(b) * dim;
      row[d] = static_cast<float>(m * inv * static_cast<double>(norm_p[d]));
      pooled_p[static_cast<long long>(b + 1) * dim + d] = dead_p[d];
      if (d == 0) {
        bpos_p[0] = base + b * r;
      }
      it.barrier(sycl::access::fence_space::global_and_local);

      // In-place NeoX on the completed row. Read both partners, then write.
      // Thread d owns pair d. The halves are disjoint.
      if (half > 0 && d < half) {
        int rot = base + b * r;
        if (mtab != nullptr) {
          rot = mtab[static_cast<long long>(rot) * 3 + (d % 3)];
        }
        const long long toff = static_cast<long long>(rot) * half + d;
        const float a = row[d];
        const float bp = row[half + d];
        const float c = cos_p[toff];
        const float s = sin_p[toff];
        row[d] = a * c - bp * s;
        row[half + d] = a * s + bp * c;
      }
    });
  });
  return pooled;
}

// ---------------------------------------------------------------------------
// index_scores. One thread per pooled row, grid sized to max blocks.
// Open row reads `dead`. Tail cells (n_kv not a multiple of the block) get
// +1e9. Dots are fp32, per head, then ReLU, then the sum. No 1/sqrt(d).
// `block` is geometry, a host int. The grid is max_blocks.
// ---------------------------------------------------------------------------

torch::Tensor index_scores_impl(
    const torch::Tensor& q_idx,
    const torch::Tensor& pooled,
    const torch::Tensor& dead,
    const torch::Tensor& scores,
    const torch::Tensor& step,
    const torch::Tensor& bias,
    int64_t block) {
  TORCH_CHECK(q_idx.is_xpu() && q_idx.is_contiguous() && q_idx.scalar_type() == torch::kFloat, "q_idx");
  TORCH_CHECK(q_idx.dim() == 2, "q_idx [heads, dim]");
  const int n_head = static_cast<int>(q_idx.size(0));
  const int dim = static_cast<int>(q_idx.size(1));
  TORCH_CHECK(n_head >= 1 && dim >= 1, "q_idx");
  TORCH_CHECK(
      pooled.is_xpu() && pooled.is_contiguous() && pooled.scalar_type() == torch::kFloat &&
          pooled.dim() == 2 && pooled.size(1) == dim,
      "pooled");
  TORCH_CHECK(
      dead.is_xpu() && dead.is_contiguous() && dead.scalar_type() == torch::kFloat && dead.numel() == dim,
      "dead");
  TORCH_CHECK(scores.is_xpu() && scores.is_contiguous() && scores.scalar_type() == torch::kFloat, "scores");
  TORCH_CHECK(step.is_xpu() && step.is_contiguous() && step.scalar_type() == torch::kInt32, "step");
  TORCH_CHECK(bias.is_xpu() && bias.is_contiguous() && bias.scalar_type() == torch::kFloat, "bias");
  TORCH_CHECK(block >= 2, "block");
  const int max_blocks = static_cast<int>(pooled.size(0));
  const int scores_n = static_cast<int>(scores.numel());
  const bool have_bias = bias.numel() > 0;
  const at::DeviceGuard guard(q_idx.device());
  record_all({q_idx, pooled, dead, scores, step, bias});
  auto& q = c10::xpu::getCurrentXPUStream().queue();

  const float* qp = q_idx.data_ptr<float>();
  const float* pooled_p = pooled.data_ptr<float>();
  const float* dead_p = dead.data_ptr<float>();
  float* scores_p = scores.data_ptr<float>();
  const int32_t* step_p = step.data_ptr<int32_t>();
  const float* bias_p = have_bias ? bias.data_ptr<float>() : nullptr;
  const int r = static_cast<int>(block);

  q.parallel_for(sycl::range<1>(static_cast<size_t>(max_blocks)), [=](sycl::id<1> id) {
    const long long b = static_cast<long long>(id[0]);
    const int n_bid = step_p[2];
    const int n_kv = step_p[1];
    // Constant grid. Rows past the spare do not exist for this token.
    if (b > n_bid) {
      return;
    }
    const float* key = (b == n_bid) ? dead_p : (pooled_p + b * dim);
    float score = 0.0f;
    for (int h = 0; h < n_head; ++h) {
      const float* qh = qp + static_cast<long long>(h) * dim;
      float dot = 0.0f;
      for (int i = 0; i < dim; ++i) {
        dot = dot + key[i] * qh[i];
      }
      if (dot > 0.0f) {
        score = score + dot;
      }
    }
    if (bias_p != nullptr) {
      score = score + bias_p[b];
    }
    long long lo = b * r;
    long long hi = lo + r;
    if (b == n_bid) {
      hi = n_kv;
    }
    if (hi > n_kv) {
      hi = n_kv;
    }
    if (b == n_bid && (n_kv % r) != 0) {
      score = score + 1.0e9f;
    }
    for (long long j = lo; j < hi; ++j) {
      if (j < scores_n) {
        scores_p[j] = score;
      }
    }
  });
  return scores;
}

// ---------------------------------------------------------------------------
// topk. 64-bit radix select, one work-group of 256.
// Key = ordered_f32 << 32 | (~index). Keeping key >= marginal and scanning
// ascending emits the same ids as np.lexsort((arange, -scores)).
// n_kv is the device step, so the launch does not change with the prompt.
// Strata's own topk refuses above 32768 (one block). This scan is the same
// one-group algorithm and is correct past that; the attend SLM cap is the
// limit that actually binds (kMaxAttendIds).
// ---------------------------------------------------------------------------

torch::Tensor topk(
    const torch::Tensor& scores,
    const torch::Tensor& ids,
    const torch::Tensor& step) {
  TORCH_CHECK(scores.is_xpu() && scores.is_contiguous() && scores.scalar_type() == torch::kFloat, "scores");
  TORCH_CHECK(ids.is_xpu() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32, "ids");
  TORCH_CHECK(step.is_xpu() && step.is_contiguous() && step.scalar_type() == torch::kInt32, "step");
  const int scores_n = static_cast<int>(scores.numel());
  const int ids_n = static_cast<int>(ids.numel());
  const at::DeviceGuard guard(scores.device());
  record_all({scores, ids, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const float* scores_p = scores.data_ptr<float>();
  int32_t* ids_p = ids.data_ptr<int32_t>();
  const int32_t* step_p = step.data_ptr<int32_t>();

  auto nd = sycl::nd_range<1>(kTopkThreads, kTopkThreads);
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<long long, 1> red(sycl::range<1>(kTopkThreads), h);
    h.parallel_for(nd, [=](sycl::nd_item<1> it) {
      const int t = static_cast<int>(it.get_local_id(0));
      int n = step_p[1];
      int width = step_p[3];
      if (n > scores_n) {
        n = scores_n;
      }
      if (n < 0) {
        n = 0;
      }
      if (width > n) {
        width = n;
      }
      if (width > ids_n) {
        width = ids_n;
      }
      if (width <= 0 || n <= 0) {
        return;
      }
      uint64_t prefix = 0;
      uint64_t decided = 0;
      long long need = width;
      for (int bit = 63; bit >= 0; --bit) {
        const uint64_t bitv = 1ull << bit;
        long long local = 0;
        for (int i = t; i < n; i += kTopkThreads) {
          const uint64_t key = pack_key(scores_p[i], static_cast<uint32_t>(i));
          if ((key & decided) == prefix && (key & bitv) != 0) {
            local = local + 1;
          }
        }
        red[t] = local;
        it.barrier(sycl::access::fence_space::local_space);
        for (int s = kTopkThreads / 2; s > 0; s >>= 1) {
          if (t < s) {
            red[t] = red[t] + red[t + s];
          }
          it.barrier(sycl::access::fence_space::local_space);
        }
        const long long cnt = red[0];
        it.barrier(sycl::access::fence_space::local_space);
        decided = decided | bitv;
        if (cnt >= need) {
          prefix = prefix | bitv;
        } else {
          need = need - cnt;
        }
      }
      if (t == 0) {
        int w = 0;
        for (int i = 0; i < n; ++i) {
          const uint64_t key = pack_key(scores_p[i], static_cast<uint32_t>(i));
          if (key >= prefix && w < width) {
            ids_p[w] = i;
            w = w + 1;
          }
        }
      }
    });
  });
  return ids;
}

// ---------------------------------------------------------------------------
// kv_gather. Grid covers max_ids, not this token's width. Dequant is
// fp16(code * scale) into the attend scratch.
// ---------------------------------------------------------------------------

torch::Tensor kv_gather(
    const torch::Tensor& ids,
    const torch::Tensor& k_codes,
    const torch::Tensor& v_codes,
    const torch::Tensor& k_scale,
    const torch::Tensor& v_scale,
    const torch::Tensor& page_table,
    const torch::Tensor& k_out,
    const torch::Tensor& v_out,
    const torch::Tensor& step) {
  TORCH_CHECK(ids.is_xpu() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32, "ids");
  TORCH_CHECK(k_codes.is_xpu() && k_codes.is_contiguous() && k_codes.scalar_type() == torch::kInt8, "k_codes");
  TORCH_CHECK(k_codes.dim() == 4 && v_codes.sizes() == k_codes.sizes(), "codes");
  TORCH_CHECK(v_codes.is_xpu() && v_codes.is_contiguous() && v_codes.scalar_type() == torch::kInt8, "v_codes");
  const int pages = static_cast<int>(k_codes.size(0));
  const int n_kv = static_cast<int>(k_codes.size(1));
  const int page_size = static_cast<int>(k_codes.size(2));
  const int head_dim = static_cast<int>(k_codes.size(3));
  TORCH_CHECK(head_dim % kGroup == 0, "head_dim");
  const int groups = head_dim / kGroup;
  auto scale_ok = [&](const torch::Tensor& s, const char* name) {
    TORCH_CHECK(s.is_xpu() && s.is_contiguous() && s.scalar_type() == torch::kFloat, name);
    TORCH_CHECK(
        s.dim() == 4 && s.size(0) == pages && s.size(1) == n_kv && s.size(2) == page_size &&
            s.size(3) == groups,
        name);
  };
  scale_ok(k_scale, "k_scale");
  scale_ok(v_scale, "v_scale");
  TORCH_CHECK(
      page_table.is_xpu() && page_table.is_contiguous() && page_table.scalar_type() == torch::kInt32,
      "page_table");
  TORCH_CHECK(k_out.is_xpu() && k_out.is_contiguous() && k_out.scalar_type() == torch::kFloat16, "k_out");
  TORCH_CHECK(v_out.is_xpu() && v_out.sizes() == k_out.sizes() && v_out.scalar_type() == torch::kFloat16, "v_out");
  TORCH_CHECK(k_out.dim() == 3 && k_out.size(1) == n_kv && k_out.size(2) == head_dim, "k_out [ids, kv, dim]");
  TORCH_CHECK(step.is_xpu() && step.is_contiguous() && step.scalar_type() == torch::kInt32, "step");
  const int max_ids = static_cast<int>(k_out.size(0));
  const at::DeviceGuard guard(ids.device());
  record_all({ids, k_codes, v_codes, k_scale, v_scale, page_table, k_out, v_out, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();

  const int32_t* ids_p = ids.data_ptr<int32_t>();
  const int8_t* kc = k_codes.data_ptr<int8_t>();
  const int8_t* vc = v_codes.data_ptr<int8_t>();
  const float* ks = k_scale.data_ptr<float>();
  const float* vs = v_scale.data_ptr<float>();
  const int32_t* table = page_table.data_ptr<int32_t>();
  sycl::half* ko = reinterpret_cast<sycl::half*>(k_out.data_ptr());
  sycl::half* vo = reinterpret_cast<sycl::half*>(v_out.data_ptr());
  const int32_t* step_p = step.data_ptr<int32_t>();

  const long long total = static_cast<long long>(max_ids) * n_kv * head_dim;
  const int threads = 256;
  const long long rounded = ((total + threads - 1) / threads) * threads;
  auto nd = sycl::nd_range<1>(static_cast<size_t>(rounded), static_cast<size_t>(threads));
  q.parallel_for(nd, [=](sycl::nd_item<1> it) {
    const long long i = static_cast<long long>(it.get_global_id(0));
    if (i >= total) {
      return;
    }
    const int width = step_p[3];
    const int span = n_kv * head_dim;
    const int id = static_cast<int>(i / span);
    if (id >= width) {
      return;
    }
    const int rem = static_cast<int>(i - static_cast<long long>(id) * span);
    const int head = rem / head_dim;
    const int d = rem - head * head_dim;
    const int cell = ids_p[id];
    const int page_i = cell / page_size;
    const int phys = table[page_i];
    const int slot = cell - page_i * page_size;
    const long long row = (static_cast<long long>(phys) * n_kv + head) * page_size + slot;
    const int g = d / kGroup;
    const float ksf = ks[row * groups + g];
    const float vsf = vs[row * groups + g];
    const float kf = static_cast<float>(static_cast<int>(kc[row * head_dim + d])) * ksf;
    const float vf = static_cast<float>(static_cast<int>(vc[row * head_dim + d])) * vsf;
    ko[i] = sycl::half(kf);
    vo[i] = sycl::half(vf);
  });
  return k_out;
}

// ---------------------------------------------------------------------------
// attend. One work-group per query head, local size = head_dim.
// KV head = q_head / (n_q / n_kv), integer division. Softmax in fp32 over
// fp16 K and V. width == 0 writes zeros and returns before the barrier.
// ---------------------------------------------------------------------------

torch::Tensor attend(
    const torch::Tensor& q_in,
    const torch::Tensor& k_f16,
    const torch::Tensor& v_f16,
    const torch::Tensor& attn,
    const torch::Tensor& step) {
  TORCH_CHECK(q_in.is_xpu() && q_in.is_contiguous() && q_in.scalar_type() == torch::kFloat, "q");
  TORCH_CHECK(q_in.dim() == 2, "q [heads, dim]");
  const int n_head = static_cast<int>(q_in.size(0));
  const int head_dim = static_cast<int>(q_in.size(1));
  TORCH_CHECK(head_dim >= 1 && head_dim <= 256 && (head_dim & (head_dim - 1)) == 0, "head_dim power of 2");
  TORCH_CHECK(k_f16.is_xpu() && k_f16.is_contiguous() && k_f16.scalar_type() == torch::kFloat16, "k_f16");
  TORCH_CHECK(v_f16.is_xpu() && v_f16.is_contiguous() && v_f16.sizes() == k_f16.sizes(), "v_f16");
  TORCH_CHECK(k_f16.dim() == 3 && k_f16.size(2) == head_dim, "k_f16 [ids, kv, dim]");
  const int max_ids = static_cast<int>(k_f16.size(0));
  const int n_kv = static_cast<int>(k_f16.size(1));
  TORCH_CHECK(max_ids >= 1 && max_ids <= kMaxAttendIds, "max_ids");
  TORCH_CHECK(n_kv >= 1 && n_head % n_kv == 0, "gqa");
  TORCH_CHECK(
      attn.is_xpu() && attn.is_contiguous() && attn.scalar_type() == torch::kFloat && attn.dim() == 2 &&
          attn.size(0) == n_head && attn.size(1) == head_dim,
      "attn");
  TORCH_CHECK(step.is_xpu() && step.is_contiguous() && step.scalar_type() == torch::kInt32, "step");
  const at::DeviceGuard guard(q_in.device());
  record_all({q_in, k_f16, v_f16, attn, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();

  const float* qp = q_in.data_ptr<float>();
  const sycl::half* kp = reinterpret_cast<const sycl::half*>(k_f16.data_ptr());
  const sycl::half* vp = reinterpret_cast<const sycl::half*>(v_f16.data_ptr());
  float* attn_p = attn.data_ptr<float>();
  const int32_t* step_p = step.data_ptr<int32_t>();
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  const int group = n_head / n_kv;

  auto nd = sycl::nd_range<1>(
      static_cast<size_t>(n_head) * static_cast<size_t>(head_dim), static_cast<size_t>(head_dim));
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> logits(sycl::range<1>(static_cast<size_t>(max_ids)), h);
    sycl::local_accessor<float, 1> red(sycl::range<1>(static_cast<size_t>(head_dim)), h);
    h.parallel_for(nd, [=](sycl::nd_item<1> it) {
      const int hds = static_cast<int>(it.get_group(0));
      const int t = static_cast<int>(it.get_local_id(0));
      const int wg = head_dim;
      int width = step_p[3];
      if (width > max_ids) {
        width = max_ids;
      }
      if (width < 0) {
        width = 0;
      }
      if (width == 0) {
        attn_p[static_cast<long long>(hds) * head_dim + t] = 0.0f;
        return;
      }
      const int kv_h = hds / group;
      const float* qh = qp + static_cast<long long>(hds) * head_dim;
      for (int j = t; j < width; j += wg) {
        const sycl::half* krow = kp + (static_cast<long long>(j) * n_kv + kv_h) * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
          dot = dot + static_cast<float>(krow[i]) * qh[i];
        }
        logits[j] = dot * scale;
      }
      it.barrier(sycl::access::fence_space::local_space);

      float mx = -3.402823466e+38f;
      for (int j = t; j < width; j += wg) {
        mx = sycl::fmax(mx, logits[j]);
      }
      red[t] = mx;
      it.barrier(sycl::access::fence_space::local_space);
      for (int s = wg / 2; s > 0; s >>= 1) {
        if (t < s) {
          red[t] = sycl::fmax(red[t], red[t + s]);
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      mx = red[0];
      it.barrier(sycl::access::fence_space::local_space);

      float sum = 0.0f;
      for (int j = t; j < width; j += wg) {
        const float e = sycl::exp(logits[j] - mx);
        logits[j] = e;
        sum = sum + e;
      }
      red[t] = sum;
      it.barrier(sycl::access::fence_space::local_space);
      for (int s = wg / 2; s > 0; s >>= 1) {
        if (t < s) {
          red[t] = red[t] + red[t + s];
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      const float inv = 1.0f / red[0];
      it.barrier(sycl::access::fence_space::local_space);

      float acc = 0.0f;
      for (int j = 0; j < width; ++j) {
        const float w = logits[j] * inv;
        const sycl::half* vrow = vp + (static_cast<long long>(j) * n_kv + kv_h) * head_dim;
        acc = acc + w * static_cast<float>(vrow[t]);
      }
      attn_p[static_cast<long long>(hds) * head_dim + t] = acc;
    });
  });
  return attn;
}

// ---------------------------------------------------------------------------
// gate. sigmoid(raw q second half) * attn, one round to fp16.
// qsa_gate_apply_kernel does the sigmoid and the multiply in double.
// ---------------------------------------------------------------------------

torch::Tensor gate(
    const torch::Tensor& attn,
    const torch::Tensor& q_full,
    const torch::Tensor& out) {
  TORCH_CHECK(attn.is_xpu() && attn.is_contiguous() && attn.scalar_type() == torch::kFloat, "attn");
  TORCH_CHECK(attn.dim() == 2, "attn [heads, dim]");
  const int n_head = static_cast<int>(attn.size(0));
  const int head_dim = static_cast<int>(attn.size(1));
  TORCH_CHECK(
      q_full.is_xpu() && q_full.is_contiguous() && q_full.scalar_type() == torch::kFloat &&
          q_full.dim() == 2 && q_full.size(0) == n_head && q_full.size(1) == head_dim * 2,
      "q_full [heads, 2*dim]");
  TORCH_CHECK(
      out.is_xpu() && out.is_contiguous() && out.scalar_type() == torch::kFloat16 && out.sizes() == attn.sizes(),
      "out");
  const at::DeviceGuard guard(attn.device());
  record_all({attn, q_full, out});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const float* attn_p = attn.data_ptr<float>();
  const float* full_p = q_full.data_ptr<float>();
  sycl::half* out_p = reinterpret_cast<sycl::half*>(out.data_ptr());
  const int n = n_head * head_dim;
  const int threads = 256;
  const int groups = div_up_int(n, threads);
  auto nd = sycl::nd_range<1>(static_cast<size_t>(groups) * threads, threads);
  q.parallel_for(nd, [=](sycl::nd_item<1> it) {
    const int i = static_cast<int>(it.get_global_id(0));
    if (i >= n) {
      return;
    }
    const int h = i / head_dim;
    const int d = i - h * head_dim;
    const double g =
        static_cast<double>(full_p[static_cast<long long>(h) * (head_dim * 2) + head_dim + d]);
    const double sig = 1.0 / (1.0 + sycl::exp(-g));
    const float y = static_cast<float>(static_cast<double>(attn_p[i]) * sig);
    out_p[i] = sycl::half(y);
  });
  return out;
}

}  // namespace

TORCH_LIBRARY(b70_qsa, m) {
  m.def("write_step(Tensor pos, Tensor step, int top_k, int block) -> Tensor");
  m.def(
      "kv_append(Tensor k, Tensor v, Tensor k_codes, Tensor v_codes, Tensor k_scale, "
      "Tensor v_scale, Tensor page_table, Tensor step) -> Tensor");
  m.def(
      "indexer_append(Tensor raw, Tensor k_norm, Tensor tail, Tensor dead, Tensor pooled, "
      "Tensor block_pos, Tensor cos, Tensor sin, Tensor step, Tensor mrope, int pos_base, "
      "float eps) -> Tensor");
  m.def(
      "index_scores(Tensor q_idx, Tensor pooled, Tensor dead, Tensor scores, Tensor step, "
      "Tensor bias, int block) -> Tensor");
  m.def("topk(Tensor scores, Tensor ids, Tensor step) -> Tensor");
  m.def(
      "kv_gather(Tensor ids, Tensor k_codes, Tensor v_codes, Tensor k_scale, Tensor v_scale, "
      "Tensor page_table, Tensor k_out, Tensor v_out, Tensor step) -> Tensor");
  m.def("attend(Tensor q, Tensor k_f16, Tensor v_f16, Tensor attn, Tensor step) -> Tensor");
  m.def("gate(Tensor attn, Tensor q_full, Tensor out) -> Tensor");
}

TORCH_LIBRARY_IMPL(b70_qsa, XPU, m) {
  m.impl("write_step", &write_step);
  m.impl("kv_append", &kv_append);
  m.impl("indexer_append", &indexer_append);
  m.impl("index_scores", &index_scores_impl);
  m.impl("topk", &topk);
  m.impl("kv_gather", &kv_gather);
  m.impl("attend", &attend);
  m.impl("gate", &gate);
}
