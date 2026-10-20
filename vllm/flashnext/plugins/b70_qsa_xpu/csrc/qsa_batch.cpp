// Batched QSA over one uint8 page per vLLM block.
//
// Formulas follow Strata night-20261016 and the single-sequence kernels in
// qsa_kernels.cpp. This is an attribution, not a copy of that source:
//   include/strata/kernels/qsa.hpp            step, index, top-k, attend, gate
//   src/kernels/cuda/qsa.cu                   indexer append, double RMS
//   src/kernels/cuda/kv_q8.cu                 group-64 int8, fp16 scale
//   include/strata/kernels/qsa_select.hpp     open block scores `dead`
//   sycl/src/kernels/cuda/qsa_prompt_attn_union.dp.cpp
//       union_build (cell | mask << 24). The SIMT union attend is the
//       correctness path; joint_matrix lives in qsa_xmx.cpp when it is linked.
//
// Page bytes match flashnext_ref.qsa_cache.make_layout:
//   K codes, V codes, K fp16 scales, V fp16 scales, completed pooled rows.
// The spare key is written into the page only when the next indexer state
// sits in the same vLLM block. Otherwise it stays in `dead`.
//
// pos < 0 is an inactive slot: write_step stores 0,0,0,0 and every kernel
// returns before its barrier. Grids are the batch capacity. Host ints are
// geometry (block_size, idx_block, idx_dim), constant across a capture.
// pos_base is a device tensor.

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
constexpr int kMaxAttendIds = 4096;
constexpr int kUnionRange = 16384;
constexpr int kUnionThreads = 256;
constexpr int kTile = 8;

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

uint32_t order_f32(float score) {
  uint32_t bits = std::bit_cast<uint32_t>(score);
  uint32_t mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
  return bits ^ mask;
}

uint64_t pack_key(float score, uint32_t index) {
  uint32_t inv = 0xFFFFFFFFu ^ index;
  return (static_cast<uint64_t>(order_f32(score)) << 32) | static_cast<uint64_t>(inv);
}

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

struct PageLay {
  int block_size;
  int n_kv;
  int head_dim;
  int groups;
  int idx_dim;
  int idx_block;
  int n_states;
  int k_codes_off;
  int v_codes_off;
  int k_scale_off;
  int v_scale_off;
  int pooled_off;
  int kv_bytes;
  int page_bytes;
  int pooled;
};

inline float* pooled_row(
    uint8_t* pages,
    const int32_t* table,
    int page_bytes,
    int pooled_off,
    int max_blocks,
    int block_size,
    int idx_block,
    int dim,
    int seq,
    int block_b) {
  const int token0 = block_b * idx_block;
  const int logical = token0 / block_size;
  const int state = (token0 - logical * block_size) / idx_block;
  const int phys = table[static_cast<long long>(seq) * max_blocks + logical];
  uint8_t* page = pages + static_cast<long long>(phys) * page_bytes;
  return reinterpret_cast<float*>(page + pooled_off + state * dim * 4);
}

inline float dequant_at(
    const uint8_t* pages,
    const int32_t* table,
    int page_bytes,
    int max_blocks,
    int block_size,
    int head_dim,
    int groups,
    int codes_off,
    int scale_off,
    int seq,
    int head,
    int cell,
    int d) {
  const int logical = cell / block_size;
  const int slot = cell - logical * block_size;
  const int phys = table[static_cast<long long>(seq) * max_blocks + logical];
  const uint8_t* page = pages + static_cast<long long>(phys) * page_bytes;
  const int local = head * block_size + slot;
  const int8_t code = static_cast<int8_t>(page[codes_off + local * head_dim + d]);
  const uint16_t* scales = reinterpret_cast<const uint16_t*>(page + scale_off);
  const float sc = f16_to_f32(scales[local * groups + d / kGroup]);
  return static_cast<float>(static_cast<int>(code)) * sc;
}

PageLay make_lay(
    int n_kv, int head_dim, int idx_dim, int idx_block, int block_size, int pooled, int page_bytes) {
  TORCH_CHECK(block_size >= 1 && idx_block >= 2 && block_size % idx_block == 0, "block_size");
  TORCH_CHECK(head_dim % kGroup == 0 && head_dim >= kGroup, "head_dim");
  TORCH_CHECK(n_kv >= 1 && idx_dim >= 1, "heads");
  PageLay lay{};
  lay.block_size = block_size;
  lay.n_kv = n_kv;
  lay.head_dim = head_dim;
  lay.groups = head_dim / kGroup;
  lay.idx_dim = idx_dim;
  lay.idx_block = idx_block;
  lay.n_states = block_size / idx_block;
  const int k_codes = n_kv * block_size * head_dim;
  const int k_scales = n_kv * block_size * lay.groups * 2;
  lay.k_codes_off = 0;
  lay.v_codes_off = k_codes;
  lay.k_scale_off = k_codes + k_codes;
  lay.v_scale_off = lay.k_scale_off + k_scales;
  lay.kv_bytes = lay.v_scale_off + k_scales;
  TORCH_CHECK(lay.kv_bytes % 4 == 0, "kv region alignment");
  lay.pooled_off = lay.kv_bytes;
  lay.pooled = pooled ? 1 : 0;
  const int pooled_bytes = lay.pooled ? lay.n_states * idx_dim * 4 : 0;
  const int expect = lay.kv_bytes + pooled_bytes;
  TORCH_CHECK(page_bytes == expect, "page_bytes ", page_bytes, " != ", expect);
  lay.page_bytes = page_bytes;
  return lay;
}

void check_pages(const torch::Tensor& pages, const torch::Tensor& table, int batch) {
  TORCH_CHECK(pages.is_xpu() && pages.is_contiguous() && pages.scalar_type() == torch::kUInt8, "pages");
  TORCH_CHECK(pages.dim() == 2, "pages [num_pages, page_bytes]");
  TORCH_CHECK(
      table.is_xpu() && table.is_contiguous() && table.scalar_type() == torch::kInt32 && table.dim() == 2,
      "block_table");
  TORCH_CHECK(table.size(0) == batch && table.size(1) >= 1, "block_table [B, max_blocks]");
}

void check_step(const torch::Tensor& step, int rows) {
  TORCH_CHECK(
      step.is_xpu() && step.is_contiguous() && step.scalar_type() == torch::kInt32 && step.dim() == 2 &&
          step.size(0) == rows && step.size(1) == 4,
      "step [rows, 4]");
}

// ---------------------------------------------------------------------------
// write_step_batch. One thread per sequence. Inactive pos stores zeros.
// ---------------------------------------------------------------------------

torch::Tensor write_step_batch(
    const torch::Tensor& pos, const torch::Tensor& step, int64_t top_k, int64_t block) {
  TORCH_CHECK(pos.is_xpu() && pos.is_contiguous() && pos.scalar_type() == torch::kInt32 && pos.dim() == 1, "pos");
  const int batch = static_cast<int>(pos.size(0));
  check_step(step, batch);
  TORCH_CHECK(top_k >= 1 && block >= 1, "top_k and block");
  const at::DeviceGuard guard(pos.device());
  record_all({pos, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const int32_t* pos_p = pos.data_ptr<int32_t>();
  int32_t* step_p = step.data_ptr<int32_t>();
  const int topk = static_cast<int>(top_k);
  const int r = static_cast<int>(block);
  q.parallel_for(sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> id) {
    const int b = static_cast<int>(id[0]);
    int32_t* row = step_p + static_cast<long long>(b) * 4;
    const int p = pos_p[b];
    if (p < 0) {
      row[0] = 0;
      row[1] = 0;
      row[2] = 0;
      row[3] = 0;
      return;
    }
    const int n_kv = p + 1;
    const int cap = topk + r - 1;
    row[0] = p;
    row[1] = n_kv;
    row[2] = n_kv / r;
    row[3] = n_kv < cap ? n_kv : cap;
  });
  return step;
}

// ---------------------------------------------------------------------------
// kv_append_batch. One group of 64 per (sequence, K or V, head, group).
// Scale is stored as fp16 bits. Inactive sequences return before the barrier.
// ---------------------------------------------------------------------------

torch::Tensor kv_append_batch(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& step,
    int64_t block_size,
    int64_t idx_block,
    int64_t idx_dim,
    int64_t has_pooled) {
  TORCH_CHECK(k.is_xpu() && k.is_contiguous() && k.scalar_type() == torch::kFloat && k.dim() == 3, "k");
  TORCH_CHECK(v.is_xpu() && v.is_contiguous() && v.sizes() == k.sizes(), "v");
  const int batch = static_cast<int>(k.size(0));
  const int n_kv = static_cast<int>(k.size(1));
  const int head_dim = static_cast<int>(k.size(2));
  check_pages(pages, block_table, batch);
  check_step(step, batch);
  const PageLay lay = make_lay(
      n_kv,
      head_dim,
      static_cast<int>(idx_dim),
      static_cast<int>(idx_block),
      static_cast<int>(block_size),
      has_pooled ? 1 : 0,
      static_cast<int>(pages.size(1)));
  const int max_blocks = static_cast<int>(block_table.size(1));
  const at::DeviceGuard guard(k.device());
  record_all({k, v, pages, block_table, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const float* kp = k.data_ptr<float>();
  const float* vp = v.data_ptr<float>();
  uint8_t* pages_p = pages.data_ptr<uint8_t>();
  const int32_t* table = block_table.data_ptr<int32_t>();
  const int32_t* step_p = step.data_ptr<int32_t>();
  const int groups = lay.groups;

  auto nd = sycl::nd_range<3>(
      sycl::range<3>(static_cast<size_t>(batch) * 2, static_cast<size_t>(n_kv), static_cast<size_t>(head_dim)),
      sycl::range<3>(1, 1, kGroup));
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> red(sycl::range<1>(kGroup), h);
    h.parallel_for(nd, [=](sycl::nd_item<3> it) {
      const int bx = static_cast<int>(it.get_group(0));
      const int b = bx >> 1;
      const int is_v = bx & 1;
      const int head = static_cast<int>(it.get_group(1));
      const int g = static_cast<int>(it.get_group(2));
      const int t = static_cast<int>(it.get_local_id(2));
      const int32_t* st = step_p + static_cast<long long>(b) * 4;
      // The whole group shares `b`. An inactive slot never reaches the barrier.
      if (st[1] <= 0) {
        return;
      }
      const float* src = is_v ? vp : kp;
      const float x = src[(static_cast<long long>(b) * n_kv + head) * head_dim + g * kGroup + t];
      red[t] = sycl::fabs(x);
      it.barrier(sycl::access::fence_space::local_space);
      for (int s = kGroup / 2; s > 0; s >>= 1) {
        if (t < s) {
          red[t] = sycl::fmax(red[t], red[t + s]);
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      const float amax = red[0];
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
      const int pos = st[0];
      const int logical = pos / lay.block_size;
      const int slot = pos - logical * lay.block_size;
      const int phys = table[static_cast<long long>(b) * max_blocks + logical];
      uint8_t* page = pages_p + static_cast<long long>(phys) * lay.page_bytes;
      const int local = head * lay.block_size + slot;
      const int region = is_v ? lay.v_codes_off : lay.k_codes_off;
      page[region + local * head_dim + g * kGroup + t] = static_cast<uint8_t>(static_cast<int8_t>(qv));
      if (t == 0) {
        const int scale_region = is_v ? lay.v_scale_off : lay.k_scale_off;
        uint16_t* scales = reinterpret_cast<uint16_t*>(page + scale_region);
        scales[local * groups + g] = f32_to_f16(sf);
      }
    });
  });
  return pages;
}

// ---------------------------------------------------------------------------
// indexer_append_batch. One work-group per sequence, one thread per dim.
// ---------------------------------------------------------------------------

torch::Tensor indexer_append_batch(
    const torch::Tensor& raw,
    const torch::Tensor& k_norm,
    const torch::Tensor& tail,
    const torch::Tensor& dead,
    const torch::Tensor& block_pos,
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& cos_tab,
    const torch::Tensor& sin_tab,
    const torch::Tensor& step,
    const torch::Tensor& mrope,
    const torch::Tensor& pos_base,
    double eps,
    int64_t block_size) {
  TORCH_CHECK(raw.is_xpu() && raw.is_contiguous() && raw.scalar_type() == torch::kFloat && raw.dim() == 2, "raw");
  const int batch = static_cast<int>(raw.size(0));
  const int dim = static_cast<int>(raw.size(1));
  TORCH_CHECK(dim >= 1 && dim <= 256, "idx_dim");
  TORCH_CHECK(
      k_norm.is_xpu() && k_norm.is_contiguous() && k_norm.scalar_type() == torch::kFloat && k_norm.numel() == dim,
      "k_norm");
  TORCH_CHECK(
      tail.is_xpu() && tail.is_contiguous() && tail.scalar_type() == torch::kFloat && tail.dim() == 3 &&
          tail.size(0) == batch && tail.size(2) == dim,
      "tail");
  const int r = static_cast<int>(tail.size(1)) + 1;
  TORCH_CHECK(r >= 2, "idx_block");
  TORCH_CHECK(
      dead.is_xpu() && dead.is_contiguous() && dead.scalar_type() == torch::kFloat && dead.dim() == 2 &&
          dead.size(0) == batch && dead.size(1) == dim,
      "dead");
  TORCH_CHECK(
      block_pos.is_xpu() && block_pos.is_contiguous() && block_pos.scalar_type() == torch::kInt32 &&
          block_pos.numel() == batch,
      "block_pos");
  check_pages(pages, block_table, batch);
  check_step(step, batch);
  TORCH_CHECK(cos_tab.is_xpu() && cos_tab.is_contiguous() && cos_tab.scalar_type() == torch::kFloat, "cos");
  TORCH_CHECK(sin_tab.is_xpu() && sin_tab.is_contiguous() && sin_tab.sizes() == cos_tab.sizes(), "sin");
  TORCH_CHECK(cos_tab.dim() == 2, "cos [pos, half]");
  const int half = static_cast<int>(cos_tab.size(1));
  TORCH_CHECK(half == 0 || half * 2 <= dim, "rotary dim");
  TORCH_CHECK(mrope.is_xpu() && mrope.is_contiguous() && mrope.scalar_type() == torch::kInt32, "mrope");
  const bool have_mrope = mrope.numel() > 0;
  int mrope_cells = 0;
  if (have_mrope) {
    TORCH_CHECK(mrope.dim() == 3 && mrope.size(0) == batch && mrope.size(2) == 3, "mrope [B, cells, 3]");
    mrope_cells = static_cast<int>(mrope.size(1));
  }
  TORCH_CHECK(
      pos_base.is_xpu() && pos_base.is_contiguous() && pos_base.scalar_type() == torch::kInt32 &&
          pos_base.numel() == batch,
      "pos_base");
  // n_kv heads are not in this op. The page check only needs the KV region
  // size, which kv_append already enforced. Recompute with n_kv taken from
  // the page byte size is unnecessary: pass the layout's idx fields and
  // accept the page when the pooled tail matches a 2-head... We do not know
  // n_kv here. Check pooled alignment only: page_bytes >= pooled rows and
  // the KV prefix is 4-aligned. The runner allocated both with one formula.
  const int page_bytes = static_cast<int>(pages.size(1));
  const int bs = static_cast<int>(block_size);
  TORCH_CHECK(bs % r == 0, "block_size multiple of idx_block");
  const int n_states = bs / r;
  const int pooled_bytes = n_states * dim * 4;
  TORCH_CHECK(page_bytes >= pooled_bytes && (page_bytes - pooled_bytes) % 4 == 0, "page pooled tail");
  const int pooled_off = page_bytes - pooled_bytes;
  const int max_blocks = static_cast<int>(block_table.size(1));
  const at::DeviceGuard guard(raw.device());
  record_all({raw, k_norm, tail, dead, block_pos, pages, block_table, cos_tab, sin_tab, step, mrope, pos_base});
  auto& q = c10::xpu::getCurrentXPUStream().queue();

  const float* raw_p = raw.data_ptr<float>();
  const float* norm_p = k_norm.data_ptr<float>();
  float* tail_p = tail.data_ptr<float>();
  float* dead_p = dead.data_ptr<float>();
  int32_t* bpos_p = block_pos.data_ptr<int32_t>();
  uint8_t* pages_p = pages.data_ptr<uint8_t>();
  const int32_t* table = block_table.data_ptr<int32_t>();
  const float* cos_p = cos_tab.data_ptr<float>();
  const float* sin_p = sin_tab.data_ptr<float>();
  const int32_t* step_p = step.data_ptr<int32_t>();
  const int32_t* mtab = have_mrope ? mrope.data_ptr<int32_t>() : nullptr;
  const int32_t* base_p = pos_base.data_ptr<int32_t>();
  const double eps_d = eps;
  const int tail_rows = r - 1;

  auto nd = sycl::nd_range<1>(
      static_cast<size_t>(batch) * static_cast<size_t>(dim), static_cast<size_t>(dim));
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<double, 1> mean(sycl::range<1>(static_cast<size_t>(dim)), h);
    h.parallel_for(nd, [=](sycl::nd_item<1> it) {
      const int seq = static_cast<int>(it.get_group(0));
      const int d = static_cast<int>(it.get_local_id(0));
      const int32_t* st = step_p + static_cast<long long>(seq) * 4;
      if (st[1] <= 0) {
        return;
      }
      const int pos = st[0];
      const int slot = pos % r;
      const float* raw_row = raw_p + static_cast<long long>(seq) * dim;
      float* tail_row = tail_p + (static_cast<long long>(seq) * tail_rows) * dim;
      float* dead_row = dead_p + static_cast<long long>(seq) * dim;

      if (slot < r - 1) {
        tail_row[static_cast<long long>(slot) * dim + d] = raw_row[d];
      }
      if (pos == 0) {
        const double p = static_cast<double>(raw_row[d]);
        double ss = 0.0;
        for (int i = 0; i < dim; ++i) {
          const double v = static_cast<double>(raw_row[i]);
          ss = ss + v * v;
        }
        const double inv = 1.0 / sycl::sqrt(ss / static_cast<double>(dim) + eps_d);
        float y = static_cast<float>(p * inv * static_cast<double>(norm_p[d]));
        if (half > 0 && d < half * 2) {
          y = y * cos_p[d % half];
        }
        dead_row[d] = y;
        pooled_row(pages_p, table, page_bytes, pooled_off, max_blocks, bs, r, dim, seq, 0)[d] = y;
      }
      // slot == r-1 is uniform across the group. Inactive already returned.
      if (slot != r - 1) {
        return;
      }
      double m = 0.0;
      for (int j = 0; j < r - 1; ++j) {
        m = m + static_cast<double>(tail_row[static_cast<long long>(j) * dim + d]);
      }
      m = m + static_cast<double>(raw_row[d]);
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
      float* row = pooled_row(pages_p, table, page_bytes, pooled_off, max_blocks, bs, r, dim, seq, b);
      row[d] = static_cast<float>(m * inv * static_cast<double>(norm_p[d]));
      const int logical = (b * r) / bs;
      const int logical_s = ((b + 1) * r) / bs;
      if (logical_s == logical) {
        pooled_row(pages_p, table, page_bytes, pooled_off, max_blocks, bs, r, dim, seq, b + 1)[d] = dead_row[d];
      }
      if (d == 0) {
        bpos_p[seq] = base_p[seq] + b * r;
      }
      it.barrier(sycl::access::fence_space::global_and_local);

      if (half > 0 && d < half) {
        int rot = base_p[seq] + b * r;
        if (mtab != nullptr) {
          rot = mtab[(static_cast<long long>(seq) * mrope_cells + rot) * 3 + (d % 3)];
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
  return pages;
}

// ---------------------------------------------------------------------------
// index_scores_batch. One thread per (query, indexer block). Open block
// reads dead[seq]. Dots are fp32. No barrier.
// ---------------------------------------------------------------------------

torch::Tensor index_scores_batch(
    const torch::Tensor& q_idx,
    const torch::Tensor& pages,
    const torch::Tensor& dead,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_slot,
    const torch::Tensor& scores,
    const torch::Tensor& step,
    const torch::Tensor& bias,
    int64_t block_size,
    int64_t idx_block) {
  TORCH_CHECK(q_idx.is_xpu() && q_idx.is_contiguous() && q_idx.scalar_type() == torch::kFloat && q_idx.dim() == 3, "q");
  const int n_q = static_cast<int>(q_idx.size(0));
  const int n_head = static_cast<int>(q_idx.size(1));
  const int dim = static_cast<int>(q_idx.size(2));
  TORCH_CHECK(
      dead.is_xpu() && dead.is_contiguous() && dead.scalar_type() == torch::kFloat && dead.dim() == 2 &&
          dead.size(1) == dim,
      "dead");
  const int batch = static_cast<int>(dead.size(0));
  check_pages(pages, block_table, batch);
  TORCH_CHECK(
      seq_slot.is_xpu() && seq_slot.is_contiguous() && seq_slot.scalar_type() == torch::kInt32 &&
          seq_slot.numel() == n_q,
      "seq_slot");
  TORCH_CHECK(
      scores.is_xpu() && scores.is_contiguous() && scores.scalar_type() == torch::kFloat && scores.dim() == 2 &&
          scores.size(0) == n_q,
      "scores");
  check_step(step, n_q);
  TORCH_CHECK(bias.is_xpu() && bias.is_contiguous() && bias.scalar_type() == torch::kFloat, "bias");
  const int r = static_cast<int>(idx_block);
  const int bs = static_cast<int>(block_size);
  TORCH_CHECK(r >= 2 && bs % r == 0, "idx_block");
  const int page_bytes = static_cast<int>(pages.size(1));
  const int n_states = bs / r;
  const int pooled_bytes = n_states * dim * 4;
  TORCH_CHECK(page_bytes >= pooled_bytes, "pooled tail");
  const int pooled_off = page_bytes - pooled_bytes;
  const int cells = static_cast<int>(scores.size(1));
  const int max_ib = (cells + r - 1) / r;
  const int max_blocks = static_cast<int>(block_table.size(1));
  const bool have_bias = bias.numel() > 0;
  const at::DeviceGuard guard(q_idx.device());
  record_all({q_idx, pages, dead, block_table, seq_slot, scores, step, bias});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const float* qp = q_idx.data_ptr<float>();
  const uint8_t* pages_p = pages.data_ptr<uint8_t>();
  const float* dead_p = dead.data_ptr<float>();
  const int32_t* table = block_table.data_ptr<int32_t>();
  const int32_t* slot_p = seq_slot.data_ptr<int32_t>();
  float* scores_p = scores.data_ptr<float>();
  const int32_t* step_p = step.data_ptr<int32_t>();
  const float* bias_p = have_bias ? bias.data_ptr<float>() : nullptr;
  const int bias_n = have_bias ? static_cast<int>(bias.numel()) : 0;
  const long long total = static_cast<long long>(n_q) * max_ib;

  q.parallel_for(sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> id) {
    const long long gid = static_cast<long long>(id[0]);
    const int qi = static_cast<int>(gid / max_ib);
    const int ib = static_cast<int>(gid - static_cast<long long>(qi) * max_ib);
    const int32_t* st = step_p + static_cast<long long>(qi) * 4;
    const int n_kv = st[1];
    const int n_bid = st[2];
    if (n_kv <= 0 || ib > n_bid) {
      return;
    }
    const int seq = slot_p[qi];
    const float* key;
    if (ib == n_bid) {
      key = dead_p + static_cast<long long>(seq) * dim;
    } else {
      const int token0 = ib * r;
      const int logical = token0 / bs;
      const int state = (token0 - logical * bs) / r;
      const int phys = table[static_cast<long long>(seq) * max_blocks + logical];
      const uint8_t* page = pages_p + static_cast<long long>(phys) * page_bytes;
      key = reinterpret_cast<const float*>(page + pooled_off + state * dim * 4);
    }
    float score = 0.0f;
    const float* qrow = qp + static_cast<long long>(qi) * n_head * dim;
    for (int h = 0; h < n_head; ++h) {
      const float* qh = qrow + static_cast<long long>(h) * dim;
      float dot = 0.0f;
      for (int i = 0; i < dim; ++i) {
        dot = dot + key[i] * qh[i];
      }
      if (dot > 0.0f) {
        score = score + dot;
      }
    }
    if (bias_p != nullptr && ib < bias_n) {
      score = score + bias_p[ib];
    }
    long long lo = static_cast<long long>(ib) * r;
    long long hi = lo + r;
    if (ib == n_bid) {
      hi = n_kv;
      if ((n_kv % r) != 0) {
        score = score + 1.0e9f;
      }
    }
    if (hi > n_kv) {
      hi = n_kv;
    }
    float* out = scores_p + static_cast<long long>(qi) * cells;
    for (long long j = lo; j < hi; ++j) {
      if (j < cells) {
        out[j] = score;
      }
    }
  });
  return scores;
}

// ---------------------------------------------------------------------------
// topk_batch. One work-group of 256 per query. width 0 returns first.
// ---------------------------------------------------------------------------

torch::Tensor topk_batch(const torch::Tensor& scores, const torch::Tensor& ids, const torch::Tensor& step) {
  TORCH_CHECK(scores.is_xpu() && scores.is_contiguous() && scores.scalar_type() == torch::kFloat && scores.dim() == 2, "scores");
  TORCH_CHECK(ids.is_xpu() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32 && ids.dim() == 2, "ids");
  const int n_q = static_cast<int>(scores.size(0));
  TORCH_CHECK(ids.size(0) == n_q, "ids batch");
  check_step(step, n_q);
  const int scores_n = static_cast<int>(scores.size(1));
  const int ids_n = static_cast<int>(ids.size(1));
  const at::DeviceGuard guard(scores.device());
  record_all({scores, ids, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const float* scores_p = scores.data_ptr<float>();
  int32_t* ids_p = ids.data_ptr<int32_t>();
  const int32_t* step_p = step.data_ptr<int32_t>();
  auto nd = sycl::nd_range<1>(
      static_cast<size_t>(n_q) * kTopkThreads, static_cast<size_t>(kTopkThreads));
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<long long, 1> red(sycl::range<1>(kTopkThreads), h);
    h.parallel_for(nd, [=](sycl::nd_item<1> it) {
      const int qi = static_cast<int>(it.get_group(0));
      const int t = static_cast<int>(it.get_local_id(0));
      const int32_t* st = step_p + static_cast<long long>(qi) * 4;
      int n = st[1];
      int width = st[3];
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
      const float* row = scores_p + static_cast<long long>(qi) * scores_n;
      int32_t* out = ids_p + static_cast<long long>(qi) * ids_n;
      uint64_t prefix = 0;
      uint64_t decided = 0;
      long long need = width;
      for (int bit = 63; bit >= 0; --bit) {
        const uint64_t bitv = 1ull << bit;
        long long local = 0;
        for (int i = t; i < n; i += kTopkThreads) {
          const uint64_t key = pack_key(row[i], static_cast<uint32_t>(i));
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
          const uint64_t key = pack_key(row[i], static_cast<uint32_t>(i));
          if (key >= prefix && w < width) {
            out[w] = i;
            w = w + 1;
          }
        }
      }
    });
  });
  return ids;
}

// ---------------------------------------------------------------------------
// kv_gather_batch. No barrier. fp16(code * stored scale).
// ---------------------------------------------------------------------------

torch::Tensor kv_gather_batch(
    const torch::Tensor& ids,
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_slot,
    const torch::Tensor& k_out,
    const torch::Tensor& v_out,
    const torch::Tensor& step,
    int64_t block_size,
    int64_t idx_block,
    int64_t idx_dim,
    int64_t has_pooled) {
  TORCH_CHECK(ids.is_xpu() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32 && ids.dim() == 2, "ids");
  const int n_q = static_cast<int>(ids.size(0));
  const int id_cap = static_cast<int>(ids.size(1));
  TORCH_CHECK(k_out.is_xpu() && k_out.is_contiguous() && k_out.scalar_type() == torch::kFloat16 && k_out.dim() == 4, "k_out");
  TORCH_CHECK(v_out.is_xpu() && v_out.is_contiguous() && v_out.sizes() == k_out.sizes(), "v_out");
  TORCH_CHECK(k_out.size(0) == n_q && k_out.size(1) == id_cap, "k_out [Q, cap, kv, dim]");
  const int n_kv = static_cast<int>(k_out.size(2));
  const int head_dim = static_cast<int>(k_out.size(3));
  TORCH_CHECK(
      seq_slot.is_xpu() && seq_slot.is_contiguous() && seq_slot.scalar_type() == torch::kInt32 &&
          seq_slot.numel() == n_q,
      "seq_slot");
  const int batch = static_cast<int>(block_table.size(0));
  check_pages(pages, block_table, batch);
  check_step(step, n_q);
  const PageLay lay = make_lay(
      n_kv,
      head_dim,
      static_cast<int>(idx_dim),
      static_cast<int>(idx_block),
      static_cast<int>(block_size),
      has_pooled ? 1 : 0,
      static_cast<int>(pages.size(1)));
  const int max_blocks = static_cast<int>(block_table.size(1));
  const at::DeviceGuard guard(ids.device());
  record_all({ids, pages, block_table, seq_slot, k_out, v_out, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const int32_t* ids_p = ids.data_ptr<int32_t>();
  const uint8_t* pages_p = pages.data_ptr<uint8_t>();
  const int32_t* table = block_table.data_ptr<int32_t>();
  const int32_t* slot_p = seq_slot.data_ptr<int32_t>();
  sycl::half* ko = reinterpret_cast<sycl::half*>(k_out.data_ptr());
  sycl::half* vo = reinterpret_cast<sycl::half*>(v_out.data_ptr());
  const int32_t* step_p = step.data_ptr<int32_t>();
  const long long span = static_cast<long long>(n_kv) * head_dim;
  const long long total = static_cast<long long>(n_q) * id_cap * span;
  const int threads = 256;
  const long long rounded = ((total + threads - 1) / threads) * threads;
  auto nd = sycl::nd_range<1>(static_cast<size_t>(rounded), static_cast<size_t>(threads));
  q.parallel_for(nd, [=](sycl::nd_item<1> it) {
    const long long i = static_cast<long long>(it.get_global_id(0));
    if (i >= total) {
      return;
    }
    const long long per = static_cast<long long>(id_cap) * span;
    const int qi = static_cast<int>(i / per);
    const long long rem = i - static_cast<long long>(qi) * per;
    const int id = static_cast<int>(rem / span);
    const int inner = static_cast<int>(rem - static_cast<long long>(id) * span);
    const int width = step_p[static_cast<long long>(qi) * 4 + 3];
    if (id >= width) {
      return;
    }
    const int head = inner / head_dim;
    const int d = inner - head * head_dim;
    const int cell = ids_p[static_cast<long long>(qi) * id_cap + id];
    const int seq = slot_p[qi];
    const int logical = cell / lay.block_size;
    const int slot = cell - logical * lay.block_size;
    const int phys = table[static_cast<long long>(seq) * max_blocks + logical];
    const uint8_t* page = pages_p + static_cast<long long>(phys) * lay.page_bytes;
    const int local = head * lay.block_size + slot;
    const int g = d / kGroup;
    const int8_t kc = static_cast<int8_t>(page[lay.k_codes_off + local * head_dim + d]);
    const int8_t vc = static_cast<int8_t>(page[lay.v_codes_off + local * head_dim + d]);
    const uint16_t* ks = reinterpret_cast<const uint16_t*>(page + lay.k_scale_off);
    const uint16_t* vs = reinterpret_cast<const uint16_t*>(page + lay.v_scale_off);
    const float ksf = f16_to_f32(ks[local * lay.groups + g]);
    const float vsf = f16_to_f32(vs[local * lay.groups + g]);
    ko[i] = sycl::half(static_cast<float>(static_cast<int>(kc)) * ksf);
    vo[i] = sycl::half(static_cast<float>(static_cast<int>(vc)) * vsf);
  });
  return k_out;
}

// ---------------------------------------------------------------------------
// attend_batch. One work-group per (query, head). width 0 writes zeros.
// ---------------------------------------------------------------------------

torch::Tensor attend_batch(
    const torch::Tensor& q_in,
    const torch::Tensor& k_f16,
    const torch::Tensor& v_f16,
    const torch::Tensor& attn,
    const torch::Tensor& step) {
  TORCH_CHECK(q_in.is_xpu() && q_in.is_contiguous() && q_in.scalar_type() == torch::kFloat && q_in.dim() == 3, "q");
  const int n_q = static_cast<int>(q_in.size(0));
  const int n_head = static_cast<int>(q_in.size(1));
  const int head_dim = static_cast<int>(q_in.size(2));
  TORCH_CHECK(head_dim >= 1 && head_dim <= 256 && (head_dim & (head_dim - 1)) == 0, "head_dim power of 2");
  TORCH_CHECK(k_f16.is_xpu() && k_f16.is_contiguous() && k_f16.scalar_type() == torch::kFloat16 && k_f16.dim() == 4, "k");
  TORCH_CHECK(v_f16.is_xpu() && v_f16.is_contiguous() && v_f16.sizes() == k_f16.sizes(), "v");
  TORCH_CHECK(k_f16.size(0) == n_q && k_f16.size(3) == head_dim, "k [Q, cap, kv, dim]");
  const int max_ids = static_cast<int>(k_f16.size(1));
  const int n_kv = static_cast<int>(k_f16.size(2));
  TORCH_CHECK(max_ids >= 1 && max_ids <= kMaxAttendIds, "max_ids");
  TORCH_CHECK(n_kv >= 1 && n_head % n_kv == 0, "gqa");
  TORCH_CHECK(
      attn.is_xpu() && attn.is_contiguous() && attn.scalar_type() == torch::kFloat && attn.dim() == 3 &&
          attn.size(0) == n_q && attn.size(1) == n_head && attn.size(2) == head_dim,
      "attn");
  check_step(step, n_q);
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
  const long long groups = static_cast<long long>(n_q) * n_head;
  auto nd = sycl::nd_range<1>(
      static_cast<size_t>(groups) * static_cast<size_t>(head_dim), static_cast<size_t>(head_dim));
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> logits(sycl::range<1>(static_cast<size_t>(max_ids)), h);
    sycl::local_accessor<float, 1> red(sycl::range<1>(static_cast<size_t>(head_dim)), h);
    h.parallel_for(nd, [=](sycl::nd_item<1> it) {
      const int wg_id = static_cast<int>(it.get_group(0));
      const int t = static_cast<int>(it.get_local_id(0));
      const int qi = wg_id / n_head;
      const int hds = wg_id - qi * n_head;
      int width = step_p[static_cast<long long>(qi) * 4 + 3];
      if (width > max_ids) {
        width = max_ids;
      }
      if (width < 0) {
        width = 0;
      }
      float* out = attn_p + (static_cast<long long>(qi) * n_head + hds) * head_dim;
      if (width == 0) {
        out[t] = 0.0f;
        return;
      }
      const int kv_h = hds / group;
      const float* qh = qp + (static_cast<long long>(qi) * n_head + hds) * head_dim;
      const long long qbase = static_cast<long long>(qi) * max_ids * n_kv;
      for (int j = t; j < width; j += head_dim) {
        const sycl::half* krow = kp + ((qbase + static_cast<long long>(j) * n_kv + kv_h) * head_dim);
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
          dot = dot + static_cast<float>(krow[i]) * qh[i];
        }
        logits[j] = dot * scale;
      }
      it.barrier(sycl::access::fence_space::local_space);
      float mx = -3.402823466e+38f;
      for (int j = t; j < width; j += head_dim) {
        mx = sycl::fmax(mx, logits[j]);
      }
      red[t] = mx;
      it.barrier(sycl::access::fence_space::local_space);
      for (int s = head_dim / 2; s > 0; s >>= 1) {
        if (t < s) {
          red[t] = sycl::fmax(red[t], red[t + s]);
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      mx = red[0];
      it.barrier(sycl::access::fence_space::local_space);
      float sum = 0.0f;
      for (int j = t; j < width; j += head_dim) {
        const float e = sycl::exp(logits[j] - mx);
        logits[j] = e;
        sum = sum + e;
      }
      red[t] = sum;
      it.barrier(sycl::access::fence_space::local_space);
      for (int s = head_dim / 2; s > 0; s >>= 1) {
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
        const sycl::half* vrow = vp + ((qbase + static_cast<long long>(j) * n_kv + kv_h) * head_dim);
        acc = acc + w * static_cast<float>(vrow[t]);
      }
      out[t] = acc;
    });
  });
  return attn;
}

// ---------------------------------------------------------------------------
// gate_batch. Sigmoid in double, one fp16 store.
// ---------------------------------------------------------------------------

torch::Tensor gate_batch(const torch::Tensor& attn, const torch::Tensor& q_full, const torch::Tensor& out) {
  TORCH_CHECK(attn.is_xpu() && attn.is_contiguous() && attn.scalar_type() == torch::kFloat && attn.dim() == 3, "attn");
  const int n_q = static_cast<int>(attn.size(0));
  const int n_head = static_cast<int>(attn.size(1));
  const int head_dim = static_cast<int>(attn.size(2));
  TORCH_CHECK(
      q_full.is_xpu() && q_full.is_contiguous() && q_full.scalar_type() == torch::kFloat && q_full.dim() == 3 &&
          q_full.size(0) == n_q && q_full.size(1) == n_head && q_full.size(2) == head_dim * 2,
      "q_full");
  TORCH_CHECK(
      out.is_xpu() && out.is_contiguous() && out.scalar_type() == torch::kFloat16 && out.sizes() == attn.sizes(),
      "out");
  const at::DeviceGuard guard(attn.device());
  record_all({attn, q_full, out});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const float* attn_p = attn.data_ptr<float>();
  const float* full_p = q_full.data_ptr<float>();
  sycl::half* out_p = reinterpret_cast<sycl::half*>(out.data_ptr());
  const long long n = static_cast<long long>(n_q) * n_head * head_dim;
  const int threads = 256;
  const long long groups = (n + threads - 1) / threads;
  auto nd = sycl::nd_range<1>(static_cast<size_t>(groups) * threads, threads);
  q.parallel_for(nd, [=](sycl::nd_item<1> it) {
    const long long i = static_cast<long long>(it.get_global_id(0));
    if (i >= n) {
      return;
    }
    const int hd2 = head_dim * 2;
    const int per = n_head * head_dim;
    const int qi = static_cast<int>(i / per);
    const int rem = static_cast<int>(i - static_cast<long long>(qi) * per);
    const int h = rem / head_dim;
    const int d = rem - h * head_dim;
    const double g = static_cast<double>(full_p[(static_cast<long long>(qi) * n_head + h) * hd2 + head_dim + d]);
    const double sig = 1.0 / (1.0 + sycl::exp(-g));
    const float y = static_cast<float>(static_cast<double>(attn_p[i]) * sig);
    out_p[i] = sycl::half(y);
  });
  return out;
}

// ---------------------------------------------------------------------------
// union_build. One work-group per 8-query tile. Bitmap passes of 16384
// cells, same packing as qsa_prompt_attn_union.dp.cpp: cell | mask << 24.
// ---------------------------------------------------------------------------

torch::Tensor union_build(
    const torch::Tensor& ids, const torch::Tensor& step, const torch::Tensor& ulist, const torch::Tensor& ucount) {
  TORCH_CHECK(ids.is_xpu() && ids.is_contiguous() && ids.scalar_type() == torch::kInt32 && ids.dim() == 2, "ids");
  const int n_q = static_cast<int>(ids.size(0));
  const int cap = static_cast<int>(ids.size(1));
  TORCH_CHECK(cap >= 1 && cap <= kMaxAttendIds, "cap");
  check_step(step, n_q);
  const int tiles = (n_q + kTile - 1) / kTile;
  TORCH_CHECK(
      ulist.is_xpu() && ulist.is_contiguous() && ulist.scalar_type() == torch::kInt32 && ulist.dim() == 2 &&
          ulist.size(0) == tiles && ulist.size(1) == kTile * cap,
      "ulist [tiles, 8*cap]");
  TORCH_CHECK(
      ucount.is_xpu() && ucount.is_contiguous() && ucount.scalar_type() == torch::kInt32 && ucount.numel() == tiles,
      "ucount");
  const at::DeviceGuard guard(ids.device());
  record_all({ids, step, ulist, ucount});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const int32_t* ids_p = ids.data_ptr<int32_t>();
  const int32_t* step_p = step.data_ptr<int32_t>();
  int32_t* ulist_p = ulist.data_ptr<int32_t>();
  int32_t* ucount_p = ucount.data_ptr<int32_t>();
  auto nd = sycl::nd_range<1>(static_cast<size_t>(tiles) * kUnionThreads, static_cast<size_t>(kUnionThreads));
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> smask(sycl::range<1>(kUnionRange / 4), h);
    h.parallel_for(nd, [=](sycl::nd_item<1> it) {
      const int tile = static_cast<int>(it.get_group(0));
      const int t = static_cast<int>(it.get_local_id(0));
      auto grp = it.get_group();
      const int t0 = tile * kTile;
      int nq = n_q - t0;
      if (nq > kTile) {
        nq = kTile;
      }
      if (nq < 0) {
        nq = 0;
      }
      int nkv = 0;
      for (int j = 0; j < nq; ++j) {
        const int v = step_p[static_cast<long long>(t0 + j) * 4 + 1];
        if (v > nkv) {
          nkv = v;
        }
      }
      int32_t* out = ulist_p + static_cast<long long>(tile) * kTile * cap;
      int total = 0;
      for (int r0 = 0; r0 < nkv; r0 += kUnionRange) {
        for (int w = t; w < kUnionRange / 4; w += kUnionThreads) {
          smask[w] = 0u;
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int k = t; k < kTile * cap; k += kUnionThreads) {
          const int j = k / cap;
          const int i = k - j * cap;
          if (j >= nq || i >= step_p[static_cast<long long>(t0 + j) * 4 + 3]) {
            continue;
          }
          const int id = ids_p[static_cast<long long>(t0 + j) * cap + i] - r0;
          if (id < 0 || id >= kUnionRange) {
            continue;
          }
          sycl::atomic_ref<
              uint32_t,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space>
              a(smask[id >> 2]);
          a.fetch_or((1u << j) << (8 * (id & 3)));
        }
        it.barrier(sycl::access::fence_space::local_space);
        constexpr int kWpt = (kUnionRange / 4) / kUnionThreads;
        int cnt = 0;
        for (int w = 0; w < kWpt; ++w) {
          const uint32_t m = smask[t * kWpt + w];
          cnt += (m & 0xFFu) != 0;
          cnt += (m & 0xFF00u) != 0;
          cnt += (m & 0xFF0000u) != 0;
          cnt += (m & 0xFF000000u) != 0;
        }
        const int base = sycl::exclusive_scan_over_group(grp, cnt, sycl::plus<int>());
        const int sum = sycl::reduce_over_group(grp, cnt, sycl::plus<int>());
        int o = total + base;
        for (int w = 0; w < kWpt; ++w) {
          const uint32_t m = smask[t * kWpt + w];
          if (m == 0u) {
            continue;
          }
          for (int b = 0; b < 4; ++b) {
            const uint32_t mb = (m >> (8 * b)) & 0xFFu;
            if (mb) {
              out[o++] = static_cast<int32_t>(
                  (static_cast<uint32_t>(r0 + (t * kWpt + w) * 4 + b)) | (mb << 24));
            }
          }
        }
        total += sum;
        it.barrier(sycl::access::fence_space::local_space);
      }
      if (t == 0) {
        ucount_p[tile] = total;
      }
    });
  });
  return ulist;
}

// ---------------------------------------------------------------------------
// union_attend. SIMT. Each query attends the union cells whose mask bit is
// set, in ascending cell order. Unselected cells are skipped, which is the
// -inf softmax. fp16 dequant, fp32 dots.
// ---------------------------------------------------------------------------

torch::Tensor union_attend(
    const torch::Tensor& q_in,
    const torch::Tensor& pages,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_slot,
    const torch::Tensor& ulist,
    const torch::Tensor& ucount,
    const torch::Tensor& attn,
    const torch::Tensor& step,
    int64_t block_size,
    int64_t idx_block,
    int64_t idx_dim,
    int64_t has_pooled) {
  TORCH_CHECK(q_in.is_xpu() && q_in.is_contiguous() && q_in.scalar_type() == torch::kFloat && q_in.dim() == 3, "q");
  const int n_q = static_cast<int>(q_in.size(0));
  const int n_head = static_cast<int>(q_in.size(1));
  const int head_dim = static_cast<int>(q_in.size(2));
  TORCH_CHECK(head_dim >= 1 && head_dim <= 256 && (head_dim & (head_dim - 1)) == 0, "head_dim");
  const int tiles = (n_q + kTile - 1) / kTile;
  TORCH_CHECK(ulist.is_xpu() && ulist.is_contiguous() && ulist.scalar_type() == torch::kInt32 && ulist.dim() == 2, "ulist");
  TORCH_CHECK(ulist.size(0) == tiles && ulist.size(1) % kTile == 0, "ulist");
  const int stride = static_cast<int>(ulist.size(1));
  const int id_cap = stride / kTile;
  TORCH_CHECK(id_cap <= kMaxAttendIds, "union cap");
  TORCH_CHECK(
      ucount.is_xpu() && ucount.is_contiguous() && ucount.scalar_type() == torch::kInt32 && ucount.numel() == tiles,
      "ucount");
  TORCH_CHECK(
      seq_slot.is_xpu() && seq_slot.is_contiguous() && seq_slot.scalar_type() == torch::kInt32 &&
          seq_slot.numel() == n_q,
      "seq_slot");
  TORCH_CHECK(
      attn.is_xpu() && attn.is_contiguous() && attn.scalar_type() == torch::kFloat && attn.sizes() == q_in.sizes(),
      "attn");
  check_step(step, n_q);
  // n_kv is the unique head count that makes this page's KV region match
  // K and V codes plus fp16 scales. has_pooled's low bit selects the tail.
  const int pooled_flag = static_cast<int>(has_pooled) & 1;
  const int bs = static_cast<int>(block_size);
  const int r = static_cast<int>(idx_block);
  const int idim = static_cast<int>(idx_dim);
  const int groups = head_dim / kGroup;
  const int page_bytes = static_cast<int>(pages.size(1));
  const int n_states = bs / r;
  const int pooled_bytes = pooled_flag ? n_states * idim * 4 : 0;
  const int kv_bytes = page_bytes - pooled_bytes;
  const int per_head = bs * (2 * head_dim + 4 * groups);
  TORCH_CHECK(per_head > 0 && kv_bytes % per_head == 0, "page does not match head_dim");
  const int n_kv = kv_bytes / per_head;
  TORCH_CHECK(n_kv >= 1 && n_head % n_kv == 0, "gqa");
  const PageLay lay = make_lay(n_kv, head_dim, idim, r, bs, pooled_flag, page_bytes);
  const int batch = static_cast<int>(block_table.size(0));
  check_pages(pages, block_table, batch);
  const int max_blocks = static_cast<int>(block_table.size(1));
  const at::DeviceGuard guard(q_in.device());
  record_all({q_in, pages, block_table, seq_slot, ulist, ucount, attn, step});
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  const float* qp = q_in.data_ptr<float>();
  const uint8_t* pages_p = pages.data_ptr<uint8_t>();
  const int32_t* table = block_table.data_ptr<int32_t>();
  const int32_t* slot_p = seq_slot.data_ptr<int32_t>();
  const int32_t* ulist_p = ulist.data_ptr<int32_t>();
  const int32_t* ucount_p = ucount.data_ptr<int32_t>();
  float* attn_p = attn.data_ptr<float>();
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  const int group = n_head / n_kv;
  const long long wgs = static_cast<long long>(n_q) * n_head;
  auto nd = sycl::nd_range<1>(static_cast<size_t>(wgs) * static_cast<size_t>(head_dim), static_cast<size_t>(head_dim));
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int, 1> sel(sycl::range<1>(static_cast<size_t>(id_cap)), h);
    sycl::local_accessor<int, 1> nsel(sycl::range<1>(1), h);
    sycl::local_accessor<float, 1> logits(sycl::range<1>(static_cast<size_t>(id_cap)), h);
    sycl::local_accessor<float, 1> red(sycl::range<1>(static_cast<size_t>(head_dim)), h);
    h.parallel_for(nd, [=](sycl::nd_item<1> it) {
      const int wg_id = static_cast<int>(it.get_group(0));
      const int t = static_cast<int>(it.get_local_id(0));
      const int qi = wg_id / n_head;
      const int hds = wg_id - qi * n_head;
      const int tile = qi / kTile;
      const int jbit = qi - tile * kTile;
      const int U = ucount_p[tile];
      const int32_t* list = ulist_p + static_cast<long long>(tile) * stride;
      if (t == 0) {
        int w = 0;
        const uint32_t bit = 1u << jbit;
        for (int c = 0; c < U; ++c) {
          const uint32_t e = static_cast<uint32_t>(list[c]);
          if ((e >> 24) & bit) {
            if (w < id_cap) {
              sel[w] = static_cast<int>(e & 0x00FFFFFFu);
              w = w + 1;
            }
          }
        }
        nsel[0] = w;
      }
      it.barrier(sycl::access::fence_space::local_space);
      const int width = nsel[0];
      float* out = attn_p + (static_cast<long long>(qi) * n_head + hds) * head_dim;
      if (width <= 0) {
        out[t] = 0.0f;
        return;
      }
      const int kv_h = hds / group;
      const int seq = slot_p[qi];
      const float* qh = qp + (static_cast<long long>(qi) * n_head + hds) * head_dim;
      for (int j = t; j < width; j += head_dim) {
        const int cell = sel[j];
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot = dot + dequant_at(
                          pages_p,
                          table,
                          lay.page_bytes,
                          max_blocks,
                          lay.block_size,
                          head_dim,
                          lay.groups,
                          lay.k_codes_off,
                          lay.k_scale_off,
                          seq,
                          kv_h,
                          cell,
                          d) *
                      qh[d];
        }
        logits[j] = dot * scale;
      }
      it.barrier(sycl::access::fence_space::local_space);
      float mx = -3.402823466e+38f;
      for (int j = t; j < width; j += head_dim) {
        mx = sycl::fmax(mx, logits[j]);
      }
      red[t] = mx;
      it.barrier(sycl::access::fence_space::local_space);
      for (int s = head_dim / 2; s > 0; s >>= 1) {
        if (t < s) {
          red[t] = sycl::fmax(red[t], red[t + s]);
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      mx = red[0];
      it.barrier(sycl::access::fence_space::local_space);
      float sum = 0.0f;
      for (int j = t; j < width; j += head_dim) {
        const float e = sycl::exp(logits[j] - mx);
        logits[j] = e;
        sum = sum + e;
      }
      red[t] = sum;
      it.barrier(sycl::access::fence_space::local_space);
      for (int s = head_dim / 2; s > 0; s >>= 1) {
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
        const int cell = sel[j];
        acc = acc + w * dequant_at(
                          pages_p,
                          table,
                          lay.page_bytes,
                          max_blocks,
                          lay.block_size,
                          head_dim,
                          lay.groups,
                          lay.v_codes_off,
                          lay.v_scale_off,
                          seq,
                          kv_h,
                          cell,
                          t);
      }
      out[t] = acc;
    });
  });
  return attn;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(b70_qsa, m) {
  m.def("write_step_batch(Tensor pos, Tensor step, int top_k, int block) -> Tensor");
  m.def(
      "kv_append_batch(Tensor k, Tensor v, Tensor pages, Tensor block_table, Tensor step, "
      "int block_size, int idx_block, int idx_dim, int has_pooled) -> Tensor");
  m.def(
      "indexer_append_batch(Tensor raw, Tensor k_norm, Tensor tail, Tensor dead, Tensor block_pos, "
      "Tensor pages, Tensor block_table, Tensor cos, Tensor sin, Tensor step, Tensor mrope, "
      "Tensor pos_base, float eps, int block_size) -> Tensor");
  m.def(
      "index_scores_batch(Tensor q_idx, Tensor pages, Tensor dead, Tensor block_table, Tensor seq_slot, "
      "Tensor scores, Tensor step, Tensor bias, int block_size, int idx_block) -> Tensor");
  m.def("topk_batch(Tensor scores, Tensor ids, Tensor step) -> Tensor");
  m.def(
      "kv_gather_batch(Tensor ids, Tensor pages, Tensor block_table, Tensor seq_slot, Tensor k_out, "
      "Tensor v_out, Tensor step, int block_size, int idx_block, int idx_dim, int has_pooled) -> Tensor");
  m.def("attend_batch(Tensor q, Tensor k_f16, Tensor v_f16, Tensor attn, Tensor step) -> Tensor");
  m.def("gate_batch(Tensor attn, Tensor q_full, Tensor out) -> Tensor");
  m.def("union_build(Tensor ids, Tensor step, Tensor ulist, Tensor ucount) -> Tensor");
  m.def(
      "union_attend(Tensor q, Tensor pages, Tensor block_table, Tensor seq_slot, Tensor ulist, "
      "Tensor ucount, Tensor attn, Tensor step, int block_size, int idx_block, int idx_dim, "
      "int has_pooled) -> Tensor");
}

TORCH_LIBRARY_IMPL(b70_qsa, XPU, m) {
  m.impl("write_step_batch", &write_step_batch);
  m.impl("kv_append_batch", &kv_append_batch);
  m.impl("indexer_append_batch", &indexer_append_batch);
  m.impl("index_scores_batch", &index_scores_batch);
  m.impl("topk_batch", &topk_batch);
  m.impl("kv_gather_batch", &kv_gather_batch);
  m.impl("attend_batch", &attend_batch);
  m.impl("gate_batch", &gate_batch);
  m.impl("union_build", &union_build);
  m.impl("union_attend", &union_attend);
}
