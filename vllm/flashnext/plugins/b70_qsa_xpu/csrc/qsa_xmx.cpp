// XMX union QSA attention, mode 1 only.
//
// Port of Strata night-lab STRATA_NIGHT_PATTN_XMX=1, with attribution:
//   sycl/src/kernels/cuda/qsa_prompt_attn_union.dp.cpp  (union_attn)
//   include/strata/kernels/qsa_prompt_attn_union.hpp
// from /mnt/vm_8tb/b70/build/strata-ab/src-ep. The joint_matrix schedule,
// packed B layout, online exp2 softmax, and 8-query / 32-cell tiling are
// that kernel. Mode 2 (S^T = K Q^T, 64-cell chunks) is not ported.
//
// This file does not include Strata headers or dpct. KV addressing is the
// B70 uint8 page, not Strata's flat row
// r = (page * n_kv + head) * page_size + slot.
// local_r = head * block_size + slot. The V row is the K row plus a
// constant byte delta (the K code region). Group-64 fp16 scales are not
// a constant delta from the code row.
//
// Q, K, V, and P enter the products as FP16. The result is not bitwise
// versus the CPU reference or the SIMT union. ok[0] is 1 when this kernel
// is launched and 0 when the shape is not 24/2/256 (the caller uses SIMT).

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>
#include <torch/library.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>

namespace {

namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace jmi = sycl::ext::intel::experimental::matrix;

constexpr int kHd = 256;
constexpr int kG = 12;
constexpr int kTile = 8;
constexpr int kRows = kTile * kG;
constexpr int kSg = 16;
constexpr int kNsg = kRows / 8;
constexpr int kThreads = kSg * kNsg;
constexpr int kCh = 32;
constexpr int kQs = kHd;
constexpr int kSs = kCh + 1;
constexpr int kPs = kCh + 8;
constexpr int kGroup = 64;

struct Lay {
  static constexpr size_t q = (size_t)kRows * kQs * 2;
  static constexpr size_t s = (size_t)kNsg * 8 * kSs * 4;
  static constexpr size_t al = (size_t)kNsg * 8 * 4 * 2;
  static constexpr size_t rows = (size_t)2 * kCh * 8;
  static constexpr size_t mask = (size_t)2 * kCh * 4;
  static constexpr size_t bytes = q + s + al + rows + mask + 16 * 6;
};

static_assert(8 * kPs * 2 <= 8 * kSs * 4, "P fits in place of S");
static_assert(kThreads == 192, "12 sub-groups x 16");

using TA = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major>;
using TB = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed>;
using TC = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16>;

template <typename T>
inline auto lptr(T* p) {
  return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

template <typename T>
inline auto gptr(T* p) {
  return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

void record_use(const torch::Tensor& t) {
  if (!t.defined() || t.numel() == 0 || !t.is_xpu()) {
    return;
  }
  c10::xpu::XPUCachingAllocator::recordStream(t.storage().data_ptr(), c10::xpu::getCurrentXPUStream());
}

// One work-group per (tile, KV head). 12 sub-groups x 8 rows = 8 queries x 12 heads.
void xmx_union_attn(
    sycl::nd_item<3> it,
    const float* __restrict__ q,
    const uint8_t* __restrict__ pages,
    const int32_t* __restrict__ table,
    const int32_t* __restrict__ seq_slot,
    const int32_t* __restrict__ ulist,
    const int32_t* __restrict__ ucount,
    float* __restrict__ attn,
    sycl::half* __restrict__ gstage,
    uint8_t* smem,
    int n_q,
    int cap,
    int n_kv_heads,
    int block_size,
    int max_blocks,
    int num_pages,
    int page_bytes,
    int v_delta,
    int ks_off,
    int vs_off,
    int groups,
    float scale_log2) {
  auto sg = it.get_sub_group();
  uint8_t* at = smem;
  auto take = [&](size_t b) {
    uint8_t* r = at;
    at += (b + 15) & ~(size_t)15;
    return r;
  };
  sycl::half* qs = (sycl::half*)take(Lay::q);
  float* s_all = (float*)take(Lay::s);
  float* al_all = (float*)take(Lay::al);
  long long* srow2 = (long long*)take(Lay::rows);
  uint32_t* smk2 = (uint32_t*)take(Lay::mask);

  const int tile = (int)it.get_group(2);
  const int kvh = (int)it.get_group(1);
  const int t = (int)it.get_local_id(2);
  const int lane = (int)sg.get_local_id()[0];
  const int sgid = (int)sg.get_group_id()[0];
  const int t0 = tile * kTile;
  const int n_head = n_kv_heads * kG;
  const int32_t* list = ulist + (size_t)tile * kTile * cap;
  const int U = ucount[tile];
  const int seq = seq_slot[t0];
  float* sS = s_all + (size_t)sgid * 8 * kSs;
  sycl::half* sP = (sycl::half*)sS;
  float* sA = al_all + (size_t)sgid * 16;
  sycl::half* kp = gstage + ((size_t)tile * n_kv_heads + kvh) * (2 * kCh * kHd);
  sycl::half* vp = kp + kCh * kHd;

  // krow is the byte offset of this cell's K codes. local_r sits above the
  // 8-bit query mask. V is krow + v_delta. An invalid cell stores -1.
  auto entry = [&](int c, long long& krow, uint32_t& packed) {
    krow = -1;
    packed = 0u;
    if (c >= U) {
      return;
    }
    const uint32_t e = (uint32_t)list[c];
    const int cell = (int)(e & 0xFFFFFFu);
    const int logical = cell / block_size;
    const int slot = cell - logical * block_size;
    if (logical < 0 || logical >= max_blocks) {
      return;
    }
    const int phys = table[(long long)seq * max_blocks + logical];
    if (phys < 0 || phys >= num_pages) {
      return;
    }
    const int local = kvh * block_size + slot;
    krow = (long long)phys * page_bytes + (long long)local * kHd;
    packed = (e >> 24) | ((uint32_t)local << 8);
  };

  for (int i = t; i < kRows * (kHd / 4); i += kThreads) {
    const int r = i / (kHd / 4);
    const int d4 = (i % (kHd / 4)) * 4;
    const int j = r / kG;
    const int h = r % kG;
    sycl::float4 v(0.0f);
    if (t0 + j < n_q) {
      v = *reinterpret_cast<const sycl::float4*>(
          q + (size_t)(t0 + j) * n_head * kHd + (size_t)(kvh * kG + h) * kHd + d4);
    }
    sycl::half* dst = qs + (size_t)r * kQs + d4;
    dst[0] = sycl::half(v.x());
    dst[1] = sycl::half(v.y());
    dst[2] = sycl::half(v.z());
    dst[3] = sycl::half(v.w());
  }
  TC O[kHd / 16];
#pragma unroll
  for (int d = 0; d < kHd / 16; ++d) {
    jm::joint_matrix_fill(sg, O[d], 0.0f);
  }
  float m_run = -INFINITY;
  float l_run = 0.0f;
  const int myrow = lane >> 1;
  const int half_ = lane & 1;
  const int qrow_j = (sgid * 8 + myrow) / kG;
  if (t < kCh) {
    long long krow;
    uint32_t packed;
    entry(t, krow, packed);
    srow2[t] = krow;
    smk2[t] = packed;
  }
  it.barrier(sycl::access::fence_space::local_space);
  const sycl::half* qa = qs + (size_t)sgid * 8 * kQs;
  constexpr int kQr = 8;
  TA Qr[kQr];
#pragma unroll
  for (int kk = 0; kk < kQr; ++kk) {
    jm::joint_matrix_load(sg, Qr[kk], lptr(qa + kk * 16), (size_t)kQs);
  }

  for (int c0 = 0, buf = 0; c0 < U; c0 += kCh, buf ^= 1) {
    const long long* srow = srow2 + buf * kCh;
    const uint32_t* smk = smk2 + buf * kCh;
    for (int i = t; i < kCh * (kHd / 16); i += kThreads) {
      const int n = i % kCh;
      const int k0 = (i / kCh) * 16;
      const long long krow = srow[n];
      uint32_t* dst = reinterpret_cast<uint32_t*>(kp) + (size_t)(k0 / 2) * kCh + n;
      if (krow >= 0) {
        const int local = (int)(smk[n] >> 8);
        const uint8_t* page = pages + (krow - (long long)local * kHd);
        const sycl::vec<int8_t, 16> c =
            *reinterpret_cast<const sycl::vec<int8_t, 16>*>(pages + krow + k0);
        const uint16_t bits =
            *reinterpret_cast<const uint16_t*>(page + ks_off + ((size_t)local * groups + (size_t)(k0 / kGroup)) * 2);
        const float sc = (float)sycl::bit_cast<sycl::half>(bits);
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          dst[(size_t)u * kCh] = sycl::bit_cast<uint32_t>(
              sycl::half2((float)c[2 * u] * sc, (float)c[2 * u + 1] * sc));
        }
      } else {
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          dst[(size_t)u * kCh] = 0u;
        }
      }
    }
    for (int i = t; i < (kCh / 2) * (kHd / 16); i += kThreads) {
      const int cp = i / (kHd / 16);
      const int d0 = (i % (kHd / 16)) * 16;
      float v0[16], v1[16];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const long long krow = srow[2 * cp + e];
        float* v = e ? v1 : v0;
        if (krow >= 0) {
          const int local = (int)(smk[2 * cp + e] >> 8);
          const uint8_t* page = pages + (krow - (long long)local * kHd);
          const sycl::vec<int8_t, 16> c =
              *reinterpret_cast<const sycl::vec<int8_t, 16>*>(pages + krow + v_delta + d0);
          const uint16_t bits =
              *reinterpret_cast<const uint16_t*>(page + vs_off + ((size_t)local * groups + (size_t)(d0 / kGroup)) * 2);
          const float sc = (float)sycl::bit_cast<sycl::half>(bits);
#pragma unroll
          for (int u = 0; u < 16; ++u) {
            v[u] = (float)c[u] * sc;
          }
        } else {
#pragma unroll
          for (int u = 0; u < 16; ++u) {
            v[u] = 0.0f;
          }
        }
      }
      sycl::vec<uint32_t, 8> o0, o1;
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        o0[u] = sycl::bit_cast<uint32_t>(sycl::half2(v0[u], v1[u]));
        o1[u] = sycl::bit_cast<uint32_t>(sycl::half2(v0[u + 8], v1[u + 8]));
      }
      uint32_t* dst = reinterpret_cast<uint32_t*>(vp) + (size_t)cp * kHd + d0;
      *reinterpret_cast<sycl::vec<uint32_t, 8>*>(dst) = o0;
      *reinterpret_cast<sycl::vec<uint32_t, 8>*>(dst + 8) = o1;
    }
    it.barrier(sycl::access::fence_space::global_and_local);
    long long nkrow = -1;
    uint32_t npacked = 0u;
    if (t < kCh) {
      entry(c0 + kCh + t, nkrow, npacked);
    }
    {
      TC S0, S1;
      jm::joint_matrix_fill(sg, S0, 0.0f);
      jm::joint_matrix_fill(sg, S1, 0.0f);
      TB Bc0, Bc1, Bn0, Bn1;
      jm::joint_matrix_load(sg, Bc0, gptr(kp), (size_t)(kCh * 2));
      jm::joint_matrix_load(sg, Bc1, gptr(kp + 16 * 2), (size_t)(kCh * 2));
#pragma unroll
      for (int kk = 0; kk < kHd / 16; kk += 2) {
        jm::joint_matrix_load(sg, Bn0, gptr(kp + (size_t)((kk + 1) * 8) * (kCh * 2)), (size_t)(kCh * 2));
        jm::joint_matrix_load(sg, Bn1, gptr(kp + (size_t)((kk + 1) * 8) * (kCh * 2) + 16 * 2), (size_t)(kCh * 2));
        if (kk < kQr) {
          jm::joint_matrix_mad(sg, S0, Qr[kk < kQr ? kk : 0], Bc0, S0);
          jm::joint_matrix_mad(sg, S1, Qr[kk < kQr ? kk : 0], Bc1, S1);
        } else {
          TA A;
          jm::joint_matrix_load(sg, A, lptr(qa + kk * 16), (size_t)kQs);
          jm::joint_matrix_mad(sg, S0, A, Bc0, S0);
          jm::joint_matrix_mad(sg, S1, A, Bc1, S1);
        }
        if (kk + 2 < kHd / 16) {
          jm::joint_matrix_load(sg, Bc0, gptr(kp + (size_t)((kk + 2) * 8) * (kCh * 2)), (size_t)(kCh * 2));
          jm::joint_matrix_load(sg, Bc1, gptr(kp + (size_t)((kk + 2) * 8) * (kCh * 2) + 16 * 2), (size_t)(kCh * 2));
        }
        if (kk + 1 < kQr) {
          jm::joint_matrix_mad(sg, S0, Qr[kk + 1 < kQr ? kk + 1 : 0], Bn0, S0);
          jm::joint_matrix_mad(sg, S1, Qr[kk + 1 < kQr ? kk + 1 : 0], Bn1, S1);
        } else {
          TA A2;
          jm::joint_matrix_load(sg, A2, lptr(qa + (kk + 1) * 16), (size_t)kQs);
          jm::joint_matrix_mad(sg, S0, A2, Bn0, S0);
          jm::joint_matrix_mad(sg, S1, A2, Bn1, S1);
        }
      }
      jm::joint_matrix_store(sg, S0, lptr(sS), (size_t)kSs, jm::layout::row_major);
      jm::joint_matrix_store(sg, S1, lptr(sS + 16), (size_t)kSs, jm::layout::row_major);
    }
    sycl::group_barrier(sg);
    float x[16];
    float mx = -INFINITY;
#pragma unroll
    for (int u = 0; u < 16; ++u) {
      const int c = half_ * 16 + u;
      const bool ok = ((smk[c] & 0xFFu) >> qrow_j) & 1u;
      x[u] = ok ? sS[myrow * kSs + c] * scale_log2 : -INFINITY;
      mx = sycl::fmax(mx, x[u]);
    }
    mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 1));
    const float m_new = sycl::fmax(m_run, mx);
    float alpha = 1.0f;
    float sum = 0.0f;
    sycl::half ph[16];
    if (m_new != -INFINITY) {
      alpha = m_run == -INFINITY ? 0.0f : sycl::exp2(m_run - m_new);
#pragma unroll
      for (int u = 0; u < 16; ++u) {
        const float e = x[u] == -INFINITY ? 0.0f : sycl::exp2(x[u] - m_new);
        ph[u] = sycl::half(e);
        sum += e;
      }
    } else {
#pragma unroll
      for (int u = 0; u < 16; ++u) {
        ph[u] = sycl::half(0.0f);
      }
    }
    sum += sycl::permute_group_by_xor(sg, sum, 1);
    l_run = l_run * alpha + sum;
    m_run = m_new;
    const bool rescale = sycl::any_of_group(sg, alpha != 1.0f);
    sycl::group_barrier(sg);
#pragma unroll
    for (int u = 0; u < 16; ++u) {
      sP[myrow * kPs + half_ * 16 + u] = ph[u];
    }
    if (half_ == 0) {
      sA[myrow] = alpha;
    }
    sycl::group_barrier(sg);
    if (rescale) {
#pragma unroll
      for (int d = 0; d < kHd / 16; ++d) {
        jmi::joint_matrix_apply(sg, O[d], [=](float& v, size_t row, size_t) { v *= sA[row]; });
      }
    }
    {
      TA P0, P1;
      jm::joint_matrix_load(sg, P0, lptr(sP), (size_t)kPs);
      jm::joint_matrix_load(sg, P1, lptr(sP + 16), (size_t)kPs);
      TB Vc0, Vc1, Vn0, Vn1;
      jm::joint_matrix_load(sg, Vc0, gptr(vp), (size_t)(kHd * 2));
      jm::joint_matrix_load(sg, Vc1, gptr(vp + (size_t)8 * (kHd * 2)), (size_t)(kHd * 2));
#pragma unroll
      for (int d = 0; d < kHd / 16; d += 2) {
        jm::joint_matrix_load(sg, Vn0, gptr(vp + (size_t)(d + 1) * 16 * 2), (size_t)(kHd * 2));
        jm::joint_matrix_load(sg, Vn1, gptr(vp + (size_t)8 * (kHd * 2) + (d + 1) * 16 * 2), (size_t)(kHd * 2));
        jm::joint_matrix_mad(sg, O[d], P0, Vc0, O[d]);
        jm::joint_matrix_mad(sg, O[d], P1, Vc1, O[d]);
        if (d + 2 < kHd / 16) {
          jm::joint_matrix_load(sg, Vc0, gptr(vp + (size_t)(d + 2) * 16 * 2), (size_t)(kHd * 2));
          jm::joint_matrix_load(sg, Vc1, gptr(vp + (size_t)8 * (kHd * 2) + (d + 2) * 16 * 2), (size_t)(kHd * 2));
        }
        jm::joint_matrix_mad(sg, O[d + 1], P0, Vn0, O[d + 1]);
        jm::joint_matrix_mad(sg, O[d + 1], P1, Vn1, O[d + 1]);
      }
    }
    if (t < kCh) {
      srow2[(buf ^ 1) * kCh + t] = nkrow;
      smk2[(buf ^ 1) * kCh + t] = npacked;
    }
    it.barrier(sycl::access::fence_space::global_and_local);
  }
  if (half_ == 0) {
    sA[8 + myrow] = l_run;
  }
  sycl::group_barrier(sg);
  const int rbase = sgid * 8;
#pragma unroll
  for (int d = 0; d < kHd / 16; ++d) {
    jmi::joint_matrix_apply(sg, O[d], [=](float& v, size_t row, size_t col) {
      const int rr = rbase + (int)row;
      const int j = rr / kG;
      const int h = rr % kG;
      if (t0 + j < n_q) {
        const float l = sA[8 + row];
        attn[(size_t)(t0 + j) * n_head * kHd + (size_t)(kvh * kG + h) * kHd + d * 16 + col] =
            l > 0.0f ? v / l : 0.0f;
      }
    });
  }
}

class b70_qsa_union_attn_xmx;

torch::Tensor union_attend_xmx(
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
  auto ok = torch::zeros({1}, q_in.options().dtype(torch::kInt32));
  const int n_q = (int)q_in.size(0);
  const int n_head = (int)q_in.size(1);
  const int head_dim = (int)q_in.size(2);
  // SIMT covers every other shape. Do not launch a kernel that would be wrong.
  if (n_q <= 0 || n_head != 24 || head_dim != kHd) {
    return ok;
  }
  TORCH_CHECK(block_size >= 1 && idx_block >= 2 && block_size % idx_block == 0, "block_size");
  TORCH_CHECK(
      attn.is_xpu() && attn.is_contiguous() && attn.scalar_type() == torch::kFloat && attn.sizes() == q_in.sizes(),
      "attn");
  TORCH_CHECK(
      seq_slot.is_xpu() && seq_slot.is_contiguous() && seq_slot.scalar_type() == torch::kInt32 &&
          seq_slot.numel() == n_q,
      "seq_slot");
  TORCH_CHECK(
      step.is_xpu() && step.is_contiguous() && step.scalar_type() == torch::kInt32 && step.dim() == 2 &&
          step.size(0) == n_q && step.size(1) == 4,
      "step");
  const int tiles = (n_q + kTile - 1) / kTile;
  TORCH_CHECK(
      ulist.is_xpu() && ulist.is_contiguous() && ulist.scalar_type() == torch::kInt32 && ulist.dim() == 2 &&
          ulist.size(0) == tiles && ulist.size(1) % kTile == 0,
      "ulist");
  const int cap = (int)ulist.size(1) / kTile;
  TORCH_CHECK(cap >= 1 && cap <= 4096, "cap");
  TORCH_CHECK(
      ucount.is_xpu() && ucount.is_contiguous() && ucount.scalar_type() == torch::kInt32 && ucount.numel() == tiles,
      "ucount");
  TORCH_CHECK(pages.is_xpu() && pages.is_contiguous() && pages.scalar_type() == torch::kUInt8 && pages.dim() == 2, "pages");
  TORCH_CHECK(
      block_table.is_xpu() && block_table.is_contiguous() && block_table.scalar_type() == torch::kInt32 &&
          block_table.dim() == 2,
      "block_table");
  const int page_bytes = (int)pages.size(1);
  const int num_pages = (int)pages.size(0);
  const int max_blocks = (int)block_table.size(1);
  const int bs = (int)block_size;
  const int groups = kHd / kGroup;
  const int pooled_flag = ((int)has_pooled) & 1;
  const int n_states = bs / (int)idx_block;
  const int pooled_bytes = pooled_flag ? n_states * (int)idx_dim * 4 : 0;
  TORCH_CHECK(page_bytes > pooled_bytes && (page_bytes - pooled_bytes) % 4 == 0, "page");
  const int kv_bytes = page_bytes - pooled_bytes;
  const int per_head = bs * (2 * kHd + 4 * groups);
  TORCH_CHECK(per_head > 0 && kv_bytes % per_head == 0, "page does not match head_dim");
  const int n_kv = kv_bytes / per_head;
  if (n_kv != 2 || n_head != n_kv * kG) {
    return ok;
  }
  TORCH_CHECK(page_bytes % 16 == 0, "page alignment");
  const int k_codes = n_kv * bs * kHd;
  const int k_scales = n_kv * bs * groups * 2;
  const int v_delta = k_codes;
  const int ks_off = k_codes + k_codes;
  const int vs_off = ks_off + k_scales;
  TORCH_CHECK(vs_off + k_scales == kv_bytes, "scale region");

  const at::DeviceGuard guard(q_in.device());
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  if (queue.get_device().get_info<sycl::info::device::local_mem_size>() < Lay::bytes) {
    return ok;
  }
  auto stage = torch::empty(
      {(long long)tiles * n_kv * (2 * kCh * kHd)},
      torch::TensorOptions().dtype(torch::kFloat16).device(q_in.device()));
  for (const torch::Tensor& t : {q_in, pages, block_table, seq_slot, ulist, ucount, attn, step, ok, stage}) {
    record_use(t);
  }
  ok.fill_(1);
  const float scale_log2 = 1.4426950408889634f / std::sqrt((float)kHd);
  const float* qp = q_in.data_ptr<float>();
  const uint8_t* pages_p = pages.data_ptr<uint8_t>();
  const int32_t* table = block_table.data_ptr<int32_t>();
  const int32_t* slot_p = seq_slot.data_ptr<int32_t>();
  const int32_t* ulist_p = ulist.data_ptr<int32_t>();
  const int32_t* ucount_p = ucount.data_ptr<int32_t>();
  float* attn_p = attn.data_ptr<float>();
  sycl::half* gstage = reinterpret_cast<sycl::half*>(stage.data_ptr());
  auto nd = sycl::nd_range<3>(
      sycl::range<3>(1, (size_t)n_kv, (size_t)tiles * kThreads), sycl::range<3>(1, 1, kThreads));
  queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(Lay::bytes), h);
    h.parallel_for<b70_qsa_union_attn_xmx>(
        nd,
        sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(kSg)]] {
          xmx_union_attn(
              it,
              qp,
              pages_p,
              table,
              slot_p,
              ulist_p,
              ucount_p,
              attn_p,
              gstage,
              slm.get_multi_ptr<sycl::access::decorated::no>().get(),
              n_q,
              cap,
              n_kv,
              bs,
              max_blocks,
              num_pages,
              page_bytes,
              v_delta,
              ks_off,
              vs_off,
              groups,
              scale_log2);
        });
  });
  return ok;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(b70_qsa, m) {
  m.def(
      "union_attend_xmx(Tensor q, Tensor pages, Tensor block_table, Tensor seq_slot, Tensor ulist, "
      "Tensor ucount, Tensor attn, Tensor step, int block_size, int idx_block, int idx_dim, "
      "int has_pooled) -> Tensor");
}

TORCH_LIBRARY_IMPL(b70_qsa, XPU, m) { m.impl("union_attend_xmx", &union_attend_xmx); }
