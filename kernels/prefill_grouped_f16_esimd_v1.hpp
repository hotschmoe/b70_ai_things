// Source-only bounded prototype: original FP16 operands, FP32 output, alpha1/beta0.
// No oneMKL metadata, host_task, allocation, normal wait, SwiGLU or scatter change.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace b70::prefill_esimd_v1 {
namespace esimd = sycl::ext::intel::esimd;
struct Group {
    const sycl::half* x;
    const sycl::half* w;  // Original dequantized W[N][K], not permanently repacked.
    float* y;
    int rows;
    std::uint64_t first_tile;
    std::uint64_t end_tile;
};
struct Batch { std::array<Group,8> group{}; int count; int n; int k; };
static_assert(std::is_trivially_copyable_v<Batch>);

// Caller owns USM operands in q's context until the returned event completes.
// q must be the SAME in-order compute queue used by the dequant/SwiGLU/down ring.
// This bounded prototype accepts actual GU/down shapes plus two small operators.
inline sycl::event launch(sycl::queue& q,const std::uint16_t*const* x,
                          const std::uint16_t*const* w,float*const* y,
                          const std::int64_t* rows,int count,int n,int k) {
    if (!q.is_in_order() || !x || !w || !y || !rows || count<1 || count>8 ||
        !((n==1280 && k==2560)||(n==2560 && k==640)||
          (n==16 && k==16)||(n==32 && k==32)))
        throw std::invalid_argument("bounded direct F16 DPAS shape/queue required");
    Batch batch{}; batch.count=count; batch.n=n; batch.k=k;
    std::uint64_t total=0;
    for (int g=0;g<count;++g) {
        if (!x[g] || !w[g] || !y[g] || rows[g]<1 || rows[g]>64 ||
            (reinterpret_cast<std::uintptr_t>(x[g])&15) ||
            (reinterpret_cast<std::uintptr_t>(w[g])&15) ||
            (reinterpret_cast<std::uintptr_t>(y[g])&15))
            throw std::invalid_argument("owned aligned USM rows1..64 required");
        const auto next=total+std::uint64_t((rows[g]+7)/8)*(n/16);
        batch.group[g]={reinterpret_cast<const sycl::half*>(x[g]),
                        reinterpret_cast<const sycl::half*>(w[g]),y[g],
                        int(rows[g]),total,next};
        total=next;
    }
    // All pointer/shape descriptors are kernel captures BY VALUE. The ordinary
    // caller arrays may disappear after launch; pointed-to ring buffers may not.
    return q.parallel_for(sycl::range<1>(total),
        [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
            const std::uint64_t tile=id[0]; int selected=0;
            for (int i=0;i<batch.count;++i)
                if (tile>=batch.group[i].first_tile && tile<batch.group[i].end_tile)
                    selected=i;
            const Group g=batch.group[selected];
            const std::uint64_t local=tile-g.first_tile;
            const int m0=int(local/(batch.n/16))*8;
            const int n0=int(local%(batch.n/16))*16;
            esimd::simd<float,128> acc(0.0f);
            for (int k0=0;k0<batch.k;k0+=16) {
                esimd::simd<sycl::half,128> a(sycl::half(0.0f));
                esimd::simd<sycl::half,256> b(sycl::half(0.0f));
#pragma unroll
                for (int r=0;r<8;++r) {
                    if (m0+r<g.rows) {
                        auto xr=esimd::block_load<sycl::half,16>(
                            g.x+std::int64_t(m0+r)*batch.k+k0);
                        a.template select<16,1>(r*16)=xr;
                    }
                }
#pragma unroll
                for (int c=0;c<16;++c) {
                    auto wr=esimd::block_load<sycl::half,16>(
                        g.w+std::int64_t(n0+c)*batch.k+k0);
#pragma unroll
                    for (int kk=0;kk<16;++kk)
                        b[(kk/2)*32+c*2+(kk%2)]=wr[kk];
                }
                // A:8x16 row-major. B:16x16 vertical FP16 pairs (VNNI).
                // Current Intel header deduces fp16 precision from sycl::half.
                acc=esimd::xmx::dpas<8,8,float>(acc,b,a);
            }
#pragma unroll
            for (int r=0;r<8;++r) {
                if (m0+r<g.rows) {
                    esimd::simd<float,16> result=acc.template select<16,1>(r*16);
                    esimd::block_store<float,16>(
                        g.y+std::int64_t(m0+r)*batch.n+n0,result);
                }
            }
        });
}
} // namespace b70::prefill_esimd_v1
