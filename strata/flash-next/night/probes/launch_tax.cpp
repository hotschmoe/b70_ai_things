// Night-lab probe: fixed per-kernel cost on Arc Pro B70 (Level Zero), eager vs SYCL graph replay.
// Build: icpx -fsycl -O2 [-fsycl-targets=spir64_gen -Xs "-device bmg-g31"] launch_tax.cpp -o launch_tax
// Run:   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./launch_tax
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
namespace sx = sycl::ext::oneapi::experimental;
using clk = std::chrono::steady_clock;

static double now_us() { return std::chrono::duration<double, std::micro>(clk::now().time_since_epoch()).count(); }

// tiny dependent kernel: one work-group, reads and writes a few floats
static void tiny(sycl::queue& q, float* p, int wg) {
  q.parallel_for(sycl::nd_range<1>(wg, wg), [=](sycl::nd_item<1> it) {
    int i = it.get_global_id(0);
    p[i] = p[i] * 0.5f + 1.0f;
  });
}

// memory-bound kernel: G work-groups x 256, reads n floats (strided per WG), writes one partial per WG
static void streamk(sycl::queue& q, const float* src, float* dst, size_t n, int groups) {
  q.parallel_for(sycl::nd_range<1>((size_t)groups * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    size_t g = it.get_group(0), l = it.get_local_id(0);
    size_t per = n / groups;
    const sycl::float4* s = reinterpret_cast<const sycl::float4*>(src + g * per);
    size_t n4 = per / 4;
    float acc = 0.f;
    for (size_t i = l; i < n4; i += 256) { sycl::float4 v = s[i]; acc += v.x() + v.y() + v.z() + v.w(); }
    acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
    if (l == 0) dst[g] = acc;
  });
}

template <class F>
static double run_eager(sycl::queue& q, int n, int reps, F&& f) {
  for (int i = 0; i < n; i++) f(q); q.wait();
  double t0 = now_us();
  for (int r = 0; r < reps; r++) { for (int i = 0; i < n; i++) f(q); }
  q.wait();
  return (now_us() - t0) / (double)(n * reps);
}

template <class F>
static double run_graph(sycl::queue& q, int n, int reps, F&& f) {
  sx::command_graph g{q.get_context(), q.get_device()};
  g.begin_recording(q);
  for (int i = 0; i < n; i++) f(q);
  g.end_recording();
  auto ex = g.finalize();
  q.ext_oneapi_graph(ex); q.wait();
  double t0 = now_us();
  for (int r = 0; r < reps; r++) q.ext_oneapi_graph(ex);
  q.wait();
  return (now_us() - t0) / (double)(n * reps);
}

int main() {
  sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order()}};
  auto dev = q.get_device();
  std::printf("device: %s, CUs %u\n", dev.get_info<sycl::info::device::name>().c_str(),
              dev.get_info<sycl::info::device::max_compute_units>());
  size_t N = (size_t)1 << 27;  // 512 MB
  float* big = sycl::malloc_device<float>(N, q);
  float* small = sycl::malloc_device<float>(1 << 20, q);
  float* part = sycl::malloc_device<float>(1 << 16, q);
  q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
    uint32_t x = (uint32_t)i[0] * 2654435761u; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    big[i] = sycl::bit_cast<float>((x & 0x007fffffu) | 0x3f000000u);  // random mantissa, incompressible
  }).wait();
  q.fill(small, 1.0f, 1 << 20).wait();

  // 1) tiny kernels: per-kernel cost
  for (int wg : {16, 256}) {
    double e = run_eager(q, 2000, 5, [&](sycl::queue& qq) { tiny(qq, small, wg); });
    double gph = run_graph(q, 2000, 5, [&](sycl::queue& qq) { tiny(qq, small, wg); });
    std::printf("tiny wg=%d: eager %.2f us/kernel, graph %.2f us/kernel\n", wg, e, gph);
  }
  // 2) memory kernels: bytes x groups sweep, graph replay, chain of 200 (distinct slices so no cache reuse)
  std::printf("\nstream kernels (graph, chain of 200, each reads a distinct slice):\n");
  std::printf("%10s %7s %10s %10s %10s\n", "bytes", "groups", "us/kern", "GB/s", "ideal_us");
  for (size_t bytes : {(size_t)128 << 10, (size_t)512 << 10, (size_t)1 << 20, (size_t)2 << 20, (size_t)4 << 20,
                       (size_t)16 << 20, (size_t)64 << 20}) {
    for (int groups : {32, 128, 512, 2048}) {
      size_t n = bytes / 4;
      if (n / groups < 4 * 16) continue;
      int chain = (int)std::min<size_t>(200, N / n);
      int slot = 0;
      double t = run_graph(q, chain, 5, [&](sycl::queue& qq) {
        streamk(qq, big + (size_t)(slot++ % chain) * n, part, n, groups);
      });
      std::printf("%10zu %7d %10.2f %10.1f %10.2f\n", bytes, groups, t, bytes / t / 1e3, bytes / 608e3);
    }
  }
  // 3) eager with profiling: device execution vs host-observed per kernel for a 1 MB read
  {
    sycl::queue qp{dev, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
    size_t n = (1 << 20) / 4;
    std::vector<sycl::event> evs;
    for (int i = 0; i < 400; i++)
      evs.push_back(qp.parallel_for(sycl::nd_range<1>(128 * 256, 256), [=](sycl::nd_item<1> it) {
        size_t g = it.get_group(0), l = it.get_local_id(0); size_t per = n / 128;
        float acc = 0; for (size_t k = l; k < per; k += 256) acc += big[g * per + k];
        if (acc == 12345.f) part[g] = acc;
      }));
    qp.wait();
    double exec = 0, gap = 0; uint64_t prev_end = 0; int cnt = 0;
    for (auto& e : evs) {
      uint64_t s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      uint64_t en = e.get_profiling_info<sycl::info::event_profiling::command_end>();
      if (prev_end) { gap += (double)(s - prev_end); exec += (double)(en - s); cnt++; }
      prev_end = en;
    }
    std::printf("\neager+profiling 1MB/128wg: device exec %.2f us, gap between kernels %.2f us (n=%d)\n",
                exec / cnt / 1e3, gap / cnt / 1e3, cnt);
  }
  return 0;
}
