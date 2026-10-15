#define DPCT_PROFILING_ENABLED
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace pf = strata::prefill;
namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
constexpr int64_t N = 2560, FF = 640, C = 12288, K = 10;
constexpr size_t GUARD = 256;
static void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
struct Buffer {
    sycl::queue& q; size_t bytes; uint8_t* base; const char* name;
    Buffer(sycl::queue& queue, size_t n, const char* tag) : q(queue), bytes(n), name(tag) {
        base = sycl::malloc_device<uint8_t>(bytes + 2 * GUARD, q);
        require(base != nullptr, "device allocation failed");
        q.memset(base, 0xD7, bytes + 2 * GUARD);
    }
    void* data() const { return base + GUARD; }
    template<class T> T* as() const { return reinterpret_cast<T*>(base + GUARD); }
    void check() const {
        std::array<uint8_t, GUARD> a{}, b{};
        q.memcpy(a.data(), base, GUARD);
        q.memcpy(b.data(), base + GUARD + bytes, GUARD).wait_and_throw();
        for (size_t i = 0; i < GUARD; ++i)
            if (a[i] != 0xD7 || b[i] != 0xD7) throw std::runtime_error(std::string("guard: ") + name);
    }
    ~Buffer() { try { q.wait_and_throw(); sycl::free(base, q); } catch (...) {} }
};
static void poison_tail(const Buffer& b, size_t from, uint8_t poison) {
    std::vector<uint8_t> h(1 << 20);
    for (size_t at = from; at < b.bytes; at += h.size()) {
        const size_t n = std::min(h.size(), b.bytes - at);
        b.q.memcpy(h.data(), b.as<uint8_t>() + at, n).wait_and_throw();
        for (size_t i = 0; i < n; ++i) require(h[i] == poison, "unused-capacity overwrite");
    }
}
struct Layer { int number; cpu::NativeFmt f; const strata::TensorInfo* t[3]; };
struct Expert { int layer, expert; cpu::NativeFmt f; size_t blob; int64_t offset, ne; };
struct Drain { sycl::queue& q; ~Drain() { try { q.wait_and_throw(); } catch (...) {} } };
struct Timing { double enqueue, drain, complete, device_span; };

static int run(int argc, char** argv) {
    require(argc == 4 && std::string(argv[2]) == "--mean", "usage: peer_gather_oracle shard1.gguf --mean 80|240");
    const int mean = std::atoi(argv[3]);
    require(mean == 80 || mean == 240, "mean must be 80 or 240");
    const int selected = 363;
    const int64_t T = mean == 80 ? 4096 : 12288;
    strata::GgufFile gguf(argv[1]);
    std::map<std::pair<int,int>, Layer> formats;
    for (int l = 0; l < 48; ++l) {
        Layer a{}; a.number = l;
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& t : gguf.tensors())
            for (int r = 0; r < 3; ++r)
                if (t.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") a.t[r] = &t;
        require(a.t[0] && a.t[1] && a.t[2], "expert tensor missing from shard1");
        std::string err;
        require(a.t[0]->type == a.t[1]->type, "gate/up format mismatch");
        for (int r = 0; r < 3; ++r) {
            const auto& dims = a.t[r]->shape;
            require(dims.size() == 3 && dims[2] == 512, "bad expert tensor rank/count");
            require(dims[0] == uint64_t(r == 2 ? FF : N) && dims[1] == uint64_t(r == 2 ? N : FF), "bad expert tensor shape");
        }
        if (!cpu::native_fmt(int(a.t[0]->type), int(a.t[2]->type), N, FF, a.f, err)) throw std::runtime_error(err);
        formats.emplace(std::make_pair(a.f.gu_type, a.f.d_type), a);
    }
    require(formats.size() == 10, "expected ten real format pairs");
    for (int gt : {16,17,18,21,22}) for (int dt : {20,42})
        require(formats.count({gt,dt}) == 1, "expected real quant format pair absent");
    std::vector<Layer> layers;
    for (const auto& a : formats) layers.push_back(a.second);
    std::vector<int64_t> sizes(selected, 1);
    const int64_t rows = int64_t(selected) * mean;
    sizes[0] = 8 * mean + 1; sizes[1] = 1;
    int64_t left = rows - sizes[0] - 1 - (selected - 2);
    int64_t weight_sum = 0;
    for (int j = 2; j < selected; ++j) weight_sum += 1 + (j * 17) % 7;
    int64_t assigned = 0;
    for (int j = 2; j < selected; ++j) {
        const int64_t add = left * (1 + (j * 17) % 7) / weight_sum;
        sizes[j] += add; assigned += add;
    }
    for (int j = 2; assigned < left; ++j, ++assigned) ++sizes[j];
    require(*std::max_element(sizes.begin(), sizes.end()) == sizes[0] && sizes[1] == 1, "skew fixture failed");
    std::vector<Expert> experts;
    std::vector<uint8_t> blobs;
    int64_t offset = 0, max_ne = 0;
    for (int j = 0; j < selected; ++j) {
        const auto& a = layers[j % layers.size()];
        const int e = (j * 17 + 7) % 512;
        const size_t pos = (blobs.size() + 255) & ~size_t(255);
        blobs.resize(pos + a.f.bytes, 0xC3);
        std::memcpy(blobs.data() + pos, gguf.tensor_data(*a.t[0]) + size_t(e) * a.f.up_off, a.f.up_off);
        std::memcpy(blobs.data() + pos + a.f.up_off, gguf.tensor_data(*a.t[1]) + size_t(e) * a.f.up_off, a.f.up_off);
        std::memcpy(blobs.data() + pos + a.f.down_off, gguf.tensor_data(*a.t[2]) + size_t(e) * (a.f.bytes - a.f.down_off), a.f.bytes - a.f.down_off);
        experts.push_back({a.number, e, a.f, pos, offset, sizes[j]});
        offset += sizes[j]; max_ne = std::max(max_ne, sizes[j]);
        std::printf("{\"expert\":%d,\"layer\":%d,\"physical_expert\":%d,\"gu\":%d,\"down\":%d,\"offset\":%lld,\"ne\":%lld}\n", j, a.number, e, a.f.gu_type, a.f.d_type, (long long) experts.back().offset, (long long) sizes[j]);
    }
    require(offset == rows && rows <= T * K && max_ne <= T, "row/T capacity failed");
    const size_t out_capacity = size_t(C * K * N) * sizeof(float);
    const size_t out_used = size_t(rows * N) * sizeof(float);
    require(size_t(rows * N) * sizeof(uint16_t) <= out_capacity && out_used <= out_capacity, "alias capacity failed");
    require(dpct::device_count() == 1, "need exactly one exposed GPU");
    const sycl::device dev = dpct::dev_mgr::instance().get_device(0);
    require(dev.has(sycl::aspect::ext_intel_pci_address), "PCI address unavailable");
    std::string pci = dev.get_info<sycl::ext::intel::info::device::pci_address>();
    std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    require(pci == "0000:44:00.0", "wrong card: require physical card1 PCI0000:44:00.0");
    const sycl::property_list props{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
    sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(), dev, props);
    require(q.is_in_order(), "queue must be in_order");
    std::vector<uint16_t> source(size_t(T * N));
    for (size_t i = 0; i < source.size(); ++i) source[i] = uint16_t(0x2800 + (i * 13 + i / N * 17) % 1024) | uint16_t((i % 3 == 0) ? 0x8000 : 0);
    std::vector<int32_t> src(size_t(rows), 0);
    for (int64_t i = 0; i < rows; ++i) src[size_t(i)] = int32_t((i * 37 + i / 7 * 13 + 17) % T);
    Buffer weights(q, blobs.size(), "weights"), input(q, source.size() * 2, "input"), ids(q, src.size() * 4, "src");
    Buffer output(q, out_capacity, "rows_reg"), xs(q, size_t(T * N) * 2, "Xs");
    Buffer gu(q, size_t(T * 1280) * 4, "GUf"), h(q, size_t(T * FF) * 2, "Hh");
    Buffer dqgu(q, size_t(1280 * N) * 2, "dq_gu"), dqd(q, size_t(N * FF) * 2, "dq_d");
    std::array<Buffer*,9> buffers{&weights,&input,&ids,&output,&xs,&gu,&h,&dqgu,&dqd};
    require(reinterpret_cast<uintptr_t>(output.data()) % 16 == 0, "output alignment failed");
    pf::Gemm gemm; std::string err;
    if (!gemm.init(&q, 1 << 20, err)) throw std::runtime_error(err);
    Drain drain{q};
    q.memcpy(weights.data(), blobs.data(), blobs.size());
    q.memcpy(input.data(), source.data(), source.size() * 2);
    q.memcpy(ids.data(), src.data(), src.size() * 4);
    q.wait_and_throw();
    // Byte oracle is separate: no extra host wait/read in the candidate compute arm.
    q.memset(output.data(), 0xA1, out_capacity);
    pf::gather_rows16(input.as<uint16_t>(), ids.as<int32_t>(), output.as<uint16_t>(), rows, N, &q);
    std::vector<uint16_t> gathered(size_t(rows * N));
    q.memcpy(gathered.data(), output.data(), gathered.size() * 2).wait_and_throw();
    for (int64_t r = 0; r < rows; ++r)
        require(std::memcmp(gathered.data() + r * N, source.data() + int64_t(src[size_t(r)]) * N, size_t(N) * 2) == 0, "all-row gather byte mismatch");
    poison_tail(output, gathered.size() * 2, 0xA1);
    for (const auto* b : buffers) b->check();
    gathered.clear(); gathered.shrink_to_fit();
    auto compute = [&](const Expert& e, const uint16_t* x) {
        const auto* b = weights.as<uint8_t>() + e.blob;
        strata::kernels::iq_dequant_expert_f16(e.f.gu_type, e.f.d_type, b, b + e.f.up_off, b + e.f.down_off, FF, N, dqgu.as<uint16_t>(), dqd.as<uint16_t>(), &q);
        gemm.f16(x, dqgu.as<uint16_t>(), gu.as<float>(), e.ne, 1280, N);
        pf::swiglu_interleaved(gu.as<float>(), h.as<uint16_t>(), e.ne, &q);
        gemm.f16(h.as<uint16_t>(), dqd.as<uint16_t>(), output.as<float>() + e.offset * N, e.ne, N, FF);
    };
    auto arm = [&](bool reverse, std::vector<float>& got) -> Timing {
        const uint8_t poison = reverse ? 0xA5 : 0x5A;
        q.memset(output.data(), poison, out_capacity).wait_and_throw();
        const auto t0 = Clock::now();
        const auto begin = q.ext_oneapi_submit_barrier();
        if (reverse) {
            pf::gather_rows16(input.as<uint16_t>(), ids.as<int32_t>(), output.as<uint16_t>(), rows, N, &q);
            for (auto it = experts.rbegin(); it != experts.rend(); ++it)
                compute(*it, output.as<uint16_t>() + it->offset * N);
        } else {
            for (const auto& e : experts) {
                pf::gather_rows16(input.as<uint16_t>(), ids.as<int32_t>() + e.offset, xs.as<uint16_t>(), e.ne, N, &q);
                compute(e, xs.as<uint16_t>());
            }
        }
        const auto finish = q.ext_oneapi_submit_barrier();
        const auto te = Clock::now();
        q.wait_and_throw();
        const auto td = Clock::now();
        double span = -1.0;
        try {
            span = double(finish.get_profiling_info<sycl::info::event_profiling::command_end>() - begin.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1e6;
        } catch (const sycl::exception&) {
            std::fprintf(stderr, "WARN device-marker profiling unavailable; correctness checks continue\n");
        }
        got.resize(size_t(rows * N));
        q.memcpy(got.data(), output.data(), out_used).wait_and_throw();
        for (float x : got) require(std::isfinite(x), "nonfinite output");
        poison_tail(output, out_used, poison);
        for (const auto* b : buffers) b->check();
        return {ms(t0,te), ms(te,td), ms(t0,td), span};
    };
    std::vector<float> baseline, candidate;
    const Timing a = arm(false, baseline), b = arm(true, candidate);
    size_t mismatches = 0, first = baseline.size();
    for (size_t i = 0; i < baseline.size(); ++i)
        if (std::memcmp(&baseline[i], &candidate[i], sizeof(float))) { ++mismatches; first = std::min(first,i); }
    auto timing = [](const char* tag, const Timing& t) {
        char device_span[64] = "null";
        if (t.device_span >= 0.0) std::snprintf(device_span, sizeof(device_span), "%.6f", t.device_span);
        std::printf("{\"arm\":\"%s\",\"host_enqueue_ms\":%.6f,\"drain_wait_ms\":%.6f,\"queue_completion_wall_ms\":%.6f,\"device_marker_span_ms\":%s}\n", tag, t.enqueue, t.drain, t.complete, device_span);
    };
    timing("baseline_forward",a); timing("all_gather_reverse_alias",b);
    std::printf("{\"status\":\"%s\",\"mean\":%d,\"T\":%lld,\"allocated_C\":%lld,\"selected\":%d,\"rows\":%lld,\"max_ne\":%lld,\"format_pairs\":%zu,\"output_capacity_bytes\":%zu,\"output_used_bytes\":%zu,\"weight_bytes\":%zu,\"gather_bytes_PASS\":true,\"finites_PASS\":true,\"guards_PASS\":true,\"FP32_bit_mismatches\":%zu,\"first_mismatch_element\":%zu}\n", mismatches ? "FAIL" : "PASS", mean, (long long)T, (long long)C, selected, (long long)rows, (long long)max_ne, formats.size(), out_capacity, out_used, blobs.size(), mismatches, first);
    q.wait_and_throw();
    return mismatches ? 5 : 0;
}
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try { return run(argc,argv); }
    catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 2; }
}
