// SOURCE ONLY: graph API/ordering/freshness discriminator; no MTP/model math.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_doorbell.hpp"
#include <array>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace s103 {
constexpr size_t N = 257, G = 32, STRIDE = N + 2 * G, WORDS = 6 * STRIDE;
constexpr uint32_t GUARD = 0xa5a5a5a5u;
class Step0; class Step1; class Step2; class Step3; class Step4; class Step5;
static void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
static void save(const fs::path& p, const void* data, size_t bytes) {
    need(!fs::exists(p), "refuse overwrite");
    std::ofstream f(p, std::ios::binary);
    need(bool(f), "open output"); f.write(static_cast<const char*>(data), bytes);
    f.close(); need(bool(f), "close output");
}
static uint32_t input_value(size_t i, unsigned phase) {
    return (uint32_t(i) * 747796405u + (phase == 1 ? 0x1234567bu : 0x91e10da5u)) ^ 0xa341316cu;
}
static uint32_t arithmetic(uint32_t previous, uint32_t input, unsigned step) {
    return previous * 1664525u + (input ^ (step * 0x9e3779b9u)) + 1013904223u;
}
template<unsigned S, class Name>
static void enqueue(sycl::queue& q, const uint32_t* mapped, uint32_t* planes) {
    q.parallel_for<Name>(sycl::nd_range<1>(512, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const size_t i = it.get_global_id(0);
        if (i >= N) return;
        const uint32_t x = strata::sys_load_mapped(mapped + i);
        uint32_t previous = uint32_t(i);
        if constexpr (S > 0) previous = planes[(S - 1) * STRIDE + G + i];
        planes[S * STRIDE + G + i] = previous * 1664525u + (x ^ (S * 0x9e3779b9u)) + 1013904223u;
    });
}
static void operation(unsigned step, sycl::queue& q, const uint32_t* mapped, uint32_t* planes) {
    switch (step) {
    case 0: enqueue<0, Step0>(q, mapped, planes); break;
    case 1: enqueue<1, Step1>(q, mapped, planes); break;
    case 2: enqueue<2, Step2>(q, mapped, planes); break;
    case 3: enqueue<3, Step3>(q, mapped, planes); break;
    case 4: enqueue<4, Step4>(q, mapped, planes); break;
    case 5: enqueue<5, Step5>(q, mapped, planes); break;
    default: throw std::runtime_error("invalid operation");
    }
}
struct Graph {
    sycl::queue& q;
    dpct::experimental::command_graph_ptr graph = nullptr;
    dpct::experimental::command_graph_exec_ptr exec = nullptr;
    explicit Graph(sycl::queue& queue): q(queue) {}
    template<class F> void capture(F record, const fs::path& dot) {
        q.wait_and_throw();
        dpct::experimental::begin_recording(&q);
        record();
        dpct::experimental::end_recording(&q, &graph);
        need(graph != nullptr, "null graph");
        graph->print_graph(dot.string(), true);
        need(fs::file_size(dot) > 0, "empty DOT");
        exec = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());
    }
    void submit() { q.ext_oneapi_graph(*exec); }
    ~Graph() { try { q.wait_and_throw(); delete exec; delete graph; } catch (...) {} }
};
struct Storage {
    sycl::queue& q;
    uint32_t* host = nullptr;
    uint32_t* output = nullptr;
    explicit Storage(sycl::queue& queue): q(queue) {
        host = static_cast<uint32_t*>(strata::host_malloc_polled(STRIDE * sizeof(uint32_t), q));
        need(host != nullptr, "mapped allocation");
        { std::lock_guard<std::mutex> lock(strata::detail::uncached_mu());
          need(strata::detail::uncached_set().count(host) == 1, "require actual zeMemAllocHost uncached allocation, no fallback"); }
        output = sycl::malloc_device<uint32_t>(WORDS, q);
        need(output != nullptr, "device allocation");
        // Exactly one common app queue fill, outside capture/replay.
        q.memset(output, 0xa5, WORDS * sizeof(uint32_t)).wait_and_throw();
    }
    ~Storage() {
        try { q.wait_and_throw(); if (output) sycl::free(output, q); if (host) strata::host_free_polled(host, q); } catch (...) {}
    }
};
static void run(unsigned mode, const fs::path& out) {
    need(dpct::device_count() == 1, "one visible GPU");
    const sycl::device dev = dpct::dev_mgr::instance().get_device(0);
    need(dev.has(sycl::aspect::ext_intel_pci_address), "PCI identity required");
    auto pci = dev.get_info<sycl::ext::intel::info::device::pci_address>();
    std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char x) { return char(std::tolower(x)); });
    need(pci == "0000:0b:00.0", "physical card0 required");
    sycl::queue q(dpct::dev_mgr::instance().get_device(0).get_context(), dev, {sycl::property::queue::in_order()});
    need(q.get_backend() == sycl::backend::ext_oneapi_level_zero, "Level Zero required");
    Storage data(q); // Graphs below die before storage and queue.
    std::vector<std::unique_ptr<Graph>> graphs;
    for (unsigned component = 0; component < (mode ? 1u : 6u); ++component) {
        auto g = std::make_unique<Graph>(q);
        g->capture([&] {
            if (mode) for (unsigned s = 0; s < 6; ++s) operation(s, q, data.host + G, data.output);
            else operation(component, q, data.host + G, data.output);
        }, out / ("capture-" + std::to_string(component) + ".dot"));
        graphs.push_back(std::move(g));
    }
    std::vector<uint32_t> first;
    for (unsigned phase = 0; phase < 3; ++phase) {
        q.wait_and_throw(); // Prior replay and host readback are complete before mutation.
        std::vector<uint32_t> host_expected(STRIDE, GUARD);
        for (size_t i = 0; i < N; ++i) host_expected[G + i] = input_value(i, phase);
        std::memcpy(data.host, host_expected.data(), STRIDE * sizeof(uint32_t));
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (auto& g: graphs) g->submit(); // Mode0 six, mode1 one; no intervening waits.
        q.wait_and_throw();
        std::vector<uint32_t> actual(WORDS), expected(WORDS, GUARD);
        // Exactly one common full-output readback per A/B/A replay.
        q.memcpy(actual.data(), data.output, WORDS * sizeof(uint32_t)).wait_and_throw();
        need(std::memcmp(data.host, host_expected.data(), STRIDE * sizeof(uint32_t)) == 0, "mapped input/guards changed");
        for (size_t i = 0; i < N; ++i) {
            uint32_t v = uint32_t(i);
            for (unsigned s = 0; s < 6; ++s) {
                v = arithmetic(v, host_expected[G + i], s);
                expected[s * STRIDE + G + i] = v;
            }
        }
        need(actual == expected, "strict full-output/intermediate ordering/guard mismatch");
        if (phase == 0) first = actual;
        if (phase == 1) need(actual != first, "B must change output");
        if (phase == 2) need(actual == first, "restored A differs");
        save(out / ("phase-" + std::to_string(phase) + ".bin"), actual.data(), WORDS * sizeof(uint32_t));
        std::printf("{\"phase\":%u,\"mode\":%u,\"full_bytes\":%zu,\"strict\":true,\"readonly\":true,\"guards\":true,\"six_step_order\":true}\n", phase, mode, WORDS * sizeof(uint32_t));
    }
    q.wait_and_throw();
    const std::string result = "{\"status\":\"PASS_GRAPH_API_FRESHNESS_ONLY\",\"mode\":" + std::to_string(mode) + ",\"graphs\":" + std::to_string(graphs.size()) + ",\"replays\":3,\"kernel_operations\":18,\"application_graph_submits\":" + std::to_string(mode ? 3 : 18) + ",\"application_setup_fills\":1,\"application_readbacks\":3,\"mapped_A_B_A\":true,\"device_pci\":\"0000:0b:00.0\",\"MTP_numeric_qualified\":false,\"speed_qualified\":false}\n";
    save(out / "result.json", result.data(), result.size());
}
} // namespace s103
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        s103::need(argc == 3, "usage: paired_graph_s103 MODE NEW_OUT");
        s103::need(std::string(argv[1]) == "0" || std::string(argv[1]) == "1", "mode0/1 only");
        s103::need(fs::create_directory(argv[2]), "new output directory required");
        s103::run(unsigned(argv[1][0] - '0'), argv[2]);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "S103 STOP: %s\n", e.what()); return 5; }
}
