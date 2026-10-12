// NEW pair diagnostic reuses the frozen single-device byte/ring fixture.
#define main source52_single_device_main
#include "copy_engine_prefill52_harness_v1.cpp"
#undef main
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <dlfcn.h>
#include <variant>
#include <vector>

struct PairOwned {
    sycl::queue* q[2] = {};
    ~PairOwned() {
        for (auto*& p : q) if (p) {
            try { p->wait_and_throw(); strata::release_copy_engine_queue(p); p = nullptr; }
            catch (const std::exception& e) { std::fprintf(stderr, "FAIL source52 pair drain; native owner retained: %s\n", e.what()); }
        }
    }
};

static void route(sycl::queue& q, unsigned device) {
    auto native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);
    if (!std::holds_alternative<ze_command_list_handle_t>(native))
        throw std::runtime_error("queue is not an immediate native command list");
    const auto cl = std::get<ze_command_list_handle_t>(native);
    void* loader = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!loader) throw std::runtime_error("native loader absent");
    using GetOrdinal = ze_result_t (ZE_APICALL *)(ze_command_list_handle_t, uint32_t*);
    using GetDevice = ze_result_t (ZE_APICALL *)(ze_command_list_handle_t, ze_device_handle_t*);
    auto get_ordinal = (GetOrdinal)dlsym(loader, "zeCommandListGetOrdinal");
    auto get_device = (GetDevice)dlsym(loader, "zeCommandListGetDeviceHandle");
    uint32_t ordinal = UINT32_MAX;
    ze_device_handle_t actual = nullptr;
    if (!get_ordinal || !get_device || get_ordinal(cl, &ordinal) != ZE_RESULT_SUCCESS ||
        get_device(cl, &actual) != ZE_RESULT_SUCCESS)
        throw std::runtime_error("native queue route query failed");
    const auto expected = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    if (actual != expected) throw std::runtime_error("native queue belongs to another device");
    uint32_t count = 0;
    if (zeDeviceGetCommandQueueGroupProperties(actual, &count, nullptr) != ZE_RESULT_SUCCESS)
        throw std::runtime_error("group count query failed");
    std::vector<ze_command_queue_group_properties_t> groups(count);
    for (auto& g : groups) { g = {}; g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES; }
    if (zeDeviceGetCommandQueueGroupProperties(actual, &count, groups.data()) != ZE_RESULT_SUCCESS || ordinal >= count)
        throw std::runtime_error("native ordinal outside enumerated groups");
    const auto flags = groups[ordinal].flags;
    if (!(flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) ||
        (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) || !groups[ordinal].numQueues)
        throw std::runtime_error("native ordinal is not a copy-only engine");
    dlclose(loader);
    std::printf("ROUTE source52 device=%u ordinal=%u flags=%u copy_only=1 device_match=1\n", device, ordinal, unsigned(flags));
}

static void small_copy(sycl::queue& q) {
    uint64_t* host = sycl::malloc_host<uint64_t>(1, q);
    uint64_t* dev = sycl::malloc_device<uint64_t>(1, q);
    if (!host || !dev) throw std::bad_alloc();
    *host = 0x123456789abcdef0ull;
    q.memcpy(dev, host, sizeof(uint64_t)).wait_and_throw();
    *host = 0;
    q.memcpy(host, dev, sizeof(uint64_t)).wait_and_throw();
    const bool matches = *host == 0x123456789abcdef0ull;
    sycl::free(dev, q); sycl::free(host, q);
    if (!matches) throw std::runtime_error("surviving/reopened device copy differs");
}

int main(int argc, char** argv) try {
    if (argc != 2 || std::strcmp(argv[1], "--synthetic")) return 2;
    if (dpct::device_count() != 2) throw std::runtime_error("exactly two visible devices required");
    const char* flag = std::getenv("STRATA_COPY_ENGINE");
    const bool on = flag && !std::strcmp(flag, "1");
    for (unsigned i = 0; i < 2; ++i) {
        dpct::select_device(i);
        if (source52_single_device_main(argc, argv)) throw std::runtime_error("device ring fixture failed");
    }
    if (on) {
        PairOwned owned;
        auto& q = owned.q;
        for (unsigned i = 0; i < 2; ++i) {
            dpct::select_device(i);
            q[i] = strata::make_copy_engine_queue("source52 pair lifecycle");
            if (!q[i]) throw std::runtime_error("pair queue unavailable");
            route(*q[i], i);
            auto* second = strata::make_copy_engine_queue("source52 pair second owner");
            if (second != q[i]) throw std::runtime_error("pair shared owner differs");
            strata::release_copy_engine_queue(second);
        }
        if (q[0] == q[1] || q[0]->get_device() == q[1]->get_device())
            throw std::runtime_error("devices share a native queue");
        strata::release_copy_engine_queue(q[0]); q[0] = nullptr;
        if (!strata::copy_engine_active() || !strata::is_copy_engine_queue(q[1]))
            throw std::runtime_error("releasing device0 retired device1");
        small_copy(*q[1]);
        dpct::select_device(0);
        q[0] = strata::make_copy_engine_queue("source52 pair reopened device0");
        if (!q[0]) throw std::runtime_error("device0 recreate failed");
        route(*q[0], 0); small_copy(*q[0]);
        strata::release_copy_engine_queue(q[1]); q[1] = nullptr;
        if (!strata::copy_engine_active() || !strata::is_copy_engine_queue(q[0]))
            throw std::runtime_error("releasing device1 retired device0");
        small_copy(*q[0]);
        strata::release_copy_engine_queue(q[0]); q[0] = nullptr;
    }
    if (strata::copy_engine_active()) throw std::runtime_error("final pair queue not retired");
    std::printf("PASS source52 pair copy_engine=%d devices=2 ring_bytes_exact=1 native_routes=%d cross_device_retirement=1\n", int(on), on ? 3 : 0);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL source52 pair: %s\n", e.what());
    return 1;
}
