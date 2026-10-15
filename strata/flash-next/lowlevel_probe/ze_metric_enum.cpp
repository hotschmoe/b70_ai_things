// ze_metric_enum.cpp - Level Zero metric CAPABILITY enumerator (read-only; no sampling, no streamer, no driver change).
//
// Lists, for every Level Zero GPU device visible after ZE_AFFINITY_MASK: the device identity (name, PCI BDF, timer
// resolution, timestamp valid bits), the driver's extension names, every zet metric group (name, description,
// sampling type, domain) and every metric in it (name, component, type, result type, units).  Nothing is hardcoded:
// whatever the metrics library reports is printed.  Also samples zeDeviceGetGlobalTimestamps a few times so the
// host/device clock relation and the device timestamp resolution are visible.
//
//   ZET_ENABLE_METRICS=1 ZE_AFFINITY_MASK=N ONEAPI_DEVICE_SELECTOR=level_zero:* ./ze_metric_enum [--json out.json]
//
// Without ZET_ENABLE_METRICS=1 the metrics library is not loaded and zetMetricGroupGet reports 0 groups (or an
// error); the tool says so instead of failing silently.  Exit code 0 when every call succeeded, 1 otherwise.
#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
const char* zr(ze_result_t r) {
    switch (r) {
        case ZE_RESULT_SUCCESS: return "SUCCESS";
        case ZE_RESULT_ERROR_UNINITIALIZED: return "ERROR_UNINITIALIZED";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "ERROR_UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "ERROR_INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE: return "ERROR_DEPENDENCY_UNAVAILABLE";
        case ZE_RESULT_ERROR_NOT_AVAILABLE: return "ERROR_NOT_AVAILABLE";
        case ZE_RESULT_ERROR_UNSUPPORTED_VERSION: return "ERROR_UNSUPPORTED_VERSION";
        case ZE_RESULT_ERROR_DEVICE_LOST: return "ERROR_DEVICE_LOST";
        default: return "ERROR";
    }
}
bool ok(ze_result_t r, const char* what) {
    if (r == ZE_RESULT_SUCCESS) return true;
    std::fprintf(stderr, "ze_metric_enum: %s -> %s (0x%x)\n", what, zr(r), (unsigned) r);
    ++g_fail;
    return false;
}
const char* sampling_type(uint32_t f) {
    static std::string s;
    s.clear();
    if (f & ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED) s += "EVENT_BASED|";
    if (f & ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED) s += "TIME_BASED|";
    if (f & ~(ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED | ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED))
        s += "OTHER(0x" + std::to_string(f) + ")|";
    if (s.empty()) s = "NONE";
    else s.pop_back();
    return s.c_str();
}
const char* metric_type(zet_metric_type_t t) {
    switch (t) {
        case ZET_METRIC_TYPE_DURATION: return "DURATION";
        case ZET_METRIC_TYPE_EVENT: return "EVENT";
        case ZET_METRIC_TYPE_EVENT_WITH_RANGE: return "EVENT_WITH_RANGE";
        case ZET_METRIC_TYPE_THROUGHPUT: return "THROUGHPUT";
        case ZET_METRIC_TYPE_TIMESTAMP: return "TIMESTAMP";
        case ZET_METRIC_TYPE_FLAG: return "FLAG";
        case ZET_METRIC_TYPE_RATIO: return "RATIO";
        case ZET_METRIC_TYPE_RAW: return "RAW";
        default: return "OTHER";
    }
}
const char* value_type(zet_value_type_t t) {
    switch (t) {
        case ZET_VALUE_TYPE_UINT32: return "UINT32";
        case ZET_VALUE_TYPE_UINT64: return "UINT64";
        case ZET_VALUE_TYPE_FLOAT32: return "FLOAT32";
        case ZET_VALUE_TYPE_FLOAT64: return "FLOAT64";
        case ZET_VALUE_TYPE_BOOL8: return "BOOL8";
        default: return "OTHER";
    }
}
std::string jesc(const char* s) {
    std::string o;
    for (const char* p = s; *p; ++p) {
        if (*p == '"' || *p == '\\') { o += '\\'; o += *p; }
        else if (*p == '\n') o += "\\n";
        else if ((unsigned char) *p < 0x20) o += ' ';
        else o += *p;
    }
    return o;
}
}  // namespace

int main(int argc, char** argv) {
    const char* json_path = nullptr;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    const char* em = std::getenv("ZET_ENABLE_METRICS");
    const char* am = std::getenv("ZE_AFFINITY_MASK");
    std::printf("ze_metric_enum: ZET_ENABLE_METRICS=%s ZE_AFFINITY_MASK=%s ONEAPI_DEVICE_SELECTOR=%s\n",
                em ? em : "(unset: metric groups will not be reported)", am ? am : "(unset)",
                std::getenv("ONEAPI_DEVICE_SELECTOR") ? std::getenv("ONEAPI_DEVICE_SELECTOR") : "(unset)");
    if (!ok(zeInit(ZE_INIT_FLAG_GPU_ONLY), "zeInit")) return 1;
    uint32_t nd = 0;
    if (!ok(zeDriverGet(&nd, nullptr), "zeDriverGet(count)")) return 1;
    std::vector<ze_driver_handle_t> drivers(nd);
    if (nd > 0) ok(zeDriverGet(&nd, drivers.data()), "zeDriverGet");
    std::printf("drivers: %u\n", nd);
    std::string js = "{\"drivers\":[";
    bool first_drv = true;
    for (uint32_t di = 0; di < nd; ++di) {
        ze_driver_handle_t drv = drivers[di];
        ze_driver_properties_t dp = {ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES};
        ok(zeDriverGetProperties(drv, &dp), "zeDriverGetProperties");
        ze_api_version_t ver = ZE_API_VERSION_CURRENT;
        ok(zeDriverGetApiVersion(drv, &ver), "zeDriverGetApiVersion");
        std::printf("driver %u: version 0x%x api %u.%u\n", di, dp.driverVersion, ZE_MAJOR_VERSION(ver), ZE_MINOR_VERSION(ver));
        uint32_t ne = 0;
        ok(zeDriverGetExtensionProperties(drv, &ne, nullptr), "zeDriverGetExtensionProperties(count)");
        std::vector<ze_driver_extension_properties_t> ext(ne);
        if (ne) ok(zeDriverGetExtensionProperties(drv, &ne, ext.data()), "zeDriverGetExtensionProperties");
        js += (first_drv ? "" : ",");
        first_drv = false;
        js += "{\"driverVersion\":" + std::to_string(dp.driverVersion) + ",\"extensions\":[";
        std::printf("  extensions (%u):", ne);
        for (uint32_t i = 0; i < ne; ++i) {
            std::printf(" %s(0x%x)", ext[i].name, ext[i].version);
            js += std::string(i ? "," : "") + "\"" + jesc(ext[i].name) + "\"";
        }
        std::printf("\n");
        js += "],\"devices\":[";
        uint32_t ndev = 0;
        ok(zeDeviceGet(drv, &ndev, nullptr), "zeDeviceGet(count)");
        std::vector<ze_device_handle_t> devs(ndev);
        if (ndev) ok(zeDeviceGet(drv, &ndev, devs.data()), "zeDeviceGet");
        std::printf("  devices after ZE_AFFINITY_MASK: %u\n", ndev);
        for (uint32_t k = 0; k < ndev; ++k) {
            ze_device_handle_t dev = devs[k];
            ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
            ok(zeDeviceGetProperties(dev, &p), "zeDeviceGetProperties");
            ze_pci_ext_properties_t pci = {ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
            const bool have_pci = zeDevicePciGetPropertiesExt(dev, &pci) == ZE_RESULT_SUCCESS;
            char bdf[32];
            if (have_pci) std::snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.%x", pci.address.domain, pci.address.bus, pci.address.device, pci.address.function);
            else std::snprintf(bdf, sizeof bdf, "unavailable");
            std::printf("  device %u: \"%s\" vendor 0x%x deviceId 0x%x type %d subdevice=%d BDF %s\n"
                        "    timerResolution %lu (%s) timestampValidBits %u kernelTimestampValidBits %u numSlices %u "
                        "numSubslicesPerSlice %u numEUsPerSubslice %u numThreadsPerEU %u coreClockRate %u MHz\n",
                        k, p.name, p.vendorId, p.deviceId, (int) p.type, (p.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) ? 1 : 0, bdf,
                        (unsigned long) p.timerResolution,
                        p.stype == ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 ? "cycles/s (DEVICE_PROPERTIES_1_2)" : "ns per tick (DEVICE_PROPERTIES)",
                        p.timestampValidBits, p.kernelTimestampValidBits, p.numSlices, p.numSubslicesPerSlice,
                        p.numEUsPerSubslice, p.numThreadsPerEU, p.coreClockRate);
            // host/device global timestamps: three samples, so the device tick rate and the host offset are visible
            std::printf("    global timestamps (host ns, device ticks):");
            for (int s = 0; s < 3; ++s) {
                uint64_t h = 0, d = 0;
                if (zeDeviceGetGlobalTimestamps(dev, &h, &d) == ZE_RESULT_SUCCESS) std::printf(" [%lu, %lu]", (unsigned long) h, (unsigned long) d);
                else std::printf(" [unavailable]");
            }
            std::printf("\n");
            js += std::string(k ? "," : "") + "{\"index\":" + std::to_string(k) + ",\"name\":\"" + jesc(p.name) + "\",\"deviceId\":" +
                  std::to_string(p.deviceId) + ",\"bdf\":\"" + bdf + "\",\"timerResolution\":" + std::to_string(p.timerResolution) +
                  ",\"timestampValidBits\":" + std::to_string(p.timestampValidBits) + ",\"kernelTimestampValidBits\":" +
                  std::to_string(p.kernelTimestampValidBits) + ",\"metricGroups\":[";
            uint32_t ng = 0;
            const ze_result_t rg = zetMetricGroupGet(dev, &ng, nullptr);
            if (rg != ZE_RESULT_SUCCESS) {
                std::printf("    zetMetricGroupGet -> %s%s\n", zr(rg), em ? "" : " (ZET_ENABLE_METRICS=1 not set)");
                ++g_fail;
                js += "]}";
                continue;
            }
            std::vector<zet_metric_group_handle_t> groups(ng);
            if (ng) ok(zetMetricGroupGet(dev, &ng, groups.data()), "zetMetricGroupGet");
            std::printf("    metric groups: %u%s\n", ng, ng == 0 && !em ? " (ZET_ENABLE_METRICS=1 not set)" : "");
            for (uint32_t g = 0; g < ng; ++g) {
                zet_metric_group_properties_t gp = {ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES};
                if (!ok(zetMetricGroupGetProperties(groups[g], &gp), "zetMetricGroupGetProperties")) continue;
                std::printf("    [%u] %s | %s | sampling %s | domain %u | metrics %u\n", g, gp.name, gp.description,
                            sampling_type(gp.samplingType), gp.domain, gp.metricCount);
                js += std::string(g ? "," : "") + "{\"name\":\"" + jesc(gp.name) + "\",\"description\":\"" + jesc(gp.description) +
                      "\",\"samplingType\":\"" + sampling_type(gp.samplingType) + "\",\"domain\":" + std::to_string(gp.domain) +
                      ",\"metrics\":[";
                uint32_t nm = gp.metricCount;
                std::vector<zet_metric_handle_t> ms(nm);
                if (nm && !ok(zetMetricGet(groups[g], &nm, ms.data()), "zetMetricGet")) nm = 0;
                for (uint32_t m = 0; m < nm; ++m) {
                    zet_metric_properties_t mp = {ZET_STRUCTURE_TYPE_METRIC_PROPERTIES};
                    if (!ok(zetMetricGetProperties(ms[m], &mp), "zetMetricGetProperties")) continue;
                    std::printf("        %-40s %-12s %-16s %-8s %-12s tier %u | %s\n", mp.name, mp.component, metric_type(mp.metricType),
                                value_type(mp.resultType), mp.resultUnits, mp.tierNumber, mp.description);
                    js += std::string(m ? "," : "") + "{\"name\":\"" + jesc(mp.name) + "\",\"component\":\"" + jesc(mp.component) +
                          "\",\"type\":\"" + metric_type(mp.metricType) + "\",\"resultType\":\"" + value_type(mp.resultType) +
                          "\",\"units\":\"" + jesc(mp.resultUnits) + "\",\"tier\":" + std::to_string(mp.tierNumber) +
                          ",\"description\":\"" + jesc(mp.description) + "\"}";
                }
                js += "]}";
            }
            js += "]}";
        }
        js += "]}";
    }
    js += "],\"failures\":" + std::to_string(g_fail) + "}\n";
    if (json_path) {
        if (FILE* f = std::fopen(json_path, "w")) { std::fputs(js.c_str(), f); std::fclose(f); std::printf("json: %s\n", json_path); }
        else { std::fprintf(stderr, "ze_metric_enum: cannot write %s\n", json_path); ++g_fail; }
    }
    std::printf("ze_metric_enum: %s (%d failed calls)\n", g_fail ? "INCOMPLETE" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
