// head_cols_probe.cpp (S25 revision) - the lm_head GEMV at the served shape (Q5_K, n_in 2560 -> n_out 248320) for
// C = 1,2,3,4,6,8 columns through exactly the verifier's two calls (verify.cpp:1663: native_quantize_q8_1 then
// native_mmvq(type 13, ..., T)), with: a device gate (exactly one visible GPU, name / driver / PCI BDF printed and
// optionally required), the ACTUAL output.weight bytes from the served GGUF (--weights FILE ABS_OFFSET, 437043200
// bytes, FNV-1a64 of the bytes printed so the sha256 recorded beside it can be cross-checked), poisoned outputs with
// tail guards, a full per-C oracle (every column c of the C-column result vs row c alone through C = 1: bitwise,
// argmax, max |delta|, finiteness), bounded iterations (20..50), timing of back-to-back launches (an event-barrier
// span: queue + kernel + inter-launch gaps, NOT isolated service; unitrace gives per-kernel service), all rounds
// kept, quantize timed separately, and a JSON record (--json PATH).  Exit 0 only when every oracle passes.
//
//   head_cols_probe [--iters N] [--weights FILE ABS_OFFSET] [--expect-bdf 0000:0b:00.0] [--json out.json]
#include <sycl/sycl.hpp>
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
uint64_t fnv1a64(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}
std::string jstr(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int iters = 30;
    const char* wfile = nullptr; long long woff = -1;
    const char* expect_bdf = nullptr; const char* json_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--weights") && i + 2 < argc) { wfile = argv[i + 1]; woff = std::atoll(argv[i + 2]); i += 2; }
        else if (!std::strcmp(argv[i], "--expect-bdf") && i + 1 < argc) expect_bdf = argv[++i];
        else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
        else { std::printf("head_cols_probe: unknown argument %s\n", argv[i]); return 2; }
    }
    if (iters < 20) iters = 20;
    if (iters > 50) iters = 50;
    const int TYPE = 13, N_IN = 2560, N_OUT = 248320, CMAX = 8, GUARD = 4096;
    const int cols[] = {1, 2, 3, 4, 6, 8};
    if (!K::native_mmvq_supported(TYPE)) { std::printf("head_cols_probe: type 13 not supported\n"); return 2; }
    const size_t wbytes = K::native_mmvq_weight_bytes(TYPE, N_IN, N_OUT);
    std::printf("head_cols_probe: Q5_K (ggml type 13) %d -> %d, weight bytes %zu, iters %d (bounded 20..50), weights %s\n",
                N_IN, N_OUT, wbytes, iters, wfile ? wfile : "SYNTHETIC");
    if (wbytes != 437043200ull) { std::printf("head_cols_probe: unexpected weight byte count\n"); return 2; }

    // ---- device gate: exactly one visible GPU after ZE_AFFINITY_MASK, identity printed, BDF optionally required
    std::vector<sycl::device> gpus;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (p.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
    std::printf("visible Level Zero GPUs: %zu\n", gpus.size());
    if (gpus.size() != 1) { std::printf("head_cols_probe: GATE FAIL: expected exactly one visible GPU (ZE_AFFINITY_MASK / ONEAPI_DEVICE_SELECTOR)\n"); return 3; }
    const sycl::device dev = gpus[0];
    const std::string dname = dev.get_info<sycl::info::device::name>();
    const std::string drv = dev.get_info<sycl::info::device::driver_version>();
    std::string bdf = "unavailable";
    if (dev.has(sycl::aspect::ext_intel_pci_address)) bdf = dev.get_info<sycl::ext::intel::info::device::pci_address>();
    std::printf("device: \"%s\" driver %s BDF %s\n", dname.c_str(), drv.c_str(), bdf.c_str());
    if (expect_bdf != nullptr && bdf != expect_bdf) { std::printf("head_cols_probe: GATE FAIL: BDF %s != expected %s\n", bdf.c_str(), expect_bdf); return 3; }

    // ---- weights: the served tensor bytes (or synthetic Q5_K blocks with finite scales)
    std::vector<uint8_t> w(wbytes);
    if (wfile) {
        FILE* f = std::fopen(wfile, "rb");
        if (!f || woff < 0 || std::fseek(f, woff, SEEK_SET) != 0 || std::fread(w.data(), 1, wbytes, f) != wbytes) {
            std::printf("head_cols_probe: cannot read %zu bytes at %lld from %s\n", wbytes, woff, wfile); return 2;
        }
        std::fclose(f);
    } else {
        std::mt19937_64 rng(11);
        for (size_t i = 0; i < wbytes; ++i) w[i] = (uint8_t) rng();
        for (size_t b = 0; b + 176 <= wbytes; b += 176) { const uint16_t d = 0x2400, dmin = 0x1c00; std::memcpy(&w[b], &d, 2); std::memcpy(&w[b + 2], &dmin, 2); }
    }
    const uint64_t wfnv = fnv1a64(w.data(), wbytes);
    std::printf("weights: %s offset %lld bytes %zu fnv1a64 %016llx\n", wfile ? wfile : "synthetic", woff, wbytes, (unsigned long long) wfnv);

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) CMAX * N_IN);
    for (auto& v : x) v = nd(rng);

    sycl::queue q{dev, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
    uint8_t* dw = sycl::malloc_device<uint8_t>(wbytes, q);
    float* dx = sycl::malloc_device<float>(x.size(), q);
    const size_t xq_bytes = K::native_q8_1_bytes(N_IN, CMAX);
    void* dxq = sycl::malloc_device<uint8_t>(xq_bytes + 256, q);
    const size_t ysz = (size_t) CMAX * N_OUT + GUARD;   // CMAX x N_OUT outputs + a poisoned tail guard
    float* dy = sycl::malloc_device<float>(ysz, q);
    if (!dw || !dx || !dxq || !dy) { std::printf("head_cols_probe: device allocation failed\n"); return 2; }
    q.memcpy(dw, w.data(), wbytes).wait();
    q.memcpy(dx, x.data(), x.size() * 4).wait();
    const uint32_t POISON = 0xffbadbad;   // a NaN pattern: any surviving poison is visible as non-finite and bitwise
    std::vector<float> poison(ysz); std::memcpy(poison.data(), std::vector<uint32_t>(ysz, POISON).data(), ysz * 4);
    auto repoison = [&] { q.memcpy(dy, poison.data(), ysz * 4).wait(); };

    // ---- reference: each of the 8 rows alone through C = 1 (finite required)
    std::vector<float> ref((size_t) CMAX * N_OUT), out(ysz);
    int ref_nonfinite = 0;
    for (int r = 0; r < CMAX; ++r) {
        repoison();
        K::native_quantize_q8_1(dx + (size_t) r * N_IN, dxq, N_IN, 1, &q);
        K::native_mmvq(TYPE, dw, dxq, dy, N_IN, N_OUT, 1, &q);
        q.wait();
        q.memcpy(ref.data() + (size_t) r * N_OUT, dy, (size_t) N_OUT * 4).wait();
    }
    for (float v : ref) if (!std::isfinite(v)) ++ref_nonfinite;
    std::printf("reference (C=1, 8 rows): %d non-finite of %zu\n", ref_nonfinite, ref.size());
    bool all_ok = ref_nonfinite == 0;

    std::string js = "{\"device\":\"" + jstr(dname) + "\",\"driver\":\"" + jstr(drv) + "\",\"bdf\":\"" + bdf + "\",\"weights\":\"" +
                     jstr(wfile ? wfile : "synthetic") + "\",\"abs_offset\":" + std::to_string(woff) + ",\"bytes\":" + std::to_string(wbytes) +
                     ",\"fnv1a64\":\"" + ([&] { char b[20]; std::snprintf(b, sizeof b, "%016llx", (unsigned long long) wfnv); return std::string(b); })() +
                     "\",\"iters\":" + std::to_string(iters) + ",\"ref_nonfinite\":" + std::to_string(ref_nonfinite) +
                     ",\"timing_label\":\"event-barrier span over back-to-back launches / iters: queue + kernel + inter-launch gaps, not isolated service\",\"C\":[";
    double us1 = 0;
    std::printf("%2s %11s %11s %9s %8s | %s\n", "C", "mmvq us", "quant us", "us/row", "C1/row", "oracle: columns 0..C-1 vs rows alone (C=1); rows >= C and tail must keep poison");
    bool firstc = true;
    for (int C : cols) {
        repoison();
        K::native_quantize_q8_1(dx, dxq, N_IN, C, &q);
        K::native_mmvq(TYPE, dw, dxq, dy, N_IN, N_OUT, C, &q);
        q.wait();
        q.memcpy(out.data(), dy, ysz * 4).wait();
        bool bitwise = true, argmax_ok = true, finite = true, guard_ok = true;
        double maxd = 0;
        for (int c = 0; c < C; ++c) {
            int am_o = 0, am_r = 0;
            for (int i = 0; i < N_OUT; ++i) {
                const float a = out[(size_t) c * N_OUT + i], b = ref[(size_t) c * N_OUT + i];
                if (!std::isfinite(a)) finite = false;
                if (std::memcmp(&a, &b, 4) != 0) { bitwise = false; maxd = std::max(maxd, (double) std::fabs(a - b)); }
                if (a > out[(size_t) c * N_OUT + am_o]) am_o = i;
                if (b > ref[(size_t) c * N_OUT + am_r]) am_r = i;
            }
            if (am_o != am_r) argmax_ok = false;
        }
        for (size_t i = (size_t) C * N_OUT; i < ysz; ++i) { uint32_t u; std::memcpy(&u, &out[i], 4); if (u != POISON) { guard_ok = false; break; } }
        const bool ok = bitwise && argmax_ok && finite && guard_ok;
        all_ok = all_ok && ok;
        // timing: all rounds kept; mmvq-only and quantize-only, back-to-back `iters` launches between two barriers
        double rm[3], rq[3];
        for (int round = 0; round < 3; ++round) {
            for (int i = 0; i < 5; ++i) { K::native_quantize_q8_1(dx, dxq, N_IN, C, &q); K::native_mmvq(TYPE, dw, dxq, dy, N_IN, N_OUT, C, &q); }
            q.wait();
            sycl::event e0 = q.ext_oneapi_submit_barrier();
            for (int i = 0; i < iters; ++i) K::native_mmvq(TYPE, dw, dxq, dy, N_IN, N_OUT, C, &q);
            sycl::event e1 = q.ext_oneapi_submit_barrier();
            e1.wait();
            rm[round] = 1e3 * ((e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                e0.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e6) / iters;
            sycl::event f0 = q.ext_oneapi_submit_barrier();
            for (int i = 0; i < iters; ++i) K::native_quantize_q8_1(dx, dxq, N_IN, C, &q);
            sycl::event f1 = q.ext_oneapi_submit_barrier();
            f1.wait();
            rq[round] = 1e3 * ((f1.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                f0.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e6) / iters;
        }
        const double best_m = std::min(rm[0], std::min(rm[1], rm[2])), best_q = std::min(rq[0], std::min(rq[1], rq[2]));
        if (C == 1) us1 = best_m;
        std::printf("%2d %11.1f %11.1f %9.1f %8.2f | %s%s%s%s%s\n", C, best_m, best_q, best_m / C, us1 > 0 ? us1 / (best_m / C) : 0.0,
                    bitwise ? "bitwise equal" : "BITWISE DIFFERS", bitwise ? "" : (" max|delta| " + std::to_string(maxd)).c_str(),
                    argmax_ok ? "" : " ARGMAX DIFFERS", finite ? "" : " NON-FINITE", guard_ok ? "" : " GUARD/POISON OVERWRITTEN");
        char b[512];
        std::snprintf(b, sizeof b, "%s{\"C\":%d,\"mmvq_us_rounds\":[%.2f,%.2f,%.2f],\"mmvq_us_min\":%.2f,\"quant_us_rounds\":[%.2f,%.2f,%.2f],\"quant_us_min\":%.2f,"
                      "\"us_per_row\":%.2f,\"bitwise\":%s,\"argmax\":%s,\"finite\":%s,\"guard\":%s,\"max_abs_delta\":%.6g}",
                      firstc ? "" : ",", C, rm[0], rm[1], rm[2], best_m, rq[0], rq[1], rq[2], best_q, best_m / C,
                      bitwise ? "true" : "false", argmax_ok ? "true" : "false", finite ? "true" : "false", guard_ok ? "true" : "false", maxd);
        js += b; firstc = false;
    }
    js += "],\"all_ok\":" + std::string(all_ok ? "true" : "false") + "}\n";
    if (json_path) { if (FILE* f = std::fopen(json_path, "w")) { std::fputs(js.c_str(), f); std::fclose(f); std::printf("json: %s\n", json_path); } }
    std::printf("head_cols_probe: %s\n", all_ok ? "OK (every C bitwise, argmax, finite, guards intact)" : "FAIL (see lines above)");
    sycl::free(dw, q); sycl::free(dx, q); sycl::free(dxq, q); sycl::free(dy, q);
    return all_ok ? 0 : 1;
}
