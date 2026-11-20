#!/usr/bin/env python3
"""Prepare a diagnostic-only verifier source copy; do not alter the source tree."""
from pathlib import Path
import hashlib
import json

source = Path('/mnt/vm_8tb/b70/build/strata-0216/src/engine/sycl/src/core/verify.cpp')
out = Path('/mnt/vm_8tb/b70/results/conc-0216/route-audit-verify.cpp')
s = source.read_text()
anchor = '#ifdef STRATA_SYCL_WORKLOAD_TRACE\n// Diagnostic graph copies preserve'
assert s.count(anchor) == 1
s = s.replace(anchor, '''#ifdef STRATA_SYCL_WORKLOAD_TRACE
// Diagnostic only: save each layer's actual router IDs before scratch is reused.
struct RouteIdsAudit {
    int32_t* ids = nullptr;
    FILE* output = nullptr;
    sycl::queue* queue = nullptr;
    int rows = 0, k = 0, layers = 0, experts = 0;
    RouteIdsAudit(int r, int topk, int nl, int ne, sycl::queue* q, const char* path)
        : queue(q), rows(r), k(topk), layers(nl), experts(ne) {
        output = std::fopen(path, "a");
        if (!output) throw std::runtime_error("route IDs audit output unavailable");
        ids = sycl::malloc_host<int32_t>((size_t)rows * k * layers, *queue);
        if (!ids) { std::fclose(output); output = nullptr; throw std::bad_alloc(); }
        std::fill(ids, ids + (size_t)rows * k * layers, -1);
    }
    ~RouteIdsAudit() {
        if (ids) { try { sycl::free(ids, *queue); } catch (...) {} }
        if (output) std::fclose(output);
    }
    int32_t* slot(int layer, int row) { return ids + ((size_t)layer * rows + row) * k; }
    void emit(int n, int lo, int hi, int64_t window, int64_t position) {
        if (n < 1 || n > rows) throw std::runtime_error("route IDs audit row count invalid");
        for (int layer = lo; layer < hi; ++layer) {
            std::map<int, int> occurrences;
            for (int i = 0; i < n * k; ++i) {
                const int expert = slot(layer, 0)[i];
                if (expert < 0 || expert >= experts) throw std::runtime_error("route IDs audit invalid expert ID");
                ++occurrences[expert];
            }
            const auto& layout = strata::kernels::cpu::expert_layout();
            if (!layout.native || (size_t)layer >= layout.fmt.size()) throw std::runtime_error("route IDs audit requires native format");
            const auto& fmt = layout.fmt[(size_t)layer];
            const uint64_t weight_bytes = 2ull * (uint64_t)fmt.n_ff * fmt.gu_row + (uint64_t)fmt.n_embd * fmt.d_row;
            std::fprintf(output, "{\\\"window\\\":%lld,\\\"position\\\":%lld,\\\"rows\\\":%d,\\\"layer\\\":%d,\\\"entries\\\":%d,\\\"unique\\\":%zu,\\\"gu_type\\\":%d,\\\"down_type\\\":%d,\\\"weight_bytes_per_expert\\\":%llu,\\\"multiplicities\\\":[",
                         (long long)window, (long long)position, n, layer, n*k, occurrences.size(), fmt.gu_type, fmt.d_type,
                         (unsigned long long)weight_bytes);
            bool first = true;
            for (const auto& entry : occurrences) {
                std::fprintf(output, "%s%d", first ? "" : ",", entry.second);
                first = false;
            }
            std::fprintf(output, "]}\\n");
        }
        std::fflush(output);
    }
};
std::map<const Verifier*, std::unique_ptr<RouteIdsAudit>> route_id_audits;

// Diagnostic graph copies preserve''')
s = s.replace('''    routing_audits.erase(this);
#endif''', '''    routing_audits.erase(this);
    route_id_audits.erase(this);
#endif''')
anchor = '''#ifdef STRATA_SYCL_WORKLOAD_TRACE
    if (const char* audit = std::getenv("STRATA_SYCL_ROUTING_AUDIT_FILE"); audit && *audit) {'''
assert s.count(anchor) == 1
s = s.replace(anchor, '''#ifdef STRATA_SYCL_WORKLOAD_TRACE
    if (const char* audit = std::getenv("STRATA_SYCL_ROUTE_IDS_FILE"); audit && *audit)
        route_id_audits[this] = std::make_unique<RouteIdsAudit>(max_t, (int)ss.k, (int)g.n_layers, (int)g.n_expert, cs_, audit);
    if (const char* audit = std::getenv("STRATA_SYCL_ROUTING_AUDIT_FILE"); audit && *audit) {''')
anchor = '''        if (expert_usage_counts_d_)
            count_expert_routes(cs, ids_ + tb * K'''
assert s.count(anchor) == 1
s = s.replace(anchor, '''#ifdef STRATA_SYCL_WORKLOAD_TRACE
        if (auto audit = route_id_audits.find(this); audit != route_id_audits.end())
            cs->memcpy(audit->second->slot((int)l, tb), ids_ + tb * K, (size_t)n * K * sizeof(int32_t));
#endif
        if (expert_usage_counts_d_)
            count_expert_routes(cs, ids_ + tb * K''')
anchor = '''    if (wait_failed(pos0, T, G, false, err)) return false;   // #5160: a wait ran out: the window's tokens are not used
'''
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '''#ifdef STRATA_SYCL_WORKLOAD_TRACE
    if (auto audit = route_id_audits.find(this); audit != route_id_audits.end())
        audit->second->emit(T, (int)lb_, (int)le_, windows, pos0);
#endif
''')
anchor = '''        if (wait_failed(pos[0], S, 1, false, err)) return false;   // #5160
        if (ep_on() && !ep_check(err)) return false;'''
assert s.count(anchor) == 1
s = s.replace(anchor, '''        if (wait_failed(pos[0], S, 1, false, err)) return false;   // #5160
        if (ep_on() && !ep_check(err)) return false;
#ifdef STRATA_SYCL_WORKLOAD_TRACE
        if (auto audit = route_id_audits.find(this); audit != route_id_audits.end())
            audit->second->emit(S, (int)lb_, (int)le_, windows, pos[0]);
#endif''')
anchor = '''    if (wait_failed(pos[0], S, 1, false, err)) return false;   // #5160
    if (prof_on_) collect_profile();'''
assert s.count(anchor) == 1
s = s.replace(anchor, '''    if (wait_failed(pos[0], S, 1, false, err)) return false;   // #5160
#ifdef STRATA_SYCL_WORKLOAD_TRACE
    if (auto audit = route_id_audits.find(this); audit != route_id_audits.end())
        audit->second->emit(S, (int)lb_, (int)le_, windows, pos[0]);
#endif
    if (prof_on_) collect_profile();''')
out.write_text(s, encoding='ascii')
manifest = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'prepared': str(out), 'prepared_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
            'generator': str(Path(__file__).resolve()), 'built_or_run': False}
out.with_suffix('.provenance.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + '\n')
print(out)
