#!/usr/bin/env python3
"""Prepare diagnostic-only MTP timing source without modifying the tree."""
from pathlib import Path
import hashlib
import json

source = Path('/mnt/vm_8tb/b70/build/strata-0216/src/engine/sycl/src/core/mtp.cpp')
out = Path('/mnt/vm_8tb/b70/results/conc-0216/draft-trace-mtp.cpp')
s = source.read_text()
a = 'using Clock = std::chrono::steady_clock;'
assert s.count(a) == 1
s = s.replace(a, a + '\nthread_local bool g_draft_first_trace = false;')
a = '''    const bool cp = coupled_active_;   // coupled draft sampling for this request: its own graphs
    if (!capture_round(T, cp, err)) return false;'''
b = '''    const bool cp = coupled_active_;   // coupled draft sampling for this request: its own graphs
    const auto trace_capture_start = g_draft_first_trace ? Clock::now() : Clock::time_point{};
    if (!capture_round(T, cp, err)) return false;'''
assert s.count(a) == 1
s = s.replace(a, b)
a = '''    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {'''
b = '''    const Clock::time_point t0 = Clock::now();
    if (g_draft_first_trace)
        std::fprintf(stderr, "draft-first-detail capture_ms=%.3f cell=%lld\\n",
                     std::chrono::duration<double, std::milli>(t0 - trace_capture_start).count(), (long long)p);
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {'''
assert s.count(a) == 2
s = s.replace(a, b, 1)
a = '''    if (!stage_source_R(T, err)) return false;

    SessionState* const pss = ple_ss_ ? ple_ss_ : ss_;'''
b = '''    const auto trace_stage_start = g_draft_first_trace ? Clock::now() : Clock::time_point{};
    if (!stage_source_R(T, err)) return false;
    if (g_draft_first_trace)
        std::fprintf(stderr, "draft-first-detail source_stage_ms=%.3f cell=%lld\\n",
                     std::chrono::duration<double, std::milli>(Clock::now() - trace_stage_start).count(), (long long)p);

    SessionState* const pss = ple_ss_ ? ple_ss_ : ss_;'''
assert s.count(a) == 1
s = s.replace(a, b)
a = '''    if (min_p <= 0.0f && max_steps > 0) {
        if (DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph('''
b = '''    if (min_p <= 0.0f && max_steps > 0) {
        const auto trace_graph_start = g_draft_first_trace ? Clock::now() : Clock::time_point{};
        if (DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph('''
assert s.count(a) == 1
s = s.replace(a, b)
a = '''        const int last = max_steps - 1;
        drafts[last] = ((volatile int32_t*) h_out_)[last];'''
b = '''        if (g_draft_first_trace)
            std::fprintf(stderr, "draft-first-detail graph_prefetch_wait_ms=%.3f cell=%lld\\n",
                         std::chrono::duration<double, std::milli>(Clock::now() - trace_graph_start).count(), (long long)p);
        const int last = max_steps - 1;
        drafts[last] = ((volatile int32_t*) h_out_)[last];'''
assert s.count(a) == 1
s = s.replace(a, b)
a = '''    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;'''
b = '''    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const char* trace_env = std::getenv("STRATA_DRAFT_FIRST_TRACE");
    const bool trace = trace_env && trace_env[0] && trace_env[0] != '0';
    const auto trace_start = trace ? Clock::now() : Clock::time_point{};
    const int64_t HCN = g_->hc * g_->n_embd;'''
assert s.count(a) == 1
s = s.replace(a, b)
a = '''    std::vector<int32_t> toks((size_t) T, token);
    return draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
}'''
b = '''    const auto trace_after_residual = trace ? Clock::now() : Clock::time_point{};
    std::vector<int32_t> toks((size_t) T, token);
    g_draft_first_trace = trace;
    const bool ok = draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
    g_draft_first_trace = false;
    if (trace)
        std::fprintf(stderr, "draft-first-detail residual_wait_ms=%.3f draft_ms=%.3f cell=%lld ok=%d\\n",
                     std::chrono::duration<double, std::milli>(trace_after_residual - trace_start).count(),
                     std::chrono::duration<double, std::milli>(Clock::now() - trace_after_residual).count(),
                     (long long)cell, (int)ok);
    return ok;
}'''
assert s.count(a) == 1
s = s.replace(a, b)
out.write_text(s.replace('\u201c', '"').replace('\u201d', '"'), encoding='ascii')
manifest = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'prepared': str(out), 'prepared_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
            'generator': str(Path(__file__).resolve()), 'built_or_run': False}
out.with_suffix('.provenance.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + '\n')
print(out)
