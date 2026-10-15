# S51: HC duplicate-store predicate in serving (2026-10-15)

CONFIG -> Retain HC as a bounded experimental candidate: the three matched,
traced C2 coding pairs take about 1.03% less time. All hard output, native-work,
functional, identity and teardown gates pass. This is one fresh server per arm,
fixed control-then-candidate order; clean speed, stability and shelf promotion
remain unqualified.

The one-line patch selects the existing LO_ONCE=true implementation only for
non-GRF/non-ILP default C16 UP. Group0 writes global lo once; every workgroup
keeps its private SLM work, arithmetic and launch geometry. Other column counts
and GRF/ILP branches keep their old selection. S48/S50 establish prior bounded
numerical/graph evidence, not identity of this production AOT image or a forecast
serving percentage. Actual serving callsites use disjoint lo/R/rs/xn buffers.

Both arms keep EP149/363, all24576 main experts resident, batch2/depth2,
headpairON/spec4, Direct PLE, int8 KV, context200000 and32GiB/four-entry parking.
S49 prefetch remains OFF in both; head RPW remains1. Both use the new verified
NVMe native shard AND its exact sibling PLE pathname, the same immutable image,
STRATA_TRACE1/HOST_BOUNDARY1 and diagnostic suffix, unchanged Python source.
The serving interface binds localhost, stable hotschmoe-dd first plus the exact
research alias cross-checked against evals/configs/models.yaml.

COMMAND -> Refresh upstream/contribution status; verify source pins, apply exact
HC patch. One168s CPU-only offline incremental build: one fused_gr TU, kernel
archive and final AOT link; no configure/clean/driver change or kernel device.
Other core/engine/prefill/spec archives, S49 sources, Python and CMake cache
remain exact. Initialize oneAPI before shell strict mode; disk TMPDIR, UID1000,
four jobs, no network/DRI. Main source stays HC-patched for the next experiment;
frozen qualified152055a6 and experimental control0c30e8aa remain recoverable.

Full both-card/compiled collective prehealth, then two fresh leased servers.
Each runs one warm128x2 pair, three fresh128x2 pairs, concurrent323/667,
reasoning391 and the actual tool roundtrip17passed/0failed. For each pair,
submit request0 alone, observe its flushed native prompt-start from a fresh
engine-log offset, then submit request1. Save marker position/times and actual
overlapping client lifetimes. This orders logical requests; it does not assume
fixed native slots. Paired activation and exact native work are separately
checked. Candidate warm full text/SSE-content/usage must match before timing.

The GPU lease spans each server lifetime. CPU lease covers startup, quiet client
measurement and health only, outside any GPU lease when both are needed. Clients
do not acquire a nested GPU lease. TERM to the Python server invokes native QUIT;
wait for normal wrapper0/container removal before per-card and compiled10
collective post-health. No forced stop, retry, reset, daily restore, full-context
rerun, extra microprobe or profiler campaign. Parse each saved segment once.

Raw P:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/hc-lo-once-serving-s51-prep`.
Exact commands: `bash P/build-root.sh`; after build audit,
`python3 P/harness-prep/make_configs.py --control-exe <0c30 frozen path> --candidate-exe <0143 frozen path> --candidate-sha256 <full SHA below>`.
Prehealth uses `bin/cpu-run --quiet bin/gpu-run python3 P/root-prehealth.py`;
startup `bin/cpu-run python3 P/harness-prep/start_arm.py ARM`, warm/timed
`bin/cpu-run --quiet python3 P/harness-prep/client.py ARM PHASE`, functionals
`python3 P/harness-prep/client.py ARM functional`, closure
`bin/cpu-run --quiet python3 P/harness-prep/stop_and_health.py ARM`.
Run the retained analyze-host-boundary.py once per saved segment and compare.py
after both post-health closures. Scripts and exact command records are retained;
these are historical recipes, not permission to overwrite completed evidence.

| Artifact | SHA256 |
| --- | --- |
| Control frozen binary | 0c30e8aa9625dbd35966752d1bfcf4191f2c4fb1b028335dd4f78170c407ca23 |
| HC candidate frozen binary | 0143d8a4da90a848597c32fdd40383e083d6fc901286595c5446c713da40dcb7 |
| Candidate kernel archive | 29e80abab55ab8e5bcdfebbbd5a995d4652a9e8a03f74a34d0dcfffb2816f3c8 |
| Candidate fused_gr source | d62bcb252eecdbcaae81095907c2241e39c4798b7854bb499fb5643fecb24e99 |
| Production patch | b3b4a4c925cf10c1448b53ab957351b6c991cee125b70b2e5b3305cee9e2239f |

Image is sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2.
The exact retained patch is
[experimental-hc-lo-once-s51.patch](../strata/flash-next/patches/experimental-hc-lo-once-s51.patch).
Build plan/hash inputs, server configs, payload stat/SHA provenance, client
lifetimes, native log segments and parser outputs are retained under P.

RESULT -> PASS_MATCHED_DISCRIMINATOR: all four pair comparisons match request,
full text/SSE content/usage, native draft/accepted counts, windows, pair/solo,
slot-copy order and graph-capture/replay work. All markers reconcile native counts,
complete with dropped0. Eight corresponding coding output pairs/16 responses
match, including S16 warm prefixes. Actual raw native token IDs and full serving
logits are not observed. Both functions pass, normal QUIT/container removal,
wrapper0 and both-card/compiled10 post-health0; no reset.

| Pair | Control wall s | HC wall s | Change | Windows / paired / solo, both arms |
| --- | ---: | ---: | ---: | --- |
| Warm, excluded from median | 4.969418 | 4.663759 | diagnostic | 48 / 45 / 2 |
| Repeat1 | 4.832553 | 4.782742 | -1.031% | 49 / 46 / 2 |
| Repeat2 | 4.830802 | 4.765393 | -1.354% | 48 / 45 / 2 |
| Repeat3 | 4.845678 | 4.800632 | -0.930% | 48 / 46 / 1 |

Median wall4.832552512->4.782741671s, delta49.811ms, versus within-arm
ranges14.877/35.238ms. All three candidate observations are below all three
control observations. Reciprocal256-output-token/wall medians52.974->53.526
aggregate tok/s are instrumentation-scoped diagnostics, not the clean150 per-stream
goal or a new clean baseline. Three pairs nested in one process/variant and fixed
order do not establish significance or stability.

Logical first stream client-decode values32.518-32.910->32.812-33.284tok/s;
second49.369-50.389->49.732-50.777. These intervals include the other request's
prefill/admission pause. Per-request TTFT/completion and content-event gap p50/p95
are retained in s51-root-report.json/comparison.json; content events are not
individual tokens. Logical first TTFT0.905-0.929->0.880-0.917s; second2.060-2.099
->2.030-2.082s. Three requests/stream do not qualify tail latency.

Same-count hot replay whole-window CPU medians by repeat46.165/46.219/45.966ms
->45.753/45.761/45.482ms. These include all verifier/draft/CPU waits and are not
HC GPU service measurements or additive saved budgets. No new per-HC GPU timer,
dispatch trace, memory-traffic or occupancy attribution is available. Whole-process
PLE counters differ46240->46288 requested rows/15475->15491 demand reads, while
prefetch93gathers/372tokens is identical. These aggregate counters include warm,
admission and functionals; do not assert physical IO equality or coding-only
causality from them.

NVMe starts reached readiness84.130/54.115s, observed at2s polling granularity.
Paths/payloads match the verified S53 copy; fixed order/cache differences prevent
causal NVMe or HC startup savings. No extra startup benchmark is justified here.

VERDICT -> Retain HC in both arms of the next distinct head-only RPW2 experiment,
without daily/shelf promotion. The small positive traced signal is sufficient to
carry this bounded candidate, not claim targets reached or extrapolate S50's6.29%
primitive gain to serving. No HC rerun/knob sweep/full200k/profiler/daily restore.
Actual Opus/Fable post-result decisions and whole-stage manifest follow below.

Actual resumed claude-opus-5-5/claude-fable-5-1 post-result reviews exit0 in
36.753/48.120s. Root read both full responses; both retain the bounded candidate
and hold HC fixed, with no extra HC run or promotion. Root rejects Fable's imported
2% threshold (S51 declares no such gate), its unproved location of cumulative-row
differences, and Opus's additive wall decomposition/reversed-order stability
implication. Raw responses stay intact; corrected limitations leave the bounded
retention decision unchanged, so no further review round was spent.

Frozen S51 prep/execution plus separate postreviews: 187files/11306846bytes.
External `hc-serving-s51-manifest.json` SHA256
`13c15150968a5a7b3ea2359634e822a14d1c5ff7d5a3863ce67153c2664f3893`.
