# Strata CPU boundary profile

Date: 2026-10-15. S32, diagnostic CPU timestamps without unitrace.

CONFIG -> Frozen `strata-host-boundary-s32`, SHA256
`595f2ff54e48882985b91972d64ab27d01241c009b385bf29d9f96721fdde826`.
S31's headpairON/depth2, CLI spec4, two200k slots,32GiB/4 parking and
all24576 experts7152/17424. Python carry correction unchanged. Opt-in
`STRATA_HOST_BOUNDARY=1` buffers CPU steady_clock timestamps in one
thread-local state across three translation units. No new GPU event,
submission or wait; no per-window diagnostic output. Records flush after
all slots become idle. The default option is OFF.

COMMAND -> CPU-only offline SYCL build with no DRI devices, freeze/hash,
one leased start, one128+128 coding pair, then existing functional checks.
Normally stop and verify lease exit0, both-card health and ten compiled
two-rank4x5120 BF16/P2P0 collective cases. All pass, no reset. Reuse S31's
passing pre-health because no GPU work intervened; do not repeat full200k
cache qualification for a diagnostic-only marker change.

RESULT -> Both128-token responses match retained S16 prefixes. Native
45paired/2solo; expert residency unchanged. Concurrent arithmetic323/667,
reasoning391 and tool call/continuation pass. Complete timing dump has
1684 records, dropped0,48 complete windows. Forty full S6/A2/depth2
windows survive conservative first-observed capture-role exclusions;
38 consecutive same-dump joins have both endpoints selected. All cold
windows remain available; selection uses source roles, never duration.

| CPU ordering span | Samples | Median | p95 |
| --- | ---: | ---: | ---: |
| Batched PLE gather | 40 | 0.829 ms | 1.085 ms |
| Verifier graph submission sequence | 40 | 55.0 us | 75.0 us |
| Paired drafter graph submission sequence | 40 | 45.0 us | 55.2 us |
| Paired slots' existing idle calls | 40 | 1.358 us | 2.196 us |
| Paired mapped-input stores/fence | 40 | 0.316 us | 0.566 us |
| Commit wrapper | 40 | 1.591 ms | 1.617 ms |
| Pair wait-return to next verifier-submit | 38 | 0.866 ms | 1.112 ms |
| Pair return to next window-enter | 38 | 1.07 us | 6.68 us |

The last row includes intervening host work and packing, which occurs
before window-enter; it is not a packing-only measurement. The existing
queue wait median39.193ms belongs to verifier execution/waiting. Paired
poll/prefetch and final queue-wait medians2.703/2.584ms likewise include
GPU progress. The commit wrapper includes GPU submission/waiting. These
are not additive CPU overhead or removable latency budgets. Mapped-input
stores are not the later GPU copy operation.

PLE gather dominates the measured interval from pair wait-return to the
next submission. This establishes a current blocking host-side boundary,
not its complete I/O/dequant decomposition. Direct gather may issue or
collect reader work and decode rows; a cache-hit ratio alone cannot assign
that cost. No kernel fusion or PLE optimization is qualified here.

The repeated S31 recorded-GPU-activity gaps were3.300-3.941ms after final
MTP-select and1.555-1.648ms before mapped-input copy. S32 measures different
CPU endpoints without unitrace. Final GPU completion and CPU wait-return
remain unjoined, and tracing can change execution inside existing waits
and submissions. Do not subtract these results to claim tracer overhead,
physical GPU idle, or saved production latency. A multi-millisecond
post-wait packing/staging delay is not observed in this bounded run.

## Source review and evidence closure

Removing diagnostic includes/calls and the diagnostic window-reset line
recovers the exact pre-change bytes of all three translation units. Both
builds pass; linker/nm evidence confirms one TLS entity in the executable.
This proves retained source work/order, not OFF executable identity or
zero instrumentation overhead. The reusable opt-in patch is
[host-boundary-profile-s32.patch](../strata/flash-next/patches/host-boundary-profile-s32.patch).

Actual Claude Opus5.5 and Fable5.1 reviewed the supplied initial patch,
responding model IDs verified, completing in39.6/47.9s. Opus found that
window IDs survived between windows; root resets them at window-end and
moves begin after the empty-slot return. The final fix changes only the
program translation unit and required one incremental build. Both agents'
stronger inference that a small CPU span fully locates the prior GPU gap
is rejected: the endpoints and instrumentation differ.

The first harness snapshot raced the buffered stderr dump: header1684,
actual1214, ending mid-window35. Parser run1 correctly rejected it. After
normal shutdown, the complete engine log supplies all1684 records; extract
through that first complete dump, preserve the partial snapshot and record
source hash plus byte/line bounds. Parser run2 passes. No GPU rerun or
weakened validation was needed. Future snapshots must wait for the exact
declared record count, not merely the earlier pair-summary line.

VERDICT -> CPU boundary hypothesis narrowed; no new speed result or daily
promotion. Keep S29 clean coding and S30 full cache as separate evidence.
Do not optimize packing or mapped stores from the older trace gaps. Next
bounded candidate is a smaller runtime prefill scheduling quantum while
retaining12288-token allocation/ownership, addressing S30's7.753s decode
content gap during fresh32K arrival. State continuity/callbacks need proof
before adopting it; no scheduling change is included in S32.

Loop: one short serving start,256 measured output tokens, one functional
check, no old GPU arm, no full-cache repeat, no daily restore. CPU reviewers
ran alongside the initial build; their scope correction cost one extra
121.5s incremental build after the135.5s initial build. Resolve narrow
scope reviews before future builds when that avoids recompilation.

Raw root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/host-boundary-s32/`.
Key evidence: inputs.json, source-equivalence.json, source-change.json,
model-review-closure.json, window.json, pair-log-closure.json,
analysis/run2/{validation,analysis}.json, functional/, normal-stop.json,
post-health.json and s32-report.json. All prior operation CSVs and S22
rollups remain unchanged; the data map adds separate S32 CPU quotes.
