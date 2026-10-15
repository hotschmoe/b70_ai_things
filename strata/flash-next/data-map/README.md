# Strata Flash-Next data map

Open `index.html` in any browser (file://, offline, no network). It shows
where data lives and moves for the Flash-Next IQ3_XXS Strata EP serving
path, which operations are expensive or wait, and how strong the evidence
is for each claim. `operation-data.json` is the source of truth;
`index.html` embeds a copy of it.

Snapshot: 2026-10-15, S35 (S22 reference operation trace plus readiness, real-head,
production PLE, EP publication fix/native ISA, paired-head serving,
draft-counter carry correction, full winning-candidate RAM recovery and
separate current paired-head replay measurements, CPU boundary markers
and a fixed-allocation prefill quantum tradeoff; historical prefill dispatch
intervals are separately attributed, with a standalone peer-gather alias gate).
Runtime source:
`/mnt/vm_8tb/b70/build/strata-ab/src-ep` (upstream Strata 296b59d2 base
plus working-tree overlays for EP/PUSH/accounting/S19-S21 and the
default-OFF S26 head-pair prototype and S28 explicit EP barrier scope).
S30 also updates the Python optional draft-counter carry helpers.
Timings reference preserved binaries, not
the changing source directory. Evidence:
`docs/20261014_flashnext_strata_optimization.md` S10 and S20-S22, plus raw
summaries under
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.

## The data path in one pass

1. Request: HTTP (serve/server.py) -> chat template -> tokenize -> native
   admission. Admission picks one of two 200000-token slots and checks
   resident slots and the 32 GiB / 4-entry RAM parking cache.
2. Cache: a parked conversation is restored into the slot
   (conversation_kv_restore; S20: about 420-430 ms per full restore at 200k,
   3.17 GB API payload, not physical PCIe bytes). A miss means prefill.
3. Prefill (S10 evidence, older binary): chunked sparse attention + GDN on
   GPU0, expert dequant + GEMM on both cards, PUSH FP32 reduction from GPU1
   posted into GPU0 memory.
4. Decode, one batch window: host packs 6 rows (2 slots x seed + 2 proposals
   at depth 2) -> 48 layers of [hyper-connections (hc 4, low rank 320) ->
   QSA attention (12 layers) or GDN (36 layers) -> dense/shared mmvq ->
   Q8_1 quant -> router top-10 of 512 -> EP push to GPU1 / local experts on
   GPU0 -> GPU1 experts -> rows returned -> combine] -> six-row Q5_K
   full-vocab head (N 2560, V 248320) -> verify/accept -> commit -> four
   serial one-row draft heads -> emit.
5. Park: when a slot is overwritten, the completed slot is saved to RAM.

That decode sequence describes the S22 trace. S29's opt-in variant pairs
the two same-depth draft heads. S31 measures two C2 calls per paired
cycle and exports159 current operations separately from S22. S32 adds
CPU ordering spans: PLE gather0.829ms median, pair-wait-return to next
verifier submission0.866ms; no GPU completion clock join or new speed claim.

The EP link is Level Zero IPC-mapped device memory with kernel-side flag
counters (sycl/include/strata/core/ep_mem.hpp), not a collective.

## Reading the numbers

- Status: measured / inferred / source-only / unknown / parked / rejected.
  Every box and quote carries one.
- Trace numbers come from the S22 control unitrace: 8 interior replay
  cycles, BOOTTIME minus MONOTONIC_RAW shift. Per-cycle values are
  summed service times / 8. They overlap and are NOT a latency budget.
  ep_wait time is spin, not work.
- Tracing slows the cycle (interior median 59.17 ms vs clean native 49.12-
  49.27 ms; the cold average of 130 ms is misleading). Use the clean
  benchmark table for performance claims.
- Logical bytes are arithmetic from shapes and block sizes, labelled as
  estimates. No physical PCIe bytes or bandwidth saturation are claimed.
- Production CPU-path costs are not estimated; standalone synthetic preparation
  is labeled separately. See [readiness probes](../lowlevel_probe/README.md).

## Maintenance

All commands run from this directory with system python3, no dependencies.

    python3 -I build_map.py --import-ops   # re-read S22 steady summaries -> operations[], stage_rollups
    python3 -I build_map.py                # re-embed operation-data.json into index.html
    python3 -I build_map.py --check        # exit 1 if embed stale, JSON broken, or non-ASCII

- Edit stages, gaps, hypotheses and benchmarks by hand in
  operation-data.json, then re-embed.
- Kernel-to-stage assignment lives in RULES in build_map.py (first regex
  match wins). Unmatched kernels land in `unattributed`; assign them from
  source rather than guessing.
- For a new trace, point CONTROL/ASYNC at the new summarize_trace.py
  output (never the raw 300 MB trace) and update CYCLES and the cycle facts.
- Keep files ASCII-only.

Validation: embedded JSON/ASCII checks and a CPU-only DOM run exercised all
40 detail views, 5 phase filters, text filtering and 13 sortable columns.
Browser layout has not been visually inspected. PLE is enabled on NVMe through the resolved second GGUF symlink; its
row-cache/O_DIRECT pread worker path is mapped, while joined request costs
remain unmeasured. The primary IQ3_XXS GGUF is on SATA. Source paths marked
project-repository refer to this B70 repo; base/overlay paths refer to Strata.

S25 adds the actual-weight C1/2/3/4/6/8 head curve and bounded production
PLE miss-ticket waits. These are separate scopes from the S22 operation
rollups; source-only C2 batching remains a hypothesis until serving tests.

S26 records an unresolved standalone EP flag-mode freshness failure and
a CPU-built, GPU-untested head-pair prototype. The reference trace remains
the frozen S21 binary. The live src-ep source now has the optional S26
prototype; reference source snapshots/provenance are recorded in the JSON.
Both-card/compiled-collective post-health passes. Failed-arm timing data
does not establish a serving gain or serving-output corruption.

S27 observes the intended wait epoch in debug failures: first receiver
copy consumes current poison while immediate second/third copies pass.
Native/debug arms and post-health are recorded separately. This narrows
the question without proving its cause. S28 corrects the S27 claim that
native BMG ISA was unavailable: hidden .text.*.asm files were present.

S28's same-binary default mirror fails112/648 first-read poison words;
explicit system scope on the producer workgroup barrier passes65/65.
The actual patched EP translation unit with native wait/no extra reads
also passes65/65. Decoded preflag UGM fence changes none.group to
invalidate.gpu; no compiler-bug/legality conclusion follows. Frozen engine
152055a6 passes small C2 output prefixes, drafting, math/reasoning/tools,
normal teardown and final both-card/compiled-collective health. Headpair
remains OFF and unqualified. No new speed/fullcache/shelf claim. See
[S28 evidence](../../../docs/20261015_strata_ep_publication_scope.md).

S29 tests that same frozen152055a6 engine with headpairOFF/ON: median C2
coding97.881->102.279 aggregate tok/s (+4.493%,3pairs), all six exact
request/output hashes match, both functionals/normalteardown/health PASS.
Native emit weighted7.033->5.380ms/window, ON1675paired/19solo. New C2
per-kernel times and driver-call deltas are not traced; S22's fourC1 heads
remain the reference operation rollup. Full depth2/headpair cache and
long-context performance are still unqualified. A pre-existing batch->solo
counter carry bug loses prior draft totals, so acceptance comparisons are
invalid; client wall-time and output comparisons remain valid. See stageS29.

S30 fixes that carry loss for new requests; native summed344accepted/
417offered equals finalserver/API after a real batch-to-solo handback.
Six CPU tests and functionals pass. Same frozen152055a6/headpairON/depth2
now passes four199501-token histories, two active256-token replies, and
four forced RAM returns:199839held reused/31new/0retained-history replay.
Draft KV byte-readback passes, peaknativeRSS16.255GiB, zero swap/OOM.
Native663.3-685.1ms/clientTTFT2.992-6.918s are correctness-run diagnostics,
not a clean speed comparison. Normal teardown/health is in stage evidence.
The overlap gate is depth-aware: mean kept rows must exceed3 at depth2.

Existing native timers from the first full pair show selection224/225ms
at the first12288-token block ->6278/6284ms at block16; attention grows
2291/2303->2630/2636ms. These are two ordinal sequences without A/B
attribution or exact context offsets, not a joined GPU budget. Value-only
cost remains unisolated. All170 S22 operation rows remain unchanged.

S31 adds [159 current replay operations](20261015_s31_replay_operations.csv)
with counts, median/p95/range, quant/template labels and grid/work-group.
Five reviewed capture-free cycles have twoC2draft heads (2.416ms summed
service/cycle), oneC6target head (2.262ms), and nine actual L0
ImmediateAppendCommandListsExp calls. The full trace's18C2heads reconcile
nine native paired windows. Both32-token replies match retained prefixes;
normal teardown and final per-card/compiled-collective health pass.

S31 quotes supplement the old S22 table; CSV host counts use complete
intervals, while the dated report also gives API-start counts and boundary
crossers. Host wait durations overlap GPU work, and invocation counts do
not establish logical barrier counts. The serving GR-down/reduce gap is
only0.209us median; the old standalone9.5us gap is not reused as a saving.
Primary recorded-activity gaps after MTPselect3.300-3.941ms and before
mapped-input copies1.555-1.648ms need source/uninstrumented verification;
no physical idle or production saving is inferred. See
[S31 report](../../../docs/20261015_strata_paired_head_profile.md).

S33 keeps the12288-token allocation and compares scheduling quantum0/4096.
The smaller quantum lowers maximum mixed decode content gap7.824->4.059s,
but increases incomingTTFT29.526->37.581s and decodercompletion41.619->46.612s.
KeepQ0; both state/prefix/functional/health screens pass, without new full200k
qualification. The primary gather region includes host peer enqueueing;
MMQ weight-gather stubs are unreachable in the actual SYCL build.

S34 reuses the old S10 trace: across 48 layers, the primary activation
gather takes median 0.149 ms, followed by median 52.987 ms without recorded
primary GPU operations. Matching host spans contain 58,716 peer kernel
appends. These overlapping scopes are not additive, not current S33
latencies, and not proof of physical GPU idle. Current bounded host
markers are still needed. The rejected S7 per-layer async arm is not
repeated. A one-gather/reverse-expert alias oracle is prepared only; no
build, GPU test, serving speedup or IPC qualification yet. See the
[S34 audit](../../../docs/20261015_strata_ep_prefill_dispatch.md).

S35 passes that standalone gate for T4096/12288: ten real format pairs,
363 synthetic-routed experts, 297,369,600 FP32 values bit-identical,
byte-gather/finite/guard/tail checks and per-card pre/post health PASS.
Ordinary device USM does not qualify production IPC or concurrent readers.
Reference-first/candidate-second timings have cold/warm order bias; no
serving speedup claim. See [S35 report](../../../docs/20261015_strata_peer_gather_alias.md).


S38 adds current CPU submission spans and actual IPC alias serving. First
4049-token diagnostic: peer-loop median15.826OFF->14.830ONms/layer, total
859.107->815.047ms/48layers; gather-return->firstDQ-entry15.947->14.963ms.
These are cold first-request CPU call spans, not GPU service or idle.
Later timed requests retain inactive compiled checks but no timestamp/log
capture:3x4K/3x32K/3C2 pairs perarm. All13 hashes/token/cache counts match;
ON48active/OFF48fallback, functionals/normalteardown/fullhealth PASS.
Prefill1303.0->1326.2/1546.1->1556.1 tok/s, C2 aggregate104.516->104.017.
Keep aliasOFF; no fullcontext/shelf promotion. Old170 operation rows and
rollups remain unchanged; no new GPU-clock join. See
[S38 report](../../../docs/20261015_strata_peer_gather_serving.md).


S39 tests the existing HC GRF option without changing the kernels:48 full
snapshot pairs/35,893,248 bytes and changed-input graphs PASS. Actual1260
DOWN/UP calls eachmode and kernel properties show128->256 registers,
private/spill0 both. All12 instrumented whole-read cases are slower; keep
default/no model-server test or fullcache repeat. Resources are reported
metadata, not dynamic occupancy or cause. No new serving speed result.
See [S39 report](../../../docs/20261015_strata_grf_resource_probe.md).


S41 adds the existing packedQ6 numerical discriminator to dense projections.
FirstactualC6 pair failsstrictbits/maxabs9.54e-7; selectedindependentCPU
references supportrounding explanation only. Bothhealth/teardownPASS, no
serving/tolerance/speedpromotion. Coldsinglekernelservice retained in raw,
not a speed forecast. Original170operationrows/rollups unchanged;52eligible
C6calls/3.569979ms source-supportedpool is ranking, not a wallbudget. See
[S41 report](../../../docs/20261015_strata_q6_dense_packed_probe.md).


S42/S43 retain40 stages/170 reference operations and prior rollups. Peer
prefill adds the grouped GU strict failure; dense adds the independent
full first-C6 CPU diagnosis. No new serving/speed claim. Different reduction
trees need independent accuracy checks; memory and same-path replay stay
strict. Prior failures remain evidence. Reports:
[S42](../../../docs/20261015_strata_grouped_gemm_probe.md),
[S43](../../../docs/20261015_strata_q6_cpu_reference.md).


S44 adds four-shape/six-column/changed-input Q6 arithmetic coverage:72cases,
2,623,488outputs satisfy declared source bounds; same-path graph/memory and
health/closure PASS. WarmC6 batch means show no useful packing benefit and
three shapes take79-90percent longer as configured. No main-owner integration
or serving run; S46 stays source-only. Repeated-weight cache regime differs
from serving and remains unmeasured. Reference170operations/rollups unchanged.
[S44 report](../../../docs/20261015_strata_q6_collection_and_warm_probe.md).


S47 actual CPU reader passes88cases: key-based matching removes extra
permutation/duplicate request tickets with identical decoded bytes. Source
audit finds missing batch-slot prefetch metadata, so helper alone has no
current C2 speed claim. Wiring remains source-only; no GPU window. Existing
170operation rows/rollups unchanged.
[S47 report](../../../docs/20261015_strata_ple_prefetch_readiness.md).


S48 same-geometry HC store predicate passes36 full numerical pairs and
preserved S39 hashes/ownchangedgraphs, health/teardown. Clean directwarm
T6 medians favorcandidate but failthepredeclaredspreadscreen; performance
is inconclusive, no integration. Graph-onlydiscriminator is separately
prepared. Reference170operation rows/rollups unchanged.
[S48 report](../../../docs/20261015_strata_hc_lo_once_probe.md).


S59 adds actual IQ4_XS C6 primitive evidence to dense_proj: strict/graph/health
PASS, tiny-.646percent quantize+projection observation, keepRPW1/no serve/retry.
S60 records the supported build boundary only, not an inference speed result.
S61 annotates small shared activation-quant pools and dormantQFUSE source
blockers. Historical170operation rows/rollups remain unchanged. See the dated
S59/S60/S61 reports linked from provenance.


S65/S68 add strict real-weight IQ3_S C6 SWAR evidence: gate primitive faster,
ssm_out slower, no global integration or serving forecast. Static code shrinks
and removes32 byte predicates without changing48 DP4A or three reconstruction
loads; those counts do not explain latency causally. S67 records a cheaper
future candidate measurement protocol and its weaker capture/layout visibility.
Historical170 operation rows/rollups remain unchanged. See dated S65/S67/S68
reports through provenance.


S73 closes the combined HC MORE_GROUPS primitive: all276 full-state files,
60 DOTs,312 checks and36 S39 anchors per process pass, with complete card1
health/removal. MORE1 is slower for T1/2/6 by16.08/19.10/27.35percent using
the median of two process medians, three nested20-read graph samples each.
KeepMORE0/no serve/retry/sweep; no production regression or physical cause
claim. Actual Opus/Fable post-reviews accept that bounded decision.
[S73 report](../../../docs/20261015_strata_hc_more_groups_probe.md).

S74 is offline compile/static ISA only: the BMG image contains one signed
INT8-to-INT32 DPAS, but no binary was executed and no128-output runtime or
canonical-IQ projection gate passed. No performance/occupancy inference.
[S74 report](../../../docs/20261015_strata_int8_xmx_capability.md).
Historical170 operations, stage rollups and operation metadata remain unchanged.


S75 adds a runtime-qualified canonical-Q8 IQ3_S C6 INT8 matrix path for two
real-weight, seeded/changed-input projections. All118 snapshots/92 binaries,
10 DOTs, memory/public-byte/health/cleanup gates pass in one38s card0 lease.
The current conservative local-ledger mapping is7.7-7.9x slower: keep main
unchanged, no serving/retry/sweep. Static DPAS and source barriers do not
establish a physical bottleneck or reject all integer-matrix designs.
[S75 report](../../../docs/20261015_strata_iq3s_int8_dpas_probe.md).
The170 historical operation rows and rollups remain unchanged.


S81-S85 separate indexer selection from attention and qualify two private
primitive candidates. S84 tiled scorer+unchangedtopk improves long/full256-query
cases but regresses short contexts; no main integration. S85 Q6 C6 packed
centering preserves all strict outputs/graph bindings and reduces two
repeated-weight projection times by22.8/22.1percent. These are nested primitive
observations, not serving speed, p95, measured cache residency or physical
latency causes. Small defaultOFF exact-path main-source preparation is next.
Historical170operations/rollups/metadata remain exact.
[Source, ISA and runtime report](../../../docs/20261015_strata_qsa_tiling_and_q6_centering.md).


S87 adds actual-main Q6 AOT and matched C2 serving evidence to dense_proj:
103.088->104.279 aggregate tok/s (+1.155%, three pairs in one OFF/ON sequence).
Full outputs/observable work/functionals/teardown/health match; keep guarded
defaultOFF opt-in. Source/AOT and one-time host marker are not independent
GPU dispatch or complete per-window layout proof. Historical170 operations,
rollups and metadata stay unchanged. See the dated QSA/Q6 report.
