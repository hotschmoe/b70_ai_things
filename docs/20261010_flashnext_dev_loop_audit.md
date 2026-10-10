# Flash-Next development-loop timing audit

CONFIG -> Read-only/CPU audit of finalized source35/C113 V8 native-GEN runs.
Exact UD-Q4_K_XL geometry unchanged. No GPU, Docker, compiler, model-payload
read, active controller edit, health-tool edit or qualification-policy change.

COMMAND -> Read AGENTS.md, full goal prompt, current completion audit and
recent JOURNAL. Parse parent-qualification.json, pre/post-health.json and
post-model-identity.json under the result root below. Inspect immutable V8
controllers and current health source. Execute both existing basic receipt
bindings on CPU and a bounded synthetic-memory SHA throughput control.

RESULT -> The dominant repeatable overhead is lifecycle/identity, followed by
engine startup; raw prerequisite recollection is a much smaller target.

Result root:
`/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/`

| Finalized directory | Total s | Pre-health s | Child interval s | Post-health s | Full4 s | Other s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| serial-cache-v8-onecard-basic | 713.2 | 60.7 | 442.4 | 65.0 | 142.4 | 2.7 |
| serial-cache-v8-pair-basic | 454.5 | 62.1 | 210.2 | 64.1 | 115.4 | 2.7 |
| serial-cache-v8-onecard-root | 449.4 | 60.7 | 168.2 | 64.9 | 152.8 | 2.7 |
| serial-cache-v8-pair-root | 330.3 | 60.5 | 84.1 | 61.6 | 121.4 | 2.6 |
| serial-cache-v8-onecard-pin | 499.0 | 58.9 | 202.2 | 64.0 | 171.2 | 2.8 |
| serial-cache-v8-pair-pin | 350.3 | 59.7 | 116.1 | 60.4 | 109.4 | 4.8 |

Total and health/identity times use explicit JSON epoch intervals. Child
interval is parent.child_terminal_epoch minus pre-health.finished_epoch;
it includes launch, prerequisite checks, process loading, requests, extraction
and teardown. It is NOT GPU execution time. Other is the arithmetic remainder.
Pair-root gates take 243.5s, approximately 74 percent of its total.
Four shard bytes total 111,334,654,784 (103.69 GiB), not 198 GB.

Each strict pair probe takes approximately 26-31s; compiled pair takes 31-35s.
These are upper intervals from command-file mtime to the next command or
health.finished_epoch, not instrumented per-process timestamps. Strict source
runs two separately pinned containers sequentially. Collective source already
persists TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR. It is a torch.compile
probe, not a freshly rebuilt standalone C++ binary on every invocation.

CPU basic_receipt_binding took 2.654s for one-card and 2.728s for paired basic,
including raw recollection and source/ELF association checks. Do not prioritize
replacing these meaningful checks with a cached Boolean.

Root request time sums were 45.23s one-card and 30.21s paired; pin sums 87.57s
and 39.27s. The command-file-to-first-warmup-start upper intervals were
109.37/34.76s for root and 99.37/59.32s for pin (one/pair). These estimates
derive first request file mtime minus final stdout_seconds and command mtime.
They show startup matters, but variable page residency and asynchronous setup
make them insufficient to attribute a particular loading bottleneck.

Ranked actions:

UPDATE -> The later source/protocol preflight below found V8 parked/eviction
requests cannot populate the RAM cache. Do not launch existing V8 all unchanged.
The grouping savings remain valid; the NEW V9 remaining suite is the current
candidate, pending independent review and coordinator-owned qualification.

1. Use the existing `--group all` suite when the remaining cases are ready.
   `serial_prefix_qualification_v8.py:run` already executes basic, root, pin,
   turn, parked, eviction, cancel, cancel_decode, cancel_prefill_isolation and
   real_live under a single parent. Each run_process still creates/removes a
   fresh owned engine; all six-request capture bounds, same-source flag arms,
   per-group numerical/state comparisons and model identity checks remain.
   Parent retains pre/post health, new complete four-shard scan AFTER all child
   work and post-health, known pages before/after, source closure and teardown.
   This is an already-defined experiment scope, not borrowing an old hash.
   For seven remaining groups, paired gates would cost about 7*243=1701s
   separately. One all-suite costs about243s gates but repeats basic/root/pin
   child work (210+84+116=410s observed). Estimated net saving: about1048s,
   or17 minutes per paired complete campaign. This is a projection, not an
   observed speedup. Failure yields whole-suite FAIL and retains partial raw
   evidence; do not present unfinished groups as independently qualified.
   First verify explicit group roster, capture bounds, prerequisite freshness
   and fail-fast behavior with source/CPU checks. Frozen V8 remains unchanged.
   Source review confirms ten groups produce twelve process results (three
   basic arms plus nine cache groups); exact armed rosters remain checked.
   Fresh pre-health is checked once at suite admission (within300s), not
   reinterpreted as fresh before every later case. Default child deadline is
   7200s; each readiness/request wait remains900s, with parent deadline owning
   the overall bound and interrupted work rejected. Previously unrun parked,
   eviction and cancellation routes can fail early; grouping trades failure
   localization/retry cost for fewer gates. Keep per-group artifacts and
   never skip an early failure to force a suite PASS. Final all receipt sets
   bounded_serial_prefix_passed only; complete_prefix_qualified stays false.
   Existing basic_receipt_binding accepts group basic, so keep the finalized
   old basic receipts as prerequisites instead of feeding an all receipt back.

2. If avoiding already-completed group repetitions matters, create a NEW
   immutable suite controller/plan with an explicit remaining-group roster.
   Keep a fresh engine per group and exactly one final owned parent gate for
   that declared suite. Seven groups could save about6*243=1458s, or24 minutes
   on paired work, before implementation cost/failure retries. Validate exact
   coverage, fail-fast/interrupt/deadline cleanup, no fresh-health expiry at
   suite admission, per-group source identity and final all-container-terminal
   chronology. No edits to frozen V8 or promotion of its partial receipts.
   A long-lived shared engine across unrelated groups adds state-isolation
   complexity and is a later option, not needed for this saving.

3. In a new parent, launch existing strict.sh --card0 and --card1 concurrently
   under the same held pair lease, then run compiled-pair sequentially. Both
   per-card matmuls/synchronizations, sentinels, timeout/error classification
   and owned cleanup remain required; no nested gpu-run or lease bypass.
   Approximate opportunity:13-15s each pre/post,26-30s per parent. Source is
   `vllm/int4/diagnostics/xpu_health_strict.sh`; dispatch is currently
   `qualify_serial_prefix_v8.py:health`. Parent ownership filters must cover
   both supervisor PIDs, and one failure must drain both before any collective.
   Validate CPU lifecycle negatives, then matched actual health repetitions.
   Existing bin tools need no edit; bin changes require shelf smoke checks.

4. Prototype bounded buffered SHA parallelism or a read/hash pipeline in a
   NEW identity helper. `qualify_serial_prefix_v8.py:full_buffered_identity`
   currently reads then hashes8MiB serial blocks and processes shards serially.
   Preserve all new bytes, per-shard SHA/size/stat-before/after, lock identity,
   postterminal/posthealth start bound and both source pages. No stat-only
   trust, digest reuse, direct IO, invalidation or model mutation. A synthetic
   1GiB memory-only control measured1.79GiB/s serial versus3.58GiB/s with two
   workers; real scans measured about0.61-0.95GiB/s, so IO/cache pressure may
   limit or reverse the gain. No numeric real-scan saving is established.
   Test tiny fixtures, truncation, changed bytes, changed stats, reordered
   shards and pre-health chronology rejection, then coordinator-owned matched
   serial/two-worker scans AFTER terminal/post-health, never beside a model run.

5. Keep arithmetic debugging on synthetic leaf programs/CPU replay until a
   full-model replay answers a new question. Fresh linked GDN35 leaf compiled
   in20.71s and HC35 projection leaf in20.06s, using source-indexed current SDK
   libraries. Continue deliberate ABI/source closure and real device checks.
   Original-owned and conditional diagnostic roles remain separate. Fullmodel
   fidelity, cache/concurrency and final qualification still need actual runs.

VERDICT -> Start with the existing all-suite option: it has the largest grounded
saving and needs no source replacement. Parallel strict health is the first
small implementation candidate. SHA parallelism needs measurement before use.
Fresh full4 identity, known-page guards, strict and compiled health, coherence,
model/source identity and healthy owned teardown remain mandatory. Skipping
these checks or treating earlier hashes as new proof would be a policy change,
not a compatible optimization. No model-speed or full-goal completion claim.

## Actual avoidable metadata failure

CONFIG -> Frozen HC35 synthetic projection V1, thirteen cases/two negative
controls, unchanged public GPU primitive. Coordinator-owned device run.

COMMAND -> Read `hc-projection35-v1-owned/parent-qualification.json`; derive
`len(fixtures())` and `sum(case['m']*case['t'] for case in fixtures())` on CPU;
count preserved actual raw output file extents. No GPU or model read.

RESULT -> Corpus has13 cases and36 output F32 words; actual13 raw files total
36 words. Frozen fixture plan and leaf prepared metadata claim38 words;
parent consequently rejects `Complete raw13/38words and2 negatives required`.
Coordinator raw recollection reports bitwise matches and both negatives.
Parent still correctly remains FAIL; normal post-health/full4 costs were
already incurred. Entire failed parent took225.071s. Parent SHA256:
`8717b749a1109c1c959712a61771fcbbabe7087d50c85b2f709a4cdc531608f6`.

VERDICT -> Before any GPU health launch, derive corpus case/word/negative counts
from source and compare them with fixture plan, input manifest, prepared leaf
and parent expectation. A tiny mutation test should prove mismatches reject
BEFORE health dispatch. This avoids an observed3.75-minute false failure,
without weakening mathematical checks. A new V2 is being prepared separately;
this audit does not claim its implementation or qualification is complete.
Retain V1 evidence rather than editing its historical38 to36.

Additional CPU preflight on both current serial V8 prepared plans passed:
all eleven required static token-case keys exist; IDs are integers inside the
248320 vocabulary; every static prompt plus64 output-token bound fits2048;
root21/pin191/turn63 prefixes match their paired token arrays. Actual lengths
are49/49/45,209/209,70/71 and55/55/51/55. Preserve these exact fixtures.
These checks should precede costly suite health/model loading. They cannot
predict parked-cache restore, eviction, cancellation timing or dynamic real
continuation correctness; those still require the declared GPU cases.

## Avoid a second predictable GPU failure: parked/eviction protocol

CONFIG -> Actual source35 SDK `sycl/src/program/generate.cpp` and immutable V8
serial protocol. Batch0; fresh engine per group; warmups all fresh1.

COMMAND -> Read cache construction7432, prefix ceiling9772 and lookup guards,
serial outgoing park9878 and reset10036-10048. Compare V8 parked/eviction
request branches. Independent source reviewer confirmed the same reachability.

RESULT -> Serial park_current is guarded by !req_fresh. V8 parked referenceB
fresh1 followed by independent-contextA fresh1 never saves outgoing B. V8
eviction referenceB and otherA/C/D all fresh1 never admit that victim into RAM.
The other cache.put belongs to inactive batch parking. These tests therefore
cannot demonstrate their required RAM restore/eviction routes as written.
This is a client recipe defect, not an engine mathematics change. No GPU run
was launched to rediscover it.

VERDICT -> New source-only V9 uses fresh0 with explicit pin0 for precisely
parked independent-contextA (both rounds) and eviction otherA/C/D. Pin0 gives
lookup ceiling0/cold input reconstruction while !fresh permits outgoing
parking BEFORE reset. All restore/victim/memory/output gates remain intact.
Raw extraction now requires exact cold GEN command, candidate_resume0,
actual_reused0 and all prompt rows evaluated; finalization recollects those
five requests, their exact label roster and JSON-normalized metadata equality
against the newly extracted raw vectors. V8 source and qualified basic prerequisites remain unchanged.

New artifacts: `serial_prefix_qualification_v9.py`,
`qualify_serial_prefix_v9.py`, `serial-prefix-source35-v9-source-plan.json`,
`test_serial_prefix_v9_cpu.py`. Suite roster is exactly the seven remaining
groups. CPU-only prepare clones the genuine frozen V8 plan and records its
digest/source association; research alias appends V9 remaining generation.
Parent preflight checks source/plan/roster and finalized raw V8 prerequisites
outside GPU try/finally, before any health dispatch. Ten CPU protocol/dispatch/
negative admission tests pass, including execution of the production AST request
branches with CPU fake requests (not a numerical/device qualification).
Temporary CPU cloning of actual one/pair plans passed; no model payload read.
Source-plan SHA256:
`e7ae64b9a588c4e75f3a82429bfd6a552f8406e18423dd1b482f618ede51cc10`.
Independent review and actual coordinator-owned runtime remain outstanding.

## Preserve completed GPU evidence across a reader failure

CONFIG -> Actual V9 one-card remaining-seven suite, source35 model unchanged.
Original parent remains FAILED. All seven numerical/group teardown results
passed; strict/compiled post-health, kernel and new complete4/page gates passed.

COMMAND -> Coordinator run55814 ended after approximately1521s. Read frozen
finalizer traceback and request records; prepare NEW read-only adjudicator and
V10 reader normalization. No GPU replay or edits to old result/source files.

RESULT -> V9 finalizer compares individual request JSON written BEFORE extract
with requests.json containing raw AFTER extract adds lifecycle_committed_ids.
That compare order was a reader bug introduced in the V9 strengthening. Exact
failure: `Cold raw request copy changed`, frozen finalizer line502; parent
errors contain only `V9 finalize rejected qualification`. Neither field may be
ignored. Recollect a COPY through actual frozen extract first, then compare
the complete derived raw object and complete newly derived metadata.

VERDICT -> NEW `adjudicate_serial_prefix_v9_v1.py` may establish an additional
bounded suite proof only after the exact original failure and all closed
source/plan/SDK/basic/health/kernel/owned-command/teardown/new4/page gates,
forty raw requests, seven production branches, full vector comparisons and
exact-template continuation recollect. It hashes the original evidence tree
before/after and writes only to a new destination. Original FAIL is retained.
Coordinator execution and final independent review are still required.

NEW V10 driver/parent normalize copied raw before full equality for future
paired testing. Five CPU tests exercise actual frozen extract's committed-ID
derivation, preservation and mutations; ten V10 protocol tests and four
adjudication fail-closed admission tests pass. This targets avoiding a26-minute
GPU rerun for a pure reader defect without claiming unproved numerical results.
Host Python lacks regex; full continuation recollection needs the coordinator's
qualified CPU Python environment. No GPU needs to be exposed to that reader.


## Coordinator adjudication completed; paired V10 started

CONFIG -> Exact source35/C113 original artifact; preserved failed V9 one-card
remaining-seven GPU run. NEW CPU-only reader in pinned Python image c388186d,
network none, 4GiB cap, no device grants; all original inputs mounted read-only.

COMMAND -> Root executes adjudicate_serial_prefix_v9_v1.py against closed V9
run in NEW serial-cache-v9-onecard-readonly-adjudication-v1/adjudication.
Root repeats all19 CPU controls; actual two-stream preserved-log replay uses
NEW replay_batch_native_failure_v1.py. Prepare fresh V10 paired plan from
frozen genuine V8 and start leased remaining suite (session47979).

RESULT -> Actual read-only adjudication PASS, all40 requests/seven groups;
complete raw derivation, first-head/all48 first-window vectors, cache routes,
eviction victims, cancellation/isolation, exact continuation tokens, complete
owned commands, health/kernel/source/SDK/new4/page chronology recollected.
Original evidence tree unchanged and original parent/child remain FAILED.
Turn reuse63; parked restore42/49 twice; real-live reuse51/79. Fresh references
and cached outputs match. Reader container exits0 and is removed normally.
Report SHA256: 0ba8b1eacb88a687fdf058e702656d8307a74521ab611a677d386a7c940f44a2.
Actual two-stream diagnostic replay reproduces old whole-trace AssertionError;
NEW ARM-scoped parser admits31 armed events/196vectors. Actual target results
are cancel6 and survivor32 length; diagnostic native budget is32, not64 and
not natural completion. No serial numerical equivalence has been proved.

VERDICT -> Bounded one-card serial remaining suite now has an additional
qualified closed-evidence proof. This is not complete prefix/API/concurrency/
model-quality/latency qualification. Paired V10 runtime started with fresh
pre/post health and a NEW terminal/post-health complete4 scan required. Grouping
reduces repeated gate dispatch by construction; no matched end-to-end saving
has been measured. Whole model fidelity, API concurrent cache identity, memory
tradeoffs, 1/2/4/bounded6 streams, critical-path profiling, clean latency and
verified shelf remain required. Never relabel old failed V9/V6 reports.
