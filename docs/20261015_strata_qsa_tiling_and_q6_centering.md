# Strata QSA query tiling and Q6 integer centering

Date: 2026-10-15. Backend: Strata upstream main296b59d2, two B70s.
These stages investigate small changes to the existing execution path.
The bounded RAM-recovery qualification from S30 remains retained. Serving
throughput targets remain unmet; source counts do not establish savings.

## S81: prefill scorer source audit

CONFIG -> Retained S30 phase summaries, S10 operation census, S22 decode
census and current QSA prefill source. No trace reparse, build or GPU run.
COMMAND -> Follow indexer append, score generation, top-k and attention;
check existing multi-query score arithmetic and public object availability.
RESULT -> Ordinary prefill passes a positive active-block count with up to
256 queries. That dispatch bypasses the existing eight-query scorer, which
can reuse each pooled-key load across queries. Both paths use the same
four-product expression, XOR reduction and per-query dead-key/tail rule.
An eight-query tiled launch can preserve those operations and score stride.
The original scorer accounts for only24.052ms in the historical4K trace;
this is primarily a longer-context lead. S30's increasing select buckets
include queue intervals and top-k; they are not scorer-only savings budgets.
VERDICT -> Prepare one private eager tiled scorer with exact score/ID gates.
Do not change attention, top-k or decode dispatch, or repeat the full200K
cache campaign to evaluate this primitive.

Raw evidence: `qsa-prefill-select-s81-prep/{findings.md,source-excerpts.txt,source-pins.json}`
under the shared result root below. Findings SHA256
`c2d5d5f821843bf969ffc3b047558540f76801c37f6f5313dcbf03870b1b8ab0`.

## S82: shared-expert scheduling source audit

CONFIG -> Existing native EP IPC/PUSH prefill, unchanged scalar expert math.
COMMAND -> Inspect routing readback, shared expert producers, scratch
ownership and final combine consumers against retained host/phase summaries.
RESULT -> Shared work is currently enqueued before the routing/IPC drain.
Moving it just after that drain, before CPU routing and peer submission,
could overlap independent primary work with host submission. Scratch and
consumers permit a narrow guarded ordering change. Vendor GEMM enqueue may
still block internally, and the measured router/shared bucket is a small
combined region rather than a movable shared-only latency budget.
VERDICT -> Park this lower-priority lead while S84 and S85 discriminate
larger, concrete kernel mechanisms. No patch, build or GPU result.

Raw evidence: `prefill-criticalpath-s82-prep/`; findings SHA256
`69fa0cb8147372dc7371d863eb38de90dac847ce9dc2735bd2cd66f0e22493b6`.

## S83: Q6 centering source and compiled-code audit

CONFIG -> Existing Q6 C6/RPW1/A2 kernel, SG32. Historical S31 dense census:
71calls/3.8256372ms. This is operation service, not request wall savings.
COMMAND -> Trace reconstructed six-bit bytes through signed-byte sub_sat;
seek/decode only the selected existing S55 device image under cpu-run,
pinned image/no DRI/no network/four CPU threads. No candidate compilation.
RESULT -> Each byte is0..63, so subtracting32 cannot saturate. The unsigned
packed identity `(u + 0x60606060u) ^ 0x80808080u` gives exactly the same
signed-byte bits without byte carries. The actual compiled original retains
32 subtract/overflow-mask/compare/select sites per static loop body before
48DP4A sites. The target has no call/calla; its extra320-byte symbol-table
kernel is separately inventoried and decoded. These are static sites,
not dynamic issue counts or a latency prediction.

The first extractor stopped before decoding because it incorrectly required
only one text section. Its evidence is retained. A versioned exact-target
selector and unexecuted Docker argument typo fix passed in one10s CPU lease.
Only51151bytes were read; no full binary/trace scan or GPU run was needed.
Target text16576bytes, SHA256
`ce58ba51362f5619ddea5f45ec2d6d3ac1fa943eb53325fef04619061a99b309`.
VERDICT -> Prepare a private two-expression Q6 candidate, preserving weight
format, A2 loads, Q8 inputs, dot calls, FP ledger and reduction tree. Inspect
its compiled code/resources and require full strict public/original anchors
before a short completed-wall comparison. No serving change follows yet.

Raw evidence: `decode-criticalpath-s83-prep/`,
`decode-criticalpath-s83-loop-supplement/`, `q6-center-native-isa-s83-root/`.
Completed ISA manifest SHA256
`5c2b3e04dcaaf9712695f4865668d90b0770491c7048a55697703a88357add84`.

Shared result root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.

Exact small source-audit records are retained in
[source_audits_s81_s83](../strata/flash-next/lowlevel_probe/source_audits_s81_s83/README.md).

## S84: strict tiled scorer passes, only long contexts improve

CONFIG -> Private eight-query tiled scorer using the byte-identical existing
multi-score body. Same public top-k follows each scorer on one in-order
queue. Five cases:256 queries ending at256/4096/32768/200000cells, plus a
ragged9-query case ending at200000. Deterministic finite/signed-zero inputs,
score stride50002, pooled50001rows including the spare row. No model load.

COMMAND -> Actual independent Claude Opus5.5/Fable5.1 reviews, responding
IDs `claude-opus-5-5`/`claude-fable-5-1`,63.895/152.729seconds. Both found
the unused private root-sync property concerning and otherwise accepted the
source. Root removed that private property in version3 without changing the
multi body. The public scorer has the same property with a large grid, so
the reviewers' cooperative-launch explanation is not asserted as measured
local behavior. The pinned SDK forwards the property to runtime; headers
alone did not settle its runtime policy. Version2/3 device wrappers are
byte-identical. Public top-k reads are independently confirmed within each
query's block bounds, with zero weight for an empty tail.

The first22s build stopped on a host noncopyable device handle. Version2
changes that declaration to a reference; its29s build passed. Version3
removes only the unused launch property and updates controller paths/pins;
its28s build passed. Failures/versions are retained. The final build uses a
small public-QSA prelink plus private adapter and ordinary host link; no
main source or native common-wrapper rebuild. Exact public object/archive
member, map, compiler and target identities pass. Two image registration
descriptors have distinct scorer/top-k targets; eight generic memcpy/fill
IDs overlap across wrappers, so this is not a blanket whole-program ODR
equivalence claim. Candidate staticSIMD32/GRF128/SLM16512/barrier_count1;
absent spill fields do not establish runtimezero or occupancy.

RESULT -> One92s quiet card0 window, leaf66s, all5cases PASS before timing:
all valid score bits and selected IDs, initial/changed/repeat/restored
queries, known observable score1->0, inactive poison, readonly inputs and
all guards.40binary exports total275523200bytes. Pre/post-health0, own full
container ID and name absent, wrappers0. No graphs, tracing, server or
collective. Final executable SHA256
`dd63db5a427f2670e59bd2e1f17d4ab6471076299d9667bef83aaf2bcd32832e`.

| Ending context / queries | Public median us | Tiled median us | Median paired change |
|---|---:|---:|---:|
| 256 /256 |59.344|164.228|+177.194%|
| 4096 /256 |866.923|997.833|+15.045%|
| 32768 /256 |4722.318|4363.269|-7.472%|
| 200000 /256 |21490.156|17130.835|-17.677%|
| 200000 /9 |495.059|482.254|-2.587%|

Each row has four balanced AB/BA/BA/AB pairs after three warm calls per
arm. Timings include submit through completion of scorer plus unchanged
top-k; reset/readback/checks are outside. Every timed output passes full
byte checks. Long200K paired changes are-17.555/-17.798/-23.015/-6.049%;
variation remains visible rather than hidden by a median. These are nested
observations in one process, not independent population samples, scorer-only
service or end-to-end serving throughput.

VERDICT -> Retain the private long-context/full256-query candidate; keep
the original path for short contexts and tails. No production integration,
threshold sweep, serving claim or full200K rerun. Any later integration
needs explicit production-domain bounds and a default-OFF comparison.
The larger measured mechanism is useful evidence; source-level eightfold
key reuse does not establish eightfold traffic reduction or a latency cause.

Both actual reviewers accept this bounded decision in their subsequent S85
source consults (Opus40.131s/Fable74.111s), avoiding a separate duplicated
post-review call. Root corrects Fable's near0 noise wording: all four near0
observations regress by176-189percent. Byte-identical device code does not
settle the runtime launch-property policy. Cleanup targets only the created
full container ID, while both ID and name must be absent before health.
The S84 completed manifest binds187files/282165114bytes, SHA256
`33047e7ca2aaacf7793ad02209ea08f63c3ab4a2fc5601cd634ac49f5255adb3`;
the later combined review records remain separately bound.

## S85: Q6 centering passes strict gates and reduces projection time

CONFIG -> Private canonical Q6 C6/RPW1/A2/SG32 original and candidate,
changing only the two integer centering expressions. Two actual weight
slices: `blk.3.attn_output.weight`6144->2560 and
`blk.4.attn_qkv.weight`2560->10240. Saved S44 initial/changed canonical
Q8_1 inputs, first6 of8columns, plus saved public output anchors. Inputs
are deterministic synthetic fixtures, not sampled serving activations.
The retained FP32 buffer is a readonly witness; kernels read Q8 and weights.

COMMAND -> Actual fresh Opus40.131s/Fable74.111s source reviews both
GO_BOUNDED_BUILD, root full-read. First16s build stopped on a missing include
for `strata::dp4a`; version2 adds only the original TU's pinned
`strata/sycl_math.hpp` include. Inverse removal reconstructs the original
header; leaf code is unchanged. Corrected offline build24s reuses the exact
common8f device wrapper and frozen archives. No common AOT regeneration,
network, DRI or main-source mutation. Adapter compile10.211s,
adapter image prelink0.587s, ordinary final host link0.359s.

An11s offline audit indexes only the137120-byte new adapter bundle and
binds both selected whole images byte-for-byte into the actual executable.
It reads205907bytes, with no common-image reparse. Original and candidate
both haveSIMD32/GRF128/48DP4A sites; spill/scratch fields are absent in both,
not measuredzero. The candidate removes32 static subtract/overflow/compare
sequences and has exactly8packed adds plus8XORs. Actual ASM prints these
immediates as signed decimal. Target text16576->10880bytes. The FP source
ledger/load/dot/tree remains unchanged; machine scheduling can change.
Separate symbol helpers are inventoried; no whole-program equivalence claim.

RESULT -> One41s quiet card0 window, leaf12s. Both shapes pass all146
snapshots: public matches saved S44, original matches public, candidate
matches original for initial/changed/repeated/restored inputs and own single/
20-call graphs. Q8/weights readonly, all guards/output tails and finiteness
pass. All correctness groups precede timing.10DOTs bind86kernel nodes and
76edges with the same three operand pointers and correct target names.
148binary artifacts total13343616bytes. Pre/post-health0, own full container
ID and name absent, analysis/identity/wrappers0.

| Projection | Original median us/call | Candidate median us/call | Median paired change |
|---|---:|---:|---:|
| 6144->2560 |63.172075|48.725638|-22.7841%|
| 2560->10240 |73.817238|57.480712|-22.0951%|

Eight balanced OCCO/COOC round means per shape, each arm comprising two
completed20-projection graph submissions. All16rounds favor the candidate;
paired ranges are-23.207..-22.479% and-22.425..-22.004%. These are nested
completed host-wall observations in one process. Quantization, reset,
readback and checks are outside timing. Repeated-weight cache behavior
differs from a model traversing many matrices. Barrier spans include their
boundaries; neither those spans nor static instruction counts prove an
isolated kernel latency cause, actual traffic or full-model gain.

Final executable SHA256
`a0a1756b2670422caab13da80cc7e45a7beabd1fc694d684b0e34789b80112e8`.
Selected ISA manifest SHA256
`f298be19d7a040feb9191dc27d424c0531356ecaa598948bb8b9ea48f689f3f4`.
Raw runtime records: `q6-center-s85-prep-v2/run-clean/`; root decision:
`q6-center-s85-runtime-decision.json` under the shared result root.

VERDICT -> Prepare one small default-OFF main-source option for exactly
the tested C6/RPW1/A2/SG32 and dimensions, preserving every other fallback.
This does not integrate or promote the private binary. Root will review
the actual source patch and build identity before a matched serving
comparison. No full-context/cache rerun or primitive retry is needed.

Actual post-review responses from Opus15.159s/Fable27.485s both accept this
limited decision. Root binds the explicit runtime comparison, which already
validates public DOT names/pointers and all snapshot phases. The earlier ISA
report's pending language describes its pre-run state. Repeated weights are
observed; cache residency and a bandwidth-bound cause are not measured. The
next full-model comparison must use the existing two-card EP configuration.
Root does not adopt the reviewers' proposed one-card substitution or another
streaming microbenchmark as a new gate.

Completed S85 evidence:323files/86839920bytes, manifest SHA256
`dfb4b31b54bc8b43fdd3da04de8c5636721e8a2aef2630c1d373a6c397d3111f`.
Small exact records: [S84](../strata/flash-next/lowlevel_probe/qsa_prefill_tiled_s84/README.md),
[S85](../strata/flash-next/lowlevel_probe/q6_center_s85/README.md).

## S86: default-OFF serving proposal is prepared, unapplied

The48-line source patch adds one compile-time specialization and a
process-latched `STRATA_Q6_CENTER` flag. Eligibility is exactly PF0, C6,
RPW1, A2, SG32 and either6144->2560 or2560->10240. Unsupported and nonzero-PF
paths retain their existing dispatch. A root correction explicitly excludes
unsupported positive-PF fallthrough, which does not take the old prefetch
early returns. No arithmetic or launch change affects the original branch.

Source SHA256 remains81558e95. Root read the complete changed function and
launcher, verified inverse reconstruction and ran `git apply --check`
against the external tree. No patch is applied, built or GPU-tested here.
The main executable/objects still contain the old S79 prototype and must not
be served as a current restored-source build. Next work is fresh upstream
and actual source review, an audited main rebuild including restored host
dependencies, then same-new-binary OFF/ON two-card serving.

[Exact unapplied patch and contract](../strata/flash-next/lowlevel_probe/q6_center_serving_s86/README.md).
Patch SHA256
`deb6e0c4f58f59095aa8d8f186b64784a44646be2f0a4f7ad4640b655993d71c`.

## Development-loop decisions

Use existing small summaries and real saved fixtures. Reuse compiler-supported
device-image prelinks where applicable; QSA needs its own small public-object
prelink because the retained native common wrapper contains no QSA kernels.
Keep source probes private, and overlap reversible offline builds with actual
Claude reviews after root source inspection. Resolve review findings before
GPU execution. Eager prefill needs no graph/DOT campaign. Announce short GPU
windows in the shared coordination log, use the CPU quiet lock outside the GPU
lease, and verify own-container removal before post-health. Do not restore
the daily server or repeat full context qualification between primitives.


## S87: guarded Q6 serving comparison passes with a small bounded gain

CONFIG -> Same fresh executable, `STRATA_Q6_CENTER=0/1`, native two-card EP,
C2/depth2/paired heads, common HC LO_ONCE, PUSH and NVMe Direct PLE. All24576
experts remain resident. Explicit WIDE1/MIN0/MAXC8/A2/RPW1/PF0/packedOFF;
head RPW2/S49 OFF. Only the Q6 flag and arm-local paths/names differ.
`STRATA_TRACE=1` CPU logging is matched; CPU rings/device profiling absent.

COMMAND -> Fresh upstream check remains296b59d2. Actual Opus27.688s and
Fable46.809s source reviews accept the bounded build. Root applies the small
S87 patch, including the once-per-process first-eligible host marker. The
219-second CPU-only incremental build performs three compiles, two archive
updates and one main AOT link. It recompiles the restored S79 verifier and
generator alongside the native kernel; generic build output now matches the
fresh frozen executable, rather than the old S79 experiment. Other archives,
HC object, CMake/ABI cache and build rules remain unchanged.

Executable SHA256:
`a1e35c4ddecff6cfcca8cd61945d0243274f4867e8eff4d134c4ad535b920900`.
Native source SHA256:
`c1cdeabe747754e5f0b3d2f4710c06d4dc50911fc692100be5191559a9ac8584`.

The12-second selected AOT audit finds unique original/candidate targets771/772
in the actual main binary. Both use SIMD32/GRF128 and48 static DP4A sites.
Original32 byte-saturation sequences disappear, replaced by8 packed adds and
8 XORs; target text16576->10880 bytes. Scratch/spill metadata is absent, not
measured zero. Only two ISA payloads are decoded; indexing1601 metadata headers
reads9.52MB. No whole-program ISA walk or common-archive rebuild is repeated.

Each arm starts once, warms one paired request, then completes three fresh
107-token/1500-output paired coding requests. Request1 is admitted only after
request0's flushed native prompt-start marker. Full requests, emitted text,
SSE content, usage, five required native work fields, windows, paired/solo
counts, rounded row summaries and logged new captures match for all four pairs.
Functionals cover concurrent math, reasoning and a tool round trip. Both normal
QUITs, own full CID/name removal, lease exits and per-card plus compiled
collective post-health pass. Arm1 reuses arm0's immediate full post-health;
there is no intervening GPU workload or stack change.

RESULT -> Timed pair wall29.1013->28.7691 seconds median; aggregate output
103.0883->104.2786 tok/s (+1.1546%). All three matched pair walls improve:
-1.1414/-1.2352/-1.2669%. These are three paired prompt samples within one
OFF-then-ON process sequence, without independent process-order replication.

| Arm / stream | Median decode tok/s | TTFT p50 / p95, s | Completion p50 / p95, s |
|---|---:|---:|---:|
| OFF / 0 | 53.516 | 0.931 / 0.935 | 28.945 / 28.976 |
| OFF / 1 | 55.867 | 2.081 / 2.083 | 28.893 / 29.273 |
| ON / 0 | 54.119 | 0.891 / 0.896 | 28.589 / 28.642 |
| ON / 1 | 56.475 | 2.026 / 2.041 | 28.560 / 28.907 |

TTFT/completion percentiles use linear interpolation over only three requests
per stream/arm, so they are descriptive samples, not population tail guarantees.
Per-request SSE event-gap p50/p95/max are retained in the root decision; events
may contain multiple tokens and are not token ITL. Successful warm execution,
source/AOT binding and exact host markers establish the selected host path;
there is no independent GPU-dispatch trace or complete per-window/cache-hit
layout observation. Rounded average rows are not an exact row sequence.

VERDICT -> Retain the guarded default-OFF opt-in source and frozen executable.
This is a small bounded serving gain, not the S85 projection-only22% result,
a default/daily-shelf promotion, or new full-context qualification. Decode is
still54-56 tok/s per stream, below150; prefill targets remain open. Keep the
S30 RAM-recovery evidence and avoid another full-cache campaign here. Existing
model hashes plus unchanged NVMe paths/device/inode/size/mtime avoid another
75GB read; this is continuity evidence, not a fresh full payload hash.

Source parsing enables only when the first character is1; this comparison uses
literal0/1. The flag latches on the first eligible host call, possibly during
capture. The old comment's before-capture wording is accepted as an overstatement,
without a comment-only rebuild. `aggregate-proof.json`'s legacy `requests`
field counts PLE rows; the raw final diagnostic is authoritative. Nonzero
common prefetch counters do not establish that S49 was enabled.

Raw root: `/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`
`q6-center-serving-s87`, with sibling build, main-AOT and actual review directories.
[Small frozen records](../strata/flash-next/lowlevel_probe/q6_center_serving_s87/README.md)
bind the retained large external evidence. Next candidate is guarded long/full256
prefill QSA tiling from S84; short/decode paths stay original. No additional
primitive run, full cache run, daily restore or timing retry was added.

Actual postreviews: Opus16.922s/Fable33.389s both ACCEPT_DECISION. Root
read both completions and retains defaultOFF opt-in. Specific admission-gap
cause and verify critical-path share remain unmeasured; a reversed-order
repeat is an optional future discriminator, not a new run scheduled here.
Ratio of arm medians is+1.155%; median of paired gains is+1.251%.

Completed external evidence:260files/231808577bytes, manifest SHA256
`40d31d4466bb47f78fdfabb72542057fc6bd2a5b8de6d90e3e05cb94e0b11f9d`.
Repository curation retains83 exact small records plus index/README (85files,
332868bytes); large responses, images and binaries remain external.


## S88: measure the missing 16K boundary before choosing a guard

The first conservative integration proposal would activate only when the last
query reaches32768 cells. Retained fresh32K requests have32721 main-prefill
tokens; their final full256-query batch ends at32512, followed by a209-query
tail. That guard would activate zero times. The proposal remains unapplied;
a33K-only request was not substituted for the32K goal.

One new16K case reuses the exact S84v3 scorer and public reference, with256
queries ending at16384 (first16129, active4097). Only the host case and truthful
result counts change. A20-second CPU build reuses the public prelink and proves
the private device wrapper byte-identical. Both complete image bundles occur
exactly once in the new executable; no repeated ISA audit is needed.

All valid score bits/selected IDs, changed/repeat/restore, known observable
1->0, inactive poison, read-only inputs and guards pass:8 exports/25050112bytes.
One quiet card0 window41s includes26s leaf and post-health. Root reuses S87
prehealth after checking no intervening GPU/stack change, current boot/kernel,
leases and containers. Own full CID/name removal, post-health and wrappers pass.

Three warm calls per arm precede four balanced AB/BA/BA/AB samples. Completed
scorer plus unchanged top-k medians are2433.2445->2351.358us; median paired
change-3.6612%, with all four negative (-3.039/-1.371/-5.751/-4.283%). This is
one seeded process and host completion, not isolated scorer service, physical
traffic, significance or a serving gain.

The result supports source preparation of an earlier active-block guard4097,
which means the LAST query reaches16384; earlier queries can be below it.
Short4K, tails and decode remain original. Actual Opus35.397s/Fable49.602s
source reviews pass; combined S88post/S89source Opus28.091s/Fable41.468s accept
this bounded decision. Main build and serving qualification belong to S89.

[Exact small records](../strata/flash-next/lowlevel_probe/qsa_prefill_boundary_s88/README.md)
retain54 byte-exact copies plus index/README,56files/193832bytes. Completed
external evidence138files/27720705bytes, manifest SHA256
`abadf5f91f98ce6a0ff1e24fe1cdc1fb7b0ae5dbac999ce66318152af3f3dddc`.
The development-loop saving is concrete: the source coverage check prevented
an ineffective main build/serve; one boundary case reused its public AOT and
health evidence, and the post review shares the next source consultation.


## S89: actual-main guarded QSA serving comparison is a small directional result

CONFIG -> Same new executable, two-card EP149/363, all24576 experts resident,
depth2/headpairON, two200000 slots and32GiB parking. Only literal QSA tiled0/1
varies. Q6center0 in both arms; the separate S87 Q6 result is not combined.
The exact256-query/4x128/R4/stride50002 guard activates at active4097..50001;
short4K, ragged tails and decode keep the original path. One first-eligible
host marker records selection, without per-operation instrumentation.

COMMAND -> Incremental CPU build169s (Ninja155.654s): only QSA TU compile,
kernel archive update and main AOT link. All other archive members, native,
restored verifier/generator objects, HC and CMake/build rules remain exact.
New executable SHA256:
`b2a610ea23856ad290cbca2a0dcdec4de676f3cdb519ddc0a01c5321942636be`.
QSA source SHA256:
`28117295f242483d62c23e060adf0c3b49be9c1868d80669bff58b7fcdbfd5a1`.

The10s selected main-AOT audit independently finds candidate image550,
SIMD32/GRF128/SLM16512/barrier_count1/text22592B/no CALL. Original scorer
image551 is byte-exact S84. Wide/ref top-k compiled bytes differ across
standalone/main links; both serving arms use the SAME new top-k/executable.
Absent scratch/spill metadata is not zero. These are static resource facts,
not occupancy, dynamic barrier counts or independent dispatch evidence.

Each arm starts once, warms one107/1500 C2 coding pair for coherence, then
runs three exact fresh4059-token and three32728-token requests,256 outputs
each. All six full requests/text/SSE/usage/five native-work fields and actual
native part intervals match; all cache_n0. Warm work/windows/captures and
admission overlap match. First eligible marker is enabled0/1, first_active4097;
source-derived subchunks are not observed GPU dispatches. Arithmetic, reasoning,
tool round trips, normal QUIT, own full CID/name removal and per-card/compiled
collective post-health pass in BOTH arms before aggregation.

RESULT -> At32K all three paired prompt times improve: -0.5474/-0.4345/-0.7528%,
median paired -0.5474%. Native prompt medians21148.8->20989.6ms; throughput
1547.5->1559.2tok/s (+0.7561% ratio of medians, descriptive). Client TTFT
21.2129->21.0539s, about159ms lower; request wall24.7185->24.6308s.
The inactive4K path is null/mixed: paired +0.9830/+0.0995/-0.8704%, not a
kernel gain. Its throughput medians1289.3/1300.7 are retained descriptively.
One fixed OFF-then-ON process sequence with three samples per shape cannot
separate this small effect from order/thermal/cache drift or establish
significance, stability or population tail latency. SSE gaps are event gaps,
not exact token inter-token latency. No isolated kernel cause is inferred.

VERDICT -> Retain guarded default-OFF opt-in and frozen executable for research;
no default/daily/shelf, combined-Q6 or new full-cache promotion. The paired
32K effect is the headline;4K is null. Absolute2000+tok/s prefill and150+tok/s
per-stream decode goals remain unmet. Existing S30 RAM recovery remains valid.
Actual postreviews Opus25.288s/Fable42.218s accept; root corrects Fable's
mistaken2000ms target to2000tok/s, and both answers' shorthand scorer-only
for S88 means completed scorer PLUS unchanged top-k. All raw answers persist.

The loop omits timed C2 repeats, extra32K warm, cache rerun and daily restore.
Both GPUs are released. Next prepared discriminator is ONE separate current32K
operation/readiness trace with existing S10/S31 unitrace/clock/stop-flush tools,
current149/363 ownership and saved request capped32 outputs; NOT RUN in S89.
This should distinguish GPU service from submission/wait/readiness intervals,
without equating recorded gaps to physical idle or guessing PCIe traffic.
[Exact small S89 records](../strata/flash-next/lowlevel_probe/qsa_prefill_serving_s89/README.md).

S89 completed external evidence:266files/211396305bytes, manifest SHA256
`560858c7e1b06d235398a5a907452f4add5c92b034512b0065d64d5fa1792cad`.
Repository curation retains68 exact small copies plus index/README,70files/293254bytes.
