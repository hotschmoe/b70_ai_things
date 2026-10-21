# Q1 - QSA real serving integration

## CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: resume X6 after the power outage. Own b70_qsa_xpu and x6/q1
harnesses. No GPU, builds, DD lifecycle, commits or submissions yet.
Preserve the shared fork WIP; root owns post-outage health and baseline.

COMMAND: inspect HANDOFF.md, BRIEF.md, last campaign milestones, X6
report/scripts and retained x6/tiny.log. Run existing CPU nested serve
and KV-spec tests with UV CPython 3.12.13 and campaign venv packages.

RESULT: QSA_SERVE_OK and QSA_SPEC_OK. The pre-outage synthetic engine
did run and exited 1 at 05:25:41 during V2 graph-memory profiling:
attention metadata existed but the runner could not bind QSA pages.
Native pre/post card-1 health passed in that old window. Current source
has a more useful expected/actual page diagnostic than the failed log.
No x6/strata-references.json exists; the reference window was queued.

VERDICT: component graph and CPU spec gates do not establish real-engine
integration. Retry the bounded synthetic engine before full Flash-Next.
The tiny harness now asserts all 12 live uint8 page aliases after engine
init and C1/C2 generation, and refuses a private-pool fallback.

## Prepared bounded window

CONFIG: physical card 1 pinned by xpu-run, random dummy 48-layer model,
12 QSA 24/2/256 layers, narrow trunk/experts, PLE off, no EP, maxlen
4096, chunk 128, FULL_DECODE_ONLY captures 1/2. No latency/quality claim.

COMMAND: `Q1_TAG=tiny-r1 bash /mnt/vm_8tb/b70/build/vllm-flashnext/q1/tiny-window.sh`.
180-second Python alarm; 240-second outer timeout; native pre/post
card-1 health. Expected 2-4 minutes. Raw files under q1/tiny-r1.

RESULT: prepared, syntax checked, waiting for root authorization after
its baseline and the bounded D1/N1 probes. No GPU touched by Q1.

VERDICT: preserve sources while baseline is loaded. Only after the tiny
gate passes, request coordinated Strata references and full QSA serve.

CPU planner cross-check: q1/cpu-planner.py uses the current fork's real
DefaultKVCacheConfigBuilder and allocate_kv_cache with 48 layers, including
36 GDN state fixtures and 12 QSA layers. At block 2752/page 3258368 and
two minimal blocks it returns all 12 contiguous uint8 views accepted by
as_pages. Q1_CPU_PLANNER_OK; raw q1/cpu-planner.log. Existing CPU serve
and spec tests also pass with the fork first on PYTHONPATH; source path
and serve results are recorded in q1/cpu-tests.log, and the independent
spec process in q1/cpu-fork-spec.log. Running those two test scripts in
one interpreter fails the spec fixture's expected byte slope because the
serve fixture deliberately leaves the backend's class-wide indexer
geometry set to its small 32-wide geometry. The intended independent
test processes pass; this is not evidence of a real-model failure.

Both model_hook.py and its CPU test were last modified at 05:28:15,
after the 05:25 tiny failure. Current code retains the worker config
from KV-spec collection and reads its late-aligned block size outside
the config context. That may already repair the old mismatch; only the
bounded real-engine retry can establish that.

Prepared reference window: q1/strata-window.sh, both cards, <=8-minute
timeout plus native health, no service lifecycle. Its research config
q1/strata-reference-config.json preserves the frozen Strata model and
QSA settings but sets --spec 0 and STRATA_BATCH_MTP=0, removing the MTP
model/draft arguments. It writes q1/strata-references.json for the exact
existing x6/reference-prompts.json token fixtures. Not run or authorized
yet; this is a correctness reference, with no latency comparison claim.

Prepared full serve: q1/serve-window.sh uses the O4+F1 no-MTP recipe,
maxlen 65536/chunk 12288 and required q1 Strata references. The x6 serve
and probe now propagate failed greedy/latency/repeat/reference/source/
teardown/health gates to a nonzero window status. Fixture hashes and
reference token counts are checked. The wrapper runs the short bench
and 4K/16K/32K solo-repeat/C2 reference gates, skipping the redundant F1
128K diagnostics on the first QSA integration window. Syntax and diff
whitespace checks pass. Not started; the root's recovered baseline still
fails strict repeat, so an eventual N1 determinism flag can be supplied
through O4_SERVE_EXTRA before interpreting a QSA exact-repeat result.

## 16:23 - synthetic full-engine integration passes

CONFIG: tiny-r1 exactly as prepared above, physical card 1/logical 0,
unchanged QSA plugin/kernel/fork, no performance measurement. Root gave
explicit authorization after D1/N1 closed their bounded windows.

COMMAND: `Q1_TAG=tiny-r1 bash /mnt/vm_8tb/b70/build/vllm-flashnext/q1/tiny-window.sh`.

RESULT: TINY_RC 0 / HEALTH_RC 0. Native card-1 health HEALTHY before and
after; all source hashes unchanged; no crash or reset. V2 graph-memory
profiling binds 12 real QSA layers with two pages each and captures FULL
sizes 1/2. Final allocation rebinds all 12 layers to uint8
[109,1,1,3258368] at block 2752 and captures FULL 1/2 again. Live alias
assertions pass after initialization and after generation. Both C1 and
C2 return eight tokens per request; X6_TINY_FULL_DECODE_OK. Raw
q1/tiny-r1/{tiny.log,summary.json,health-before.txt,health-after.txt,
source-after-check.txt}. Card released to root at 16:23 UTC.

VERDICT: real KV-planner/profiling/final-capture lifecycle now passes.
The existing 05:28 worker-config repair suffices for the old tiny
failure; no speculative source change was required. This remains a
random-weight integration gate. Flash-Next coherence, reference match,
latency and full-model memory behavior remain unmeasured with QSA.

## Frozen CPU review of the reference path

CONFIG: D1 matched pair owns both GPUs and freezes all source/config
files. Read-only review of Strata reference argv, tokenizer, image and
source; report edits only.

COMMAND: compare prepared q1 config to
strata/flash-next/night/configs/dd-nomtp-c1.json; inspect native-pack and
serve gates in src-ep/src/program/generate.cpp:3026 and :7240, the plain
decode selector at :12678, and StrataEngine sampling/EOS handling.

RESULT: the prepared --spec 0 is invalid for this native serve: spec>=2
is required as verify-buffer capacity. No --mtp plus suffix-draft 0 /
lookup-chain 0 selects one token per round, independent of that buffer
capacity. The existing canonical dd-nomtp-c1 config already expresses
this qualified no-draft recipe with parallel 1; its environment matches
the prepared reference exactly. It uses the same release executable,
GGUF/PLE/pack/profile paths and KV int8. The pinned image e41d59c0 is
exactly nl_launch.sh's image; native tokenizer files exist and their
fixture equality already passed. HF generation_config and Strata both
stop on IDs 248044/248046; temperature 0 selects greedy in both.

VERDICT: root authorized replacing the scratch reference config with
the canonical dd-nomtp-c1 config after the D1 freeze, changing only the
research alias/log. Do not run the currently prepared spec0 config.
Retain engine_info, engine_last and executable SHA in every reference,
and require runtime zero-draft evidence. These C1 references do not prove
Strata C2 parity; the later vLLM C2 equality check remains its own gate.

## 16:43 - canonical reference CPU preparation ready

CONFIG: root authorized scratch q1 reference config/launcher changes
only during the corrected D1 treatment freeze. Shared campaign harness,
plugins, fork and registry remain untouched. No GPU/build/weight load.

COMMAND: copy canonical dd-nomtp-c1.json to the scratch reference config,
change only aliases/log; compare resolved Strata engine_args; run frozen
x6_strata_reference.py --check-tokenizer with UV CPython 3.12.13 and
Strata venv packages. Disable bytecode writes. Syntax-check scratch
launcher and host reference-output validator.

RESULT: Q1_CANONICAL_REFERENCE_CONFIG_OK. Only aliases/log differ;
resolved argv equals the canonical argv exactly: spec2, no MTP,
suffix0, lookup0, parallel1, no injected --batch flag. All prompt hashes
and prompt+output context limits pass. X6_STRATA_TOKENIZER_OK at exactly
4096/16384/32768 fixture IDs. Raw q1/cpu-reference-config.json and
q1/cpu-reference-tokenizer.log. No native executable was started.

VERDICT: ready for explicit root both-card authorization after D1/N1.
Exact command: `bash /mnt/vm_8tb/b70/build/vllm-flashnext/q1/strata-window.sh`.
Expected 2-4 minutes, 8-minute body timeout plus native pre/post health.
The scratch launcher now checks initial RAM, hashes config/canonical/
helper/fixtures before and after, and propagates health/source/output
failures. Its scratch validate-references.py requires all three matching
fixture hashes/counts, nonempty outputs, stored engine_info/engine_last,
consistent executable SHA and zero offered/accepted draft counters. That
runtime gate is prepared and syntax checked, not yet exercised. It saves
a separate reference summary without changing the frozen generator.

## 17:08 - real Strata references close PASS

CONFIG: root-authorized both-card canonical dd-nomtp-c1 engine, release
52d1778c909b90123a8947a3a2a9700370b76a2f64d293765f0e2510526fdad2,
parallel1, spec2 verify buffer, no MTP/suffix/lookup drafts. Research
alias/log only differ from the canonical config; no DD service start.

COMMAND: q1/strata-window.sh, 8-minute body cap, native pre/post health,
input/source hash checks, host output validator.

RESULT: STRATA_RC0 / VALIDATION_RC0 / HEALTH_RC0 / SOURCE_RC0. All three
4096/16384/32768-token prompts generate 64 output IDs. Each actual
engine_last reports zero drafts offered/accepted/reused and full-length
prompt_read. Engine info, raw and visible IDs, text hash, executable SHA
and config identity are retained. Native health HEALTHY before/after,
normal exit, no reset, inputs unchanged. Actual Strata EP split is
231/281 experts per layer. Canonical config SHA 2e0caec54286b589...;
research config SHA 9cac99e468e317c3.... All raw q1/strata-* artifacts
and q1/strata-references.json are under the campaign build root.

VERDICT: qualified independent C1 reference material for the first QSA
serve comparison. No Strata C2 parity or matched latency claim. The
64-token outputs include reasoning, so this is an exact continuation
comparison, not a completed needle-answer score.

## Full QSA attempt q1-r1 - running

CONFIG: original O4+F1 recipes plus QSA, explicitly N1/D1/oneDNN-det
flags0, original c7b FA library path, no new model/kernel edits. Both
cards, maxlen65536/chunk12288, FULL_DECODE_ONLY1/2/4, no MTP. Required
Strata refs and 4K/16K/32K solo-repeat/C2 comparisons; no redundant128K.

COMMAND: `Q1_TAG=q1-r1 bash /mnt/vm_8tb/b70/build/vllm-flashnext/q1/serve-window.sh`.
Root authorized one bounded attempt; start17:09:08UTC, expected6-10min,
40-minute outer/20-minute server cap. Monitor post-ready progress with
the root's120s no-progress abort bound, allowing firstJIT/32KC2 work.

RESULT: pre-health HEALTHY; native engine loading normally at17:10.
Actual12-layer binding, stable alias,323/667, long refs, engine-error
scan, normal teardown and post-health remain required.

VERDICT: first integration/fidelity attempt only. Original baseline
strict nondeterminism remains explicit; no promotion or stability claim.
The harness preserves root's ENGINE_ERROR_RC gate and now also records
12unique QSA bindings/no private pool and active GGUF/FA/harness/reference
source hashes. Shared active sources frozen for the run.

## 17:17 - full QSA integration completes; fidelity unqualified

CONFIG: the single q1-r1 attempt above, original O4+F1 math, N1/D1 off,
same 4K/16K/32K fixtures and independent canonical Strata C1 references.

COMMAND: the approved q1/serve-window.sh, with cpu-run --quiet around
the short benchmark and finite long comparison. No retry or extra serve.

RESULT: ready17:13:06,12unique actual QSA page bindings, no private pool,
stable hotschmoe-dd first plus registered x6-qsa alias. 323/667 and short
coherence pass. Native graph profiling/final initialization complete;
card0 reports8.55GiB available KV/567978token capacity at configured
maxlen65536 (a capacity diagnostic, not2x200K qualification).

The Strata comparisons below are per output, not the aggregate boolean:

| Prompt | Solo0/solo1 vs Strata | C2stream0/1 vs Strata | Solo repeat | C2 vs solo0 |
|---|---|---|---|---|
| 4096 | PASS/PASS | FAIL/PASS | PASS | FAIL/PASS |
| 16384 | FAIL/FAIL | FAIL/FAIL | FAIL | FAIL/FAIL |
| 32768 | FAIL/FAIL | FAIL/FAIL | PASS | PASS/PASS |

All streams complete64output tokens. At32K both concurrent outputs equal
the repeated solo output but differ from Strata. The4K aggregate
strata_exact=false is caused by only one C2 stream; its solo outputs do
match Strata. The original dense baseline already fails strict repeat,
so these results do not establish QSA as the cause of drift.

Diagnostic wire ITL p50: shortbench C1=18.18ms,C2=22.71/22.74ms;
long4K C1=30.22ms,C2=35.65/35.65ms;16K C1=47.60ms,C2=53.25/53.24ms;
32K C1=70.84ms,C2=77.10/77.12ms.32KC2stream0 has p99=8972.91ms and
mean386.78ms, versus stream1 mean76.49ms. Preserve that scheduling/timing
diagnostic without assigning cause. Historical client saves percentiles
but not individual arrival timestamps; no tail mechanism is proven.

Closure: X6_PROBE_RC1 (fidelity/repeat), QSA_BINDING_RC0,GREEDY_RC0,
BENCH_RC0,SERVE_RC0,ENGINE_ERROR_RC0,HEALTH_AFTER_RC0,SOURCE_UNCHANGED.
Actual EngineCore SIGTERM/drain at17:16:54 logs request processing
complete/resource teardown, then engine manager stopped at17:16:55;
no force-kill. Container exits0 after446s. One Python resource-tracker
semaphore warning appears at API exit. Both native post-health probes
HEALTHY; no crash/reset. Window closes1 and both leases release by17:17:30.
Raw x6/flashnext-ista-iq3-xxs-ep-qsa-q1-r1; derived per-output summary
q1/full-qsa-r1-summary.json. Raw JSON/logs remain unchanged.

VERDICT: full QSA serving integration works through coherent requests,
finite long-context requests and normal teardown. Fidelity/repeat gates
fail, so QSA remains unqualified and no promotion/stability/speed claim
is made. NEXT: same original-FA recipe with QSA disabled, same fixture
IDs/lengths, maxlen65536/chunk12288,64outputs,C1 repeats+C2. Compare its
per-output Strata matches and repeat behavior before attributing drift to
QSA. Later timing work must save content-chunk arrivals in a shared client
for both treatments. Loaded-context liveness localization and default-off
P1 serving integration remain separate coordinated campaign work.

Reproduction ports (CPU-only, after teardown): harness/q1_serve.sh,
q1_strata_reference.sh,q1_strata_reference.json,q1_validate_references.py.
The reference launcher copies the preserved canonical config to scratch,
then uses the same frozen generator and explicit zero-draft validator.
Generate fixtures with the campaign venv and
`python harness/x6_make_prompts.py /mnt/vm_8tb/b70/build/vllm-flashnext/x6/reference-prompts.json`
from vllm/flashnext. References: `bash harness/q1_strata_reference.sh`;
QSA: `Q1_QSA=1 Q1_TAG=q1-r1 bash harness/q1_serve.sh`; dense control:
`Q1_QSA=0 Q1_TAG=q1-dense-control-r1 bash harness/q1_serve.sh`.
GPU commands require the parent-coordinated pair lease and DD stopped.

## 17:30 - QSA-disabled control reproduces fidelity/repeat failures

CONFIG: original O4+F1/c7b Flash Attention, QSA/prefill-QSA off, N1/D1/
oneDNN-det0, otherwise same maxlen65536/chunk12288/capture1,2,4/maxseq4,
same fixture IDs and64outputs. No prefix cache/MTP. Root reconfirmed DD
stopped and both leases free. Mandatory native pre-health was retained.

COMMAND: `Q1_QSA=0 Q1_TAG=q1-dense-control-r1 bash harness/q1_serve.sh`
from vllm/flashnext. Start17:23:06,ready17:27:11. Root's shared client
adds arrival offsets/individual ITL samples/wall time only; request body,
sampling and existing timing summaries are unchanged. Per-output Strata
booleans are now saved directly as strata_solo_exact/strata_c2_exact.
Original QSA raw JSON/client version remains preserved by the prior
checkpoint and source manifest; it lacks the new arrival arrays.

RESULT: runtime Flash Attention selection and FLASH_ATTN cache block-size
lines present; zero QSA registration/binding lines. Registered dense alias,
stable hotschmoe-dd first,323/667 and short coherence pass. All12finite
long streams complete64outputs,64content chunks and63individual ITLs.

| Prompt | Solo0/solo1 vs Strata | C2stream0/1 vs Strata | Solo repeat | C2 vs solo0 |
|---|---|---|---|---|
| 4096 | PASS/FAIL | PASS/PASS | FAIL | PASS/PASS |
| 16384 | FAIL/FAIL | FAIL/FAIL | FAIL | FAIL/FAIL |
| 32768 | FAIL/FAIL | FAIL/FAIL | FAIL | FAIL/FAIL |

At4K both backends match Strata on3/4streams, but the failed streams and
failed texts differ. At16K both backends produce3unique texts with no
shared complete continuation between the arms. At32K QSA produces one
repeated text across4outputs; dense produces3unique texts, none equal to
that QSA text. Thus attention changes the continuation, while the exact
fidelity/repeat failure is already present with QSA disabled. No unique
QSA causality, baseline correctness, or QSA determinism benefit is proven.

Diagnostic latency for the same serving settings/fixtures follows. This
is NOT the canonical512-token benchmark or a fidelity/speed promotion.
There are2solo requests and oneC2 pair per length. TTFT is observed first
content in seconds; C1 uses solo1. Other columns are milliseconds; pairs
are C2stream0/stream1. Every stream has64content chunks, so TTFT is not a
request-wall fallback. The two arms differ in attention semantics and
cache geometry, and the historical QSA arm has no individual arrivals.

| Context/backend | C1 TTFT s | C2 TTFT s | C1 p50/mean ms | C2 p50 ms | C2 mean ms | C2 p99 ms |
|---|---:|---|---|---|---|---|
| 4K original FA | 2.125 | 4.215/4.236 | 17.81/17.80 | 22.89/22.89 | 22.85/22.79 | 23.38/23.43 |
| 4K QSA | 2.807 | 5.630/5.651 | 30.22/30.23 | 35.65/35.65 | 35.35/35.50 | 36.18/36.25 |
| 16K original FA | 7.033 | 13.474/13.499 | 24.21/24.07 | 29.15/29.15 | 28.97/28.95 | 29.79/29.95 |
| 16K QSA | 12.531 | 24.564/24.592 | 47.60/47.52 | 53.25/53.24 | 52.72/53.02 | 53.83/54.03 |
| 32K original FA | 14.112 | 20.399/28.309 | 32.67/32.39 | 37.23/37.22 | 160.76/36.75 | 3593.79/38.07 |
| 32K QSA | 31.321 | 43.078/62.838 | 70.84/70.57 | 77.10/77.12 | 386.78/76.49 | 8972.91/78.56 |

Observed QSA C1 p50 cost above original FA is12.41/23.39/38.17ms per
token at4K/16K/32K, or1.70/1.97/2.17x total ITL. This is a diagnostic
cost curve, not an accepted speed comparison. QSA int8 increases planned
KV capacity from345553 to567978tokens at these settings (FA8.53GiB and
QSA8.55GiB available KV); this is planned capacity, not qualified200K
serving. The real QSA serve is currently slower on these fixtures.

Dense32KC2stream0 arrival offsets begin20.399s,25.774s,28.276s: gaps of
5374.92ms then2502.12ms after its first and second content chunks. The
other stream's first content is28.309s. The long gaps therefore overlap
the other stream awaiting first content. That supports a scheduling/
interference hypothesis only; neither engine causality nor absence of
client/network contribution is proven. QSA's old aggregate tail cannot
be placed without arrival samples. Raw arrays remain in the dense JSON.

Closure: probe1 due fidelity/repeat; backend/greedy/bench/serve/engine-error/
post-health0,SOURCE_UNCHANGED. Actual EngineCore drains and manager stops
normally; no forced kill. Container exits0 after341s; both native health
probes HEALTHY. Both leases free by17:30:10; no reset or retry. Raw
x6/flashnext-ista-iq3-xxs-ep-dense-q1-dense-control-r1; derived comparison
q1/dense-control-r1-summary.json preserves text hashes, exact cross-arm
matches and largest-gap placement without changing raw results.

VERDICT: the matched control narrows interpretation: baseline repeatability
and long-context fidelity still require work with QSA off. QSA remains
unqualified, and no speed/stability promotion is supported. Parent now
owns subsequent P1 GPU windows; Q1 is CPU-only with all shared source frozen.

## Ranked QSA profiling plan - CPU review only

1. Measure topk_batch first: qsa_batch.cpp:680 uses one256-thread group
   per query,64full-context radix scans with reductions/barriers, then
   thread0 scans the context again to emit ordered IDs. That is a concrete
   source candidate for the rising context-dependent cost, not a timed
   attribution. Profile actual M1/M2 captured decode at4K/16K/32K before
   changing it. An exact multi-pass/radix or pooled-cell implementation
   must preserve pack_key ordering/ties, forced open-block priority and
   ascending output-ID order; require exact IDs and graph replay first.
2. Measure index_scores_batch at the same shapes: it performs serial
   four-head x128 dots per pooled cell and writes repeated token scores.
   Capture query/pool/score/selected-ID evidence on identical inputs
   before attributing continuation drift or reducing arithmetic precision.
3. Measure kv_gather_batch dequant plus attend_batch, then projection/
   side-state/gate overhead. Geometry is24query/2KV heads,dim256,indexer
   4x128,compress4,budget2048,cap2051,real block2752 and12full layers.
   The SIMT attention path has only one group per query/head and scalar
   dot/value loops; compare selected-KV tiling/GQA reuse only after timing.

Rank the summed12-layer captured critical path, including readiness and
cross-operation gaps, with real paged state and loaded-model VRAM pressure.
Avoid per-op synchronizations that manufacture a new execution path. Keep
component numerical/capture checks and loaded-pressure health/lifecycle
checks distinct; existing low-VRAM probes did not reproduce prior liveness
failures. No profiling code, build or GPU run is authorized by this plan.

Reference-port preservation fix: q1_strata_reference.sh now rejects existing
reference/window/config/engine log artifacts before any mkdir/copy/GPU
call. A fresh absolute Q1_REFERENCE_DIR controls config/log/output and
validator paths. Default rerun against current artifacts exits5 with all
four reference/config/log hashes unchanged. Syntax/AST/ASCII pass. Future
reference command: `Q1_REFERENCE_DIR=/mnt/vm_8tb/b70/build/vllm-flashnext/q1-reference-fresh bash harness/q1_strata_reference.sh`.
Existing canonical references remain unchanged under q1 for current controls.

## Isolated top-k reduction candidate - CPU preparation

CONFIG: native QSA topk_batch uses64full-context unsigned-key radix scans
and a256-thread manual reduction. First candidate changes only that
reduction to sycl::reduce_over_group.64passes, packed score/index bit
semantics, ties, prefix selection, clamps and serial ascending-ID output
remain intact. All copied kernels have explicit unique names under
b70_q1_topk_r1; native and copied tree/group variants load together.
No shared QSA/model/kernel source changes. Build/library/tmp/results stay
under /mnt/vm_8tb/b70/build/vllm-flashnext/q1/topk-r1.

COMMAND: B70_AGENT=Q1 MAX_JOBS=4 bin/cpu-run bash
vllm/flashnext/harness/q1_topk/build.sh. This is an offline BMG AOT build
using the existing oneAPI2026.1/compiler and torch ABI. CPU oracle command
is in harness/q1_topk/README.md and uses host Python3.12 with existing venv
packages. Future GPU command, ONLY after a coordinated grant:
`Q1_PROFILE_TAG=profile-r1 bash vllm/flashnext/harness/q1_topk/window.sh`.
Physical card1 and matching device pin; expected2-4min with240s body cap,
native pre/post health, quiet CPU lock outside GPU lease, source/SO hash
checks, no-overwrite output and actual container teardown.

RESULT: CPU build exit0 in57s; candidate SHA256
`eb83f79dbc6256e9a99d9b8871b6a174a06209e13885336773df57d3f81f84e9`.
CPU oracle63cases PASS: independent full unsigned-key sort versus radix
selection, ties preferring lower ID, NaN payloads, signed zero, infinities,
extremes, width/n clamps and inactive canaries. AST/ASCII/bash-n PASS.
Seven source/build/profile/oracle/window/README files ported under
harness/q1_topk before D1/N1 source freeze; CPP byte-identical to the built
scratch source and compiled native library unchanged. GPU not yet run.

Prepared GPU validation measures native/copied-tree/group top-k alone and
complete captured one-layer QSA decode chains at4K/16K/32K, capacity4 with
active M1/M2, real24/2/256 geometry, indexer4x128,compress4,budget2048,
cap2051,block2752. Deterministic page fixtures contain signed int8 KV,
positive fp16 scales and finite f32 pooled rows. Exact selected IDs,
inactive canaries, gated outputs and changed-input graph replay must pass.
Native unitrace logs uninstrumented whole-chain replays. Auxiliary event
boundaries report all operations plus commit_length and prior gaps with no
host wait between operations. Interpret those boundaries alongside the
uninstrumented trace because event insertion can perturb scheduling.

VERDICT: CPU-ready only. No kernel speed, serving improvement, model
fidelity, loaded-model pressure/liveness or fault-clearance claim.
Component inputs are seeded populated fixtures, not saved real-model
activations. Full-model pressure and serving gates remain separate.
No GPU until parent grants the post-N1 card1 window; shared sources now
frozen for D1's N1 diagnostic, reports/JOURNAL only.

## Top-k profile-r1 - unitrace launcher rejection

CONFIG: authorized physicalcard1/logical0 component window, unchanged
candidate/native libraries,240s body cap. No full serve.
COMMAND: Q1_PROFILE_TAG=profile-r1 bash harness/q1_topk/window.sh.
RESULT: nativepreHEALTHY; unitrace rejects session q1topkprofile-r1
because its session argument must be alphanumeric. Python probe never
started, so no numerical, graph or timing data. Body exit1 after12s;
container gone normally, nativepostHEALTHY, all source/SO hashes unchanged.
Window closed17:55:05 and card1 released. Raw q1/topk-r1/profile-r1.
VERDICT: launcher configuration failure, not a numerical reject or GPU
fault. Preserve raw attempt; propose PID-based alphanumeric session and
fresh profile-r2 only after parent source/retry grant. No rebuild needed.

## Top-k profile-r2 - exact component PASS, minimal reduction benefit

CONFIG: physicalcard1/logical0, same compiled candidate and native QSA,
real capacity4 with active M1/M2, contexts4096/16384/32768. Populated
seeded int8 KV/fp16 scales/f32 index rows; not real-model activations or
full-model VRAM pressure. Entire one-layer QSA decode graph is captured.
Only launcher change from preserved failedr1: session uses q1topk${BASHPID}.

COMMAND: Q1_PROFILE_TAG=profile-r2 bash harness/q1_topk/window.sh.
Fresh source hashes, default native prehealth, CPU quiet outside one-card
GPU lease. Candidate SHA remains
`eb83f79dbc6256e9a99d9b8871b6a174a06209e13885336773df57d3f81f84e9`.
Native QSA SHA: `2ee734458f9ed0ac0ebd827c409c3771ba315fab15dfdf004bd19da24f67bf71`.
Container image: `sha256:ea234fbf7123cba58b387f4046a0c32192dd5b6febead3684dbd8690882bf572`.

RESULT: start17:56:30; new sessionq1topk71277 accepted; numerical63cases
PASS across native/copied-tree/group (189comparisons). Every real-shape
chain passes integer-key ID references, inactive canaries, exact gated
output parity and changed-input top-k plus whole-chain graph replay.
No numerical rejection or runtime fault. Body0, container leftover0,
nativepre/postHEALTHY, sourcecheck0; closed/card1released17:57:20.

Graph event averages below are milliseconds per layer,12replays each.
Top-k alone and whole-chain capture are distinct measurements. These are
component timings, not serving claims.

| Context/active | Native top-k/chain | Copied tree top-k/chain | Group top-k/chain |
|---|---:|---:|---:|
| 4096/1 | 0.7416/1.1710 | 0.4698/1.0839 | 0.4653/1.0796 |
| 4096/2 | 0.8062/1.5575 | 0.4646/1.1100 | 0.4601/1.1045 |
| 16384/1 | 2.0988/2.5911 | 1.6062/2.2211 | 1.6036/2.2167 |
| 16384/2 | 2.1214/2.6561 | 1.6349/2.2863 | 1.6307/2.2812 |
| 32768/1 | 4.0246/4.6392 | 3.2762/3.8930 | 3.2719/3.8883 |
| 32768/2 | 3.9763/4.6361 | 3.2280/3.8891 | 3.2239/3.8846 |

The actual reduction substitution saves only about4-5us per layer against
its copied-tree control, approximately0.1-0.5% of chain time here. This
is not a compelling serving candidate. The copied tree itself differs
substantially from the existing native binary. The tracked native build
recipe is O2/intel_gpu_bmg_g31; isolated build is O3/spir64_gen with
-device bmg and a smaller compilation unit. The native-to-copy difference
is confounded by build context and must not be attributed to group reduce.
No full model serve is justified for this minimal candidate.

Native uninstrumented trace critical path, mean of3replays per phase:

| Context/active | Chain span ms | Top-k ms | Attention ms | Gather ms | Index score ms | Within-chain gaps ms |
|---|---:|---:|---:|---:|---:|---:|
| 4096/1 | 1.1663 | 0.5505 | 0.4884 | 0.0745 | 0.0315 | 0.0086 |
| 4096/2 | 1.1912 | 0.5454 | 0.4959 | 0.0963 | 0.0313 | 0.0091 |
| 16384/1 | 2.5856 | 1.9695 | 0.4876 | 0.0746 | 0.0318 | 0.0089 |
| 16384/2 | 2.6480 | 1.9985 | 0.4960 | 0.0982 | 0.0318 | 0.0091 |
| 32768/1 | 4.6348 | 4.0166 | 0.4884 | 0.0746 | 0.0323 | 0.0089 |
| 32768/2 | 4.6301 | 3.9685 | 0.4979 | 0.0983 | 0.0397 | 0.0089 |

Trace contains exactly54complete11-operation replays, three for each of
18known shape/variant phases. Every replay's top-k symbol validates the
variant assignment. Host log and GPU trace clocks have an offset, so
phase assignment uses complete ordered operation structure, not naive
host-walltime bounds. All raw trace events are retained. AtM1 top-k is
47/76/87% of measured chain span at4K/16K/32K. Attention stays near0.488ms,
gather0.075ms, score0.032ms, and internal readiness gaps near9us. Top-k,
not index scoring or transfer gaps, is the dominant rising component cost
on these populated inputs. A12-layer multiplication is only a projection:
M1 native chain14.0/31.0/55.6ms, top-k6.6/23.6/48.2ms. This is not an
observed model critical path or an explanation of every full-serve gap.

Auxiliary uncaptured per-op event spans include Python/host launch delay
and are much larger than device kernel durations; use them only as
readiness diagnostics, not kernel-time attribution. Native unitrace
above has no per-op event overlay or per-op host synchronization.
Raw q1/topk-r1/profile-r2 contains result.json, trace/python.34.json,
trace-summary.json, source/SO manifests, image ID, run/window/health logs.

VERDICT: numerical/capture component PASS; preserve candidate as research,
no serving/fidelity/pressure-liveness qualification. Minimal group reduce
has little benefit over same-build tree. Next CPU-prepared candidate is
an exact packed-int64 parallel selector using upstream Torch XPU topk and
ascending-ID sort before investing in multi-bit SYCL radix. No new GPU
or full serve follows without a separate coordinated grant.

## Torch packed-key parallel selector - CPU prepared

CONFIG: next exact selector uses upstream Torch topk and stable sort,
no custom GPU build. Reinterpret raw f32 bits, apply native order_f32,
pack unsigned64score/index key and XOR sign bit to signed int64 ordering.
Mask logical cells>=device n to signedMIN, strictly below valid keys.
Select fixed2051 descending keys, mask ranks>=device width before stable
ascending-ID sort, then guarded copy preserving IDs beyond width. Width0
and inactive rows remain canaries. Score columns fixed147456 in every
real-shape native/Torch arm; do not benchmark only the active prefix.

COMMAND: CPU test under bin/cpu-run, hostPy312/existing Torch packages;
future coordinated GPU command:
`Q1_PROFILE_TAG=profile-r1 bash /mnt/vm_8tb/b70/build/vllm-flashnext/q1/torch-r1/window.sh`.
Physicalcard1/logical0, default nativepre/posthealth, CPUquiet outsidelease,
240s body cap and source/identity/container closure. Expected2-4min.
All new candidate/probe/window files remain scratch; campaign unchanged.

RESULT: CPU oracle66cases PASS in3s: original63adversarial cases plus
3padded147456-column mixed-width/context cases. Raw NaN/Inf/signedzero,
integer-key ordering, clamps and tail/inactive canaries exact. AST/ASCII/
bash-n PASS. Per4x147456 int64 intermediate4.7MB; multiple full-width
bit/mask passes require tens of MB plus topk/sort workspace. CPU checks
cannot establish Torch XPU int64/topk/stable-sort support or graph capture;
those are the first capability gates inside the proposed GPU window.
Whole-chain/gated parity and changed-input replay remain required.

VERDICT: CPU-ready design only; await explicit GPU grant. Parallel topk
is worth a bounded capability/profile check before a complex multi-bit
SYCL radix implementation. No full serve, pressure/fidelity or speed claim.

## Torch packed-key selector profile-r1 - exact captured component PASS

CONFIG: physicalcard1/logical0, original native top-k versus scratch
upstream Torch packed-int64 selector. Fixed147456score columns in both
arms, capacity4 with active M1/M2, contexts4K/16K/32K, real QSA geometry
and populated seeded pages. Same input tensors and integer-key semantics;
no reduction-precision or output-order change. No model/pressure load.

COMMAND: Q1_PROFILE_TAG=profile-r1 bash
/mnt/vm_8tb/b70/build/vllm-flashnext/q1/torch-r1/window.sh.
Source/candidate hashes frozen before run, nativepre/posthealth and CPUquiet
outside card1lease. Alphanumeric unitrace session;240s body cap.

RESULT: started18:03:36, body20s/exit0, normal container disappearance,
nativepre/postHEALTHY/sourcecheck0; released18:04:26. CPU66case log saved;
GPU63adversarial cases across native/Torch (126comparisons) plus all6real
147456-column shapes PASS. Torch XPU int64 topk/stable sort and entire
chain graph capture supported. Integer IDs, inactive/tail canaries,
changed-input top-k and whole-chain replay, exact gated output parity PASS.
No unsupported-operator fallback, numerical rejection or hardware fault.

Graph event averages (12replays/arm), milliseconds per layer:

| Context/active | Native top-k/chain | Torch top-k/chain |
|---|---:|---:|
| 4096/1 | 0.8193/1.4128 | 0.8063/1.4949 |
| 4096/2 | 0.9310/1.4778 | 0.8020/1.5282 |
| 16384/1 | 1.9761/2.6114 | 0.8138/1.4658 |
| 16384/2 | 2.0048/2.6764 | 0.8152/1.5273 |
| 32768/1 | 4.0232/4.6604 | 0.8105/1.5100 |
| 32768/2 | 3.9750/4.9686 | 0.8110/1.8943 |

Native unitrace contains36complete replays, alternating3native11-operation
and3Torch136-operation chains for all6shapes. Verify every chain's initial
write-step, gather boundary and operation count before ordinal association;
do not align host and GPU walltimes naively. Selector boundary covers all
126Torch operations, not only the final topk/sort call.

Uninstrumented per-chain device trace means (3replays), milliseconds:

| Context/active | Native/Torch chain span | Native/Torch selector span | Native/Torch index scores | Native/Torch internal gaps |
|---|---:|---:|---:|---:|
| 4096/1 | 1.1856/1.4843 | 0.5505/0.8042 | 0.0533/0.0550 | 0.0083/0.1269 |
| 4096/2 | 1.2155/1.5198 | 0.5435/0.8059 | 0.0541/0.0560 | 0.0082/0.1263 |
| 16384/1 | 2.6132/1.4580 | 1.9696/0.8021 | 0.0544/0.0571 | 0.0087/0.1291 |
| 16384/2 | 2.6707/1.5176 | 1.9983/0.8036 | 0.0539/0.0609 | 0.0087/0.1281 |
| 32768/1 | 4.6552/1.5026 | 4.0166/0.8070 | 0.0535/0.0608 | 0.0083/0.1271 |
| 32768/2 | 4.9584/1.8824 | 3.9683/0.8047 | 0.3678/0.3797 | 0.0085/0.1258 |

Torch selector stays near0.80-0.81ms; trace shows96segmented radix-sort
kernels for full key ordering and ascending-ID ordering, plus bit packing,
masks/copies/guards. The two sort sets sum near0.60ms. Torch's126selector
operations introduce about0.127ms total intra-chain gaps versus native's
0.008ms. This is measurable parallelism despite more launches; it avoids
the original single-group context scaling. It is not an optimized partial
selection implementation on this local Torch dispatch.

At4K the uninstrumented chain regresses approximately0.30ms/layer; event
averages show only0.05-0.08ms regression because short native timings have
extra inter-replay/host overhead relative their device spans. Keep both
measurements; do not use the smaller difference as a universal estimate.
At16K M1 trace improves2.613->1.458ms and at32K4.655->1.503ms.
Those are about44/68% component reductions, not model speed gains. A12layer
projection saves13.9/37.8ms at16K/32K but costs3.6ms at4K; loaded-model
scheduling, pressure and full-prefill behavior remain unmeasured.

At32KM2, index_scores is0.368ms native/0.380ms Torch versus roughly0.054ms
in other native shapes at147456score columns. The rise occurs in both
arms; retain as a shape/working-set diagnostic, no causal attribution.
Current kernel/rank shows selector first, then attention near0.49-0.57ms;
this larger M2index-score cost is worth later profiling. Previous topk-r1
used score width=context+4, so cross-run total-chain comparisons are not
matched and cannot establish a selector gain.

Raw q1/torch-r1/profile-r1 contains all result/trace/source/image/health
logs, trace-summary.json and passed parity gates. Selector/source remains
scratch. Actual primitive semantics and graph support are established only
for the tested shapes, not the whole model or200K concurrency.

VERDICT: supports a separately authorized default-off serving integration
experiment with the original O4/F1 recipe and no N1/D1 overlay. Preserve
short-context regression and both native/Torch raw traces. No promotion,
full-model fault clearance, fidelity/stability or serving-speed claim.
No further GPU work queued; parent owns next scheduling decision.

## Fused pack and threshold compaction - CPU ready

CONFIG: preserve measured Torch-only control in separate harness/q1_torch/
with source hashes linked to measured scratch/profile-r1; selector is
byte-identical, runner changes are path/import-search only. New candidate
is scratch q1/fused-r1 and two uniquely named SYCL ops under
b70_q1_packcompact_r1. One full-score-column grid packs exact uint64 keys,
then sign-XOR and bit_cast to int64. Invalid cells are signedMIN. Keep
upstream int64 topk2051 sortedTrue; compact uses device values[width-1]
threshold and a single actual-n ascending serial scan, preserving tails.
No second-ID sort, rank-mask or tensor-copy stage. Width/n/idcap/selectedK
clamps and width0 no-write guard match native semantics.

COMMAND: cpu-run/MAX_JOBS4 offline AOT build using existing oneAPI2026.1,
disk TMPDIR; compile same key.h into host CPUhelper; cpu-run cpu_test.py.
Future command, ONLY after coordinated card1grant:
`Q1_PROFILE_TAG=profile-r1 bash /mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r1/window.sh`.
Physicalcard1/logical0, default nativepre/posthealth, quietoutsidelease,
240s body and source/binary/container gates, expected2-4min.

RESULT: GPU library AOT34s PASS; initial separate CPUhelper build had
missing g++, corrected to existing icpx in0.4s without rebuilding GPUlib.
Library SHA42b7cb8eccd5c23164d7242776a4cccddcb837b317ac2f70fdf7b5f605baaf67.
Compiled common key helper4166rawbit/index pairs exact includingNaN,
signedzero/extremes; strict invalidMIN below every valid tested key.
CPU252threshold/clamp/selectedK/width0/tail cases PASS; AST/ASCII/bash-n
PASS. New operator behavior on GPU is not yet tested.

Complexity: O(capacity*scores_n) fused packing, upstream topk unchanged,
then O(active*n) serial threshold scan; no64radix scans in our kernel.
Persistent keys8*actualB*scores_n (4.7MB at4x147456), values/indices about
0.13MB plus upstream full-sort workspace. Many full-width Torch
intermediates and second-sort workspace disappear. Allocation/reserved/
peak counters and native trace are prepared for all three matched arms.
The lazy per-shape key allocation must warm before graph capture; parent
API caches per actual buffer/shape. The requested interface is
PackedTopk(scores_n,cap,device).__call__(scores,ids,step).

VERDICT: CPU gate ready only; await GPU grant. Preserve tested Torch-only
control for attribution. Custom ops currently have XPU implementation;
component graph capture alone will not prove Dynamo/FakeTensor support.
Full API/serve integration belongs to parent after component and memory
review; no native/plugin/upstream source edits or full-model claim.

## Fused serial v1 component - exact PASS, performance REJECT

CONFIG: physicalcard1/logical0, native/Torch-only/fused serial matched
147456scorecols/capacity4/activeM1,M2 at4K/16K/32K, same populated pages.
COMMAND: Q1_PROFILE_TAG=profile-r1 bash q1/fused-r1/window.sh (scratch).
RESULT: started18:13:27; body23s/exit0/container gone/nativeprepostHEALTHY,
sourcecheck0; released18:14:19. All189GPU adversarial comparisons and6real
shapes pass exact integer IDs, inactive/tail canaries, changed top-k and
whole-chain graph replay, exact gated outputs. No numerical or GPU fault.

Graph event means, milliseconds per layer:

| Context/active | Native selector/chain | Torch selector/chain | Fused serial selector/chain |
|---|---:|---:|---:|
| 4096/1 | 0.7987/1.3857 | 0.8058/1.4960 | 0.9843/1.6763 |
| 4096/2 | 0.9909/1.4795 | 0.8141/1.5336 | 1.0099/1.6934 |
| 16384/1 | 1.9773/2.6131 | 0.8090/1.4676 | 2.0301/2.6785 |
| 16384/2 | 2.0047/2.6778 | 0.8115/1.5613 | 2.1642/2.9420 |
| 32768/1 | 4.0247/4.6586 | 0.8093/1.4900 | 3.5507/4.3279 |
| 32768/2 | 3.9763/4.9616 | 0.8148/1.9276 | 3.7211/4.8381 |

Native trace validates54complete chains:11native/136Torch/63fused GPU
operations per replay,3replays per18phases. Fused M1 pack costs5-6us;
upstream first-sort kernels sum0.514-0.517ms; serial compaction itself
costs0.416/1.447/3.091ms at4K/16K/32K. It dominates the attempted savings.
Fewer kernels and lower memory are insufficient when the serial
ascending scan returns as the critical path. Torch-only stays around
0.805ms per selector. Fused selector spans0.992/2.020/3.668ms inM1trace.

Raw memory counters include retained graph pools, current/prior case
objects and cached selectors; reserved memory is arm/order dependent.
On warmed cases peak-minus-before allocated is approximately42.64MB
Torch versus19.04MB fused (23.59MB less), with another4.72MB fused keys
on first actual147456shape allocation. These are component allocator
measurements, not model capacity or loaded-pressure evidence. Capturing
both whole-chain and isolated-top-k graphs is additional experimental
pool pressure. The report does not multiply reserved counters into a
model-KV capacity claim. Raw all-arm allocated/reserved/peaks are saved.

Measured v1 is ported to harness/q1_fused with13source/build/CPU/profile/
window/control/README/hashlink files. Kernels/key/control are byte-identical,
runner changes paths only. B70_Q1_FUSED_SO defaults to the measured scratch
library; no binary in campaign. Tested Torch-only remains separate under
harness/q1_torch. No native QSA/upstream source change.

VERDICT: reject serial compaction for speed; do not full-serve this path.
Numerical/capture success does not clear loaded-model liveness faults.
Next approved scratch candidate sorts selected IDs directly in parallel
SLM bitonic, avoiding all actual-n serial scans; no GPU until parent grants.

## Fused bitonic r2 - CPU ready during D1 freeze

CONFIG: unique b70_q1_packcompact_r2 namespace/PackKernelR2 and
CompactBitonicKernelR2. Keep exact fused uint64 packing and upstream
int64 topk sortedTrue. Read selected-value low32bits to recover IDs for
first device width, pad to nextpow2(cap2051)=4096 with INT_MAX, sort using
256thread bitonic78stages plus initialfill barrier in16KiB SLM, write
first width cooperatively. No context scan in compaction. Width/n/cap/
selectedK clamps and inactive no-write remain exact. Tested v1/Torch
controls stay unchanged and can co-load with distinct kernel names.

COMMAND: cpu-run MAX_JOBS4/diskTMPDIR offlineAOT build; compiled key
helper and CPU bitonic reference. Future authorized command:
`Q1_PROFILE_TAG=profile-r1 bash /mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r2/window.sh`.
Expected2-4min physicalcard1/logical0;240s body/defaultnativepreposthealth.
Four matched native/Torch/v1serial/v2bitonic arms,147456cols,6realshapes,
exactness/canaries/changedgraphs/native trace and memory counters.

RESULT: CPU AOT33s exit0, SHA
23e456afdc5ab7bff44b40a45fe51744773d19fe2191eb6f3963f57871d69325.
4166compiled key pairs and252CPU bitonic/clamp/selectedK/canary cases PASS,
AST/ASCII/bash-n PASS. Compaction complexity O(cap*log^2(cap)), independent
of actual context, with16KiB SLM per active query group. Persistent keys
and upstream topk workspace unchanged from v1; no second full sort.

VERDICT: CPU-ready only. GPU execution and campaign port wait until D1
posthealth/source closure and parent's explicit grant. No shared-source
changes during its freeze. Future port is separate harness/q1_fused_r2;
parent owns API/serve integration after component/graph-memory evidence.

## Bitonic r2 component - exact captured PASS, source port complete

CONFIG: after parent recovery/health closure, physical card1/logical0;
native, preserved Torch-only, preserved serial v1 and bitonic v2 matched
at147456score columns, capacity4/activeM1,M2,4K/16K/32K populated QSA
pages and real geometry. One bounded component attempt, no model serve.

COMMAND: Q1_PROFILE_TAG=profile-r1 bash
/mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r2/window.sh.
Quiet CPU outside one-card lease; default native pre/posthealth, source
and actual binary hashes,240s body cap, unitrace and allocator counters.

RESULT: start18:26:03, body24s/exit0, no container leftover, nativeprepost
HEALTHY, sourcecheck0; closed/card1released18:26:54. GPU252adversarial
variant comparisons and all6real shapes PASS: exact integer-key reference
IDs, inactive/tail canaries, gated outputs, changed-input top-k and whole
chain graph replay. V1 and V2 co-load with distinct namespaces/kernel
names. No numerical fallback, crash, reset or loaded-model qualification.

Identity clarification: raw result.json candidate_sha256 is the SHA256
of measured selector.py source, NOT the compiled library. Raw artifact
is preserved; new identity.json explicitly records meaning, loader path,
SO identity and all measured-to-port source hashes. Selector source SHA:
`e2da5b83cadf98dc36b2349f78a3066a41877b29611a76a1ea1cfc089f7b0ca1`.
Actual measured SO:
`/mnt/vm_8tb/b70/build/vllm-flashnext/q1/fused-r2/libq1_fused_r2.so`.
SO SHA:
`23e456afdc5ab7bff44b40a45fe51744773d19fe2191eb6f3963f57871d69325`.
Both before/after source manifests validate that exact library.

Graph event means, milliseconds per one-layer operation/chain:

| Context/active | Native selector/chain | Torch selector/chain | Serial v1 selector/chain | Bitonic v2 selector/chain |
|---|---:|---:|---:|---:|
| 4096/1 | 0.9430/1.2941 | 0.8074/1.4964 | 0.9844/1.6739 | 0.7078/1.3468 |
| 4096/2 | 0.9310/1.4850 | 0.8064/1.5010 | 1.0168/1.7141 | 0.7027/1.3876 |
| 16384/1 | 1.9758/2.6107 | 0.8053/1.4808 | 2.0798/2.6994 | 0.7086/1.3665 |
| 16384/2 | 2.0048/2.6739 | 0.8195/1.5141 | 2.1633/2.8660 | 0.7037/1.4305 |
| 32768/1 | 4.0232/4.6581 | 0.8113/1.4639 | 3.6590/4.3265 | 0.7089/1.3578 |
| 32768/2 | 3.9765/4.9595 | 0.8222/1.8576 | 3.6038/4.7752 | 0.7049/1.7359 |

Device trace has exactly72complete chains,3per24phases, operation counts
11native/136Torch/63v1/63v2 and distinct compact symbols. Native trace
means below exclude inter-replay host gaps and per-op event overlays.

| Context/active | Native chain ms | Torch chain ms | V1 chain ms | V2 chain ms | V2 pack/bitonic ms |
|---|---:|---:|---:|---:|---:|
| 4096/1 | 1.1857 | 1.4872 | 1.6615 | 1.3415 | 0.0059/0.1229 |
| 4096/2 | 1.2114 | 1.4931 | 1.7010 | 1.3795 | 0.0058/0.1226 |
| 16384/1 | 2.6063 | 1.4686 | 2.6997 | 1.3600 | 0.0057/0.1222 |
| 16384/2 | 2.6679 | 1.4999 | 2.8584 | 1.4261 | 0.0055/0.1225 |
| 32768/1 | 4.6532 | 1.4558 | 4.3237 | 1.3518 | 0.0053/0.1228 |
| 32768/2 | 4.9544 | 1.8478 | 4.7738 | 1.7279 | 0.0058/0.1225 |

Bitonic compaction is approximately0.123ms independent of context,
replacing v1's0.414/1.483/3.090ms M1 serial scan. Pack is5-6us; first
upstream radix-sort kernel sum remains about0.515ms. Selector device
span is roughly0.70ms. The parallel sort removes the prior serial
critical path while preserving exact native output ordering.

V2 improves every tested shape versus Torch-only, but4K still regresses
versus native by0.156ms M1/0.168ms M2 in device chain spans (about13-14%).
Event averages show a smaller/different4K delta because short native
measurements include more inter-replay/host overhead. Preserve both;
no short-context win claim. M1native->v2 chain is2.606->1.360ms at16K
and4.653->1.352ms at32K (about48/71% component reduction). Multiplying
12layers is only a projection, not measured end-to-end improvement.
The shared32KM2 index-score rise remains0.367-0.379ms across all arms;
no new attribution. A matched serving pair must expose actual tradeoffs.

Memory: warmed per-arm peak-minus-before allocated is42.64MB Torch-only
versus19.04MB v1/v2; first v2 real-shape call adds4.72MB persistent keys.
Transient reduction is23.59MB on this component path. Reserved and graph
pool counters depend on arm order, retained graphs and fixture objects;
these are not KV-capacity or full-model pressure claims. Full serving
must record actual graph pools/workspace/available KV and activation.

Port complete:14files under harness/q1_fused_r2, with kernel/key/Torch/
v1control bytes identical to measured source. Path-only source/build/
CPUhelper/SO adaptations are hash-linked in measured-source-sha256.json.
B70_Q1_FUSED_SO explicitly pins the measured library (default same path).
Selector interface remains PackedTopk(scores_n,cap,device).__call__.
Measured v1 stays in q1_fused, Torch-only in q1_torch. Syntax/AST/ASCII
pass; no port-only GPU rerun. No further Q1 source mutation queued.

FakeTensor review: baseline compilation config is mode0/NONE with
FULL_DECODE_ONLY and two warmups; no Dynamo tracing. The validated direct
XPUGraph path therefore does not require FakeTensor registrations.
Keep measured loader unchanged for this baseline. Future Dynamo use
needs separately tested fake registrations; component capture alone
would not qualify that mode. Parent owns default-off API, buffer/shape
cache, workspace review and matched native/r2 full-serving authorization.

VERDICT: exact captured component PASS and meaningful long-context
component gain; preserve4K regression and model-pressure limits. Proceed
only through parent's separately reviewed default-off serving pair.
No serving speed, fidelity, stability,200K or fault-clearance claim.
No additional GPU work queued; nativehealth and leases cleanly closed.
