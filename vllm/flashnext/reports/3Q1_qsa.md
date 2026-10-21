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
