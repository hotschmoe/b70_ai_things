# D1 -- no-spec decode after the 2026-10-21 outage

Owner: Codex D1 (Sol 6.1 requested by user). Parent coordinates both cards,
DD lifecycle, and shared source ownership. No commits or external submission.
Scratch: /mnt/vm_8tb/b70/build/vllm-flashnext/d1.

Current verdict: original merged no-spec canonical baseline61.6/61.8
tok/s remains below Strata66.2. Compact FA plan16 is a component win but
is disabled/unqualified after full EngineCore resource failure before
canonical collection. A subsequent bounded lifetime/mixed-T21 probe
passes; it does not reproduce the full-model failure. No promotion or
external validation/submission.

## Recovered Ld trace: short-context native FA owns the growth

CONFIG: CPU-only recovery of O4's completed Ld trace. Ld used O4 f14,
max_len 131072, no MTP, dense FA, real lmx reasoning chat input, 512 output
tokens, native GGUF o4g and EP o4d. It did not use Lc route or Q8F flags
(o4/ex-Ld.txt). The trace inflates absolute latency and is diagnostic.

COMMAND: B70_AGENT=D1 bin/cpu-run python3
vllm/flashnext/harness/d1_trace_growth.py
/mnt/vm_8tb/b70/build/vllm-flashnext/o4/run-Ld/steps.pkl --output
/mnt/vm_8tb/b70/build/vllm-flashnext/d1/Ld-growth.json.
The preexisting trusted local pickle derives from the raw 1000 MiB trace
o4/run-Ld/trace/python.7.json. Select modal-count segments (1938 kernels
on card0, 576 on card1), compare complete graph steps [8:72] to last64.

RESULT:

| Card0 traced metric | Early64 | Late64 | Change |
|---|---:|---:|---:|
| graph segment span, ms | 20.837 | 21.693 | +0.856 |
| XeFMHAFw family kernel sum, ms | 0.912 | 1.741 | +0.829 |
| EP wait kernel sum, ms | 0.412 | 0.612 | +0.200 |

506 complete card0 graphs and 511 card1 graphs were retained. Other card0
kernel families change by <=1.8 us each. Card1 grows mostly in EP waits
(+0.619 ms); expert families also change with routed experts. Kernel sums
and graph spans differ, so do not add the deltas as critical-path savings.
Raw Ld window.log: RUN_RC:0, gates 323/667 pass, C2 same text, teardown
SIGUSR1, HEALTH_AFTER_RC:0 and xpu-health HEALTHY. Ld skipped its own
pre-health after Lc; it is pre-outage evidence and does not qualify the
current host or a new serving recipe.

VERDICT: target short-context full-attention decode. The native Xe2 source
at vllm-xpu-kernels/csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp:385
sets kMinBlocksForSplit=32 for tile64 and collapses all KV splits below
that threshold. The actual 1600-token page is divisible by64, therefore
contexts with ceil(KV/64)<32 (KV<=1984) still use one split even with
B70_FA_KV_SPLITS=8; the first possible32-tile case is1985 tokens.
This explains why the previous fixed-split experiment did not remove the
growth. The two KV heads expose too little parallelism in this range.
The expected benefit remains an inference until a matched GPU probe.

## Prepared bounded discriminator

CONFIG: physical card1 only; random BF16 Q/K/V, real shape24/2/256 and
page1600, graph metadata maxK131072, live KV128/512/900/1024/1536/2047/
2048/4096. Native FA vs upstream Triton normal/grouped split-K, splits4/8/
16. Independent FP32 reference, numerical error and bitwise FA comparison.
Three graph timing batches of32 replays per case. No model load or build.

COMMAND: D1_TAG=fa-probe-1 bash
vllm/flashnext/harness/d1_fa_window.sh. Uses cpu-run --quiet outside
xpu-run's card1 lease, separate leased pre/post health, Python alarm180s.
Expected <=4min including health. Root must authorize GPU scheduling.

RESULT: CPU py_compile and bash -n pass. GPU not run yet. Added native
compact-plan variants at splits4/8/16: d1_fa_plan.py computes live-length
descriptors on device, bypassing the legacy collapse through the existing
plan-driven native kernel path. Empty static-grid rows receive count0;
the actual per-sequence split count caps to the available KV tiles.

VERDICT: compare existing kernels before paying for a private native FA
rebuild. No shared FA, fork, plugin binary, or serving source changed.
Prepared isolated Python overlay via d1_prepare_fa_overlay.py at
d1/fa-overlay. B70_D1_FA_PLAN enables only pure one-token BF16 24/2/256,
page1600, causal full attention; prefill and all other shapes retain the
original interface. This is unmeasured and not enabled for any server.

## Card1 discriminator measured: select native compact plan16

CONFIG: the bounded discriminator above, physicalcard1/logical0. All
candidates captured in one process. Warm3 replays then median of three
device-event batches of32 graph replays, with live sequence length changed
between cases. This measures one attention call including the plan-fill
kernel; it is not full-model ITL. Reference is independent FP32 QK,
softmax, and AV on the same BF16 input values (atol0.003,rtol0.03).

COMMAND: D1_TAG=fa-probe-1 bash harness/d1_fa_window.sh, authorized by root
after root baseline WINDOW_DONE and post-health HEALTHY. Raw:
/mnt/vm_8tb/b70/build/vllm-flashnext/d1/fa-probe-1/{probe.log,status.txt,
health-before.txt,health-after.txt}. Body67s, no native build.

RESULT:

| KV tokens | Native FA graph us | Compact plan16 graph us |
|---:|---:|---:|
| 128 | 106.30 | 48.02 |
| 512 | 94.34 | 28.75 |
| 900 | 163.88 | 34.00 |
| 1024 | 174.18 | 34.66 |
| 1536 | 252.11 | 48.09 |
| 2047 | 138.84 | 48.49 |
| 2048 | 139.11 | 48.30 |
| 4096 | 228.34 | 69.97 |

All24 native compact-plan cases (splits4/8/16 x8 lengths) pass the
FP32 reference check. Plan16 maxabs0.003160 at128, <=0.001046 at512+;
the split path is not bitwise native FA. Native FA itself also differs
from FP32 (maxabs0.001833 at128). Upstream Triton grouped passes but is
slower (e.g. grouped16 at1024 roughly156us, at2048 271us). Upstream
Triton normal fails all3 split variants at128 (maxabs0.007892), so the
whole sweep exits1 with bad3. This is a rejected candidate, not a GPU
failure: native pre/post HEALTHY, normal container exit, no crash/reset.

VERDICT: select B70_D1_FA_PLAN=16 for a matched full-model experiment.
The native 1536-to-2047 drop demonstrates the legacy split threshold;
2047 and2048 have the same32-tile count. Cold128 absolute timings are not
a steady-state claim.
The component evidence supports the plan; model coherence, exact-repeat,
canonical512 ITL and teardown/post-health remain required.

## Baseline canonical collection and integrated callback

CONFIG: root live merged baseline, port18147, stable hotschmoe-dd first,
alias qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-o4, original O4+F1 flags,
maxlen131072, no D1 candidate. Parent requested client-only canonical
512-token collection before teardown, explicitly dirty diagnostic.

COMMAND: CPUquiet stock localmaxxing/lmx_vllm.py --measure-local,
both suites. Raw d1/baseline-lmx-r1/measure.log and command.txt.

RESULT: collector fails before any warmup/request because standard serve
does not expose /server_info?config_format=json (HTTP404). No speed data,
partial valid packet, retry or external request. Parent server was already
entering teardown, so no attempt was made to preserve it.

VERDICT: future control and plan arm use harness/d1_api.py (existing
x5_server_info read-only config route + stock CLI) and integrated
d1_lmx_callback.sh before teardown. Callback owns CPUquiet, records dirty
fork patch/untracked metadata, collects canonical warmup1+3 iterations
with authoritative usage and isolation counters, and labels local
validation ranking_eligible=false/diagnostic_only=true. Parent added
F1_API_WRAPPER,F1_PYPATH_PRE,F1_POST_PROBE_SCRIPT to its merged harness.
Candidate flags: F1_PYPATH_PRE=d1/fa-overlay, B70_D1_FA_PLAN=16. Same
N1 determinism flag and other settings must be used in both arms.

## Expanded actual-shape graph gate passes

CONFIG: same native plan16, physicalcard1/logical0. B2 ragged active
rows, B4/B8 with2 active rows and padded KV0 rows. Capture once per batch,
then change sequence lengths and all mapped KV page identities in-place.
Active lengths129/900,512/1024,2051/1536,4096/17; B2 also32768/131071.
Output storage has surrounding canary rows. FP32 reference groups12 query
heads per KV head rather than duplicating the long K/V data12 times.

COMMAND: bash harness/d1_fa_batch_window.sh; root authorized after N1 p2
closed with native health. Raw d1/fa-batch-gate-1/gate.log,status.txt,
health-before.txt,health-after.txt. Body21s, alarm180s, no timing claim.

RESULT:13/13 cases pass active-row FP32 tolerance, exact repeated graph
replays, changed-table mapping and output canaries. At32768/131071 the
maximum absolute errors are0.00010259/0.00007561. Max short-case error
is0.004269 at17 tokens (passes the declared atol0.003+rtol0.03 gate).
Inactive rows have no numerical output contract; canaries remain intact.
D1_BATCH_GATE_RC0,HEALTH_RC0,pre/post HEALTHY,normalexit0,no crash/reset.

VERDICT: the hook's enabled batch/context shape now has component
coverage. Proceed to matched full-model control and plan16, N1 OFF both.

## Matched full-model windows started

CONFIG: root source freeze after campaign checkpoint e626d490; no source
or registry edits during both arms. Original merged O4+F1 recipe only;
no route or Q8F additions. maxlen131072,chunk12288,maxseq4. N1 OFF both
arms. Control plan0/native FA; candidate plan16/private Python overlay.
64-token C1/C2 bench, promotion gate,8K prefill2 repeats,128-token C1/C2
repeat gate, canonical reasoning/code512 warmup1+3 iterations per suite.

COMMAND: bash harness/d1_matched_windows.sh. Same settings in both arms;
F1_API_WRAPPER=d1_api.py and F1_POST_PROBE_SCRIPT=d1_combined_callback.sh.
Full identity snapshot covers408 files: dirty fork HEAD/diff/untracked,
effective GGUF/EP/prefill/O4/FA shared libraries, overlay/helper sources,
plugin/harness sources and model registry. Verify after each arm. Capture
script and raw outcomes under d1/matched-r1 and f1/merged-d1-*-r1.

RESULT: active; no full-model claim yet. Strict gate failures will remain
recorded even when diagnostic canonical measurements can be collected.

VERDICT: parent authorized both-card pair, expected15-20min. Continue
second arm only with unchanged identity, greedy coherence, normal teardown
and native post-health. Parent owns DD/systemd and external submissions.

Control closed first: canonical reasoning median61.642817 tok/s
(16.222490 ms/token), code61.800831 (16.181012). Each suite has3 measured
512-token requests after1 warmup, with cache-busting nonce, default
thinking, config speculative_config=null, authoritative usage and isolated
counter checks. Local measurement validation passes; ranking_eligible=false,
diagnostic_only=true. No external validation or submission.

Control64-token bench: C1 ITL p50 15.168827 ms, C2 19.582852/19.604523.
8K prefill cold1947.93 then warmed2378.69 tok/s. Both arithmetic gates,
Paris coherence and8K/32K needles pass. The promotion gate exact-repeat
fails. The separate128-token repeat also fails: C1 stream equality
[false,false,true,true],C2[false,false,true,false]. This is the known
unpatched repeatability problem with N1 OFF, not a qualified baseline.

Control teardown SERVE_RC0; native both-card post-health HEALTHY; effective
library hash check passes; full408-file identity verification bad=[] at
fork HEAD0a6170da8ca9991972322fa1ca1d27674f0e75cd plus recorded dirty delta.
Control armRC1 preserves strict gate failure. These closure checks permit
the scheduled diagnostic plan16 arm, now loading from the same sources.

## Integration guard correction; discard plan16-r1

CONFIG: review of the actual vLLM-to-FA argument path while r1 loaded.
flash_attn.py always expands layer._k_scale and layer._v_scale to scalar
views with shape(batch,kvheads), including BF16 KV. _xpu_ops forwards
those views, and the FA interface preserves FP32 scale objects.

COMMAND: read-only source review; then parent-authorized graceful stop of
owned vllm-fn-f1-merged-d1-plan16-r1, followed by its existing teardown,
native post-health and full identity verification. No live source edit.

RESULT: D1's initial private hook incorrectly required absent KV scale
arguments. Direct helper probes passed, but did not cover this wrapper
contract. The real arm could bypass the plan. r1 is discarded/unmeasured,
including all partial client results. Graceful stop caused expected
HTTP500 shutdown errors in interrupted clients; SERVE_RC0, both-card
post-health HEALTHY,408-file source verification unchanged, no GPU crash.

VERDICT: this was D1's overly restrictive integration guard. Keep the
original c7b control evidence; its executed path is untouched. After the
window exited, parent authorized treatment-only correction: accept paired
KV scale presence, forward all normalized scale arguments unchanged to
native varlen_fwd, require matching cu/lens batch dimensions, and emit a
one-time D1_FA_PLAN_ACTIVE marker. No scale-value host read or native build.

Before/after manifests: d1/matched-r1/identity.json versus
d1/identity-scale-fix.json; delta d1/scale-fix-identity-delta.json. Exactly
four changed path entries: private helper symlink + its harness target,
private FA interface and its builder. Two new wrapper-gate files. All
fork HEAD/diff/untracked, effective SOs, original c7b interface and other
shared sources remain unchanged. New manifest410files.

Parent authorized tiny real-wrapper gate d1_fa_wrapper_window.sh and,
conditional on activation/numerical/post-health PASS, a new plan16-r2
model window with the same control settings. This gate intercepts the
native call to verify the same expanded scalar-view objects arrive,
requires plan/work descriptors and ACTIVE marker, and checks changed
lengths/pages, FP32 output tolerance and exact graph replay.

Wrapper gate1 exited1 during probe initialization, before numerical
results: importing native FA before vLLM triggered torch Inductor's
CustomDecompTable mapping error during the helper's lazy vLLM import.
Native pre/post health remained HEALTHY. The gate now initializes vLLM
first, matching actual serving and the successful batch-gate import order;
production helper/overlay code was unchanged by this probe-only correction.
Preserved raw d1/fa-wrapper-gate-1; retry d1/fa-wrapper-gate-2.

Wrapper gate2 PASS: D1_FA_PLAN_ACTIVE count16,batch2,KV scale shapes(2,2).
Two intercepted native calls (eager and capture) receive the same KV scale
objects and non-null plan/work tensors with split16. Changed live lengths
345/900,512/1024,1984/2051 and changed page tables all pass FP32 reference
and exact graph replay. Worst maxabs0.00130472. GateRC0/posthealthRC0,
native pre/post HEALTHY, normal exit, no crash/reset.

Root authorized immediate corrected plan16-r2 full-model window with the
same control flags, N1 OFF and the same integrated callbacks. Started
about16:43 UTC. New410-file manifest d1/plan16-r2/identity-before.json and
treatment-only-delta.json preserve exact attribution to the corrected
private treatment. No second control load: all sources and SOs executed
by the original c7b control are unchanged. Parent may build unrelated
scratch-only CPU candidates during model load through cpu-run; timed
client windows retain the same quiet CPU lock. r2 results pending.

## r2 fails before canonical collection; treatment remains disabled

CONFIG: corrected plan16 with actual serve activation proven:
D1_FA_PLAN_ACTIVE count16,batch4,page1600,KV scale views(4,2).

COMMAND: parent-approved plan16-r2 full window and combined callback.

RESULT: identity, arithmetic, Paris coherence and8K/32K needles pass.
The strict promotion exact-repeat fails.64-token C1 p50 15.138275 ms,
C2 20.072390/20.102467. These are diagnostics, not a stable variant.
Both8K probe requests have n_chunks1 and nonempty first-token events
(' hotel' and newline text), so their observed TTFTs are35.520950 s then
3.558202 s: input/observed-first-content rates230.46 and2300.88 tok/s.
These client rates include request overhead; they are not pure engine
prefill rates. Control also had n_chunks1 in both8K requests: observed
TTFT4.202408/3.441812 s, input rates1947.93/2378.69. Do not generalize
this validity to other probes whose completion token has no content chunk.

At16:48:53, during the separate128-token C2 repeat, EngineCore fails with
UR_RESULT_ERROR_OUT_OF_RESOURCES(40), surfacing in PLEHistory.conv's
ring gather (mtp_ple.py:175). The scheduler witness is20 new prefill
tokens plus1 cached decode token,total21,no speculation. This is before
attention in that step; it does not locate the originating device fault.
The subsequent repeat-client KeyError(c2_2) and canonical connection reset
are secondary. No canonical treatment packets exist; no lmx comparison or
beat-Strata claim can be made. Preserve d1/plan16-r2/failure-witness.json.

Normal outer serve process exit0 does not hide EngineCore failure.
Treatment armRC1, native post-health card0OK/card1OK/HEALTHY, effective
SO check0,410-file final identity bad=[]. No reset or new GPU retry.

VERDICT: plan16 stays disabled/unqualified. Parent assigned N1's next
serve to original attention only. CPU-only localization facts:

- Capture reclaim500 is enabled in both arms; card1 logs key1 refinalize
  at500/1000/1500 and key2 at500 before r2 failure. The control crossed
  key2=500 and key1 through6000 cleanly. A reclaim correlation alone does
  not establish a cause.
- Helper counts/work/zeros are capture-local tensors, while native C++
  allocates BF16 split output and FP32 LSE scratch inside each call.
  This is a lifetime hypothesis requiring a stress discriminator.
- Candidate available KV budget8.53 GiB vs control8.50; estimated graph
  reservation fraction0.0123 vs0.0136. This does not support a larger
  captured-VRAM footprint. Card1 prefill free1.28 vs1.29 GiB.
- No matching kernel journal xe/drm/error event was found over the
  failure interval; native post-health recovered without reset.

Scratch-only bounded stress probe prepared at d1/fa_lifetime_stress.py
and fa_lifetime_window.sh. It compares original, temporary-plan and
persistent-plan buffers in separate processes with real head-major,
interleaved KV views, five shared-pool graph sizes,12 FA calls per graph,
B1=1501/B2=501 replays and reclaim500. It introduces the exact-shaped
PLE BF16 ring gather atT8192 after500 replays, then real PLEHistory.conv
at mixedT21 after the B2 reclaim boundary. No model/host-table pin; no
GPU run yet. Parent will schedule after N1 closes. Expected60-100s body,
hard3x60s per-process alarms, plus import/health overhead.

## Bounded lifetime/mixed-shape probe passes; failure not localized

CONFIG: physicalcard1/logical0, three fresh processes: original native
FA, temporary plan tensors (production helper) and persistent per-layer
counts/work/zeros. No EP sidecar or near-full VRAM pressure. Real cache
storage order(block,head,page,2D),transposed/split K/V views,24/2/256;
five shared-pool captured sizes8/4/3/2/1,12 FA calls per graph. Actual
graph-reclaim500 overlay, B1=1501 and B2=501 replays. BF16 PLE ring
[233,10,10240], int64 index arrays; kernel4,dilation3,width1. Ring outer
size is inferred from r2's232-block pool; trailing dimensions/dtypes and
mixed20-prefill+1decode metadata follow the actual source and witness.

COMMAND: bash /mnt/vm_8tb/b70/build/vllm-flashnext/d1/fa_lifetime_window.sh,
explicit root authorization after N1 original-attention window closed and
native health passed. Raw d1/lifetime-probe-1/{probe.log,reclaim.log,
status.txt,health-before.txt,health-after.txt}. Body67s; stops on first
error; no campaign source modification or model/PLE-table load.

RESULT: all3 arms pass every replay boundary and grouped FP32 reference
check on real cache strides. AtB1=500, PLE8192-row gather takes about5ms;
atB2=500, the actual PLEHistory.conv mixedT21 call takes0.25-0.27s and
passes its causal-tap reference. No35s spike or resource error reproduced.
Memory allocated remains107494912bytes;FDcount6. Both plan arms reserve
150994944bytes initially, then402653184 after eager PLE8192 allocation,
and remain flat. Original reserves402653184 throughout. Reclaim log
records actual instantiate_begin/done at500/1000/1500 for each B1 graph
and500 for B2. Exit0, leased native pre/post HEALTHY, no crash/reset.

VERDICT: low-VRAM single-card evidence does not support changing to
persistent buffers as a fix. It does not rule out model-sized resource
pressure, EP queues/context interactions, a full-graph interaction, or a
different new eager kernel. Root's N1 original-attention full window also
stalled, so plan16 is not established as the unique cause. Keep plan16
disabled; no new full FA serve until a concrete cause/fix and a stronger
bounded component discriminator justify it. Parent owns next GPU slots.

Stage handoff: no further D1 GPU work or full loads. Parent will review
and curate the scratch stress source before committing it, then prioritize
P1's selected-dequant primitive and QSA's first full serve. The control
remains operationally clean but fails strict repeatability. Full plan16 is
rejected for serving; component speed and low-VRAM stress results remain
conditional. Parent's lifecycle review separately detects fatal EngineCore
errors even when the shell/outer serve process returns0.

## CPU-only loaded-context liveness localization prepared

CONFIG: parent schedules Q1/P1 GPU work first. No D1 GPU work, builds,
DD lifecycle or shared source changes. Plan16 remains disabled/unqualified.
This stage prepares an instrumented N1-original reproduction after parent
releases the source freeze and explicitly grants both cards.

COMMAND: inspect original r2 failure-witness.json, original N1 n1-env.json
and stall snapshots; map current EP/PLE/GDN/graph source; CPU-only compile,
mock activation/exception tests and byte-equivalence review of private
merged lifecycle snapshot. Scratch source is under build/vllm-flashnext/d1.

RESULT: original N1 stall evidence has 78 threads and 173% aggregate CPU,
but no individual-thread/native/Python stack. It cannot locate the stall.
The resource exception surfaced in PLEHistory.conv at mtp_ple.py:175,
before current-step GDN/FA. This does not identify the originating fault.

Actual critical paths, fork root build/vllm-flashnext/vllm/vllm:

- v1/worker/gpu/model_runner.py:1742 execute_model prepares state/metadata,
  executes eager/full graph; sample_tokens:2132 is a separate boundary.
- models/qwen4_exp/xpu/model.py:777 invokes PLE before attention;
  ple_device.py device_stage reads the pinned host table, and T21 uses
  eager PLEHistory.conv instead of the <=8-row fused tail.
- _xpu_ops.py:181/265 dispatches the installed native _xpu_C.gdn_attention;
  the enabled path is not an inferred D3/R3 kernel.
- EP hooks.py sets pending{key,sent}; native moe_ep.cpp:478 prepares and
  occasionally refinalizes the private card1 graph. Card0 instantiate then
  occurs in patches/d1_graph.py:60, followed by private-card1 submit and
  card0 replay. Host returns do not prove device completion.
- Mixed T21 routes through F1 prefill_ep.py:223, with card0/card1 Torch
  streams and separate regions/counters, not the decode private-card1
  queue. Dispatch:274, card0/card1 experts, return:326 are useful boundaries.
  The available pf_sync_op waits q1 then q0 but current apply does NOT call
  it; do not infer an actual blocking wait from this unused API.
- Decode and F1 expose separate host-USM error arrays. ep_read_errors
  (moe_ep.cpp:508) and ep_prefill_errors (prefill_ep.cpp:262) read existing
  malloc_host words and produce CPU data without queue waits/copies.
  Error11 means waiting for input;12 waiting for return. Zero errors do
  not prove queue progress. Device counters cannot be dereferenced on CPU.

Scratch d1_liveness.py uses the supported worker-extension init hook.
Unarmed, all inference method identities remain unchanged; no per-layer
wrapper/stat/timer or Torch import is introduced. One daemon checks arm
once per second. After an idle explicit arm it installs16 host-only hooks
and writes armed-PID.json; clients wait this handshake. Coarse hooks cover
execute/sample, graph/reclaim and EP prepare/submit; detail adds PLE,
GDN outer forward, FA forward and F1/expert boundaries. Late wrapping the
GDN outer forward matters because CustomOp caches its backend callable
at construction. No device events or synchronization are added. C-level
faulthandler dumps every10s even if the GIL is held. The Python watchdog
records >=8s active-phase gaps, both host-error arrays, and Python stacks.
External d1_proc_watch.py records per-thread CPU/wchan/syscall, fd count,
limits and initial mappings; denied proc fields are preserved as errors.

CPU evidence: liveness-cpu-check{,-2,-3}/result.json PASS: unarmed identity,
no Torch import, activation handshake, all16 targets, preexisting GDN
coverage, T21 host metadata, exception propagation and stack unwind.
Classifier fixtures distinguish timeout during advancing phases from a
>=8s call without a return. n1-diagnostic-prepared-r2/cpu-validation.json
checks original prefix and teardown/health suffix byte-identical, and
client sequence identical after removing the detail-arm marker. Final
portable-path snapshot is n1-diagnostic-prepared-r3; shell syntax passes.
Actual worker initialization and HTTP/GPU integration remain UNTESTED.

VERDICT: instrument the previously failing N1-original recipe once,
not an unrelated original-FA reload or a persistent-buffer workaround.
After source-freeze release, port the seven operational scratch files
plus classifier into campaign harness, then rebuild a private snapshot.
Use original c7b FA, plan0, N1 r2 flag1,131072/chunk12288/capture1,2,3,4,8,
no MTP/extra route/Q8F. Keep original greedy->promotion gate (including
8K/32K needles)->bench->F1 8K twice+long math sequence. F1_NEEDLE_LENS is
empty, matching raw n1-env; no extra random stress. Arm coarse at READY,
detail immediately before F1. Callback performs exactly N1 initial
identity and Gate.greedy(17,19) request, then stops. The first historical
callback began ~73s after READY. Bound the entire client body to120s
(+5s TERM kill grace), with teardown and native health OUTSIDE that bound.
Expected total ~6min including ~3min load and native health. Future command:

TAG=n1-liveness-r1 bash /mnt/vm_8tb/b70/build/vllm-flashnext/d1/d1_n1_diagnostic.sh

Parent will update command to the ported harness path before granting GPU.
The wrapper captures broad source/SO/private manifests and verifies after
closure. Preserve requested strict gate failures, actual EngineCore fatal/
forcekill status, no-MTP API metadata, native health and all raw stacks.
No GPU work is queued now. No reset unless parent handles a health failure.

Discriminators: no-return ep_prepare localizes graph finalization before
submit; instantiate localizes card0 graph rebuilding; ep_submit localizes
private-card1 submission; later graph/sample boundaries plus both EP
error banks distinguish decode transport from eager F1 transport. A PLE
exception still permits asynchronous earlier failure. Captured FA/GDN/EP
cannot be separated by host breadcrumbs alone: if stack points to a
native graph wait, the next bounded step is a focused native queue trace,
not a kernel rewrite. Body deadline while phases advance is budget expiry,
not evidence of lost liveness. Finite success would not qualify stability
or speed and would not establish N1 or plan16 as the unique cause.

## N1-original liveness run: EngineCore SIGSEGV, not a completed sequence

CONFIG: original c7b FA, plan0, N1 r2 flag1 (SO c90919594b0e2b355073e2b398ac76755a3ace5eb0094d014a72a05a411246d0), no P1/extra route/Q8F,
no speculation,TP1,maxlen131072/chunk12288/maxseq4,capture1/2/3/4/8,
O4+F1 original recipe. Model alias hotschmoe-dd first, registered detailed
alias second. Parent authorized both cards after P1 pair closure/Q1 port.
All campaign harness/plugins/registry frozen before445-file snapshot.

COMMAND: TAG=n1-liveness-r1 bash harness/d1_n1_diagnostic.sh.
Started17:48:16, READY17:51:49, native pre/post health inside lifecycle.
Exact original greedy->promotion->bench->F1->initial N1 callback sequence
prepared; entire postREADY client body bounded120s, lifecycle outside.
Raw build/vllm-flashnext/f1/merged-n1-liveness-r1.

RESULT: parent CPU review caught two real monitor defects before launch:
kwargs-only GDN calls and cross-thread global-LIFO cleanup. Fixed before
port: all metadata accepts positional/kwargs; all metadata/log writes are
best-effort, unknown objects become type names without str/repr/int/bool;
per-native-TID token stacks and per-thread progress replace shared LIFO.
CPU regression source/tests verify kwargs-only actual GDN signature,
metadata/sink failure preserving one original call and exception identity,
and overlapping threads finishing out of global order. Eight new harness
files ported byte-identical from scratch. CPU regression PASS, shell
syntax and original-client/lifecycle equivalence checks PASS.

Live extension initialization and coarse activation PASS. Actual API
server-info.json proves speculative_config=null,maxlen131072,TP1; live
logs confirm originalFA2/SYCL GDN and expected graph configuration. All
four effective O4/F1 SO pins matched original N1 run. Initial arithmetic
323/667 PASS. Before detail/F1, promotion exact-repeat hit HTTP500 after
EngineCore unexpected death at17:52:11. Last810th execute entry isT6,
scheduled[1,5],newreq1;809 execute returns. Serve log ends native eager
pre-JIT T6 layers0..5 after a completed card1 key1 refinalize500. No
metadata errors, no Python monitor exception, no UR40 exception and no
>=8s active-phase witness. Historical T21/F1/callback failure was NOT
reached; F1 and callback subsequently fail against the dead engine.

Saved /proc data independently identifies the primary process signal:
hostPID/TID68426 maps NSpid107. At17:52:11.298941, stat field52 exit_code11
is Linux waitstatus SIGSEGV11 (os.waitstatus_to_exitcode=-11). Prior field52
is0; previous sample main-thread wchan=do_exit. Field38=17 is parent
SIGCHLD, not this terminating signal. Native PC is unknown. Docker kill15
at17:52:12.788 and container exit143 at17:52:14.989 are later cleanup,
not the primary EngineCore signal. Kernel interval has no OOM/xe/segfault
entry; no fresh /var/crash record and coredumpctl is unavailable.

Two timed C stack dumps were produced. An earlier momentary sample shows
the main thread inside ep_submit; it does not show a sustained blocked
submit. Final dump truncates in the monitor watcher while reading pathlib,
with implausible printed line196608. Monitoring contribution remains an
explicit confound. No native PC/coredump proves where SIGSEGV originated.
Both errors banks were not sampled because the8s trigger never fired;
do not describe absent error samples as zero device errors.

The original d1-localization.json has a FALSE SUCCESS classification:
client-body shell returned0 despite recorded GATE3/BENCH1/PROBE1/CALLBACK1.
Preserve that raw artifact. Corrected offline classifier in scratch first
checks fatal-engine and individual RCs, and decodes saved process status.
It writes d1-localization-corrected.json: engine failure;SIGSEGV, lastphase
gap4.458s,0long-phase witnesses. This supersedes the raw classification.
Exact raw/parsed witnesses: d1-engine-exit-witness.json and
 d1-failure-witness.json; Docker/kernel events also preserved.

Closure: EngineCore unexpected SIGSEGV, outer SERVE_RC143,
ENGINE_ERROR_RC1, strict armRC1. NativepreHEALTHY27s and nativepostHEALTHY29s
both cards. WINDOW_DONE; SOURCE_RC0 (445files,bad=[]), N1_RC0,
PRIVATE_RC0, effective SO checks0. No reset. Both GPU leases released;
Q1 owns next component slot after parent grant. Actual server initialization
and breadcrumbs work, but this is not a clean or successful serving arm.

VERDICT: no speed/stability qualification and no N1/allocator/FA-plan
causality. The observed T6 eager-prejit surface is a localization clue,
not a cause. No further N1 full load until reviewed native-fault capture
can identify a PC/core and separate monitor contribution. No unrelated
stress was added. During Q1 source freeze, classifier repair remains
scratch-only; port only after root releases that freeze.

## Scoped native debugger CPU preparation passes; GPU capture pending

CONFIG: parent approved CPU-only debugger setup after SIGSEGV witness.
No /dev/dri in these test containers; no Torch import or GPU workload.
Host core_pattern remains Apport pipe, Yama1/core0 unchanged. Parent
image pinned ea234fbf7123cba58b387f4046a0c32192dd5b6febead3684dbd8690882bf572.

COMMAND: MAX_JOBS=4 B70_AGENT=D1 bin/cpu-run bash
build/vllm-flashnext/d1/native_capture/prepare_cpu.sh. Record parent image,
apt simulation, pre/post package versions; build diagnostic-only derived
image. Synthetic CPU C child under per-container SYS_PTRACE/default
seccomp, GDB attach, recoverable then fatal SIGSEGV. Repeat only the tiny
attach preflight after adding early-options/resume-handshake validation.
No model load or research GPU lease consumed in this stage.

RESULT: reusable CPU setup268s, greater than initial30-90s estimate;
apt and image export dominate. All parent layers retained;25 packages
newly added, zero existing package versions changed (both pre-install
plan and post-install proof). Derived image
b6a97730abd8bcb76b73f3d336c6e5b2a4bbdd096622c366c0b7f9d1f7a49564.
Original image tag/ID unchanged. CPU synthetic attach under default
seccomp PASS: two native PCs/siginfo/backtraces recorded; existing
recoverable handler runs, final child waitstatus SIGSEGV11 preserved.
Debugger portion0.315s; exact early-options wrapper9s. No package build
repeated. setup-r1/{image-build.log,package-delta.json,qualification-proof.json,
attach-early-r2/result.json,gdb.log} retain proof (gdb.log within attach dir).

Parent approved port of ten files to harness/d1_native_capture/; generated
private snapshot syntax PASS. Private xpu-run differs only HERE anchored
to original harness, derived IMAGE and +SYS_PTRACE; default seccomp and
all lease/loader/native checks remain. Its executable mode is preserved.
No shared runtime/bin/fork edit. GDB early options disable auto-load and
external symbols before attach. At READY, root GDB in the one research
container attaches EngineCore namespace PID, records a continued-event
handshake, then clients start. Catch SIGSEGV records $_siginfo (including
si_code/si_addr), PC/registers/modules/all-thread bt32 and passes signal
then continues. SIGTERM passes through for lifecycle. Recoverable fault
alone is not labeled fatal. No inferior function calls or tensor printers.
Both D1 C-timed AND Python faulthandler dumps are disabled only in the
private overlay; host metadata/proc/error-word observation remains.
Original N1 recipe/request sequence and120s client bound retained;
clock begins after GDB resume and monitor arm. Native health stays outside.

VERDICT: concrete native-PC capture is CPU-qualified, GPU integration
untested. Proposed command after Q1 component closure/source port and
parent explicit freeze-go:

TAG=n1-native-r1 bash vllm/flashnext/harness/d1_native_capture/native_window.sh

Both cards, expected~6min including model load/120s body/teardown/health.
No GPU job queued now. Reuse cached image; do not rebuild packages.
Debugger pauses can perturb EP flags, so this is fault localization,
not a matched latency/stability comparison. A clean finite arm would not
prove that removing the C watchdog fixed serving. Need actual native PC,
backtrace and final process exit before any causal interpretation.

## Native-capture arm fails before READY; recovery completed by root

CONFIG: approved n1-native-r1, both cards, originalFA/plan0/N1r2/noP1,
131072/chunk12288/capture1/2/3/4/8/no speculation. Debug imageb6a97730,
private runner changes onlyHERE/IMAGE/SYS_PTRACE, both D1 faulthandler
dump mechanisms disabled only in private overlay. GDB intended to attach
and emit continued handshake before unchanged model requests.471-source
freeze established after Q1 port/root optional x6 wrapper completion.

COMMAND: TAG=n1-native-r1 bash harness/d1_native_capture/native_window.sh.
Started18:18:13, standard nativepreHEALTHY27s both cards. Raw
build/vllm-flashnext/f1/merged-n1-native-r1. No additional probe/retrial.

RESULT: FAILED BEFORE READY at18:20:59 in process_weights_after_loading
-> EP method -> torch.ops.b70_ep.ep_init. Primary DEVICE_LOST20 follows
EP pathprobe/init prints; empty_cache memory-profiler exception is
secondary cleanup. EPinit prints only after its local allocations,
zeroing/reserve and both queue waits. It then calls ep_transport_warmup;
that summary never prints. No GDB attach/resume or diagnostic body/model
request occurred, no successful live /server_info identity proof, no
SIGSEGV nativePC captured. The intended postREADY120s clock never started.

Kernel events18:20:58 identify card0/BDF0000:0b:00.0 CCS,ASID904,
116 distinct fault pages in0xeaab52c10000..0xeaab52cfa000 (~958KiB span),
116 unsuccessfulENOENT replies and32 engine-memory CATerror18 records.
Kernel resets the CCS engine and creates a device coredump. Native
failure is in an initialization/peer-page-touch surface, not a request
or late graph boundary. kernel-events.log/kernel-fault-summary.json saved.

Dump copy was explicitly requested by root and attempted read-only via
cpu-run. /sys/class/drm/card0/device/devcoredump/data is0600 root:root;
stat_size0 is the sysfs declared size, not actual dump length. Read fails
errno13 PermissionError. No zero-byte dump artifact exists. Root's sudo-n
read also required interactive authorization, so device dump is UNAVAILABLE;
preserve devcoredump-copy.json/error and kernel logs, not a fake dump.

The failed EngineCore left API waiting startup. To avoid an unnecessary
20-minute readiness wait, gracefully stopped only this research container
with20s grace; outerSERVE_RC137/ENGINE_ERROR_RC1 recorded. Nativepost
bothHEALTHY33s; WINDOW_DONE. Strict WINDOW_RC1, SOURCE_RC0 (471,bad=[]),
N1_RC0,IMAGE_RC0,NATIVE_RC0; effectiveSO checks0. Both GPU leases fully
released. No D1 reset or additional GPU activity. Root then performed
approved non-reboot rebind66s despite passing nativepost because actual
kernel CCS faults occurred: native bothHEALTHY, compiled10iteration two-
rank collective P2P0 HEALTHY, same boot. recovery.log saved by root.

VERDICT: failed startup, not N1 SIGSEGV localization. No speed/stability
qualification. GDB never attached, so debugger attachment/stops cannot
explain this preAttach failure; derived image/additional package presence,
allocation timing and runtime context remain confounds. Keep N1 OFF for
normal follow-on work and do not infer a model or kernel fix.

CPU-only decision point after closure: native N1SO load, weightcat/cache
and bf_gemm occur only inside the first projection apply. Constructor only
shallow-copies the quant method; its weight-loading methods are preserved.
No first model forward was reached. Thus direct N1 kernel execution or
its lazy weight-cache pressure cannot explain THIS startup fault. No KV
allocation, FA graph capture/reclaim or request pressure stage was reached.
General model/hostpin/peer-VA/runtime pressure remains untested. Earlier
N1-original-r1 reached requests then stalled; liveness-r1 SIGSEGV also
reached requests with no matching kernel fault. These different surfaces
do not establish one common cause or uniquely blame N1.

One conditional future candidate, not implemented/run: if a startup peer
fault recurs after root recovery in a planned N1-OFF serve, reuse cached
GDB but attach at supported worker-init BEFORE EPinit, with one pending
breakpoint on exported ep_touch_pages. Log x86_64 remote/local/bytes args
(RSI/RCX/R8) then continue, alongside existing SIGSEGV PC/bt/pass capture.
Compare kernel fault pages with these exact region bounds. This can
separate an out-of-range address from an apparently in-range unmapped
peer region without a native rebuild or new tensor copy. It is a source-
backed discriminator (ELF exports ep_touch_pages/ep_transport_warmup),
not a reason for another blind full N1 load. No extended instrumentation
work or GPU job is queued. Q1 is now sole GPU owner under root scheduling.


## HC short-decode candidate: finite gate rejects signed zero, no timing

CONFIG: CPU saved-Ld census of506 stable1938-kernel card0 steps. Original
HCup97calls1.249625ms +collapse0.158ms, HCdown1.2315ms; route topk/dispatch/
bump0.84075ms combined; source-mapped PLEstage103kernels0.3095ms. Trace
absolute times are instrumented. Current canonical control61.642817/
61.800831 uses ORIGINAL Q8 HC with Q8F and EP_ROUTE absent (saved actual
recipe-args SHA251940ee...). Existing Lc route+Q8F is a separate research
candidate, not the current default; its old -.54ms cannot be counted as
a new kernel gain. F1EPc249cf94e contains route schema/host/device symbols
and prefill support, so that existing candidate needs no oldO4e swap/build.

COMMAND: prepared M1-only up+collapse subgroup-local scratch candidate,
same LPR8 row/chunk/FMA/XOR reductions and BF16 points, no crossSG SLM.
CPU81920ownership checks PASS. CPUbuild waited currentQ1quiet lock, active
36s, SO84b64b232579423e6b7e55f1eb0bd0a2611ffaa01ba410f003eb241b17dcb2d9.
Root granted TAG=probe-r1 then reporting-only repaired TAG=probe-r2,
physicalcard1/logical0,120s bound+nativepre/post/quietCPU+lease. Raw
B/d1/hc_subgroup_r1/{probe-r1,probe-r2}. OriginalR1source preserved45fb9bb5.
No kernel or SO change between runs. Proposed96distinct-weight timings
were conditional on strict original/existing/new comparison gate.

RESULT: R1body17s exit1 stops beforetiming on adversarial largegate weight1;
normalcontainerexit, nativepostHEALTHY15s, SOURCE0. Root-reviewed R2removes
failfast reporting ONLY; body16s exit1 with ALL9case comparisons saved,
normalcontainerexit/nativepostHEALTHY14s/SOURCE0. Leases fullyreleased
18:52:17/18:55:55. One mismatch at largegate weight1,h386: original+0
BF16=0x0000, existingfused/newcandidate-0 BF16=0x8000. count1/maxabs0/
orderedBF16ULP1/allfinite. Existingfused/newcandidate match in all9cases.
AffectedCPU payload41723bytes, stdlib raw-storage bits independently
confirm signedzero (no CPUFTZ ambiguity). No per-weight/graph timing ran.
Rawr1/r2 remainunchanged; no retry/modelserve/reset afterclosure.

VERDICT: strict COMPONENT_FAIL and timingSKIPPED retained. Newsubgroup
speed UNMEASURED; no bitexact/promotion claim. Sources ported separately to
harness/d1_hc_subgroup with measured-source-sha256.json; kernel/frozenR1
source byteidentical, currentreplay/build/window/CPUoutput path changes
only. CPUownership/ASCII/AST/bash PASS afterport, no rebuild/GPU. Parent
ported and reviewed final existing-route/Q8F finite diagnostic wrapper:
identical one-return nativeop observer in BOTHarms,240s postREADY clients,
one promo/minimalF1/shortbench/canonicalonly, all statuses distinct. That
fullpair remains root-granted work; no D1 serve queued independently.


## Retained-default flag-pair control: requests complete, teardown rejected

CONFIG: root CONTROL grant only; production freeze at campaignc215a49f.
Retained O4/F1, Q8F/EP_ROUTE absent, originalc7bFA, N1off/plan0, noP1/QSA/
HC_FUSE/FA8/speculation. Identical defaultoff opobserver intended for both
arms. Server config actually recorded: TP1,maxlen131072,chunk12288,maxseq4,
prefixoff,speculative_confignull,mode0,FULL_DECODE_ONLY[2,0],caps1/2/3/4/8.

COMMAND: ARM=control TAG=d1-existing-control-r1 bash
harness/d1_existing_route_q8f/window.sh. Bothcards, started19:04:06,
READY19:07:36. Onepromo,shortbench,minimalF1(1024/repeat1/coherence1024/
decode0), canonicalcallbackonly. PostREADY240sbound, phaseprogresssaved.
RawB/d1/existing_route_q8f/d1-existing-control-r1 and
B/f1/merged-d1-existing-control-r1. No other GPU work or code edits.

RESULT: nativeprebothHEALTHY28s. Actualw8_gemv op_return12564917052567
before fullcapture_return12577600188957, sameworkerPID108; no treatment
ops observed, branchproofPASS. Promoidentity/math/8K32Kneedles/identity-
afterPASS; exactrepeatFAIL=>gateRC3 preserved. bench/probe/canonicalRC0.
All4phases complete134.3066spostREADY; boundnotexpired. Canonicalvalid
reasoning61.5/code61.8tok/s, warmup+3samples each saved, rankingeligible
false/no submission. No speed improvement or strict-repeat qualification.

Normal request processing completed, then shutdown rejected: API SIGTERM
19:09:51; EngineCore prints request processing complete/resource teardown;
XPUWorker shutdown done19:09:52. API5sdeadline then forcekills remaining
EngineCorePID108 at19:09:56. SERVE_RC0 cannot hide this. ENGINE_ERROR_RC1,
HARNESS_RC1, OPERATIONAL_RC1; outcomeoperational_completefalse with ONLY
engineerror/fatal-evidence issues. This is observed teardown failure, not
an inference-time no-progress stall or proof of a kernel/allocator cause.

Containerabsent, nativepostbothHEALTHY30s, kerneljournal since19:04:06
clean ofGPUfault/reset/OOM. Broadsource502bad[], private/SOchecks0. Both
leases fullyreleased19:10:32. No reset/retry/treatment queued byD1.

VERDICT: finite canonical diagnostic completed; control is NOT operationally
clean due forcedEngineCore cleanup, and strictrepeat rejects. Preserve all
raw statuses; parent must decide any treatment separately. No code changes
between intendedarms and no claim of successful promotion/control closure.


## Shutdown30 control: finite natural exit, original reject preserved

CONFIG: source of prior5s deadline VERIFIED in actualcontrol serve.log63:
XPUplatform xpu.py447 maps CLIdefault0 to5 (source445 conditional). Actual
liveconfig records5. Supported --shutdown-timeout30 bypasses that branch;
privateDockerstopgrace45 +readonlyhostproc onlyduringteardown. Same
retaineddefault/config/body/observer/SOs as prior5sreject; no kernel/fork
change, no newGPUhooks/reset. Production graceport/freezeeb671679,507sources.

COMMAND: ARM=control TAG=d1-shutdown30-control-r1 bash
harness/d1_shutdown_grace/window.sh, bothcards19:18:44,READY19:22:17.
RawB/d1/shutdown_grace_r1/d1-shutdown30-control-r1 and
B/f1/merged-d1-shutdown30-control-r1. Same trimmedclients/240spostREADY.

RESULT: nativeprebothHEALTHY27s. Actual/live shutdown_timeout30 and full
no-spec config checked; originalw8_gemv return before fullcapture marker,
branchPASS. Promoidentity/math/needlesPASS, exactrepeatFAIL. bench/probe/
canonical0, all4phases132.70495s. Canonicalvalid61.5/61.8tok/s, allwarm+3
samples saved, rankingeligiblefalse. No shortdecode speedchange claimed.

Teardown watcher firstsample_savedtrue, ARM/WATCH0. No forcekill. Natural
EngineCorehostPID100775 stateZ/stat52waitstatus0 captured. SERVE0/ENGINE0,
OPERATIONAL0/issues[], HARNESS3 strictlypreserved dueexactrepeat. Passive
samples show threads78->2; mainR2-4s thenD/folio_wait_bit_common4.6-5.1s
fromwatchstart, thenZ; no nativePC/cause inferred. EngineCore worker.done
precedes cleanup_dist_env_and_memory/interpreter/nativefinalization.

Timing semantics: firstsample19:24:30.709858688->observedZ19:24:36.300774996
is5.590916308s. Actualexit liesbetween finalnonZ andZsamples: fromfirst
(5.090192782,5.590916308]s. Watchstart->Z5.597982724s. SIGTERMlog has ONLY
secondresolution19:24:31, worker.done19:24:32, managerstopped19:24:36;
5.59s is NOT an exact SIGTERM-to-exit measurement. Details in
teardown-interval.json/teardown-summary.json outside source tree.

NativepostbothHEALTHY32s, source507bad[], private/SO0, kerneljournalclean,
containerabsent/bothleasesreleased19:25:17. Original5sforcedcleanupreject
remains untouched. This finite intervention supports deadline-limited
cleanup as a hypothesis; it is not a universal cure or a liveness cause.

VERDICT: operationally complete finite control, strictrepeatFAIL and no
promotion. Root separately granted SINGLE existing route+Q8F treatment
with same30/45/frozen507sources/SOs/body/observer. Treatment underway under
new tag; no independent extra GPU work, retry or reset is authorized.


## Existing route + Q8F matched pair: +3.28 percent, still below Strata

CONFIG: control then treatment, one fixed-order dirty diagnostic pair,
frozen507sources/effective4SOs/private lifecycle bytes identical. TP1EP
sidecar, maxlen131072/chunk12288, originalc7bFA/plan0/N1off, noP1/QSA/MTP/
FA8/HC_FUSE changes. Both actualconfigs use shutdown30/Docker45; all fields
match except runinstance_id and quantconfig object's printedaddress.
Only functionalenv delta is B70_EP_ROUTE=1 +B70_DECODE_HC_Q8F=1; branch
output directory differs as evidence metadata. F1EPc249cf94e retained;
new subgroupSO84b64 is NOT loaded. Defaultrecipe remains unchanged.

COMMAND: ARM=control TAG=d1-shutdown30-control-r1 then separately granted
ARM=routeq8f TAG=d1-shutdown30-routeq8f-r1 bash
vllm/flashnext/harness/d1_shutdown_grace/window.sh. Treatment started
19:26:18/READY19:29:57, same onepromo/shortbench/minimalF1/canonicalONLY;
postREADY240s bound. Reproduce ONLYafter coordinated grant with freshTAG
using that routeq8f command. No external submission or promotion.

RESULT: every warmup+3 measured canonical request forces512 outputtokens.
Normalized requests are byte-equivalent after stripping ONLY standard
LocalMaxxing cache-bustnonce, with all other request fields equal. Actual
prompttokens vary with nonce: reasoning control341/347/346 vs treatment
347/348/348; code281/279/281 vs283/281/278. This is the faithful canonical
protocol, not exact nonce-token equality or deterministic output parity.
Raw median output rates, not rounded CLIpayload summaries:

| Suite | Control tok/s | Route+Q8F tok/s | Gain | Historical Strata tok/s | Remaining gap tok/s |
|---|---:|---:|---:|---:|---:|
| reasoning-v1 | 61.514492 | 63.534291 | 3.283% | 66.2 | 2.665709 |
| code-v1 | 61.836306 | 63.861133 | 3.274% | 65.9 | 2.038867 |

Treatment trails historical Strata by4.027%/3.094% respectively. Reciprocal
median-rate latency drops0.516800/0.512752ms. Short64token bench C1p50
15.156042->14.664565ms; C2perstream p50 19.560515/19.568130 ->
19.220978/19.235395ms (~0.33-0.34ms, ~1.7% reduction). ShortC1 is NOT a
512token target win; no matched historical no-specC2 bar was established.
Minimal1KF1 remained a coherence diagnostic, not a new prefill claim.

Treatmentactual nativeop_return evidence: w8_down_act13904965730111,
w8_up_collapse13904966113020, moe_ep_route13906263239619 all beforefull
capture_return13916294858598 inworkerPID107. Control proves original
w8_gemv beforefullcapture and no treatmentops. Observers restore native
op attributes after first success; no Tensorread/sync/perop timedlogging.
Branchproof in both outcome.json files PASS; installedflags alone unused.

Treatment all4clientphases complete129.09885spostREADY, canonicalvalid/all
samples saved, bench/probe/canonicalRC0. In BOTHarms identity/math/8K32K
needles PASS, exactrepeatFAIL=>HARNESS3 retained; OPERATIONAL0/issues[].
No bitexact/stability/fidelity promotion follows. Route weights can change
ulps/tokens; existingHC component has signedzero exception. The observed
bundle gain cannot isolate route fromHC or establish a repeated-order gain.

Treatmentnatural EngineCorehostPID103593 stateZ/stat52waitstatus0; no
forcekill, SERVE0/ENGINE0/ARM0/WATCH0. FIRSTsample->observedZ5.573460021s,
actualexit bracketfromfirst (5.072919845,5.573460021]s. Watchstart->Z
5.580213982s. SIGTERMlog19:32:07/worker.done19:32:08/managerstopped19:32:12
are secondresolution; do not call5.573s an exact SIGTERM-to-exit interval.
Control's corresponding firstsample->Z5.590916308s/bracket
(5.090192782,5.590916308]s remains separately recorded. Original5s forced
cleanup reject is preserved;30s finite natural exit is not a universalfix.

TreatmentnativeprebothHEALTHY28s/postbothHEALTHY31s, source507bad[], private/
SO0, kerneljournalclean, containerabsent/bothleasesreleased19:32:52. Control
pre/post27/32s alsoHEALTHY. Crossarm summary with source/config/env/fixture
hashes/rawrates: B/d1/shutdown_grace_r1/matched-route-q8f-summary.json.
Rawbothouterdirs inB/d1/shutdown_grace_r1 and liveartifactsB/f1/merged-*.

VERDICT: existing default-off route+Q8F gives an observed~3.28% canonical
no-spec gain in this clean finite pair, but63.53/63.86 still trail historical
66.2/65.9. StrictrepeatFAIL/dirtydiagnostic/rankingeligiblefalse/no promotion.
No newHCspeed claim; noP1/QSA combination. Keep retaineddefaults unchanged.
All authorized GPUwork complete; no furtherGPU/reset/retry/jobs queued.
