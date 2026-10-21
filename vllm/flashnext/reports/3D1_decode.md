# D1 -- no-spec decode after the 2026-10-21 outage

Owner: Codex D1 (Sol 6.1 requested by user). Parent coordinates both cards,
DD lifecycle, and shared source ownership. No commits or external submission.
Scratch: /mnt/vm_8tb/b70/build/vllm-flashnext/d1.

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
