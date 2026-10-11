# Source40 one-token decode critical path: minimum next control

CONFIG

This is a source-only plan over the unchanged consumed source40 SDK:
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T193019Z-1kl6uepo.
The user's roughly50tok/s target is for TWO B70s. The current chunk64/512
experiment is SINGLE32GiB B70: chunk512 nativehost prefill57/85.7 versus
chunk64 28.4/33.6; decode6.4/8.6 versus6.8/8.8. Root reports own-arm repeated
outputs match, short outputs match across arms, BOTH524-token outputs differ
across arms. Chunk512 parent post-gates were still pending when this plan was
written. These are diagnostic raw numbers, not an optimization qualification.
Changing prefill chunk does not provide a matched-output speedup conclusion.

COMMAND

Prepare one separate unchanged native serial process per OFF/ON arm, with four
requests short/524/524/short, fresh1/greedy/seed1/max64, native natural EOS intact.
Obtain exact inputs from independent current tokenizer/template metadata for the
same original speed messages; root must join their IDs to the actual native
engine_begin records. Do not feed captured hidden tensors into the reference.

PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/prepare_decode40_host_profile_v1.py --prepared ACTUAL_CHUNK64_PREPARED --inputs AUTHENTIC_TWO_INPUT_ID_LISTS_JSON --binding ACTUAL_SOURCE_MODEL_RUNTIME_PREREQUISITE_SHA --mode OFF --output NEW_OFF_RECIPE.json
PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/prepare_decode40_host_profile_v1.py --prepared ACTUAL_CHUNK64_PREPARED --inputs AUTHENTIC_TWO_INPUT_ID_LISTS_JSON --binding ACTUAL_SOURCE_MODEL_RUNTIME_PREREQUISITE_SHA --mode ON --output NEW_ON_RECIPE.json

Inputs JSON has exactly short and prefill_about512 integer arrays. This helper
only emits native argv/env/protocol; it does not launch or qualify a workload.
Root must independently admit current prepared/source/SDK/runtime/model/tokenizer
and current exact source40 flags before execution. Use the same pinned C388,
original source/build/pack/model read-only mounts, owned name/labels/resources,
card0 pin plus bin/gpu-run --card 0 lease, continuously drained original pipes,
READY then one GEN at a time, exact DONE before next GEN, QUIT after fourth DONE.
Create /results/host-traces before ON. Preserve original native exit/CLI EOF,
exact-owned terminal/removal, strict card+compiled pair health/kernel checks and
new full4/pages after actual model terminal. This separate native observer does
not use the standard speed parent: that parent forbids heavy flags and runs six
coherence requests, exceeding H36's four-request process bound. Do not weaken it.
Reuse its existing qualified identity/lifecycle functions in an explicitly
named root-owned diagnostic actor; no registry/SDK/math changes are needed.

ON adds only STRATA_DECODE_TIMING=1 and the existing three H36 variables.
OFF omits these entirely. Both preserve SYCL_NATIVE_HC1, HC_Q8/HC_Q8_INJECT/
QFUSE/GR_V3=0, DEVICE_PLAN1, NO_HOST1, suffix0/lookup0/noMTP/cacheOFF.
H36 only supports onecard/native serial/no pipeline, up to4 requests and64
outputs, prompt2048 maximum. It cannot cover current192-token speed requests
or the two-card route. The existing collect_host_critical_path36_v1 assumes verifier coverage for EVERY
prompt row. Source40 read_part can instead run optimized batched Prefill outside
Verify, so that collector must fail whole-prompt coverage rather than invent
missing rows. Use its raw schema and closure checks to inspect host intervals;
collect_decode40_host_trace_v1.collect makes exactly one explicit predicate
change: accept all prompt/decode Verify rows OR the complete contiguous decode
Verify positions n-1 onward, retaining all raw footer/thread/graph/token/printf/
flush/closure predicates and labeling batched prompt internals UNOBSERVED. PromptSpan still provides wall duration. Do not disable
batched prefill to make the old collector pass. Require all
OFF/ON output IDs, DONE finish/counters and prompt IDs equal; never copy ON
capture authority into OFF. If equality fails, retain it and stop attribution.
Interleave fresh processes ABBA if startup cost permits; compare first-use and
later requests separately, never mix warm graph with first capture.

RESULT: exact source path and available measurements

- generate.cpp10715..10743 starts decode after preceding prompt rows, snapshots
  host Drive counters, and runs ordinary one-token/no-draft windows. 11597
  splits host verify-call, commit/emit and draft wall intervals;11620 stops the
  decode clock BEFORE final wait_commit.11628..11637 prints DECODE_TIMING.
  These fields are host intervals, not device execution. Host-layer counters
  may overlap; ms_host charges gather separately then whole preparation.
- verify.cpp2112..2164 H36 records Verify/StageInputs/PLE prefetch/gather;
  2180..2195 graph submit;2327..2329 existing queue wait. generate11548..11585
  records actual token printf and stdout flush, LP only if actually requested.
  Graph first-use/preexisting hit is explicit. Report nested spans separately:
  sum disjoint top-level intervals or busy-time unions, not every scope sum.
  No LP control change: preserve the original invocation's policy.
- H36 ReplayWait covers the existing queue wait and may contain almost all
  outstanding compute/mapped-memory stalls. A short ReplaySubmit with long wait
  distinguishes host preparation from pending queue work, but cannot split
  GPU compute from PCIe mapped loads or queue scheduling. CLOCK_MONOTONIC host
  events are not device timestamps. No new waits/barriers are added by H36.
- verify.cpp2138/2228 skips the entire host expert-pool loop under NO_HOST1.
  Its ms_wait/ms_pool and Drive cache/offload/PCIe counters are not complete
  device-plan routing counters. Zero DONE.offload is NOT no mapped reads.
  Initial15 CPU expert workers or noAVX512 are NOT evidence of CPU expert math.
- verify.cpp1425+ builds resident_plan_bound using stage-owned mirror table.
  verify_kernels.dp.cpp resident-plan kernels choose a VRAM pointer or mapped
  host-mirror pointer for routed experts; expert arithmetic then reads it in
  the graph. No separate H2D memcpy is required for those device accesses.
  Startup51.9GB mirror is allocated coverage, NOT bytes transferred per token.
  H36 does not record all routed expert IDs/tiers or selected logical blob bytes.
- Native HC is exact original Q8_0 projections and F32 norm/injection/mix, not
  an off switch: verify995..1016 calls native_hc_submit for two sublayers in
  each of48 layers, plus output HC at1678+. hc_native_composition.cpp15..59
  has RMS, Q8 down, SiLU, Q8 up, injection, mix as separate C++ launches;
  standalone residual write is separate. This exposes a candidate many-small-
  kernel sequence, NOT measured overhead. Capture calls are not per-replay
  submissions; only an actual graph/backend trace can confirm launch execution
  and optimization. Do not disable native HC or substitute compatibility math.
- STRATA_VERIFY_DEBUG is intentionally excluded: verify2354+ under NO_HOST
  performs multiple device memcpy(...).wait() per window, plus extra vector
  readbacks. STRATA_VERIFY_PROFILE disables shared-stream overlap and its SYCL
  gpu_stamp writes0 (verify_kernels2538). EAGER adds waits. Neither is a normal
  device-stage timing control; enabling them can change the critical path.

VERDICT and next decision

First run the bounded existing H36 control to determine whether host PLE/input/
commit/stream or outstanding replay wait dominates. It is executable with the
current SDK, no new native compilation. Six CPU recipe/reader controls pass; actual
GPU/lifecycle/coverage are pending. Nativehost8.6tok/s singlecard does not test
or disprove the50tok/s TWOcard goal.

If replay wait dominates, the next minimal separate experiment is passive UR/
LevelZero command-buffer/list API tracing over the same4 requests, with exact
loaded library/tool/source/queue/context/module identities and OFF/ON outputs.
Capture-only API call durations cannot measure every replayed HC kernel.
Existing graph replay events are discarded and normal finalization is not
profiling-enabled, so device timestamps are presently UNOBSERVED. Do not call
UR host API durations device time. Per-node timestamp events require a later
explicit capability/source experiment, not toggling a hidden property here.

To distinguish expert mirror traffic from compute, future observation must
retain actual each-layer route IDs plus stage-owned d_res/mirror/blob-size
mapping AFTER existing completion, with no new per-layer barriers. Current
flags do not expose that complete route stream; a full49 observer changes the
work and is not this speed control. Classify selected logical expert bytes
separately from physical transfer. Physical PCIe bandwidth/page faults require
an independently qualified hardware trace, not RSS/startup mirror/counter0.
This source gap is explicit; no unsupported precise traffic claim is proposed.

Use the pinned x2 method: retain owned events until existing completion; match
logical IDs/outputs, preserve clock domains and dependency edges, copy completed
records before graph/ring reuse, bound quotas, reject missing coverage, then
measure observer overhead separately. The current H36 collector rejects open
scopes/foreign ownership/clock errors/overflow rather than inventing zero cost.
No model fidelity, broad quality, full concurrent cache, physical residency,
serving latency, or shelf qualification transfers from this plan.
