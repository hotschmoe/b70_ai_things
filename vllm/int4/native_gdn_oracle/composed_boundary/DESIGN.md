# Composed native boundary contract

STATUS -> HELD. Actual native standalone failed rolling-column0 conv publication
while output/SSM arithmetic passed. The assumptions below are unqualified; do
not implement or execute until the independent publication/read probe resolves
the installed native contract. Original design retained as a hypothesis.

CONFIG -> Exact phase+MRV1 image7b, nativeSHA271db0d4, workerSHA936d65f7,
TP1 K16/V48/dim128, FP16 projected/conv, FP32 recurrent state, FP8-layout
page3276800 with conv122880 and SSM3145728. Full native q4 operator, actual
Mamba context POST and PRE methods. No GPU execution is authorized by this
file. The parent owns the selected lease and strict pre/post-health lifecycle.

COMMAND -> Prepare five scalar cases below in two scheduling modes (unsynced
and explicitly synced), using identical preallocated inputs, weights, initial
states, permuted physical pages and accepted metadata. Run both modes against
an independent native no-move chain and CPU recurrence. Freeze all source and
configuration hashes before the parent launches. This is a new artifact; do
not alter standalone native or POST inputs already running.

The first native call always reads accepted1 from its four-column table and
produces four candidate state prefixes. The case's accepted count then selects
how many of those four outputs survive. Counts must not be confused with the
accepted count supplied to the first native invocation.

| computed before first call | first running col | accepted after first call | POST publication | accepted after POST | next computed | next running col | PRE action |
|---|---|---|---|---|---|---|---|
| 4796 | 2 | 4 | self col2, bias3 | 1 | 4800 | 3 | col2 to col3, bias0 |
| 4798 | 3 | 1 | none | 1 | 4799 | 3 | none |
| 4798 | 3 | 2 | col3 to col2, bias1 | 2 | 4800 | 3 | none |
| 4798 | 3 | 3 | col3 to col2, bias1 | 3 | 4801 | 3 | none |
| 4798 | 3 | 4 | col3 to col2, bias1 | 4 | 4802 | 3 | none |

POST receives scheduled4/draft3, computed as above, and actual acceptance.
Use actual ctx.run_fused_postprocess with its separate accepted-output buffer.
PRE receives the declared previous/current columns and POST's GPU accepted
output minus1, then actual ctx.run_fused_precopy. The second native call uses
accepted1 when PRE moves state, otherwise the unchanged POST output. Construct
that final accepted tensor on device or preallocate from the frozen scalar
case; do not read GPU accepted values on the host between kernels. Verify
actual POST accepted output after the entire chain.

All allocation, tensor uploads, context initialization, and CPU references
finish before the measured chain; one initial synchronization is allowed.
The unsynced chain is exactly native1 -> actual POST -> actual PRE -> native2,
with no .cpu(), .item(), tensor truth conversion, host readback, explicit wait,
or torch.xpu.synchronize between operators. Keep all input and metadata tensors
alive through the final readback. The synced control inserts explicit device
synchronization after native1, POST, and PRE. Both use final readback only after
native2. Document those fences; do not call host native return a completion.

The no-move native chain uses independent initial storage and the same first
and second qkv/ba. Its second call stays in the first running column and uses
the actual accepted count (1..4), selecting first-call SSM prefix accepted-1
and conv rows accepted-1:accepted+2. Thus it does not execute POST or PRE and
provides an independently scheduled native logical-state control. A padded
versus packed comparison remains a separate configuration, not another change
inside the first ten-case matrix.

RESULT -> Required independent gates, all retained on failure:

- First native output and second native output versus CPU recurrence, using
  the standalone frozen numerical bounds without relaxing them.
- Composed second output, z, current conv state and four SSM prefixes versus
  no-move native chain, byte-exact. Keep numerical differences as diagnostics
  if exactness fails; do not promote tolerance changes into a pass.
- Actual POST accepted output and input unchanged; final second-call accepted
  value matches the table. Same-column positive-bias publication is exercised,
  not skipped as a no-op.
- Snapshot-based CPU expected whole allocation: first native publishes four
  prefixes, POST and PRE copy declared regions, second native updates current
  conv and four prefixes. Conv, untouched slots and page padding exact; SSM
  mathematical comparison separately bounded, with untouched bytes exact.
- Synced/unsynced outputs and state exact for identical scalar case. A mismatch
  localizes synchronization sensitivity; it does not identify the faulty queue
  or prove all graph paths share that behavior.

VERDICT -> Prepared design, not executed. These five cases test normal
continuation through publication and a running-column move. Backward POST
checkpoint reuse by a different request is not included: the original request
continues from its unchanged running column. A successful result cannot rule
out request-reuse, graph replay, scheduler metadata races, or native interaction
with the model's other kernels. No production source patch follows from this
design alone.
