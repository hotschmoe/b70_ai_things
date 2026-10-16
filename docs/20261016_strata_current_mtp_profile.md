# S136: current MTP profile, preserved partial analysis

CONFIG -> One instrumented ordered107-prompt C2 pair,64 output tokens per
request, on frozen S133 executable
`fe57222eea163dd7396f2bdabb95d2c18326a729c4d7919d400a9614f42d47ee`.
EP149/363 remained all-resident; fixed suffix0/lookup0/spec4/MTPmax4/min-p0,
depth2 and paired head ON. Q5C2/C6 runtime words, owner queues, parallel argmax,
whole-pair graph and other candidate flags were0. HOST_BOUNDARY and TRACE were1;
unitrace was loaded paused before any request/capture. VERIFY_PROFILE was absent.
The saved config, actual process environment and source/profiler identities are
retained. This is an instrumented diagnostic, not a clean serving comparison.

COMMAND -> Reuse the existing controller and profiler pause/resume/stop flow,
with fresh before/after clock calibration, one pair, idle/ring inspection and
normal QUIT. One existing unprofiled flush was needed after explicit profiler
stop; it is excluded from pair counters/window. Native completion, full owned
CID/NAME removal, per-card health and compiled two-rank health passed. No retry,
reset or daily-server restore ran. The ring snapshot includes cold/capture/solo/
tail records, not just selected replay windows.

Then run the raw JSON analyzer ONCE under the CPU lease. It saved cycles,
census, operations CSV and a proposed interior window. Root interrupted only
the subsequent redundant global-gap host-overlap scan with SIGINT after a
reported113s CPU scan; original analyzer exit-2/controller assertion/log remain.
The separate CPU ring parser ran once successfully. Its validation declares820
records,820 actual records,dropped0,one dump and no errors. Native markers have
no footer/thread IDs; normal completion and idle observation are separate proof.

RESULT -> Collection/native/teardown/health passed. GPU analysis is PARTIAL,
not a full analyzer PASS. Raw trace is279154635B; its previously recorded SHA
is indexed without re-reading it. The saved census covers749782 complete events,
21 structural cycles and15 replay candidates. Whole-collection service and
union values include cold/capture/solo/tail activity; they are not per-depth or
MTP-only budgets. The proposed12-cycle interior still requires interpretation.
Global coverage-gaps.json was not produced.

The first source-compatible complete cycle is index3:5151 events,56.73325ms,
with two C2 full-vocabulary heads, one C6 head and48 primary EP-wait records.
It fits the predeclared6000-event cap without truncation or fastest selection.
The saved array was timestamp-sorted for bounded attribution because retained
array order is not strictly chronological. No raw trace was parsed again.

One j>=1 head-free interval runs from completion of the first tail's two
mtp_select records to the next C2 head start:1311.75us,204 fully-contained
primary-GPU events, no boundary crossers found. Repeated motifs are compatible
with two sequential owner pre_step(T1) bodies and shared-tail input Q8.
Actual owner/tensor pointers were not logged, so this remains source-conditioned
role interpretation, not owner/buffer identity proof.

| Disjoint recorded name family | Calls | Service sum us |
| --- | ---: | ---: |
| Q8 dense/shared projections | 18 | 330.824 |
| HC norm/down/reduce/UP | 24 | 261.762 |
| Attention chunk/merge | 4 | 183.539 |
| Other named kernels | 40 | 128.214 |
| Routed GU/down | 4 | 92.186 |
| Input Q8 quantization | 12 | 29.682 |

Six ordered HC bundles contain norm6/40.520us, down6/91.456us,
reduce6/21.039us and UP6/108.747us. Q-projection-compatible events sum121.666us;
position-compatible output projections sum73.437us. All six M2560/C1 Q8
operations also include FC/shared roles, so they cannot all be labeled o_proj.
Exact names, IDs and relative timestamps remain in the204-row bounded CSV.

Named-kernel service sums1026.207us but their coverage union is1008.084us;
overlap exists. Another102 GPU-lane intervals named eventReset/barrier sum
125.376us. These are not host API timings; profiler/engine contribution is
unknown. Union of all204 records is1095.440us. Residual216.310us is missing
collected coverage, not physical idle, data starvation or a removable host wait.
Mapped-copy service likewise does not measure CPU staging. Epoch-sized start
timestamps have finite resolution. CPU steady/client/peer clocks remain
UNJOINED; no cross-clock timing attribution is made.

VERDICT -> Preserve useful current evidence and the original interrupted
status. Within the j1 interval, HC is a concrete bounded source lead before a broader
T2 projection or attention redesign, with Q8 projections a larger recorded family;
none of these service pools is promised saved wall time. This single nested
instrumented cycle supplies no population, clean latency, bandwidth, occupancy
or hardware-overlap result. No new kernel or GPU result follows from this profile.

Dev loop -> The original analyzer scans selected events for every gap, then
keeps only50. The unexecuted source fix retains gap endpoints, stable descending
tie order, total count and strict overlap predicates, selects the same top50
first, then scans prefiltered host-L0 events only for those retained gaps.
Its patch/inverse/pins and AST-only status are preserved. It has NOT run on
this raw trace and supplies no measured new analyzer time or completed gaps.
There is no need to repeat collection or parse the279MB trace merely to fill
discarded-gap output. Source-copy and external indexes retain the larger saved
analysis identities without duplicating raw traces,5MB cycles or full SSE.


Full saved-cycle reconciliation used only the already saved cycles file in
0.093s, with no raw parse or GPU repeat. All eight spans sum56.733250ms:
recorded union46.049262ms, uncovered10.683988ms. LEFT (previous C6 end to
first C2 start) is11.600500ms, including6.719819ms without retained events.
The verifier-compatible interval is38.569313ms, including an initial
3.317063ms gap after the second selectors. These are absent records, not
physical idle or measured CPU/peer causes. Event-reset/tooling perturbation
remains unresolved. The first narrow j1 interval is not the entire MTP path.

Within the verifier-compatible interval, recorded service sums include
routed GU4.411336ms, routed down3.158937ms, HC6.159098ms and Q6 dense
3.818719ms. EP waits total3.082584ms, but only1.160050ms has no other
family active. Overlap prevents adding these pools into a wall-time budget.
The dominant two Q6 shapes account for2.906855ms; the larger parked S125
pre-expansion/DPAS design adds memory, scratch and a second launch. Root
prioritizes explaining readiness and preparing the small existing-kernel
indexer launch reduction before selecting that broader rewrite.

[Compact profile evidence](../strata/flash-next/lowlevel_probe/current_mtp_profile_s136/README.md)
and [full-cycle ranking/source obligations](../strata/flash-next/lowlevel_probe/current_mtp_profile_s136/verifier-full-cycle/findings.md)
retain exact formulas, source pins, anchors and timestamp precision limits.
Original partial analyzer status and historical operation data remain intact.
