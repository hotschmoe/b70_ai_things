# Owner-queue MTP serving discriminator: S131 (2026-10-16)

S131 closed PASS_MATCHED_DISCRIMINATOR against the completed read-only S130
control. All observable work/output and lifecycle gates passed. Median
aggregate output rate was103.145490 control versus102.808566 owner-ON tok/s
(-0.326649%). The configured host-joined owner-queue path had no observed
throughput advantage here. This does not identify a cause, prove absence of
GPU overlap or establish a population slowdown.

CONFIG -> The same S129 frozen main on both comparisons:
ee8854f8313d9610f7e79197fc4b9451e79b2a1a447a062bbf50c3a50a211702.
Only algorithm environment difference: STRATA_BATCH_MTP_OWNER_QUEUES=0
versus1. Q5 runtime word0 and MTP parallel argmax0 remained fixed. All other
experiment flags stayed OFF, with WIDE/RPW1/A2/PF0, HC S51 common/S49 OFF,
EP149+363/all24576experts, parallel2/depth2/headpair1, two200k/32GiB parking,
prefill12288 and Direct PLE unchanged. Native and HTTP policy remained
suffix0/lookup0/spec4/mtp-max-t4/min-p0.

The reviewed S127-V2 source records the two head-free forwards on their
respective queues, host-joins both before the unchanged paired head/tail, and
waits for each tail before mapped-token checks and CPU PLE hints. Queue-keyed
scratch/capture caveats remain source findings; actual allocation delta,
parallel GPU occupancy and the cost of those joins were not profiled here.

COMMAND -> Explicit --control-dir bound the already-completed S130/mode0.
No new control was generated or run. Its actual config/identity/request-work/
normal-closure artifact hashes were recorded and rechecked by the completed
S131 comparator. Only a new owner-ON process/config/container was launched.
The earlier Q5 ON run occurred between control and owner ON; temporal drift
remains possible. Exact same-binary/configuration gates do not remove it.

One107-prompt/256-output C2 warm pair was excluded from timing, followed by
three fresh107-prompt/1500-output pairs under quiet CPU. Every ON pair had
to pass immediate control request/full text/SSE content/usage/native work/
windows/pair/solo/rounded rows/verifier-capture gates before another pair or
functional request. Intended head-free MTP queue/graph identities were not
required to match the control. Existing math/thinking/tool cases followed;
normalTERM->nativeQUIT, owned fullCID/name removal and full per-card plus
compiled collective posthealth closed. No retry or extra baseline run.

RESULT -> All comparator checks passed, including admission order/client
overlap, loader provenance, final diagnostic, exact owner HOST marker once
by warm/final, and absent Q5/parallel-argmax/other experimental markers.
Owner ON normalstop+posthealth completed62.735733s, outside timed metrics.

| Timed pair | Reused control aggregate tok/s | Owner-ON aggregate tok/s | Change | Pair wall change |
| --- | ---: | ---: | ---: | ---: |
|1 |103.145490 |102.808566 |-0.326649% |+0.327719% |
|2 |103.569240 |103.216988 |-0.340112% |+0.341273% |
|3 |101.698299 |101.528745 |-0.166722% |+0.167001% |

Median pair wall was29.085130s control/29.180448s owner ON. The following
per-stream medians are over those same three nested pairs, not independent
process/server replications.

| Metric | Control stream0 | Owner stream0 | Control stream1 | Owner stream1 |
| --- | ---: | ---: | ---: | ---: |
|Client TTFT s |0.900367 |0.921457 |2.079498 |2.103110 |
|Client completion wall s |28.912415 |29.011637 |28.869379 |28.970609 |
|Client decode tok/s |53.512690 |53.363845 |55.852132 |55.730334 |
|Whole-request output tok/s |51.880828 |51.703391 |51.958166 |51.776612 |

All three timed pairs matched per-slot offered/accepted draft counts:
1128/1132 and934/933;1122/1122 and937/938;1110/1140 and943/934.
Both arms had windows/pair/solo567/562/4,562/559/2,556/553/2; every timed
request reported cache0/prompt107/predicted1500. Warm work and verifier capture
layouts also matched. Printed rounded rows and client text are observable
gates, not full KV-byte or unprinted per-window-state identity.

VERDICT -> Keep owner queues defaultOFF; no observed serving gain; no additional serving run is selected at this stage. One fixed earlier control and one
later owner-ON process give no noise band or population speed result. The
HOST marker is not a hardware-overlap proof. SSE content-event gap metrics
remain partial client diagnostics, not token ITL or a causal readiness trace.

The S129 new-Q5 body HOLD remains unresolved and Q5 was OFF here. Optional
main ISA/independent-host/four-MTP certification was deferred under root/user
direction; finite runtime PASS does not retrospectively complete those audits.
Actual Claude postreviews covering S130/S131 are pending root; none are
invented or invoked by this curator.

The companion payload includes only S131 controllers/deltas and owner-ON
artifacts, with explicit read-only control pins and a reference to the S130
q5_runtime_ones_s130 evidence (committed58ec1ef0). It preserves all four ON pair/engine-segment
records and immediate parity proofs, configs/identity projections/functionals/
normal health/removal/aggregate receipts. Full raw SSE and process maps remain
external; existing text digests are not mislabeled file hashes. Payload: strata/flash-next/lowlevel_probe/mtp_owner_queues_s131/.
