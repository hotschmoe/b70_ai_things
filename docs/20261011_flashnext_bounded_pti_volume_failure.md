# Bounded GEN4 capture still exceeds the long-prefill metadata bound

CONFIG -> Native43 mode0, short54/long524/long524/short54 prompts, four fresh
GEN4 requests; LPdisabled, H36 both arms, sampled-query tool only ON. Actor
d07f3190, actual intakee87b3e14; current registry173 and model unchanged.

COMMAND -> Sole leased parent53490, output
`/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-bounded-pti-v1-run-v1`.
Parent terminal exit1 after1616s; report SHA256
`c445a56d5ac7fd45f7e131281018845b5d6123a30d9dd47f320286606195632d`.

RESULT -> OFF completed all four requests and the H36 session, then passed
per-card/compiled-pair health and fresh four-shard integrity. ON reached READY,
completed request1 and hit the512MiBclock metadata cap during request2 after
PP448/524. Clock536870223bytes; Chrome approximately676MiB at observation.
Bounded16KiBtail contains19 instance_normalization rows and ends at
sequence663485/instance661659. This is evidence of the tail, not a complete
record-type distribution. Saved external observation4a9fa74c at
F17/source43-bounded-pti-v1-cap-observation.json.

The native result records exit86 and forced_cleanuptrue, with controlled parent
interruption. Successful explicit finalization was not reached. ON post-health
and its closing full4 passed; aggregate final health and four-shard integrity
also passed. Parent reports both native-phase failure and postwrite aggregate
raw-budget exceeded, with1265985145rawbytes. No successful capture or normal
ON retirement claim follows. Both cards are released after actual closure.

VERDICT -> Sampling query diagnostics and reducing output tokens did not make
this long-prefill trace fit. Every operation normalization remains exported;
do not drop those rows silently or waive the cap. Use an explicit paused warmup
and selected short-prompt window for the next decode diagnostic. Long-prefill
coverage needs scoped collection or lossless compact export with declared
ownership, coverage and clock limits. Preserve this failed run and do not feed
it to a success-only analyzer. Full fidelity/cache/concurrency/latency/shelf
requirements remain open.
