# S135: same-device cross-queue graph events

CONFIG -> Two distinct in-order Level Zero queues in one context on card0
(PCI 0000:0b:00.0), pinned e41 compiler/runtime, four kernel operations per
depth. This is a small standalone API oracle with no model, weights, KV or MTP.
The executable SHA is
`575d592783f7e07bd7fed0988ad954de76cff9d2525e920c040c5914d6cfa79c`.

COMMAND -> The retained CPU-only build used four CPUs, no DRI and no network.
Build and tee exited0; the wrapper logged21s and compiler/link component9.273s.
The V2 runner used one CPU-quiet card0 lease, fresh pinned pre/post health and
one owned non-auto-removed container. Two fresh processes ran control mode0
then event mode1. Source, scripts, arguments and receipts are indexed in
[source-copy-index.json](../strata/flash-next/lowlevel_probe/graph_event_api_s135/source-copy-index.json).
This stage used no timing benchmark or profiler.

RESULT -> `PASS_GRAPH_API_ONLY` in one rounded38s window. Both modes passed
all eight depth intermediates in each of three A/B/A phases. Each full phase
readback was32100B. Cross-mode outputs were byte-exact, phase0 equaled phase2,
and phase1 differed. The six larger output files stay external with their
completed analyzer hashes indexed; they are not duplicated here. Three DOTs
per mode contain one Apre node, one Bpre node and two tail nodes.

| Application-level source count, per mode | Control0 | Events1 |
| --- | ---: | ---: |
| Graph submissions | 72 | 72 |
| Kernel operations | 96 | 96 |
| Explicit barrier submissions | 0 | 45 |
| Per-depth dependency host waits | 72 | 0 |

These are source counts, not measured Level Zero command counts. Mode1 still
waits at phase/setup/cleanup boundaries. Before Bpre at depths after the first,
B's queue barrier depends on the previous tail event. A's queue barrier
depends on both pre-graph events before tail; A's next pre remains ordered
after its tail. The barriers are
outside separately captured graphs. Tail data and counter advancement are
separate kernels. All intermediate histories are checked, not just the final
output.

Native/container/wait/removal/driver/tee exits were0, final OOMKilled wasfalse,
and full CID plus NAME were absent before successful fresh card0 post-health.
No timeout, forced cleanup, reset or retry ran. V2 replaced V1's automatic
cleanup with a hold-for-root path for live or unknown owned containers. V1 was
not executed; source/build/analyzer were unchanged by that lifecycle fix.

VERDICT -> The tested SDK/runtime accepts this bounded same-device graph/event
ordering mechanism. It proves neither hardware overlap nor speed nor MTP
correctness. Real MTP tail completion, token validation, CPU PLE hint ordering,
owner buffers and error drains remain separate integration requirements. No
main source change, automatic integration or further run follows from this
result. Prior actual Opus5.5/Fable5.1 source consultation is referenced by the
payload; root separately reviewed V2 lifecycle. Post-result model disposition
was pending at this curation cutoff.

Subsequent actual Opus5.5/Fable5.1 reviews accepted this as API feasibility
only. They rank current MTP profiling and the C6 marker fix before an event-
chained MTP variant; the cause of S131's timing difference remains unmeasured.
Those independent replies are preserved with the S137 diagnosis.
