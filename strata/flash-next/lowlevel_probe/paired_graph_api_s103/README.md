# S103: six-versus-one graph replay API probe

CONFIG -> One small uint32 TU, same six kernels/arguments, same in-order
queue, fresh mapped A/B/A inputs, all six intermediate planes and guards.
Six one-operation graphs versus one re-recorded six-operation graph.
No model, MTP arithmetic, main patch or host output polling in this probe.

COMMAND -> Actual source reviews claude-opus-5-5 58.806s and
claude-fable-5-1 81.611s; root full-read and versioned malformed-CSV-row
HOLD correction before any build. One offline CPU build in pinned e41,
no DRI/network, UID1000, CPU4, disk TMP; wrapper20s, compile/link9.948s.
Binary2c7ac67aa919e30869b644164edcddae8e12ac661ec99fa2f7b3886196f6e9a9.
One card0 lease, physical device0 pin, two fresh processes, 43s wrapper.

RESULT -> Three exact A/B/A replays each mode. Actual Level Zero immediate
graph submissions18 versus3 (six versusone per replay); Step0..5 each
executes3 times per mode. Whole DOT has six nodes/five edges. Same one
7704-byte fill and three7704-byte readbacks; no unclassified device work.
Full intermediates, guards, readonly and cross-mode bytes exact. Normal
exit0/noOOM, exact owned fullCID+NAME gone before post-health, pre/post
card0 health0, driver/tee0. No reset, forced cleanup, workload retry or
raw trace parse. Root separately checks DOT and all saved snapshot guards.

VERDICT -> PASS_GRAPH_API_ONLY. Prepare minimal default-OFF depth2 main
integration. MTP numeric/state isolation, host output publication/polling,
PLE, production pointer lifetimes and serving speed remain unqualified.
No timing columns support a speed result. Main/defaults/S30 unchanged.

Development loop: this untimed exactness/count check deliberately omitted
the quiet-CPU lease while the other agent built llama.cpp under cpu-run.
The numeric root wrapper retains the exact GPU lease/pin/health/teardown.
Timing comparisons still require quiet CPU. Source wrappers using quiet
CPU remain retained as preparation, with the actual wrapper separate.

record-copies.json binds exact text copies. Original/v2 preparation,
actual review responses and root decisions, build command/hashes, profiler
summaries, DOT structures and teardown evidence are retained. Binary and
six binary snapshots stay at the on-disk paths with hashes in the records.
Compiler-produced whitespace in raw records is deliberately preserved.
See [dated findings](../../../../docs/20261015_strata_graph_capture_and_bottlenecks.md).
