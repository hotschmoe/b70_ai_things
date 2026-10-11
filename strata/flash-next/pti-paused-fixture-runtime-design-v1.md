# Paused regular-list loaded fixture

CONFIG -> Fresh successful tool build f52f9a97, injection c72c6b4f,
C388 compiler/runtime image, GPU0 workload with card0 lease and strict card0 health.
The same two-kernel SYCL graph is recorded and replayed while collection is
paused. Original named IPC control resumes collection, then the graph replays.
Retention OFF and ON are separate fresh owned processes. No model is mounted.

COMMAND -> The controller compiles the frozen fixture in a fresh no-device
container, records its original create/start/terminal/removal evidence and ELF
SHA, then runs matched OFF/ON containers under bin/gpu-run --card 0. The tool's actual
resume argv consumes the session as its value. Original strict ownership,
normal terminal, health, kernel-journal and source checks remain mandatory.

RESULT -> Source author executed nine focused CPU controls only. No fixture
compiler, tool or GPU was executed by the author. Root must observe the actual
regular-list API, two selected child kernels, raw tick ordering and normal
graph/allocation retirement before this diagnostic passes.

VERDICT -> Fixture-conditional coverage only. Retention adds execution of the
original timestamp-query paths during paused recording. Raw normalization is
recomputed, but clock alignment, physical idle, model coverage, latency and
zero added queries/waits are unqualified. The stock negative permits zero
observed instances; the ON route requires exactly the two source-known child
kernels. All observed clock/Chrome IDs remain joined, including other events.

The public reader replays original argv, named IPC transition, stdout markers,
actual child executable/namespace ownership and bounded producer-pinned raw
artifacts. Container /results paths map only to the corresponding original arm.
Failed cleanup/setup never qualifies. The host retains the exact owned object
through stop/terminal/removal and original stdout EOF; ambiguous creation does
not imply absence. No old model or helper runtime qualification transfers.
