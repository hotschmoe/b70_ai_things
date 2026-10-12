# Paired cooperative prefill: precise source safety prerequisites

CONFIG -> Full paired prototype, never stage0-only replacement. No GPU, compiler,
model payload or frozen57 modification by source author.

Concrete CPU-testable controls are now source files:
strata/flash-next/cooperative-prefill/control_v1.hpp and
kernels/prefill_cooperative_ticket_cpu_v1.cpp. They use real worker/owner threads,
owner-only callback execution, one parked ticket/ACK, failure-priority readiness,
close/unpark/join and pause/service budget accounting. Failure never becomes ready.
Root can compile with its exact NoDri compiler image using -std=c++20 -pthread
-I<cooperative-prefill-directory>; this is CPU coordination, no SYCL/model fixture.
These controls are source only and have not been executed/qualified by author.

Private-data audit:

- Prefill Alloc::take allocates distinct owned objects when no borrowed base;
  Verifier carves its own separately allocated arena. Native slot graphs bind
  own GDN/PLE/QSA banks. Dense weights/stage-mirror tables are shared immutable.
- GGUF read_into writes caller-provided stager buffers and reads immutable stage
  aliases/pread role slices. Stage alias counters are atomic. Decoder blob uses
  the same immutable alias, avoiding transient GGUF blob-ring invalidation in
  this exact full-owned-mirror scope. Generic ExpertSource is not admitted.
- Stager/issuer ring destinations and completion markers are private to Prefill.
  They may continue copying while compute is parked, but must keep all buffers
  and queue owners alive until real completion. No failed marker is forged.
- PLE table bookkeeping is mutable/shared. Layer0 launches async gather. A yield
  there can race decode's prefetch_keys/pending/cache even if its GPU queue is
  complete. Hook must require !ple_next.valid() after ple_land, and n<=m.T so
  there is no next-chunk gather. Do not add a wait at the hook to fake safety.

Two concrete gaps prevent admitting a naive paired callback:

1. Prefill m.owned contains pointers only; Verifier arena size is counted at
   allocation but not retained as an exported ownership range. There is no
   closed per-device allocation-extent comparison covering retained Prefill
   R/region/ring/host grouping versus Verifier scratch/slot banks. Small fix:
   retain already-known bytes/context with each owned allocation and arena,
   expose read-only owner extents, and reject intersections before arming the
   ticket. Existing slot and mirror owner records can supply their ranges.
   Adding this is host metadata only, no model workspace/device barrier.
2. IssuerJoin stops and joins before StagerDone.finish. Stager::wait is an
   uninterruptible ready spin; workers may await real DMA completion. A callback
   failure/cancel cannot simply throw/unwind while issuer is blocked. The legacy
   GGUF failed read also zero-fills then marks ready1, losing actual copy failure.

candidate-prefill-stager-cancellation-v1.patch provides the exact prerequisite
source change: cancelled generation stops claims/waits, readiness wait observes
issuer stop, real GGUF failure sets failedjob/ready-1 instead of zero+ready1,
consumers reject missing issue, and finish no longer manufactures issued=1<<30.
This patch is not an admitted model build or completed paired hook. GPU copy
failure propagation and exception-safe owner ACK/drain still need integration;
copy buffers must not be freed merely because CPU waits were cancelled.

Complete paired seam remains: hook existing post-router compute completion;
park stage1 with one CPU ticket; original controller pumps it from successor
future-drain; private decode only on original owner; resume same stack/device.
Debit measured service from existing share allowance, subtract entire parked
interval from prefill elapsed, and skip postchunk service when credit spent.
Failure/STOP must close ticket, retain owners, unpark and join all workers.
No new global queue wait may hide incomplete lifetime status.

VERDICT -> Concrete cancellation/ticket prerequisites are reviewable/CPU-testable.
A full paired model callback is not declared safe until owner-extents and real
copy-failure/teardown closure are integrated. No unsafe stage1 callback is offered.
