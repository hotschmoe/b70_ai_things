# Eager device-async all-reduce for the decode all_gather -- design + go/no-go

Date: 2026-07-21. No-GPU analysis (source + trace + API-header read only). ASCII only.
Files: `vllm/contrib/vllm_push_allreduce/119_xpu_push_ar_eager_async.cpp`,
`_push_ar_patch.py` (PUSH_AR_ALLGATHER_ASYNC block). Builds on `pushar_decode_gap.md`.

## Problem recap

Captured decode is ~43% collective; the dominant piece is 631 `oneccl_allreduce_pcie` calls/step
(~472 ms, 39%) emitted INSIDE vLLM's `all_gather` (MTP spec-decode). These gathers run EAGER, between
the piecewise captured subgraphs, so the capturable device-event push-AR (118 `ar_allreduce_graph`,
~0.034 ms/call) cannot reach them. Routing them to the host-BARRIER eager push-AR (118
`ar_allreduce_ptr_dt` = push.wait() + shm-spin barrier + reduce.wait(), i.e. ~2 host waits + a busy
spin) was COHERENT but 2.4x SLOWER: 631 host round-trips/step dwarf the device saving.

Target: get do_ar-class device performance for these EAGER all-reduces without capture and without a
torch upgrade.

## The idea and why the hard part is real

Put the push + the cross-card rendezvous on our OWN Level-Zero IMMEDIATE command list (a separate
stream from torch's SYCL queue Q_t), append the L0 events directly (bypassing the broken eager SYCL
interop `get_native_queue<level_zero>`), and keep the cross-card sig/wait reset CO-LOCATED in that
in-order list (race-free, exactly like the proven graph path). Per call there are two cross-stream
dependencies vs Q_t:

- (a) INPUT-READY: our push must not read `inout` until Q_t finished WRITING it.
- (b) RESULT-READY: Q_t's next op must not read the reduced result until our all-reduce finished.

(a) is solvable with NO host wait: submit an in-order barrier on Q_t (fires after the write), take its
native L0 event via `get_native<ext_oneapi_level_zero>(event)` (returns `ze_event_handle_t` -- verified
in the 2025.3 headers), and make the immediate list's first command a `zeCommandListAppendWaitOnEvents`
on it. Keep the SYCL barrier event alive in a small ring so its L0 event is not recycled under our wait.

(b) is the wall. A zero-host device-async handoff from our list back to Q_t needs an event our list
SIGNALS and Q_t WAITS on, RECYCLED every call. Recycling it race-free requires resetting it AFTER the
Q_t consumer, CO-LOCATED in Q_t's own in-order list. Eager SYCL interop cannot inject an L0 reset onto
Q_t (that is the broken `get_native_queue` path; only graph capture can put our L0 ops on Q_t). So the
list's reset and Q_t's wait are both released by the SAME upstream (the input barrier) with NO
happens-before between them:

- Q_t's wait can observe the STALE prior-call signal (an L0 event is a latch; waiting does NOT clear
  it) and proceed early -> reads unpopulated scratch -> INCOHERENT.
- An event RING does NOT fix this: the racing pair is reset(c) vs wait(c) on the SAME index at the
  SAME call, not a cross-iteration reuse.
- The monotonic-counter escape (waiter waits `mem >= c`, event never reset) needs either an L0
  wait-on-memory on Q_t (broken interop) or an EU spin-wait inside the reduce kernel (J.9-C: spin-wait
  HANGS on B70). Both dead.

Conclusion: on THIS platform the two standing constraints -- (1) cannot inject L0 onto torch's eager
queue, and (2) EU spin-wait hangs -- together forbid a race-free zero-host result-ready handoff. So a
FULLY device-async (zero host round-trip) eager all-reduce is a NO-GO. The floor for a CORRECT eager
path is exactly ONE host synchronization per call, on the result side.

## What 119 actually implements (the correct floor)

`ar_allreduce_eager_async` keeps exactly one host sync per call and pushes everything else off-host:

1. `ew = Q_t.ext_oneapi_submit_barrier()` -> `zew = get_native(ew)`. (a), no host wait.
2. On our async immediate list `g_imm`, in order:
   `[wait zew] [copy inout -> peerScratch[ring]] [signal sig[i]] [wait wait[i]] [reset wait[i]] [signal done]`.
   Cross-card sig/wait are ringed IPC events (pool spans both devices, K.5) with consumer-reset
   co-located in this in-order list -- identical safety to the graph path.
3. `zeEventHostSynchronize(done); zeEventHostReset(done)`. THE one host sync. It is the correct,
   race-free g_imm->Q_t handoff: the host owns `done` (sync then reset, no concurrent device waiter).
   Host blocks only for push + cross-card latency (~one AR device latency).
4. Submit the REDUCE (`src += scratch`) on Q_t. Scratch is populated (step 3 guaranteed it). The reduce
   is ASYNC (not host-waited); Q_t's in-order successor cannot read `inout` until it completes -> (b).

Cost vs the failed host-barrier path: input-ready host wait ELIMINATED (device event), reduce host wait
ELIMINATED (async on Q_t), shm busy-spin ELIMINATED (cross-card via co-located L0 events). One host wait
remains. So ~1 host wait vs the failed path's ~2 waits + spin -> roughly half to a third of the host
cost. A scratch RING (double/triple buffer) plus the cross-card event ring keep the two ranks in
lockstep within <1 iteration so a peer overwrite of live scratch or a lost-signal deadlock cannot occur.
`ASYNC_HOSTWAIT_INPUT=1` swaps (a) for a host `ew.wait()` (one extra host wait, NO interop) for
first-bringup safety before trusting `get_native(event)`.

## Correctness / ordering argument (summary)

- (a) push waits `zew` = Q_t write barrier -> push reads `inout` only after torch wrote it.
- inout write-after-read: reduce (Q_t) runs only after step-3 host sync, which is after the push read
  `inout` -> reduce cannot clobber `inout` before the push snapshotted it.
- scratch ready: reduce reads `scratch[i]`, populated by the peer push whose completion is guaranteed by
  the cross-card `wait[i]` inside step 3's host sync.
- scratch overwrite: peer push(c+1) targets `scratch[(c+1)%ring]`, a different buffer than reduce(c)
  reads (`scratch[c%ring]`), for ring>=2 -> no data race even if reduce(c) lags.
- (b) reduce is on Q_t; torch's next op is Q_t's in-order successor -> waits the reduce for free.
- cross-card latch: sig/wait consumer-reset is co-located in g_imm's in-order list (reset immediately
  follows the wait); host-sync-per-call + ring keep drift <1 iteration -> no signal is reset away.

## Wedge-safety / risk

The dangerous state is the cross-card sig/wait latch (a lost signal = a device list waiting forever =
potential box wedge, reboot-only on this display-attached box). It is handled EXACTLY as the proven
graph path (consumer-reset co-located, IPC pool spanning both devices) PLUS a host sync per call and an
event ring, which bound rank drift to <1 iteration -- strictly safer than the graph path's free-running
replay. Remaining failure modes and their class:

- `get_native(event)` returns a non-materialized / recycled L0 event -> g_imm's `[wait zew]` hangs ->
  step-3 host sync blocks forever. HANG (host-side, killable; if the imm engine is stuck, recover with
  bin/xpu-health + bin/xe-reset / reboot). Mitigation: START testing with ASYNC_HOSTWAIT_INPUT=1 (no
  interop) to validate the imm + cross-card machinery, THEN enable get_native.
- `get_native(event)` returns an event that never enforces the wait -> push reads stale `inout` ->
  INCOHERENT (safe: caught by the coherence gate, NOT a wedge).
- peer dies mid-rendezvous -> g_imm waits a never-coming cross-card signal -> HANG (same class as the
  existing push-AR paths; recover via reset).
- a bug in the cross-card ring -> lost-signal deadlock -> WEDGE. This is the one true wedge risk; it is
  minimized by the co-located reset + ring + per-call host sync, and is why the FIRST test must be a
  tiny standalone microbench, never the live DD, and attempts must NOT be chained without a reset.

Most failure modes are HANG or INCOHERENCE (recoverable / gate-caught), not silent corruption. The
zero-host variant was rejected precisely because its result-side race is silent incoherence.

## Expected win (honest, measurement-gated)

Upper bound if the device AR were free: 472 ms -> ~21 ms (collective share 43% -> ~7%), like the graph
path. This op does NOT reach that -- it keeps one host wait per call. Realistic: host-blocked AR time ~=
631 * (push + cross-card latency, ~34-50 us) ~= 21-32 ms/step of host-serial blocking, vs oneCCL's
472 ms. If the per-call device AR really is ~34-50 us (as do_ar measures), even one host wait per call
should beat oneCCL's 0.75 ms/call substantially. BUT the failed host-barrier path (~2x the host cost)
was 2.4x SLOWER than the oneCCL-gather baseline -- which implies the per-call host overhead in the live
serve is much larger than the raw device latency (python + ctypes + launch + the 631x count). So the
PREDICTION is: 119 roughly halves the failed path's overhead and MAY land near or modestly below
oneCCL, but is NOT guaranteed to win and is unlikely to approach the graph path's ~22x. Treat it as a
measurement, not a landing.

## VERDICT

- Zero-host device-async eager all-reduce (the literal task goal): NO-GO. Precise reason above -- the
  result-ready handoff cannot be recycled race-free without a host round-trip, given the two platform
  constraints (no L0-on-eager-queue, EU spin hangs). Do not attempt fence/RMW/spin variants; they are
  the J.9-C dead end and will wedge.
- One-host-sync eager path (119, provided + compiles): CONDITIONAL GO for a wedge-guarded, coherence-
  gated, microbench-first A/B. Worth measuring because the potential win is large and the failure modes
  are mostly recoverable; but predicted to at best modestly beat oneCCL, so it may not be worth shipping.
- The higher-value levers if 119 does not clearly win: (1) make the MTP all_gather run CAPTURED so the
  proven `ar_allreduce_graph` reaches it (the ~22x path), or (2) reduce the all_gather COUNT (orthogonal;
  pushar_decode_gap.md fact 6 -- the 631 scale with the spec/MTP verify).

## Build

Single-file, low-RAM (a few GB), NO -j / image build, NO setsid. DD RAM cap: keep total < 100 GB.

    icpx -fsycl -O2 -fPIC -shared 119_xpu_push_ar_eager_async.cpp -o \
      vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_eager.so -lze_loader -lrt

Compile-checked green in `vllm-xpu-env:int8g-v0251` (icpx 2025.3): exit 0, exports ar_ea_setup /
ar_ea_exchange / ar_allreduce_eager_async / ar_ea_teardown.

## Test recipe (coordinator -- owns the GPU)

Wedge-guard the whole session: `B70_AUTO_RESET=1`, bin/xpu-health available, do NOT chain attempts
(reset between wedge-prone starts), DD DOWN for the microbench.

0. Build the .so into `prebuilt/` (command above), DD down.
1. MICROBENCH FIRST (never the live DD): a 2-proc torch-XPU harness that sets up
   ar_ea_setup+ar_ea_exchange and runs a few hundred `ar_allreduce_eager_async` on a known input,
   comparing against a reference SUM. Start with `ASYNC_HOSTWAIT_INPUT=1` (no get_native interop) to
   validate the imm + cross-card machinery; confirm byte-exact result and no hang. Then drop
   ASYNC_HOSTWAIT_INPUT (enable get_native path) and re-confirm. If either hangs or wedges: NO-GO, stop,
   reset, report -- do not retry on the live model.
2. COHERENCE GATE (before any speed trust): serve the DD with the existing config PLUS
   `PUSH_AR_ALLGATHER_ASYNC=1` (keep PUSH_AR=1 PUSH_AR_GRAPH=1 and the graph .so for the captured
   all_reduce path; the async .so is separate). On rank 0 expect
   `[push_ar] ALLGATHER_ASYNC ENGAGED ...`. Run `bin/serve-sweep --smoke` (or the shelf sweep); it MUST
   stay green. This substitutes a collective -> verify coherence before speed.
3. RE-TRACE decode (same decode_optrace.sh / trace_driver.sh path). Expect:
   - `oneccl_allreduce_pcie` count COLLAPSES (631 -> few/zero); remaining collective under
     `vllm::all_gather` should now be our push copy + reduce, not oneCCL.
   - `parse_trace.py` collective share drops from 43% toward the host-sync floor.
   If oneCCL is UNCHANGED, the shim did not catch the 0.25.1 gather path (inert = no regression); grep
   where `vllm::all_gather` emits its `dist.all_reduce` and repoint the wrap (see pushar_decode_gap.md).
4. BENCH decode t/s vs the current DD (`research/profiling/bench_decode_completions.py` /
   `bench_code`). Only trust the number if steps 1-2 were clean. TTFT/prefill should be ~unchanged
   (prefill has almost no all_gather).

Env knobs: `PUSH_AR_ALLGATHER_ASYNC=1` (enable), `PUSH_AR_ASYNC_SO` (path; default prebuilt/),
`PUSH_AR_ASYNC_RING` (default 4), `PUSH_AR_ASYNC_CHUNK` (default 4 MiB; tensors above it fall back),
`ASYNC_HOSTWAIT_INPUT=1` (safe input-ready mode). Rollback: unset `PUSH_AR_ALLGATHER_ASYNC` -> the
block is skipped, `torch.distributed.all_reduce` untouched -> byte-for-byte the current DD.
