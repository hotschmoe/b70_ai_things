# Upstream bug report / PR draft: XPU graph replay leaks Level-Zero command-list space

Target: pytorch/pytorch (the file is pytorch core, not intel/torch-xpu-ops).
Status: root-caused + fix identified + reproduced on Intel Arc B70 (Battlemage), torch 2.12.0+xpu.

## Title
`[XPU] XPUGraph replay leaks Level-Zero command-list space (submit_with_event, never reclaimed) -> command-buffer overflow / abort under repeated replay`

## Summary
`at::xpu::XPUGraphImpl::replay()` submits the captured SYCL executable `command_graph` via
`queue.ext_oneapi_graph()`, which routes through `submit_with_event`. Each replay allocates a
host-visible `sycl::event` and appends an "execute-graph" command to the current in-order XPU
stream's Level-Zero **immediate command list**. Neither is reclaimed per replay -- the UR L0
adapter only runs its immediate-command-list cleanup pass every
`UR_L0_IMMEDIATE_COMMANDLISTS_EVENT_CLEANUP_THRESHOLD` (default 20) completed events, and even
then reclaims events, not the accumulated command-buffer segments. Under workloads that replay a
captured graph many times without a full teardown (e.g. speculative decoding, which replays the
drafter graph thousands of times), the immediate command list's backing `LinearStream` grows
monotonically until it cannot allocate the next command:

```
Abort was called at 84 line in file:
../../neo/shared/source/command_stream/linear_stream.h   (NEO LinearStream::getSpace UNRECOVERABLE_IF)
```

Observed as a hard process abort (SIGABRT) mid-run, with a faulthandler traceback landing in
`torch/xpu/graphs.py:replay`.

## Environment
- Intel Arc Pro B70 (Battlemage), Intel Compute Runtime (NEO) 26.22.38646.4, kernel 7.1.
- torch 2.12.0+xpu (pytorch COMMIT_SHA 7661cd9c6b841b62b7f411aa52ec51f05457263b), oneAPI 2025.3,
  UR L0 adapter v0.12.0. Stock upstream torch-xpu (no IPEX).
- vLLM v0.24.0 serving Qwen3.6-27B with MTP speculative decode + piecewise XPU graph capture.

## Reproduce
Any repeated replay of a captured XPU graph without full teardown. Concretely: MTP/EAGLE spec
decode under `cudagraph_mode=PIECEWISE`. The drafter's propose loop replays (num_spec-1) x
(num graph pieces) times per decode step with no host sync between draft steps; the command list
overflows after ~1e5-1e6 replays. On our box: NVFP4 27B TP=2 aborts at ~8-12k generated tokens
(~6 min single-stream); W8A8 27B TP=2 at ~96k tokens. Decode throughput visibly DEGRADES as the
list grows (43 -> 34 -> 30 -> 27 -> 20 t/s) before the abort, consistent with an ever-longer
command buffer.

Minimal-ish repro without vLLM: capture a small graph, then `for _ in range(2_000_000): g.replay()`
on the default stream and watch the process abort / `UR_L0_LEAKS_DEBUG=1` command-list growth.

## Root cause (source)
`aten/src/ATen/xpu/XPUGraph.cpp`, `XPUGraphImpl::replay()` (release/2.12, ~line 183):
```cpp
auto& queue = at::xpu::getCurrentXPUStream().queue();
queue.ext_oneapi_graph(*graph_exec_);     // all ext_oneapi_graph overloads -> submit_with_event
```
All three `queue::ext_oneapi_graph` overloads (`sycl/.../queue.hpp`) are
`submit([&](handler& h){ h.ext_oneapi_graph(Graph); }, ...)` i.e. `submit_with_event`. The returned
event is discarded but its allocation + the appended command-list segment persist until an adapter
cleanup pass runs, which under sustained replay never keeps up.

## Fix
Submit the graph event-lessly so the in-order queue recycles its command-list space each replay.
`queue::submit_without_event(...)` is a PRIVATE internal template (`queue.hpp:3777`, and it also
takes a non-deducible `bool UseFallbackAssert` template param), so use the PUBLIC event-less free
function `sycl::ext::oneapi::experimental::execute_graph(queue, G)` from
`ext/oneapi/experimental/enqueue_functions.hpp:488`, which routes through the same event-less
`submit()`. (`using namespace sycl::ext::oneapi::experimental` is already in scope in XPUGraph.cpp.)
```diff
   auto& queue = at::xpu::getCurrentXPUStream().queue();
-  queue.ext_oneapi_graph(*graph_exec_);
+  execute_graph(queue, *graph_exec_);   // event-less public submit; see enqueue_functions.hpp
```
For an in-order queue the ordering guarantee is preserved without the per-submit event.
Verified to compile with g++-13 (the host compiler for XPUGraph.cpp) + oneAPI 2025.3 SYCL headers.

Symbol lives in `libtorch_xpu.so` (`nm -D` shows `at::xpu::XPUGraphImpl::replay()` as `T`); the fix
is an incremental relink of `libtorch_xpu.so` (`cmake --build build --target torch_xpu`).

## Notes / alternatives considered (all rejected on our box)
- `torch.xpu.synchronize()` per step: does NOT reclaim (waits for completion; does not trigger the
  adapter cleanup pass).
- `UR_L0_IMMEDIATE_COMMANDLISTS_EVENT_CLEANUP_THRESHOLD=1`, `UR_L0_REUSE_DISCARDED_EVENTS=1`: reclaim
  events, not the command-buffer segments -> still aborts.
- `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0`: still aborts, slower.
- Application workaround that works: run the repeatedly-replayed graph (the spec drafter) eager
  instead of captured -- avoids the leak but loses capture speed.
