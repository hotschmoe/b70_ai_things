# Greedy64 checked SDK52 shutdown

CONFIG -> Same original target/draft greedy64 arithmetic, one card, stage mirrors,
explicit commits and original typed tensors. Executed V1 remains failed: OFF
printed64 tokens and a footer, then exited139. The controller's lowercase absence
bug is separate from that native failure. Kernel metadata records an NX jump;
the exact faulting destructor is not identified by this source correction.

COMMAND -> Prepare and freshly compile the V2 native translation unit. The hook
uses the actual SDK52 normal serve shutdown body byte-for-byte, adding only an
original draft reset immediately after all device queues finish. Ordinary return
is removed, so later GPU destructors cannot run out of the original shutdown
order. Stage mirrors are required for this bounded diagnostic configuration.

RESULT -> Ordered retirement at the hook:

| Owner/work | Completion or release | Actual SDK52 source anchor |
| --- | --- | --- |
| CPU expert callback | Dispatch returns after pool done and workers parked | src/kernels/cpu/pool.cpp694-695,835-839 |
| Target/draft GPU work | All current device queues wait_and_throw | Normal serve11980; original greedy also drains before footer |
| Original draft allocations | Reset drain-protected OwnedRuntime53 while context live | Added after queue drain; failed drain exit86 retains allocations |
| Token/session graph references | token_graph_free and session_graphs_free | Normal serve11995-11996 |
| Verifier mirror graphs | release_stage_mirror_graphs, including optional later stages | Normal serve11999-12005 |
| Batch sessions and mirror aliases | Trace retirement, clear sessions, reset hits/first mirror | Normal serve12007-12009 |
| Source aliases / owning mirror storage | trace_serving_graphs_retired, gguf_src.close, checked release | Normal serve12014-12016 |
| Optional diagnostic exports | Existing host trace/PTI finalizer checks | Normal serve12029-12035 |
| Process | Flush and checked normal _Exit0 | Normal serve12036-12038 |

Drive has no drain method in SDK52. Its expert pool callbacks complete
synchronously; no fictitious API is introduced. The original shutdown leaves
parked pool workers to process retirement. All referenced local variables are
initialized before the diagnostic hook. No target or draft math changes.

VERDICT -> A footer remains insufficient. Root must require native normalExit0,
original EOF, owned removal, per-arm health/kernel and common fresh full5.
The failed V1 is not requalified. No new native run or GPU access occurred during
source preparation, and passing equivalence/performance remains unobserved.
