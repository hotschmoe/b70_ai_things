# Default-off slot-owner trace0029 source preparation

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: New separate patch after immutable integrated60 planb4a7d4ba. No frozen
plan/header/kernel changed. STRATA_SLOT_OWNER_TRACE=1 opts into host ownership
markers; absent/0 keeps new native-context queries and diagnostic GPU allocation/
copy/wait counts zero. Existing arena allocation, session initialization, queue
wait, graph retirement, QSA host release and arena free bodies remain operative.

COMMAND: prepare_slot_owner_trace_overlay_v1.py then
 test_slot_owner_trace_cpu_v1.py. Actual new SlotSessionArena body compiles with
ordinary hostg++ ASan/UBSan against explicit host allocation/session/queue mocks
in image39992 without devices, model files or SYCL/SDK compilation.

RESULT: Normal arena plus three unique owned host allocations register exactly
once; host_kv borrowed alias is omitted. Null allocation yields no registered
owner. Actual partial session-init zero/throw paths discover the allocated QSA
host sides before their actual release. Repeated release is idempotent. Parser
rejects stale graph binding, missing owner/free/returned marker, double register/
free, wrong context/generation and requested stage/slot roster. Marker/native
context queries add zero allocation/copy/wait beyond the baseline owner body.

VERDICT: CPU source ownership evidence only. Actual owning-context UR frees,
GPU graph retirement, slots/private state/resize/source/mirror lifecycle and
concurrent serving remain unqualified until a new whole SDK and device gates.

## Marker contract and scope

Every actual stage/slot constructor receives stage ordinal and slot index from
the existing allocation loop. Each nonnull allocation attempt has a monotonically
numbered owner generation. SLOT_USM owner_register binds PID, stage, slot, device,
generation, pointer, exact bytes, native context and role. The device arena is
registered immediately after allocation before session_init. Explicitly owned
host_step, host_pos and owned_host_kv pointers register after initialization and
again before release to discover partial initialization; previously registered
pointers are not registered twice. Borrowed host_kv and all arena-derived aliases
are not independent owners.

Registration queries only the owning queue's native context when enabled. It
never queries a freed pointer type. The chronological UR parser resolves context
bridges and exact live allocation identity/kind/extent; each registered allocation
must free once in that context between the owner's release_begin/release_returned.
Null attempts cannot become successful owning-lifecycle qualification.

Successful init_slots emits graphs_bound. Normal checked shutdown emits
 graphs_retired only AFTER the existing actual verifier graph-retirement calls,
then destroys the slot owners. Partial-init/fitted-count shrink before binding
needs no graph-retirement marker. A release of a bound owner without a preceding
retirement marker fails the observer/parser proof; it does not silently invent
retirement or reinterpret unrelated engine allocations as slot memory.

The existing owner release still drains its queue and releases QSA owned-host
state before freeing its device arena. All markers follow that actual body.
Native release errors remain failures and require terminal/recovery evidence;
physical allocator reclamation is not inferred from metadata or logical frees.

## Build/runtime requirements

0029 changes the inline arena layout and native constructor/teardown call sites.
All eight SDK targets and all ABI-specific ownership/numerical oracles rebuild
from the new source generation. Integrated60/C1v7 receipts cannot transfer to the
new29 plan. Require new source390, source/full4 hashes, runtime fingerprints,
pre/post per-card+compiled P2P0 health, source known-page guards, actual registered
private slots on every stage, graph/owner lifetime, partial-init/resize negatives
and native clean terminal before concurrent harness owner claims can pass.

The 2/4/6 harness must require actual SLOT_USM owners/UR frees and exact requested
stage/slot roster, alongside SBF snapshot owners, mirror owners and full source/
health teardown. This closes the source observability gap rather than treating
process exit alone or generic UR allocation sizes as per-slot owner proof.
