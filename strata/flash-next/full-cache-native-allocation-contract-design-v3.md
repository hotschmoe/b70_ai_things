# Native cache allocation and physical residency proposal V3

CONFIG -> Unintegrated source/CPU proposal for a future default-OFF diagnostic
successor to source39 or source40. It does not modify either consumed SDK or
frozen recipe. Source39 FC38/FC39 logical reservations, snapshots, capacity and
current-main-body attribution remain separate. Observer40 target-USM bytes do
not establish model or expert residency. No actual native observations exist.

COMMAND -> PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m
unittest test_full_cache_native_allocation_contract_cpu_v3 -q

RESULT -> Ten tiny CPU controls replay explicit successful API allocations,
expert byte ownership and completed copies, release ordering and peaks. Wrong
owner/sequence, overlapping or out-of-range bytes, submitted-only copies,
added completion barriers, missing actual slot attribution and unretired owners
fail closed. Arena bytes count once; expert slot ranges do not inflate peaks.

VERDICT -> SOURCE ONLY. No device physical residency, eviction, transient peak,
normal-latency or full-cache qualification follows from these API receipts.

## Required source hooks and observations

The actual source39 SYCL ExpertCache::open allocates a guarded device-USM
arena, or reserves/maps a VMM range, or opens segmented backing. The ordinary
allocation is sycl/include/strata/sycl_queue.hpp malloc_device_guarded; the
cache is sycl/src/core/expert_cache.cpp. Device-slot ownership lives in
ExpertCache::admit/slot_of; fills and queued completion are separate operations.
The host complement, mappings, locked/pinned ranges and actual exchange
storage in ExpertSource also need their own lifetime IDs and byte scopes.

A future patch must guard at each call site before constructing records,
perform no additional OFF allocation/query/copy/wait/graph work, and append
only after actual API success. IDs are unique for the process lifetime, even
if a pointer value is reused. Allocations, address reservations and physical
handles are separate spaces. Arena subranges retain exact layer/expert/slot,
offset, extent, allocation generation and actual successful existing copy
completion. Failed allocation/map/copy/free events remain explicit failures.

All stages must bind actual PID/start ticks, physical device UUID, source/build
and original input/run identity, monotonic host observation sequence/clock,
actual request/slot generation and command phase. The FC39 current-main-body
RID must not be relabeled as the slot whose BSTEP currently runs. Capture the
whole actor from initialization through terminal and all owner retirement;
selected tensor RID caps must not suppress allocation events. Parent health,
lease, EOF, journals, exact current full4/pages and raw-source recollection are
separate mandatory admission gates.

Successful USM allocation bytes prove API ownership, not OS physical residency.
VMM mappings likewise distinguish reserved VA, owned physical handles, mapped
ranges and backing location. Per-expert residency requires allocation-range
attributable device/driver observations with validated units and semantics,
not cache slots/resident(), a pointer, aggregate free-memory data or fdinfo
metadata alone. The capability may be unavailable; report it missing.

Physical samples must bind same-process/device/range/lifetime and raw query
bytes at declared whole-actor boundaries. Sparse samples are sampled peaks;
they cannot establish every transient peak. A driver event stream or an
explicitly qualified whole-allocation backing/high-water source is required
for that stronger scope. Host owned task-wide PID/cgroup/smaps/fdinfo samples
remain non-atomic metadata, with host file/anonymous/shmem split and no swap.
No new synchronization may be added to a clean latency arm. Any device query
or barrier belongs to a separately declared diagnostic arm, preserving math
and scheduling and qualifying its own overhead before performance claims.

V2 requires a nonempty exact preregistered layer/expert roster and complete actual
placement/copy coverage before claiming observed expert bytes. Allocation/free
only, missing experts, source-generation floats and experts outside 0..511 are
rejected. Frozen V1 remains unchanged, including its zero-expert scope bug.

V3 joins actual-row owner types using canonical typed JSON, enforces integer
released expert keys, and rejects ready-expert placements on bare virtual
reservations, physical handles or host mappings. Those routes still require a
future explicit map/backing lineage producer; missing lineage is not waived.
Frozen V2 and its two source-scope counterexamples remain unchanged.
