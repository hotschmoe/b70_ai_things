# Bounded segmented stage expert mirrors

CONFIG -> Actual first whole-model C1 launch loaded3671MiB native weights and a
675430400-byte head, populated8083 expert slots in23.59GiB VRAM, then refused
readiness: the legacy mirror needed16493 missing experts,51688243200 padded bytes
(49293MiB), and its single sycl::malloc_host request returnednull. No fallback
model response was accepted under STRATA_VERIFY_NO_HOST=1.

COMMAND -> Inspect actual GgufExpertSource::mirror/mirror_stage, StageExpertMirror,
global StageMirrorBudget, stage placement and graph-owned VerifyHits. Prepare
patch0012 against frozen v5 source, verify applicability after0011 and against the
complete pristine v6 patch chain, and test actual host-only helpers under
ASan/UBSan with an explicit SYCL mock. No live source or GPU work by this agent.

RESULT -> Legacy mirror directly asks sycl::malloc_host(total,current default
queue), catches a SYCL exception or checks a null return, and reports failure
without installing any source mirror. It does not call host_malloc_polled or
request an uncached LevelZero host allocation. The failed C1 profile omitted
STRATA_STAGE_MIRRORS, so patch0005's per-stage owners and shared host budget were
inactive. Merely setting a higher global cap does not change this single-request
shape.

The parent's actual card0 allocation probe recorded device max_alloc32530182144,
global34242297856 and process memlock8388608 bytes. Host allocations at1MiB,
1GiB and8GiB returned valid pointers;48GiB returnednull. No bulk writes were made.
This supports trying bounded segments, but proves neither the exact null-return
cause nor full host backing/device-access capacity. Memlock alone does not explain
these observed returns. Health and exact original-byte access remain GPU gates.

Patch0012 adds a default-off STRATA_STAGE_MIRROR_SEGMENT_MIB option. Unset/0
retains the prior contiguous stage allocation. Positive values must be<=1024MiB
and require STRATA_STAGE_MIRRORS=1. Keep the existing stage guards: streamed native
GGUF experts, populated cache, --adapt-every0, --no-prefill-borrow, no peer/remote
or resident-CPU tier, MTP, pipeline windows or same-device split. Required C1
settings retain VERIFY_DEVICE_PLAN=1 and VERIFY_NO_HOST=1. No legacy-mirror
fallback is added.

The new CPU-testable plan groups complete256-byte-padded expert blobs into
segments at most the configured limit. An expert cannot cross a segment; an
oversized expert, malformed extent or overflow fails before allocation. Each
segment's actual requested size equals its occupied padded bytes, with no unused
full-size tail allocation. Total owner bytes and the global budget remain the sum
of exactly those logical requests. Existing raw expert GGUF bytes, tensor layout,
raw source FNV checksums, kernel arithmetic, tier counters and lookups are retained.

Every host allocation uses the owner's retained queue/context; host-USM metadata
must match that context. The full source gather writes directly to each expert's
segment address. Existing arbitrary-address device tables and source aliases
receive these exact pointers only after all reads/checksums and table upload
finish. Shared ownership is installed before aliases/counters publish, so a
publication-vector allocation failure cannot expose dangling aliases. Graphs
retain the same stage owner through VerifyHits.

Segments are freed only after the retained owning queue completes and graph
references retire. Explicit release errors propagate. Successfully freed pointers
clear individually, allowing retry after a later segment free fails without
repeating an earlier free. The existing nonthrowing destructor remains a failure
fallback; GPU lifecycle evidence must still establish every actual free and
normal process exit.

In segmented mode, all stage caches are already populated when a single global
budget preflight sums every remaining expert's padded bytes. Complete coverage
must fit that budget before any stage mirror allocation. Each stage then draws
its complete allocation from the same remaining cap. This removes fair-share
truncation for unequal stages: for example40GiB+10GiB can fit a64GiB global cap
without falsely limiting stage0 to32GiB. Disabled mode retains its old fair-share
policy. Requested caps/reserves are not multiplied by the number of stages.
The segment parameter is included in the disk-session identity contract.

CPU validation reconstructs the actual8083-resident/16493-missing profile and
native_experts layout. At64/256/512/1024MiB limits, the actual planner uses
793/194/97/49 segments, all totaling51688243200 bytes. Tests cover complete extent
coverage, alignment, immutable failure output, segment limits, arbitrary pointer
byte preservation and padding guards, asymmetric global budgets, graph-held
lifetimes, two mock contexts/devices and injected partial-free retry. These are
host contracts; the mock cannot qualify SYCL allocations or device pointer use.

VERDICT -> Final0012 SHA256:
e620578b9de2b5ba6cc2826375f2e9d8505066859f159f422aad8b17c3e56b23.
Immutable native-source-engine-build-plan-v6.json adds0011/0012 and the new
stage_mirror_segments.hpp overlay; plan SHA256:
88b337dc41ca350757a0fe91c69ae1295dadf4f41d51f0dfabec519d315022cd.
Full source compilation remains parent-controlled. Before serving, require the
measured allocator limit for the selected context, complete original expert
bytes and exact pointer-table consumption on each owning GPU, all390 HC/PLE
source images, complete VRAM-or-stage-mirror coverage for both stages, matched
tier counters, graph/context ownership, normal logical frees, teardown and
post-health. A successful small allocation or CPU plan is insufficient for the
required48GiB backing or two-stage capacity qualification.
