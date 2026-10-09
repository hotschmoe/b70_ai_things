# Stage-owned exact-source expert RAM mirrors

CONFIG -> Pinned Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf and corrected
native-HC patch series0001..0004; selected UD-Q4_K_XL unchanged. Source work
only, no external checkout edits, SYCL compilation, device discovery or GPU
work. Default-off runtime mode STRATA_STAGE_MIRRORS=1.

COMMAND -> Inspect all stage/cache population, source gather, residency,
resident_plan and graph/serve teardown paths; prepare patch0005; sequential
apply; CPU ASan/UBSan checks of actual budget and owner headers with an
explicit mock SYCL queue/free interface. Reproduce:
python3 strata/flash-next/test_stage_mirrors_cpu.py.

RESULT -> Sequential apply passes. CPU tests validate one global cap with
reserves charged once, bounded two-stage spending, parser/overflow rejection,
two device/context/residency associations, layer bounds, shared graph-owner
lifetime and mock frees in retained contexts. Mocks validate host contracts;
they do not establish real SYCL allocation, device visibility or teardown.
Patch SHA256: 22abe038d2ffdc7ac98049af9955442979ed7e55461275f0a2faa8c27917071f.

VERDICT -> Concrete source replaces first-stage-only/global-dispatch mirrors
when explicitly enabled. It allocates only uncached experts AFTER all stage
VRAM caches are populated. Ownership, budget and graph association are explicit.
Native-HC/model-use gates remain in force. Actual GGUF-to-USM byte identity,
GPU-consumed bytes, direct RAM expert arithmetic, second-stage execution,
fallback, live graph replay and healthy teardown remain unqualified. This
increment does not establish tiered serving or latency improvement.

## Runtime policy and immutable placement

Unset/0 preserves the legacy mirror source/dispatch behavior. Malformed
STRATA_STAGE_MIRRORS values fail. New mode requires --stream-experts, a native
GGUF expert source, a populated profile cache, --adapt-every 0,
--no-prefill-borrow and no asynchronous adaptation, peer/remote/resident CPU
expert tier, MTP, pipeline windows or same-device split. These are explicit
first-increment constraints. Adaptive refill/eviction and overlap need later
qualified increments; they are not implicitly supported by this code.

The foreign-GPU prompt helper is disabled in new mode because it could consume
USM owned by another stage's context. Each stage reads its own prompt experts.
Ordinary missing experts retain existing safe host/storage fallback when the
RAM cap cannot cover all experts. STRATA_VERIFY_NO_HOST rejects any missing,
unmirrored expert across ALL stages, rather than checking only stage0.

## One process-wide bounded host budget

The driver reads MemAvailable once after all VRAM caches fill. It deducts
explicit OS, future PLE, future prompt and future workspace reserves exactly
once. Defaults are8 GiB each (32 GiB total). Overrides are nonnegative MiB:

- STRATA_STAGE_MIRROR_OS_MIB
- STRATA_STAGE_MIRROR_PLE_MIB
- STRATA_STAGE_MIRROR_PROMPT_MIB
- STRATA_STAGE_MIRROR_WORKSPACE_MIB

STRATA_MIRROR_MIB sets the global requested maximum; effective cap is the
smaller of that maximum and the one available-memory snapshot minus reserves
and metadata. Invalid/negative/overflowing values fail. Zero cap is supported:
explicit zero-address device tables retain correct affinity and misses use
fallback. Existing allocated source rings/PLE/host buffers are already absent
from the snapshot's available memory; reserves cover additional future needs.
Reserve adequacy is a matched-configuration qualification question, not a
claim that every context/concurrency choice fits those defaults.

A single StageMirrorBudget tracks total committed mirror bytes. Initial fair
shares leave remaining budget for later stages; unused bytes remain available
for those stages. Each stage filters missing pairs using its OWN filled cache,
deduplicates profile entries, then considers unprofiled missing experts. No
resident expert is included in mirror construction. Padding is accounted at
256-byte blob alignment. Original source blob sizes remain unchanged.
Host address/checksum/alias metadata receives an explicit global reserve;
transient gather bookkeeping remains covered by workspace headroom. Device
address-table allocations must separately enter the matched VRAM budget.

## Exact bytes, ownership and associations

GgufExpertSource::mirror_stage accepts one disjoint layer interval and a retained
SYCL queue. It allocates USM host storage in that queue's context, gathers each
original [gate|up|down] blob using the native_experts.txt per-role GGUF offsets,
records per-blob FNV64, uploads a device address table, and publishes aliases
only after all reads and the upload finish. Source weights are never
requantized. FNV64 is an audit checksum; pinned model/shard SHA256 and direct
GPU-consumed byte checks remain mandatory. Construction errors abort instead
of silently publishing a partial stage table. Thread-creation failures join
started readers before unwinding storage.

A StageExpertMirror owns both its host allocation and device table, together
with the original queue/context/device and immutable layer range. Source,
stage and VerifyHits retain shared ownership. Captured verifier graphs cannot
outlive their allocation owner even when the source or stage drops a reference.
Each verifier binds the EXACT residency base, full dimensions, owning device,
context and layer interval before capture. Binding is synchronized for multiple
session initializations. A changed queue, residency base or stage range rejects
the operation. No cross-allocation pointer ordering/subtraction is used in the
new path.

resident_plan_bound receives the explicit owning layer's table. Legacy global
mirror pointer lookup is bypassed completely in new mode. Native stage mirrors
enable the existing device plan when applicable; an always-publish overlap
mode rejects until separately qualified. All-resident slots remain VRAM hits.
An incomplete RAM mirror causes the existing plan's missing-entry fallback,
not a stale pointer. Later-stage verifiers receive their own table/owner.

GgufExpertSource::blob/read_into can use the appropriate published RAM address
on the host without re-reading storage. New host-source counters distinguish
actual storage gathers from host reads of mirror data. They do NOT count
GPU direct RAM reads or VRAM resident hits. Separate GPU tier counters and
trace receipts remain a profiling/qualification gate; no source counter may
be mislabeled as GPU-consumed bytes.

## Graph and allocation teardown

Owners retain queue/context copies so frees use the original context even if
another device is current. Normal serve shutdown drains all stage contexts,
permanently retires their verifiers, destroys ordinary token/session and verifier graph references, drops
source aliases, then calls checked mirror release before the existing _Exit.
Checked release propagates errors; only failure-path destructors suppress
exceptions. CPU mocks verify reference lifetime/free ordering, not live driver
behavior. Existing non-mirror backend allocation teardown remains a separate
lifecycle concern and cannot be qualified by this increment.

## Remaining gates before model/tier claims

Compile corrected0001..0005 overlay in enabled/disabled configurations. Changed
SYCL TUs: generate.cpp, verify.cpp, gguf_expert_source.cpp and
verify_kernels.dp.cpp. The common VerifyHits header adds shared owner metadata;
CUDA/HIP behavior must remain unaffected in their existing paths. New source
headers are SYCL-specific and the common header uses only a forward declaration.

First qualify one-card owner/table creation and context mismatches, source/GPU
byte checks for every mirrored blob, forced mirror hits, forced storage misses,
cap0/partial/full budget, duplicate profile entries, allocation failure and
safe fallback. Then qualify both populated caches and second-stage mirrors,
ordinary/prompt/verifier expert arithmetic, graph capture/replay, independent
sessions and healthy teardown/post-health. Native HC and original model
identity/full-logit/state-isolation gates must precede full model runs.

Immutable placement remains the first accepted mode. Before allowing borrowing,
eviction/refill or overlap, demonstrate no stale table/graph identity, correct
source fallback and one global host budget through those transitions. Trace
VRAM resident hits, actual direct RAM reads and actual storage reads separately,
then measure matched interactive latency. No backend or shelf promotion follows
from source support, CPU tests or attractive mirror coverage alone.

Prerequisite source snapshot: corrected0003 stage ownership207642b5 and
0004 exact-source bindabc9772a. Native HC multi-GPU loading requires an explicit
trimmed split until auto split can defer its original-source allocations.

Parent full-engine follow-up: the first consumed-SYCL build rejected005's
VerifyHits additions because sycl/include/strata/core/verify.hpp shadows the
shared include/strata/core/verify.hpp edited by005. Prior CPU mock tests cover
budget/owner helpers only; they do not cover the consumed VerifyHits declaration
or prove the complete engine compiles. Parent prepares new009 to repair the
SYCL declarations while retaining the failed build receipt and frozen005.
