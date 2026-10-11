# Output-only PTI clock inputs

CONFIG -> Deliberate successor to freshly built PTI commit
6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033. Stock executable, library and actual
919-event capture remain immutable. This proposal adds a header and only output
calls to original ze_collector.h and unitimer.h. The exact patch reconstruction
is checked against their pinned source hashes. No original clock arithmetic,
Level Zero query, event, synchronization, kernel or helper body is replaced.

COMMAND -> Root creates a NEW isolated source checkout of that commit, verifies
the original 893-file source manifest, and applies the exact new patch. Generate:

    PYTHONDONTWRITEBYTECODE=1 python3 strata/flash-next/prepare_pti_unitrace_clock_export_v1.py > NEW_PATCH_COPY

Root applies with git apply --check, then git apply. It must verify all original
source bytes outside the two named replacements and the exact new header against
the source plan. The existing fresh build recipe's C388, compiler SHA, Release,
L0-only flags, pinned header commits and plain CMake/Ninja commands are retained
with ONLY explicit new source and new output roots. No stock source tree or ELF
is edited; root builds and inventories new ELFs before any actual GPU capture.
The reported version's empty commit string is not source identity.

ON is explicit:

    UNITRACE_CLOCK_METADATA_PATH=/results/unitrace/clock.{pid}.jsonl

OFF is absent or empty. OFF returns before writer initialization, allocation,
mutex, open and output. It still tests getenv; this is not a zero-instruction or
zero-overhead assertion. The exact output template belongs to this fixture
recipe; the output directory must exist and be owned. Each process exclusively
creates clock.PID.jsonl mode0600/nofollow. Writes are serialized, individually
bounded4096 bytes, total8MiB per process. Duplicate files, inherited process
writer, overflow, write or close errors fail with exit86 and retained stderr.
No fsync or crash durability claim is made.

RESULT -> host_clock_origin exports the original selected epoch origin and
signed boot/TAI offsets as explicit two's-complement u64 payloads. device_origin
exports the original queried host/device timestamp pair, properties1.2 frequency,
kernel/global valid bits, mask, exact F64 ns-per-cycle bits, UUID native u64 words
and device descriptor identity. It does not query another timestamp.
instance_normalization exports each original actual instance/command ID,
device/list/queue/engine and submitter thread, original append/submit host times,
original retained submit device ticks, original raw kernel start/end ticks,
frequency/mask/F64 scale, original computed host start/end and epoch origin.
The process PID and writer TID are actual POSIX identities. No command context
owner is claimed because ZeCommand does not retain that field.

VERDICT -> Source/CPU proposal only. Exact arithmetic reconstruction controls
preserve the original strict greater-than submit comparison, including its equal
branch behavior; this proposal does not fix or reinterpret that arithmetic.
Future actual collection must join every919 instance and its original Chrome
GPU/flow record, match descriptor/device/owner/current loaded library, verify
frequency-mask properties and exact original normalization outputs, and retain
all raw data before/after. Original18 outputs and current health/retirement gates
remain mandatory. A missing clock record or failed export cannot become a
calibration PASS. Exported values still require real collected evidence and
normalization validation before any clock qualification. Physical/model idle,
normal serving latency and zero added waits remain false. New CPU file I/O and
mutex can perturb this instrumented fixture and need separate overhead controls.
