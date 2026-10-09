# Batch observer 0024-v2 CPU source READY

CONFIG -> actual compiled20 source with its38-file expected ledger, frozen21
3c282ced and22-v2 1c7e7352, then new0024-v2. Original0024 draft1da9dad8,
its plan/test/auditor and CPU evidence remain unchanged. Strict native HC text
batch2/4/6/group1, no-MTP, non-pipelined captured graphs, and distinct devices
when stages are split. Existing serial fidelity/layer0 observer flags are rejected
with this observer. Numerical kernels, cache policy and scheduling stay unchanged.

COMMAND -> python3 strata/flash-next/test_batch_fidelity_observer_v2_cpu.py
reconstructs every bound input and verifies output source hashes, then compiles
actual new headers with ordinary host g++ ASan/UBSan against graph-record/replay
queue mocks in image39992. No devices, SYCL compiler flags, SDK invocation or
runtime libraries are used. A separate dry apply checks current draft23 between21
and22-v2/24-v2; final23 d86af122 is now frozen and a fresh21/23/22-v2/24-v2 apply passes.

RESULT -> CPU source reconstruction and actual-header mocks pass. The graph
controls reject21 no/stale/partial/duplicate/wrong-layout/wrong-peer/wrong-engine,
position and migrated-slot cases. A separate successful process produces196 raw
vectors, six owning stage replays and one completed multi-row event. The strict
collector accepts that synthetic full48/head coverage and rejects17 empty,
missing, duplicate, marker, ownership, row-mask, identity, span and bound cases.
Default disabled and enabled-but-unarmed processes allocate, copy and wait zero
additional diagnostic resources. This is source/helper evidence, not a SYCL TU
build, device experiment, model math or concurrency qualification.

VERDICT -> CPU source READY for full compiler review. Runtime qualification and
all actual2/4/6 numerical/cache/fairness gates remain open.

STRATA_BATCH_FIDELITY_DIAG=1 with STRATA_BATCH_FIDELITY_ARM and
STRATA_BATCH_FIDELITY_DIR opts in. ARM must exist before accepting observed BGEN
requests. There are at most six RID entries, two samples per RID,16 immutable
graph rosters per stage and8192 logical span events. Raw device storage across
all48 layers and head remains17756160 bytes; each stage additionally owns4736
metadata bytes. Two stages therefore own17765632 bytes total. Six complete
paired samples write35512320 bytes under a64MiB output cap.

A graph roster fixes admission/later phase, row count, actual ordered slot rows,
AR/doorbell variant and handoff base. Admission graphs consume the main session
row0; their dynamic target slot is bound in the frame. A later graph consumes its
exact ordered active slots. The roster contains each local layer exactly once,
and a head only on the final stage. A second seal/rebind is rejected. Constant
graph layouts upload before queue recording. Queues must be in order, and each
arm/producer/stamp checks the captured owning device and context.

Every arm binds a new epoch, native engine incarnation, protocol22 RID, observer
request generation, slot generation, row, input token, absolute position, active
slot mask, selected row mask, phase and geometry. The engine incarnation derives
from the first observer-use monotonic sample plus PID and stays fixed in the process; PID alone is insufficient across
container restarts. This is the native incarnation, not an invented API generation
on the native wire. API TLS engine_generation and original control-queue ownership
remain the frozen22 transport contract.

After each actual D2D residual or head copy, the captured graph copies that frame's
input epoch to its row/layer/head epoch cell. After all producers it copies the
complete input marker and its immutable graph-layout constant to output markers.
Before publication, dump compares the full input/output identity, graph layout
and every required/foreign row/layer/head epoch. Missing or changed data markers,
stale input, wrong layout or migration reject the frame. A failed validation
poisons the frame; a later replay cannot publish it. Explicit discard drains the
owning queue before allowing a fresh arm. Host metadata storage is persistent
owner memory and survives until the window waits or owning release drains.

The native driver binds the actual run tokens/positions/rows before launch. A
later sample requires an actual run_slot_rows event with at least two active rows;
a one-row window does not count. A monotonically numbered batch event is recorded
after the native call returns. The collector matches event, masks and geometry to
the replay proof and every raw vector. Admission samples consume prompt-1.
Both phases must cover all48 local residuals and the complete248320 head before
sampler selection. This does not substitute metadata for full raw coverage.

Source ledger completion now follows verified publication by every linked stage.
ARM disappearance or an unobserved child cannot falsely mark a request observed.
Default/unarmed selection clears stale identities. Diagnostic graphs retire
before snapshot buffers, slot arenas and expert mirrors; owner release records
pointer and returned free. Runtime logical-free traces still need qualification;
no post-free UNKNOWN pointer-type requirement is introduced.

The strict initial collector accepts distinct assigned BGEN RIDs. It validates
complete actual per-stage prompt evaluation spans from read_from to prompt end,
plus the later sampled position's owning stage spans. RESUME/read_from/reread_to
are separate observations, not complete-state cache correctness proof. Stale
migration controls are tested, but numerical BYIELD/solo migration, restart,
fullfresh, cancellation and automatic root/pin/leaf full-state cache-chain tests
remain later work. No pin/fresh batch guards are removed.

Next: freeze final23 compatibility, compile/link the whole source in the pinned
image, rerun actual source390/identity/health/teardown gates, then build a matched
serial vs2/4/6 per-request harness with these strict raw/span proofs. Source and
runtime bytes must match new receipts; old390 results cannot transfer to a new
build. The required automatic full-state cache-chain increment is separate.
Fairness still requires ongoing decode beside long prefill, request-local stream
P50/P95 measurements, eviction/cancellation isolation and complete teardown.
Diagnostic barriers/copies can affect timing or a race; these observations do not
prove their absence has no effect.
