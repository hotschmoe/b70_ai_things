# Verifier initialization owns capture streams

CONFIG -> Actual first two-card C1 reached readiness with complete24576-expert
coverage: stage0 resident8849/mirrored7535 (23557120000 bytes), stage1 resident8192
and no missing experts. The first request recorded one two-token window, then
rejected "stage mirror: verifier queue/residency changed after binding" before
stage1 math. Unwinding reported UR_ERROR_UNINITIALIZED37. Parent strict pre/post
health passed, and standing post-crash recovery is parent-controlled.

COMMAND -> Inspect the exact combined0013..0017 engine source at
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T080647Z-6_ja3xd5/source,
GpuStage construction/init order, SYCL/CUDA Verifier defaults, queue selection,
mirror binding/record guard, capture and destruction. Prepare0018 and an immutable
source build generation, actual control initializer and bounded capture oracle.
No live checkout, model, driver, GPU or old result changes by this agent.

RESULT -> Source proves a construction-device sentinel error. GpuStage embeds its
Verifier objects before entering OnDevice(stage.dev). The SYCL header initializes
ext_stream_ to the default queue of that construction device (GPU0); CUDA uses
nullptr. Later init onGPU1 binds mirror/residency toGPU1's default queue, but
compares ext_stream_ againstGPU1's default queue. The staleGPU0 pointer differs,
so it is incorrectly treated as an explicitly supplied stream and becomes cs_.
The unchanged mirror guard correctly rejects that foreign queue. The first
successful T2 capture is stage0; stage1 fails before T1 or its model math. Source
localization does not require a changed residency pointer or a nested-capture
current-device leak.

SYCL cs_, sh_cs_ and copy_ also use default-queue constructor initializers. An
unused verifier's destructor can erase the default queue through destroy_queue,
leaving other default lookups dangling. Destruction also formerly freed using the
caller's current device. Those paths can contribute to bad unwinding; source
analysis alone does not prove the specific cause of UR37 in this process.

Patch0018 restores nullable defaults for all four queue fields. Construction
performs no queue lookup and owns no runtime queue. Only explicit set_stream
borrows a queue. The actual init_capture_streams method used by full init checks
explicit queue context/device/in-order compatibility before model allocation,
creates owned copy/execution/shared queues through the explicit init device, and
binds the mirror to the selected execution queue. It does not waive or rebind an
incompatible mirror association.

Capture, record, warm, graph release and destruction select the owning device
for their scope and restore the caller. Source residency/device/context matches
remain mandatory. A rejection now logs comparison components and expected/actual
residency pointers, so any later mismatch can be localized without weakening the
predicate. Tensor identities, math, cache admission, stage ranges, mirrors and
model arguments remain unchanged.

The CPU test exercises the actual helper under ASan/UBSan with explicit mock
queue context/device/order values, rejects foreign/unordered streams, reproduces
the old construction-device comparison, checks the corrected nullable fields,
verifies engine init calls the actual initializer and preserves the mirror guard.
The complete pristine combined patch chain through0018 applies successfully.
These prove source/host control contracts, not GPU capture or serving.

The bounded GPU oracle constructs two real Verifier objects whileGPU0 is current,
initializes their actual control streams onGPU0/GPU1, and captures actual DPCT
resident-plan/tagged-mirror consumers for T2 thenT1 with the caller deliberately
on the opposite device. It replays four graphs twice, checks owning scope/caller
restoration and nine metadata negatives, and tests a compatible borrowed stream
survives borrower destruction. It loads no model tensors and executes no full
Verifier::record_window body. Synthetic tagged allocations make the queue/control
proof bounded; they do not substitute for original model source or numerical
qualification. Full two-stage model, handoff, state and history remain required.

VERDICT -> Frozen0018 SHA256:
a9f382a8861132746475b34b738b723d281f44dc33e83aa8828ae849a87987a3.
New verifier-stream-engine-build-plan-v1.json SHA256:
97243b44806418ce81557923405e8b219b33082ebb4a8f3e7b72a903413df08d.
CPU contract checks and patch applicability pass. Parent must compile the full
source generation, link the bounded oracle against those exact libraries, qualify
its actual pair capture/replay and healthy teardown, then rerun actual two-card
C1. Existing failed two-card receipts remain failed and unchanged. No serving
stability/speed or complete two-card fidelity is established by this source fix.
