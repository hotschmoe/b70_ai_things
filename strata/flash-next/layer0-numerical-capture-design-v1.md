#0021 bounded layer0 numerical observer source increment

CONFIG -> New patch against immutable combined20 source4uag6j5x and receipt
1647fdb0941d6e960bdbded793849b998de9de963eb5a6e6c687f18ea5547b00.
CPU/source only; no GPU, actual model payload, XPU/full-TU compilation or frozen
source modification. First-GDN original reference, packet contracts and old
observer source artifacts remain unchanged. Initial capture is partial26/33.

COMMAND -> python3 strata/flash-next/test_layer0_numerical_capture_cpu_v1.py.
The test reconstructs only the three changed source files and two added headers
in a private temporary overlay, applies0021 and verifies byte-exact reconstruction.
It compiles the ACTUAL new headers with a small mocked SYCL queue/graph/allocator
using ordinary CPU icpx (no-fsycl/no-devices/networknone, exact39992d70 image).
It does NOT compile the real Verifier/generate TUs or execute actual SYCL graphs.

RESULT -> Reconstruction and actual helper mock compile pass. Default off has
zero mock allocations/copies/frees.14 negative controls cover unbuilt /partial /
alternate combined graphs, wrong token position, no replay, stale request/nonce,
duplicate publication, wrong extent/context, reused state, cancellation and the
four-request bound. Three successful mock frames contain26 exact-sized files,
33 metadata fields, seven UNOBSERVED entries, current replay nonce+key, and no
model math qualification. These are CPU contract/lifecycle tests only.

## Source changes and observed fields

0021 modifies consumed sycl/core Verifier header/body and sycl/program/generate,
adds layer0_numerical_contract.hpp and layer0_numerical_observer.hpp. Only the
stage containing global layer0 reserves a snapshot. Raw GEN request begin /
actual resume count bind immutable accepted IDs; select only the first T1 input
at absolute position ids.size()-1. ARM absent warmups consume no quota. Supported
armed prefix sizes1/2/4/8, serial non-GENI, batch<=1, no MTP/pipeline windows,
no reused prefix. Unsupported requests publish a skip, not fabricated vectors.

Native HC, exact selected geometry, original expert layout, QFUSE0, graph mode
and existing T1 selfcommit are required when enabled. ONE_TOKEN_COMMIT0 is refused,
not changed: without selfcommit, state updates occur in the later commit graph
and this hook could not call the early buffer "outgoing state". VERIFY_EAGER must
be absent. Flags/RNG/cache/residency/math are never toggled by this observer.

Observe actual residual input; attention HC normalized/low/gate/inject/mixed;
ordinary input Q8_1; GDN incoming state/conv, qkv/z/decay/beta/normalized qkv,
gated output and actual output packet, projection block output, outgoing state /
conv; materialized attention HC write; FFN mixed/input packet; router logits /
IDs/weights; FFN block combine and final HC write. Copies sit after each actual
producer and before its scratch overwrite, in the same owning compute queue.
Incoming state is after preceding prefill, not assumed zero at prefixL>1.

Seven reserved fields remain explicitly UNOBSERVED: shared gate/up, shared
DERIVED hidden, shared HQ; expert entry map, expert gate/up, expert DERIVED hidden,
expert HQ. No unwritten raw hidden scratch is read or called an observation.
No derived hidden kernel is added in this increment. Outgoing residual alone
cannot qualify missing expert arithmetic. Metadata always has complete layout
and full_model_math_qualified=false; packet math proof still needs independent
supplied-input /intrinsic /affine-consumer expectations.

## Graph freshness and ownership

Four new graph keys combine resident/doorbell and existing0013 fidelity-first
selection. Both diagnostic modes together therefore keep0013 all-layer/head
copies; their caches cannot reuse a graph built without the other mode. Normal
graphs add no snapshot allocation/copy/marker nodes when the new flag is off.

Producer coverage is an immutable per-key roster, cleared at construction and
sealed only with the exact26 known hooks. It is not a sticky per-request seen
bit or reset only on the CPU while a cached graph replays. Runtime arm writes a
current PID/request nonce and clears completion using PERSISTENT class-member
host backing. GPU graph copies stamp nonce and a graph-specific device constant
only after all26 copies and real outgoing state. Dump requires current IDs /
position/request, sealed selected key, completed nonce AND key, and no duplicate
publication. Missing/alternate/partial/skipped graphs cannot publish stale data.
Actual GPU marker behavior is still a future qualification gate.

33 fields reserve7023048 bytes. One reused device snapshot plus56 bytes of
nonce/key controls requests7023104 bytes; at most four raw frames reserve
28092192 bytes below64MiB. Allocation/copy affinity binds queue context/device
and in-order property. Source buffers remain owned by the existing model/window.
Readback occurs after successful existing full-window completion; it is diagnostic
and cannot provide clean latency measurements. Metadata/file writes preserve
raw LE bits with exclusive creation and no automatic repair of partial evidence.

Combined graphs retire before either observer's storage, including explicit
fidelity release, mirror graph release and normal destruction. New graph/snapshot
release scopes the original device and waits the owner queue before free; no
postfree pointer query is added. Allocation/release markers identify stage,
pointer, requested extent and owning verifier queue. Parent chronological UR
trace/context-matched free and pre/post health remain mandatory actual gates.
The SHA binding tag is validated syntactically, not an engine self-proof: parent
must independently hash the immutable experiment/source/runtime/model manifest.

## Future execution preregistration

Build a NEW engine generation with20+0021 (and independent0022 if selected),
actual complete source reconstruction/hash receipt and all affected TU/link gates.
Pass source-upload/readiness/lifecycle gates. Compare21 flag-off with frozen20,
then SAME21 binary off/on with matching model/tokenizer/GEN IDs/source hashes,
precision/seam flags, context/chunks/cache cuts and stage placement. Existing0013
can be enabled for paired all48 residual/full logits; graph keys cover coexistence.
Do not promote any source math claim before output/logits/state/lifecycle match.

Future enable environment: STRATA_LAYER0_Q8_DIAG=1, absolute existing output
DIR, ARM pathname (create after readiness), immutable binding SHA256. Preserve
QFUSE0 and ordinary selfcommit /graph flags; do not force an easier math mode.
Use new outputs for numerical raw GEN prefix probes1/2/4/8. They are not natural
API completion or latency tests. Source identity before/after and current known
page guard remain mandatory; old stat cannot prove unchanged buffered content.

## Remaining seven producer hooks: separate next source increment

Shared hooks must bind actual raw gate/up BEFORE an unfused fallback overwrites
gate, actual HQ AFTER native packing, and declared DERIVED hardware-intrinsic
reconstruction separately. A shared-stream fork needs an explicit existing join
before compute-queue readback; never infer synchronization from host call return.

Expert hooks need the actual native grouped scratch layout and resident+mirror
call spans. Capture the actual entry token/destination map, validate all10 rank
slots are unique/covered and associate source expert IDs from the router. Preserve
actual default fused kernel/mode and obtain HQ after each producer before scratch
reuse. A diagnostic gather/reconstruction must label derived hidden, while raw
fused hidden remains UNOBSERVED. Require source/shape/mapping negatives and a new
GPU off/on+packet/math/lifecycle gate; do not infer these seven from combined
output or merely relabel this26-field receipt as complete.

VERDICT -> Source0021 and CPU contract checks are ready for parent review/build.
No full SDK compile, actual graph, hardware nonce, numerical, serving stability,
original whole-model fidelity or latency qualification is claimed.
