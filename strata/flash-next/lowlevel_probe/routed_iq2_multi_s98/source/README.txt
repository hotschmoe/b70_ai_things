S98 routed IQ2_XS GU primitive - SOURCE PREPARATION ONLY

CONFIG -> Fixed N2560/FF640/QT17, two actual S42 layer0 experts41/211.
Read only their gate/up947200B windows during a future approved run; no model
or compact weight payload was read while preparing this packet. The complete
S42 compact file is pinned to a9549c6a... using its retained export manifest.
Allocate full1868800B blobs with unread down regions set to finite byte sentinel;
all blob bytes remain read-only. Repeated group pointers alternate the two
experts: synthetic fixture aliasing, not an observed serving route distribution.

Three small layouts: mixed sizes1,2,3,4,5,6; singleton sizes1,1,1,1,1,1;
zero groups. cap_groups8/grid_groups2/cap_entries32, active firstentry2,
permuted/repeated ent_tok in6input rows. Mixed/singleton stride beyond grid2;
inactive prefix and remaining capacity must preserve finite123.0f exactly.
4096B outer guards stay byte0xd7. Finite inactive poison avoids feeding NaNs
into public phase1's unrelated all-capacity SwiGLU/quantization pass.

COMMAND -> Future ROOT only, after full source/model review and explicit GO:
  bash build.sh
ROOT audits generated host/device bindings and writes root-build-closure.json
with actual hashes/fields described in root-build-closure-required-fields.json.
Then announce one quiet physical-card0 window and, only on execution GO:
  bash ROOT_RUN.sh
The wrapper uses cpu-run --quiet OUTSIDE gpu-run --card0 with ZE mask0 and
level_zero:*. One numerical process, unique container, one pre/post per-card
health, no TP collective needed for this one-card primitive. Original300s leaf
cap is an execution safety bound, not a polling deadline. Any exit stops this
attempt; cleanup addresses only its valid CID, requires BOTH fullCID and name
absence before health, and retains CPU/GPU leases if removal is unconfirmed.
No auto reset/retry/server/production mutation. Scripts refuse prior build/run.

BUILD -> CPU-only pinned e41/noDRI/network-none/UID1000, --cpus4, OMP/MKL4,
MAX_JOBS4 and linkjobs4, disk tmp/cache. Compile ONLY the new adapter/private
header, supported private prelink, ordinary host link against existing S60
common8f5a and frozen f2b kernels/34e core. No public/kernel TU rebuild. Actual
IQ source acaa7d... matches the existing recorded source/member binding;
future build verifies source/membera61c and archives/wrapper hashes ONCE,
then saves measured actual stat continuity for runtime checks. No repeated
whole-archive/image copies. Private AOT is newly generated; public wrapper is
reused. Final host weak metadata separation must be audited (S91 precedent),
not inferred from new class spelling/device IDs. No compiled result exists yet.

RESULT -> PREPARED_NOT_EXECUTED. Public native_expert_grouped phase1 first:
gate scratch+0, up scratch+align256(32*640*4)=81920. Its following SwiGLU/quant
only reads those planes; phase1 returns before down. Full gate/up bytes from
public anchor must equal private original and natural E4/E2/singleton Multi.
Two inputsA/B, eager own-repeat and captured A/B/A checks plus batch20 final
checks; read-only weights/FP32/Q8/metadata, full finite GU, inactive capacities,
phase1 unused down-output and guards checked before any timing. Zero changed
output observability is explicitly vacuous. All three layouts pass before
warming/timing either active layout. Timing:3warm batches/arm then4balanced
original/Multi pairs,20calls captured in ONE graph submission per sample.
Reset/snapshots/readbacks remain outside each timing interval. Host enqueue,
host completed wall and profiling-barrier span are distinct observations.

Expected completed inventory:88checked snapshots,16timing samples,
3correctness records,22binary files (18fullGU eager anchors +2FP32 inputs
+2actual nativeQ8 images including their512B poison tail),10DOTs. Public
initial/changed and both private arms fullcapacity bytes are exported; later
repeat/graph/timed checks are strict in-process. The small analyzer enforces
that exact inventory and compares all eager anchor bytes again. Runtime
identity checks exact kernel7.1.0-070100-generic and current bootb785982c-4954-
4d23-808c-48a7534ce3bd pre/post, plus build binding, actual payload and source
hashes. Future records are not present or claimed by this source packet.

VERDICT -> Await root full read/actual Claude consultation/build and binding
review before any GPU GO. Stop at first strict failure; no tolerance switch.
Even full success is a warm repeated two-expert GU primitive, not full EP,
streaming-cache behavior, serving speed or production/model qualification.
Singleton-only timing exposes natural Multi resource overhead without reuse;
route frequencies and source-pool totals cannot forecast serving savings.
DOT is capture metadata, not independent dispatch proof. No unitrace sweep.
