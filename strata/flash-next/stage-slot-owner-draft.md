# Stage slot owner draft0020

CONFIG -> CPU-only source overlay on actual corrected0018 source at
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T085407Z-pzbw4jgp/source.
COMMAND -> Generate separate0020 patch; apply after drafted0019 to a new minimal
source tree; host-only ASan/UBSan compiler against explicit allocator/queue mocks.
RESULT -> Source application and mock owner controls PASS. No real SYCL/device
compilation, GPU execution or new full-engine build accompanies this draft.
VERDICT -> Ready for source review and parent-owned full compilation/ownership
oracle. No concurrent serving or numerical qualification is claimed.

The owner captures a stage's queue under explicit OnDevice, owns the raw device
arena and SessionState side allocations, validates null before session_init,
rolls back partial/throwing init, and releases all owned QSA host staging/KV plus
state metadata and device arena in owning context. QSA ordinals remain global;
second-stage release iterates qsa_ord0+j instead of accidentally freeing only
array indices0..3. Host KV allocation base/size are tracked rather than inferred
from interior typed pointers; shared host-KV byte accounting is decremented.

Normal fit shrink retains the original fitting policy but discarded owners now
free. No cache/scheduler/kernel math logic changes. Qualification must still
refuse INFO batch_slots below the requested count; a warning/fallback is not
successful parallelism. Elastic KV global registry allocations are outside this
owner's scope and explicitly rejected for batch before slot allocation. Current
full-resident FP16 one/pair profiles do not request elastic KV.

QSA host_step/host_pos/hostKV null checks prevent dereference after nonthrowing
SYCL allocation failure. Initialized graph-free sessions are freed on init
failure and resize. At normal QUIT, batch graph references are retired before
slot owners clear; stage-mirror teardown already retires those graphs, and the
non-mirror branch receives a slot-only graph retirement method. Normal scalar
serving math/cache selection and scheduling are unchanged.

RoPE has a singleton registered table per device. Freeing the latest discarded
slot can unregister it. After fit/shrink/fallback, the draft registers an
identical surviving slot/main table before graph capture. This preserves the
existing table-enabled route without retaining freed addresses or falling back
to different angle math. The mock explicitly verifies unregister then rebind.
No new table allocation or computation is introduced by rebind.

CPU tests exercise construction on card0 with allocation/init/free belonging to
stage1, global QSAordinal8, device-null with no session_init, partial zero/throw
rollback, repeated release,3-to1 resize and all owned frees, plus RoPE rebind.
The actual header and extracted source release/rebind helpers compile in the
host mock. This does not syntax-check complete real SYCL TUs or execute driver
allocation/free. The full source/library rebuild and actual queue/context/free
trace oracle remain required.

Patch0020 is separate from serial lifecycle0019. The source-only application
check currently applies0019 then0020 with offset-only hunks. Preserve all frozen
source/build evidence; append a new immutable engine plan with pristine external
source ledger and separately recorded patched overlays. Do not repeat the old
mistake of putting patched hashes into the builder's pristine-source field.

The subsequent graph-limit proposal remains separate: plain batch graph caches
can hold up to63 active subsets at six slots. Existing LRU hook can bound it, but
changing that retention/capture behavior needs its own default profile, memory
budget and cold-layout latency evidence. This draft does not silently introduce
that scheduling/capture policy.

First actual oracle should instantiate no-weight slot owners for onecard0:48 and
pair0:32/32:48, allocate/release/resize in the same current-device patterns as
production, verify distinct arenas/state storage and shifted QSA ordinal cleanup,
then parse chronological successful UR frees by owning context for all device
arenas and host staging. Postfree pointer-type unknown is not a valid gate.
Later full model slots2/4/6 need per-slot state/logit/counter identities and exact
active capacity before concurrent output or fairness measurement. See
concurrent-full-state-source-plan.md for pin/root retention, penalty/cvec scope,
API request-generation isolation and ongoing-decode/long-prefill qualification.

CPU test source binding follow-up: test_stage_slot_owner_cpu.py now reconstructs
the exact frozen combined05041bd1 plan from pristinefb58 and all20 tracked
patches in a temporary source tree. It verifies pristine and all38 expected
consumed hashes before compiling the actual header and extracted helpers. The
untracked source overlay is not consumed. Source reconstruction and host-only
ASan/UBSan controls PASS; no real SYCL/device compilation or execution.
Patch0020 and the combined engine plan remain unchanged.
