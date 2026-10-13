# GPU lease acquisition must fail closed

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Source/CPU-only urgent audit after root observed a SIGTERM'ed queued
flock return143, followed by an unleased NoDri compiler launch. The actual
attempt is invalid and preserved. Root interrupted its exact owned builder;
container retirement succeeded. No GPU devices were exposed. Live PLE and
source61 actor files are unchanged. This candidate does not edit bin/gpu-run.

COMMAND: Read actual bin/gpu-run, shelf contracts and dispatchers. Prepared
strata/flash-next/gpu_lease_acquire_fix_v1/gpu-run-acquire-failure-v1.patch
and matching gpu-run-candidate.sh. Ran bash -n plus seven real flock/process/
signal tests against tiny temporary CPU lock files, never the GPU locks.
The original wrapper snapshot is retained solely for the negative control.

RESULT: Seven tests PASS. The original snapshot reproduces unleased command
dispatch after TIMEOUT0 waiter143. Candidate rejects that failure before
owner publication or command launch, preserving unrelated owner text.
Wrapper TERM/INT really terminates its queued waiter and returns143/130;
this is verified with actual process lifecycle, not assumed Bash trap behavior.
Second-card acquisition failure cleans only its acquired card0 and preserves
foreign card1 owner. Timeout returns124 without dispatch. Active wrapper TERM
keeps foreground workload/lock/owner until child finishes, then returns143.
Success and workload exit17 cleanly preserve their actual return codes.

VERDICT: Root review required before applying the bin patch. All flock
statuses are checked, including blocking TIMEOUT0 and initial nonblocking
errors. Only the lock waiter runs in background so wait can receive shell
traps immediately; tracked queued waiter is terminated on cancellation.
Actual workload remains foreground, preserving stdin/process-group behavior.
INT/TERM no longer invoke owner cleanup and continue execution. Queued signals
exit; active signals latch until foreground retirement. Immediate active
cancellation targets the owned actor PID, whose existing handler owns actual
native/Docker teardown. EXIT alone clears owner files. Cleanup registration
occurs after successful flock but before publishing ownership.

This fixes lock acquisition safety, not arbitrary detached workload ownership.
A wrapper command must continue to own every GPU worker until its retirement.
An active signal does not authorize premature lease release. No throughput,
new serving stability or GPU health claim follows from CPU flock tests.

Required shelf gate and current static blocker

AGENTS.md165-166 requires applicable live-shelf smoke before bin commit.
bin/README.md17 and rdy_to_serve/_common/lib.sh5 spell out all-shelf
bin/serve-sweep --smoke GREEN; the sweep performs start/health/one-token/stop
using GPURUN and defaults GRAPH0. The exact candidate command would be:

  GPURUN=/absolute/path/to/reviewed/gpu-run-candidate.sh bin/serve-sweep --smoke

Do not launch it unchanged against stale routes. Static discovery finds:

- vllm/qwen36-27b-nvfp4: retained configuration control, old ABI quarantined.
- vllm/qwen38-27b-nvfp4: retained configuration control, old ABI quarantined.
- vllm/qwen38-27b-fp8: script accepts start/run/stop/config, not smoke, and
  unconditionally configures P2P_ACCESS1.

rdy_to_serve/README.md20-34 explicitly says retained NVFP4 entries are not
runnable production shelves until fresh ABI rebuild/requalification. Both
NVFP4 wrappers capture ACTION but exec the detached backend launcher without
forwarding smoke, so this sweep cannot establish health/generation/stop even
if stale dependencies happened to exist. FP8 rejects smoke with exit2.
Thus the all-shelf gate cannot honestly be called GREEN by this audit, and
restoring archived binaries or forcing the old P2P route is not authorized
by an infrastructure fix. Root must record disposition of applicable live
checks before bin commit; use actual current Flash-Next owned serving/health/
lease lifecycle for any scoped current-path smoke. No GPU smoke was run here.
The lock patch does not alter healthy foreground model execution, so this
report asserts no performance improvement and does not invent a bench result.

Runtime flat copy /mnt/vm_8tb/b70/gpu-run is a separate hand-synced deployment
(bin/README.md20 onward). Review its exact bytes and applicability separately;
changing repo bin alone does not repair a different runtime copy. No host copy
was edited by the author.
