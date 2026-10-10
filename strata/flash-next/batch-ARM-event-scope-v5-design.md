# Initial BGEN event-scope repair and failure replay

CONFIG -> preserved failed native2/diag1 source35 batchV6 run; same-process
warm RIDs1001/1002 -> HARNESS ARM -> targetRIDs2001/2002, actualconstant32
BGEN limit. Source35 generate.cpp emits event=bt_windows+1 and resets that
counter when all private batch slots become idle. Engine generation persists.

COMMAND -> read-only replay_batch_native_failure_v1.py against the failed
child log, stored plan and captures, verifying every preserved artifactSHA
and size first. Independently run roster consumption, coverage, completed
N2 event proof, exact fresh counters and serial prefix jobs. Source tests:
PYTHONDONTWRITEBYTECODE=1 python3
strata/flash-next/test_batch_arm_event_scope_cpu_v5.py.

RESULT -> actual Roster replay passes warm32/32 and target6cancel/32length.
The simulated cancellation condition first becomes eligible on targetN2
completed event1 at logline616386; command would be BSTOP0 rid2001 slotgen2.
This replay does not claim an independently recorded actual stdin write.
Frozen wholetrace V2 coverage raises AssertionError with empty message at
line19: duplicate(pid,enginegen,event) because warm and armed events1..31
share enginegen106462208427817. This is not queue.Empty, invalid emitted
IDs or cancellation failure. Event/fresh counters and four serial jobs pass.

The new ARM-scoped audit on those preserved bytes passes196 raw vectors,
11837440 bytes,3 replay proofs and31 armed events. Eleven CPU controls pass.
The failed parent/child receipts remain failed and unchanged. No new native
run, numerical serial comparison, model math, batch concurrency or speed
qualification is inferred from parser repair or preserved-log recovery.

VERDICT -> concrete source35 producer-counter scope mismatch in Python
coverage, with cheap deterministic CPU reproduction. Frozen V6 and V2
source/evidence are preserved. New source driverV7/auditorV5 are ready for
root to integrate deliberately into NEW immutable execution/parent plans.
Existing V6 source hashes/parent admissions must not be silently transferred.

AuditorV5 requires exactly one warm-to-ARM marker, no unarmed observer
request/resume/replay/row/vector/allocation data, actual completed N2/4/6
geometry in the declared private slots, single matching PID/enginegen on
both sides, and no duplicate event key WITHIN either session. The marker
provides a declared source session boundary. It then invokes the frozen
strict initial V2 target audit unchanged on the armed interval. Raw48/head,
replay epoch, GPU row identity, slots/generations, mask, consumed spans,
finite vectors and byte quotas remain strict. Source producer warm-to-ARM
idle counter reset no longer masquerades as a global duplicate.

This is an initial BGEN repair, not a universal interpretation of arbitrary
multiple post-ARM idle sessions or API migration. Later API audit integration
must retain its explicit migration/current-session identity contracts.

DriverV7 preserves V6 lifecycle/native submission behavior, adding stage,
exception type/message/repr/full traceback and a deep roster snapshot at
failure. Empty AssertionError/queue.Empty messages become identifiable;
owned close failure receives its own stage evidence. Requested plan limits
[64,64] versus actual native constant32 are recorded as an unresolved
separate contract mismatch, not inferred as this exception's cause. This
repair does not silently change the submitted limit or claim64-token runs.
