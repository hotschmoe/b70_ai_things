# Observed CPU retry startup race and owned recovery

CONFIG -> Exact e633 continuation recipe, reviewed observerV3 and wrapperV2.
Original prior swap failure95049 remains separate; strict memory guards unchanged.
COMMAND -> Actual14032 terminal1 after130s under pair CPU exclusion.
RESULT -> Idle30 passed with29 samples. Passive observer failed before its first
owned sample because its missing-object parser accepts only "No such object",
while current Docker emits lowercase "error: no such object: NAME". Command
publication preceded owned Docker creation. Observer failure caused wrapper
SIGTERM of the model parent; immediate absent-container cleanup failed.
The outstanding launch later created the owned CPU container after the parent
and wrapper exited. No continuation response or candidate was collected.
The failed report therefore never proves model/launch terminal ownership.
The final producer log changed after the failed receipt: recorded empty SHA
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
versus current add3b06d006ddbb64e375e4e04697b9036c2f87cbd2883d51e577e703c3187b3
(1242 bytes). This late write is failure evidence, not a repaired receipt.

COMMAND -> Root45070 reacquired pair exclusion, verified exact owned image,
labels and unchanged CPU runtime recipe, stopped and removed the container.
RESULT -> Recovery terminal0 after9s, stopped exit0/OOMfalse/Runningfalse.
Receipt SHA54a4debb19bf5644e06542d151013d22ae0b9b3c17836b92c226f35634e564b0.
Root81692 terminal0 after75s: new post-recovery complete four-shard hashes and
bracketing known pages PASS after owned removal. No GPU device grant/workload.
Recovery is separate from failed original teardown and never changes its PASS.

VERDICT -> Actual orchestration FAILED, owned recovery/integrity closed.
NEW observerV4 and wrapperV3 must support exact current absence grammar, retain
exclusion through actual launch ownership/descendant terminal evidence and
strict owned census, and record passive observer failure without aborting the
unchanged model screen solely for observation failure. No arbitrary Docker
error, PID terminal or observation timeout proves container absence.
Source/CPU review and new actual same-recipe all16 completions remain needed.
Full original math/cache/API1/2/4/6, x2 traces, latency/fairness and shelf remain open.
