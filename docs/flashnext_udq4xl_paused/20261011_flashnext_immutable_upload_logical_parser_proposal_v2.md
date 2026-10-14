# Immutable upload logical parser proposal V2

CONFIG -> New unintegrated source/CPU successor. Frozen V1 4ee4aa45 and all
H50/original parser/upload/evidence files remain unchanged. Exact positive
and three real negative parsing, source/runtime byte snapshots, parameters,
owner scope and all future outer live gates retain V1 semantics.

COMMAND -> PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_logical_free_immutable_epoch_cpu_v2 test_logical_free_worker_owned_cpu_v2 test_operation_evidence_snapshot_cpu_v2

RESULT -> 34 CPU controls pass:18 narrow parser controls,6 ownership/output
controls and10 unchanged generic purity controls. Tiny actual own CPU workers
were launched, interrupted, timed out and retired. No actual upload/SDK/log,
model/pack payload, Docker or GPU work occurred. V1 review found interrupt
retirement and post-buffer-only output bound gaps; its source is preserved.
Root also demonstrated a caught transport failure could produce passedTrue
without a failure latch in the drafting V2. That draft was unexecuted for
actual admission; final V2 has explicit regression controls for it.

VERDICT -> Source/CPU proposal for review, not H50 integration or runtime
qualification. The new ownership helper requires main-thread signal ownership
and an actual new CPU worker session. Every BaseException/error/timeout path
retires the child, closes stdin and both regular sinks, and proves the owned
session empty before exception propagation or reuse. A second catchable
cleanup interruption cannot cut retirement short. Temporary signal handlers
invoke prior handlers and restore them after cleanup. Unexpected descendants
cannot silently qualify, and PID/session reuse is recognized by start ticks
without killing a foreign replacement. Unknown cleanup state retains control
until emptiness is observed; there is no fallback terminal or release stamp.
Uncatchable parent death/SIGKILL cleanup is not qualified by this prototype.

The worker sets RLIMIT_FSIZE=32 MiB before parsing. Parent stdout/stderr are
regular owned temporary files, so the kernel limits writes as produced; no
unbounded pipe buffer is collected. Parent observes terminal state, closes
sinks, checks actual sizes, then bounded-reads them. This is a file output
bound, not a heap/RSS or CPU-time bound. The reader does not claim pipe EOF
for these regular sinks. Worker PID/start matches the parent's actual process
identity. Library/source/pyc/executable byte roster and generic immutable
entry/predevice/post barriers remain unchanged in meaning.

Any invocation, interruption, transport, output, JSON/protocol, worker parse
or runtime-closure error latches failure before returning control. Later memo
reuse is refused. If a caller catches the error and still seals/finalizes,
the proof remains passedFalse with typed attempt/error/retirement/output
observations. Unobserved parse counts are None and completeness is false,
never fabricated zeros. Successful and parser-rejected attempts retain actual
worker counters. No saved result, callable hidden state or generic purity
relaxation is introduced. All original source/model/SDK/health/page/time,
container, inventory and shape predicates remain outside the pure cache.

Actual-callgraph25-scan derivation and future explicit epoch routing are in
V1 design. Root's bounded6 actual cProfile will supply observed fanout/cost;
no speed, model math, concurrency, quality or actual-upload claim follows
from these tiny CPU controls. A future consumer must pin V2 in a new source
generation and retain immutable worker attempts plus all byte/owner/chronology
and original live gates. No frozen H50 helper is edited.
