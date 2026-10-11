# PTI prototype V3 protected watcher startup

CONFIG -> V2 is held unchanged after the actual tiny partial Thread.start
counterexample. V3 changes only the actor/public-reader namespace and protected
watcher startup/failure latch. Runtime intake remains unfilled and all previous
mapped-library/source-complete coverage gaps remain explicit.

COMMAND -> run_source41_pti_latency_v3.py and validate_source41_pti_latency_v3.py
retain the same future --pti-intake/model arguments; current unfilled intake
is refused before model/lease activity. No actual runtime command was run.

RESULT -> eleven tiny CPU controls pass. The new control calls real Thread.start
then raises InterruptedError. The actor raises failure, sets stop, joins the
actual non-daemon watcher and retains its original failure latch. Native phase
is not invoked. No fake thread retirement, GPU or model action occurs.

VERDICT -> source prototype pending peer, not runtime-ready. watch.start is
inside the protected try/finally. Partial startup and owned-phase exceptions
permanently mark failure, and watcher.close remains mandatory. Original native
cleanup/EOF, postwrite raw caps and all model/source/full4/health/page gates
are preserved. No mapped injection census implementation or runtime PASS is
claimed; that remains a separate follow-up.
