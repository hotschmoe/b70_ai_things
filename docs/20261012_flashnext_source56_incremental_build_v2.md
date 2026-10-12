# Direct55-to56 incremental startup repair V2

CONFIG -> Same isolated source/object/archive reuse and NoDri compiler gates as
V1. V1 and its failed tree remain unchanged.

COMMAND -> Exact root argv is in
strata/flash-next/source56-incremental-from55-build-recipe-v2.json.
Only startup changes: use the original full builder's SETVARS_COMPLETED guard
before sourcing oneAPI, and emit shell failure line/return code visibly.
Actual LF joining was already correct and remains unchanged.

RESULT -> Actual root V1 session84020 failed23s with Docker exit3, empty build.log,
no incremental sidecar files, parent closing proof passed. Exact receipt script
contains14 LF bytes and0 literal backslash-n sequences. Therefore a newline
repair would not fix the observed failure. Unconditional oneAPI re-source,
under set-e with all output redirected, is the leading startup explanation;
actual inherited environment was not recorded, so this is not claimed proven.
V2 skips re-source when oneAPI initialization already completed, matching the
existing accepted full builder. Visible errors locate any remaining failure.
One targeted CPU test constructs the actual V2 script, checks bash syntax and
actual newlines, and verifies inherited SETVARS_COMPLETED=1 skips sourcing.
No compiler, Docker, GPU or model payload was executed by the source author.

VERDICT -> New V2 startup candidate. Incremental reuse auditing/compilation has
not executed successfully yet. No incremental elapsed-time/speed claim.
