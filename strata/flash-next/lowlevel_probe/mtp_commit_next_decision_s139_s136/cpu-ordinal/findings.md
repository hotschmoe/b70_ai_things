CONFIG -> Saved S136 COMPLETE CPU diagnostic:820 records,dropped0,24windows;
existing conservative replay selection16windows/14adjacent same-dump transitions.
COMMAND -> Read parsed CPU JSON and three current pinned source files only.
No raw trace, GPU/build, new parser or CPU/GPU/peer clock join.
RESULT -> Actual order is verify_enter/stage/submit/wait/commit, THEN
slot_copy_begin/end perowner, THEN pair_call/.../pair_return. Thus same-window
pair_return->slotcopy is backwards; next-window slot copies follow another
verifier+commit. Actual tags are slot_copy_begin/end, not slotcopy.

Existing replay median CPU spans (ms):
- previous pair_return -> next window_enter0.002955 (n14).
- previous pair_return -> next stage_ple_begin0.0301475 (n14).
- previous pair_return -> next verify_submit_begin0.759122 (n14,max1.180681).
- stage_ple_begin/end0.709177 (n16,max1.149051).
- verify_submit_begin/end3.2213425 (n16,range3.062910..3.532445).
- verify_wait_begin/end43.867980 (n16); commit2.928889.
- individual slot_copy_begin/end0.0759395 (n32,range0.050387..0.114288).
- commit_return -> pair_call0.1722685 (n16,range0.164424..0.182678).
- pair_submit_begin/end1.9506695; pair_poll/PLE1.0989555;
  pair_queue_wait2.9238945 (all n16).

Source: generate.cpp9453..9465 brackets residual memcpy(...).wait(); paired
path defers draft until both copies complete. verify.cpp3249..3251 brackets
capture lookup;3274..3276 brackets PleTable::gather_batch;3322..3347 brackets
stage then peer+primary executable graph submissions plus ext_oneapi_empty.
Thus verify_submit is host-call wall duration, potentially driver/profiler
work, not GPU service or launch-only overhead. mtp.cpp2468..2482 waits queue,
reads completed tokens/prefetches before pair_return. Stage capture replay
median0.001112ms and stage_inputs_after_PLE0.000391ms are small here.
VERDICT -> The next-window pre-PLE handoff is tens of microseconds; the
multi-ms CPU-labeled intervals are submit/wait/commit/pair stages, with PLE
sub-ms median. None identifies a removable host wait or unknown-gap cause.
One instrumented run, nested windows, no causal latency/idle/overlap conclusion.
Selected records5->6 illustrate ordinal order, not GPU cycle identity.
