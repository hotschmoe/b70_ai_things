# Deferred identity composition primitive

CONFIG -> Frozen public V3 remains unchanged. The new primitive directly
executes its original _finalized_binding predicates only for an actual
off-equivalence root. It omits the separate public wrapper closing scan and
does not claim current_public_full4_rehashed or standalone qualification.
Actual arm pre/post full4, page, health, source and cleanup gates are untouched.

COMMAND -> A future reviewed caller constructs DeferredOffIdentity, calls
recollect(), completes ALL other prerequisite reads, and calls
mark_prerequisite_reads_finished(actual_source_check). It must then perform
its fresh first-arm pre-full4 and three-page brackets, call
close_with_caller_scan(path,before,after), and require_closed() before launch.
If predicates or the final source callback fail, a fresh closing scan is still
mandatory in caller finally; require_closed retains the original failure.

RESULT -> Seven tiny CPU controls exercise deferred/unqualified scope,
negative scan ordering, failed predicates/source callbacks requiring closure,
foreign PID and wrong publisher hash. Original predicates are mocked in these
controls; no actual model admission, tokenizer, publisher scan or GPU runs.
The primitive is not integrated into the live host actor or any public gate.

VERDICT -> Gate equivalence is conditional on a reviewed caller enforcing the
same-process ordering and actual first-arm scan after all reads. It is not
permission to consume an old PASS or remove a failure-closing scan.

Duplicate upload return availability -> source41 model admission currently
computes fresh41_upload directly and again through masked Baseline. The masked
Baseline return already includes actual_fresh41_upload_binding; the masked
saved report records it under baseline_binding. However masked public return
exposes only report SHA, comparison and scoped component booleans, not that
freshly revalidated upload result. A narrow NEW masked reader can return this
validated envelope, and a NEW actor can compare it with plan.fresh41_upload
instead of separately invoking the same upload gate. Saved report data alone
is not a substitute. Any reused result must remain tied to one PID-owned
complete current byte operation including all upload/oracle/raw source files;
fresh model/page and actual arm gates remain mandatory.
