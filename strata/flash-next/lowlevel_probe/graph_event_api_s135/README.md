# S135 graph/event API-only evidence

CONFIG -> One standalone card0 oracle, two same-context/device in-order queues,
control host joins versus explicit graph-event barrier dependencies.

COMMAND -> Exact source/build recipe in source/; V2 owned-container lifecycle
and analyzer in runner-v2/. V1 lineage hashes are retained, without presenting
its unexecuted cleanup as runtime evidence. The source-copy index records
absolute origins and exact-copy hashes. Final-inspect is an explicitly derived
projection. external-output-index.json retains completed analyzer hashes and
external paths for all six full outputs; no raw output, binary or compiler
image is copied or newly rehashed by this curation.

RESULT -> runtime/comparison.json and runtime/closure.json report the closed
API-only PASS, two fresh processes, exact A/B/A intermediates, actual health
and normal teardown. Mode-specific logs, DOTs and result records remain
separate. Source counts are not Level Zero measurements; there is no timing
benchmark. The build wrapper's21s and compiler/link9.273s have different scopes.

VERDICT -> Same-device dependency API precondition passed for this fixture.
No MTP/model/overlap/speed claim or automatic integration. Root owns the later
post-result disposition, source integration and GPU scheduling.
