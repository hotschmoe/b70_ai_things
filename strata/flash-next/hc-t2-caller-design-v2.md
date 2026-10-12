# Stage-local serial paired T2 HC caller

CONFIG -> Preserve frozen V1. V2 changes only its whole-model/next-null guard to
bounded-stage ownership and real handoff/next checks. ExactT2/G1, ordinary owned
in-order queue, device-plan stage mirror, no batch/G2/externalstream/remote or
observer route remain required. Default OFF and arithmetic are unchanged.

COMMAND -> Construct five copied actual52 files. Projection/composition/header
and fixture remain byte-identical to V1; only the verifier eligibility expression
changes. Fresh verifier compile/link and actual paired target checks are required.
No old single-card proof is relabeled as paired qualification.

RESULT -> Each stage has its OWN arena and max_t*K*N parts allocation; two rows
need20480 gate floats versus51200 available atK10. Layer indices remain absolute
[lb,le), and existing GDN/QSA carve/weight routing is untouched. SDK init requires
stage_mirror.lb/le to equal the verifier range (verify515-522). Native run uses
OnDevice(device) and creates own queues in init_capture_streams457-465. Pipeline
streams are supplied only in the pipeline branch(generate7086-7093); stage-mirror
always-publish overlap is rejected(verify713), and V2 keeps ext_stream null.

Each stage's previouspost combine consumes its parts before nextpre on the SAME
orderedqueue; scratch never crosses a stage boundary. The handoff contains only
R/bo/injection, not parts/gate(verify.hpp153). Upstreamrun synchronizes completed
handoff before next_->run(verify2434-2436). Stage0[0,24) requires hand_out/next;
stage1[24,48) requires hand_in and no next. Pending writes, per-row reductions,
original Q8/F32 weights, head, recurrence, PLE and commit forwarding are unchanged.

VERDICT -> No source lifetime blocker was found for this serial paired route.
The guard expansion is source-only; actual operator and paired original-MTP64
outputs/health/teardown remain required. Runtime HC share and speed are unknown.
Scheduler57/externalstream overlap is excluded rather than implicitly admitted.
