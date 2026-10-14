# Paired prefill cooperation candidate V2

CONFIG -> Same source52 parent and full paired scope as candidate V1.
Original sealed V1 source/evidence is retained.

COMMAND -> Apply candidate-paired-prefill-cooperative-core-v1.patch plus
candidate-paired-prefill-cooperative-scheduler-v2.patch, and copy
candidate_cooperative_header_v1.hpp to include/strata/prefill/cooperative.hpp.
Do not additionally apply either standalone cancellation/extent patch.
Use paired-candidate-source-plan-v2.json for exact resulting source hashes.

RESULT -> V2 adds one failure-stop correction: any failed cooperative Prefill
run sets batch_fatal after its original issuer/stager joins and paired drain.
A partially failed batch cannot silently remain eligible for later requests.
All ownership, PLE, CPU-ticket/ACK, no-new-GPU-barrier, budget and default-OFF
behavior described in the V1 candidate document is otherwise retained.
Exact patch application to the admitted source52 files passed. Root's real
CPU ticket test passed independently; this candidate was not compiled or run.

VERDICT -> Full paired source candidate only. Fresh header-dependent compile
and matched original-history, cancellation/failure, ownership, identity/page,
health/kernel and teardown evidence remain required before qualification.
