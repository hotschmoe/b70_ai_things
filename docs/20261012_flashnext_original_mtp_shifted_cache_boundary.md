# Original MTP cache: shifted prefix and first ownership

CONFIG -> Source audit actual SDK58 generate.cpp and frozen original pair V3.
Existing checkpoint/private fixtures are synthetic roundtrip/reset/continuation
checks. They do not authenticate a real MAIN prefix/cache publication.

COMMAND -> Inspect pair adapter prefill_chunk next=prompt[pos+t+1], final prompt
advance(generated0,MAIN final_R_all), and target batch IDs push(tok) before BDONE.
Actual58 slot_generations initializes0 and first admission increments to1.

RESULT -> At TARGET input prefix lengthP, original cellsP consumed NEXT tokens
[target_ids[1],...,target_ids[P-1],lookahead]. cells==ids.size alone cannot qualify
same-prefix/divergent-suffix restore. Terminal sl.x/sl.p can remain stale because
BDONE breaks before updating them; retain last actual committed pick separately.

Required shifted envelope: complete currentP original state + explicit lookahead;
complete pre-last P-1 original state; authoritative LAST hidden from TARGET input
P-1; target input IDs, source/math/model contracts and independent full digest.
Matching lookahead may reuse currentP. Different next token must restore pre-last
and recompute ONE unchanged original front/QSA/FFN/head/settle with new NEXT token
and saved actual hidden. Compression boundaries require full bank/indexer rollback,
not decrementing cells. P=1 requires genuine EMPTY pre-last state/head absent.

Reserve a private ordered-device rolling backup and saved hidden BEFORE both
expert cache budgets. Copy device state on existing ordered queue before each
actual catchup, with existing drains; export to host only at cache boundaries.
Extra workspace/traffic must be charged and profiled, never called free.

First native BSlot generation1 must bind as generation1. Prototype bootstrap gen1
is not a client request. New explicit UNBOUND owner0/0 permits first realRID/gen1;
release/reset/reuse still refuse stale actual owner tuples. Admission workspace
may explicitly release to UNBOUND after transferring its complete state, so it
never spoofs a new slot's independently assigned generation.

VERDICT -> Joint505 equality of TARGET IDs to own consumed NEXT IDs is HELD for
real serving. Its synthetic ownership/digest fixtures remain in their declared
scope. Implement shifted successor before real compound publication/restore and
serve-loop activation. Original target/draft arithmetic and publisher weights stay
unchanged; no current serving/cache/API/math qualification is claimed here.
