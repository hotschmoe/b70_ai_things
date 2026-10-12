# Real tail T1 and native serving/reset compile

CONFIG -> New bridge V2 retains frozen V1 and exact private-session V4. Source58
stage_batch accepts S>=1 through max_t; commit_slot_prefixes requires keep>=1
within each contiguous slot group. Current generate batch_mtp always packs TWO
rows, even when remaining1, so that allocator must deliberately choose ONE row
at actual tail/context boundary when integrated. No stock guard is removed here.

COMMAND -> V2 accepts row_count1 or2, sums actual packed widths<=8, permits up to
six selected slots when tails fit. Tail commits one actual row, consumes only
pick0 and never tests unused pick1/guess. T2 retains actual acceptance/EOS/context
checks. Source58 semantic fixture instantiates real V4/bridge APIs and an actual
one-row shape. Root compiles that object against exact actual58 headers.

RESULT -> New loaded reset fixture reuses the original four-session synthetic
fixture: after original interleaved/serial proof, reset USED A/B with new RID/gen,
refuse the actual old owners, preserve their private GPU allocations, and compare
full heads at every new prefix plus BOTH full banks/hidden/counts/proposals against
genuinely NEW serial reference sessions using the SAME shared original pack.
Only two reference sessions are retired/recreated; weights are not reloaded.
All sessions drain before final pack retirement. Readbacks remain diagnostic.

VERDICT -> Native compiler and loaded GPU outcomes remain root-required; source
inspection or textual tests are not semantic/runtime proof. This is not yet the
final cache solution: complete original-state checkpoint export/import plus exact
all-stage target prefix joins for rewind/shared/parked/persisted are still required
before cache-enabled serving. No fake KV-only restore or API promotion occurs.
