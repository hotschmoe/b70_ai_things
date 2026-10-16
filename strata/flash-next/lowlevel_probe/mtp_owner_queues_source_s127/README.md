# S127 source-only owner-queue MTP proposal

CONFIG -> Default-OFF paired MTP scheduling proposal; unbuilt, unapplied,
untested at this checkpoint. No measured overlap or speed result.

COMMAND -> Exact V1/V2 source patches, inverse/apply-check metadata, frozen
source/caller excerpts and bounded static audit. See source-copy-index.json
for every absolute source path and SHA. Full external source TUs are omitted.

RESULT -> V2 adds ON-only readiness/error handling without changing device
arguments/arithmetic. Existing HC maps can lazily allocate850KiB per queue;
idle waits only cs_, and ordinary DPCT wait errors return false. Actual source
reviews responded as claude-opus-5-5/claude-fable-5-1,56.541902/92.955938s;
filtered records preserve identities/session/exit/packet/response hashes,
without cumulative usage. No new consultations were run during curation.

VERDICT -> Conditional source design only; later apply/build/runtime decisions
belong to root. No completed KV byte proof, noise band, causal overlap or
forecast. See the [dated report](../../../../docs/20261016_strata_mtp_owner_queues_source.md).
