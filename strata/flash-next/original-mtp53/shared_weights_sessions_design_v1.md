# Original34 shared weightpack and private slot sessions

CONFIG -> New source-only callable ownership split from frozen paired adapter V3.
Default OFF via the unchanged STRATA_ORIGINAL_MTP53 gate. GPU1/context4096 only.
No production serving hook, target guard change or original arithmetic change.

COMMAND -> OriginalDraftWeights53(path, fresh_identity, last_queue, 1) loads the
original 34 exact typed views once, uploads unchanged tensors and losslessly
concatenates experts once. It owns a copied queue implementation and the read-only
GGUF mapping until its USM owner has drained. The caller must freshly verify the
locked whole-file SHA; the identity argument and repeated stat/path checks do not
compute or replace that proof. A canonical nonalias path is mandatory.

RESULT -> OriginalDraftSession53(shared_pack, {rid,generation}, last_queue, 1)
allocates the unchanged private scratch, two complete compress4 QSA/indexer banks,
committed hidden, incoming residual, token and persistent host staging. Sessions
share immutable typed weights only. One ordered pack queue is retained; no claim
of independent overlapping queues or thread-safe simultaneous calls per slot.
The caller serializes each slot and must complete authoritative paired target
commit before advance_actual copies the actual LAST residual into private staging.
propose_actual uses only the resulting current own-head logits. Stale RID or
generation is refused. Operational exceptions latch failure; retirement uses the
existing fatal-drain ownership route, retaining pending resources until process
exit86. Private graphs/buffers retire before the last shared weight reference.

VERDICT -> No 2.775GB upload per session. Shared static tensor bytes remain
2775258112; each capacity4096 bank is 8915972 bytes, two are 17831944 plus the
unchanged private scratch/hidden/staging. All intended sessions must be reserved
before target cache sizing. No runtime concurrency, numerical, speed or serving
qualification. No reset/rewind/cancel handoff/persisted import is supplied: legacy
KV-only serialization cannot represent full original recurrence. Such integration
requires both complete banks, indexer state, committed hidden/counts and matching
target prefix, with source-owned logits recomputed or preserved truthfully.
