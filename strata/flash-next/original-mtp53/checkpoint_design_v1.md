# Complete original private-draft checkpoint

CONFIG -> Source-only successor V5 sessions, unchanged original arithmetic and
private allocations. New QSA header adds only validated quiescent host count
restore. Source/owner/original tensor/target prefix contract is explicit.

COMMAND -> Export only exact CURRENT accepted prefix (1..4096 cells) after no
active transaction and queue drain. Payload includes BOTH complete8915972-byte
banks, including future cells, indexer tail/dead/pooled/blockpos; committed hidden
40960 bytes; full current own-head logits993280 bytes; count, accepted IDs,
sourceRID/gen, locked publisher SHA, original draft/math SHA and full target
model/source/config/state-contract SHA. No private pointers are serialized.
Serialize bounded20MiB fixed LE schema; reuse native host SHA256. Deserialize
requires an independently pinned payload digest, exact lengths/no trailing data,
finite head and valid prefix/count. It does not authenticate arbitrary caller data.

RESULT -> Import first refuses wrong independently pinned hash/prefix/draft/target
identity before any writes. Caller MUST separately restore/byte-verify BOTH target
stage checkpoints at that exact prefix/current contract. Then drain, restore both
full banks/hidden/logits, restore committed count, rebuild no-pending step, and
read back every byte. Immediate proposal uses source-owned restored current
logits, never stale scratch. Native failure latches; owner retains pending work.
Older prefixes require saved full payload, not truncating a later bank's count.

VERDICT -> Actual cache manager/branch/parked/persisted wiring still needed; this
is complete export/import data code rather than legacy KV substitution. Target
contract must bind model full4 identity, actual SDK/quant/math/profile/config and
both stage ranges; caller cache metadata must pin exported digest and accepted
prefix. No unverified serialized payload or current-target restoration is implied
by source declarations. Added CPU executable fixture tests full canonical bytes,
corruption/truncation/extra/forged-rehash/count/prefix/nonfinite controls; semantic
fixture instantiates native export/import; loaded fixture tests branch rewind and
shared-prefix copy across private owners plus prefix/hash negatives. Root compiles
and runs later. No math, serving, concurrency, speed or runtime qualification yet.
