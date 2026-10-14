# Source40 one-card versus pair prefix4 targets

CONFIG -> Exact original UD-Q4_K_XL and unchanged source40 SDK. One-card
QSA9 run uses all48 layers on card0; pair QSA9 uses32/16 across cards0/1.
Four identical fresh prefix4 requests per OFF and ON arm, MAX1, no PIN,
cacheOFF. Pair parent final qualification remains pending.

COMMAND -> Root independently consumed both topologies' completed OFF/ON
SFD target files. Key is request/kind/layer/position; physical stage must
match the declared layer split. Indices reset per request, with layers1-48
and head49. Each file was checked regular/non-symlink, bounded exact size,
and stat/read/reread bracketed. No GPU operation or reference mutation.

RESULT -> All196 complete layer/head vector pairs match BITWISE across
one/pair in OFF, and all196 match in ON. The four complete vocabulary
vectors per arm are included. Standalone evidence is
f17/qsa40-one-pair-prefix4-captured-target-comparison-v1.json, SHA256
98334e68ba64c5a54e288ea520befdfdf81d89968e94b7e8f67025ea962c3ed2.
It binds784 files; these captures are comparison targets only.
Initial exploratory indexing assumed a cross-request counter and refused;
actual file names established the per-request index used above.

VERDICT -> Concrete short-prefix cross-topology target parity. This does
not prove original-model fidelity, long-prefix equivalence, cache/concurrency
correctness or full paired lifecycle qualification. Both hardware profiles
still reproduce the original indexer half-storage discrepancy. Long524-token
speed-screen outputs differ across profiles whose context/prefill/PLE-cache
settings differ; the prepared matched-settings controller is the next
controlled comparison after active registry171 consumers finish.
