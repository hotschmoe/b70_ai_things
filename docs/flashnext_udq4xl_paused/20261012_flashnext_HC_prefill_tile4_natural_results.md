# Prefill-only HC tile4 matched model result, 2026-10-12

CONFIG -> Original UD-Q4_K_XL current56 pair24/24, P2048/pref512/
ctx4096, three natural-EOS GEN512 prompts and fresh-state repeats.
Coalesce1/globalHCtile2/mirror1/copy0/noMTP/cacheOFF/grouped0/ESIMD0
fixed. Sole PREFILL_TILE4 absent0/1 selects existing2/4 kernels for
explicit coalesced prompt reads; decode remains tile2.

COMMAND -> T512 direction benchmark and completeHC chain/replay passed.
Verified same-stack incremental build68s, one composition body compiled,
123 unchanged objects reused, alleight ELF targets freshly linked.
Producer76704 PASS1095s; independent53683 PASS.

RESULT -> All12 full512 output IDs and finish match acrossarms and
fresh-state role repeats, including actualT511 tail. All ended at length
cap; no completed-task claim. Source/all8/full4/percard/collective/kernel/
normalEOF/ownedretirement pass. Warm-repeat rows:

| Case | PP tok/s tile2 -> tile4 | PP elapsed reduction % | Pipe TTFT s tile2 -> tile4 | Decode tok/s tile2 -> tile4 |
| --- | ---: | ---: | ---: | ---: |
| prose_operations_review | 243.43 -> 255.78 | 4.83 | 8.459 -> 8.053 | 21.67 -> 21.67 |
| code_event_pipeline | 239.14 -> 253.06 | 5.50 | 8.611 -> 8.136 | 22.36 -> 22.37 |
| document_decision_memo | 240.68 -> 252.23 | 4.58 | 8.556 -> 8.162 | 20.52 -> 20.54 |

Mean warm PP elapsed reduction 4.97%. Decode is essentially unchanged.
Cold and first-new-shape rows are retained separately; their large
variation is not hidden. Ordered A/B is not ABBA/population stability.
Pipe timings are not HTTP TTFT. The incremental build is explicitly
not an all74 fresh-compilation claim; parent/dependency/config/object
checks and fresh target links close its provenance.

VERDICT -> Accepted original-arithmetic model prefill gain, current
warm252-256tok/s. Still below2000/50, no fullnumerical/concurrentservice/
cachefamily/shelf promotion. Do not add separate copy-engine gains; a
combined configuration requires its own matched output/lifecycle timing.

F17 source56-hc-prefill-tile4-natural-v1-run-v1/report.json SHA33b5bc24fbd30b2ebcb37cb12f5bf40fbde0c0d2f085ea1b6f4145ebf7f1deb1.
Independent public SHA4282bfdacff740a586cd2556333975d3a9409e6a456af51e1ae73986b0dde7f2.
