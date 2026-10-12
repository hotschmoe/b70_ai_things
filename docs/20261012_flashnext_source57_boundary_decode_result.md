# Final-part decode scheduling result, 2026-10-12

CONFIG -> Fresh57 OFF/ON native2, P2048 A512/C64 after actual A256,
defaultshare0.5, mirror0, unchanged512 chunks/pin/turn/tail boundaries.
COMMAND -> Actual16955 PASS769s; independent47281 PASS.
RESULT -> Bothfull512/64 histories exact acrossarms andaccepted52serial;
63 completed rows2 events/arm, normalexit/EOF/ownedretirement/source/
all8/full4/health/collective/kernel PASS. Longmaxgap11983.327->8515.676ms
(-28.94percent); incomingTTFT31.151->36.624s (+5.473s), completion
40.466->45.083s. LongP50 85.341->89.209ms, P95 144.892->147.438.
Producerb34cd5711cb775fd02935b1b2de5f2506fe73d7467061bb39a9739de51d60fd6; publicad88647b2f5892270216bd6c136652333bb64da45f5dab4740bcafd839f2be61.
VERDICT -> Cadence hole split, remaining chunk-level pauses stillseconds;
tradeoff explicit, no cleanABBA/stability/API/native4 promotion.

| Metric | Default OFF | Final-part decode ON |
| --- | ---: | ---: |
| Long maximum token gap ms | 11983.327 | 8515.676 |
| Long P50/P95 gap ms | 85.341 / 144.892 | 89.209 / 147.438 |
| Incoming TTFT s | 31.151 | 36.624 |
| Incoming completion s | 40.466 | 45.083 |
| Incoming P50/P95 gap ms | 131.175 / 183.328 | 122.389 / 166.568 |

The new branch services live decode after final chunks of logical prompt
parts that formerly skipped it. It reuses the existing half-prefill-time
decode budget. More decode service before incoming admission increases
incoming TTFT; this tradeoff is measured, not hidden behind throughput.
Neither median nor P95 long-gap metrics improved in this ordered pair.
The peak reduction is local evidence, not population stability or a
subsecond latency result. Remaining full-chunk work can still pause decode
for seconds. No physical GPU-idle/bandwidth attribution follows from this
client receipt clock. No direct timing comparison with older source52 is
promoted; the actual57 OFF baseline is the matched control.

Candidate remains defaultOFF and explicitly native2 scoped. Pending stop/
yield commands keep legacy handling. Cancellation/eviction/cache families,
native4/bounded6 and HTTP concurrent serving remain required. The original
2000prefill/50decode targets and complete campaign are not achieved.

Evidence root: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/
f17-source37-20261010/source57-native2-boundary-v1-run-v1.
Public reader: source57-native2-boundary-v1-public-v1.json in F17.
