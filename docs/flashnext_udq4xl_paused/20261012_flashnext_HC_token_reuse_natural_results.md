# HC Q8 token reuse matched model result, 2026-10-12

CONFIG -> FreshSDK54 originalUnsloth UD-Q4_K_XL, pair24/24, P2048/
prefill512/context4096, three authentic prose/code/document prompts each
fresh-state repeat, GEN512/naturalEOS. Mirror staging1/copyengine0/noMTP/
grouped0/ESIMD0/cacheOFF/short first chunk0. Sole delta HC_Q8_TOKEN_TILE0/2.
Original Q8 weights, peroutput FMA/reduction, old8-token caller unchanged.

COMMAND -> Producer54311 PASS988s, independent reader4483 PASS. This follows
standalone bitwise/replay/operator timing and fresh74-source/all8 ABI build.

RESULT -> All twelve output IDs/finish match acrossarms and perrole repeats.
All512/caplength; no completed task claim. Fulloriginal4/source/all8/kernel/
health/ownedEOF/retirement pass. Repeated warm rows:

| Case | PP tok/s scalar -> tile2 | PP elapsed reduction % | Pipe TTFT s scalar -> tile2 | Decode tok/s scalar -> tile2 |
| --- | ---: | ---: | ---: | ---: |
| prose_operations_review | 184.99 -> 205.56 | 10.01 | 11.117 -> 10.009 | 21.69 -> 21.69 |
| code_event_pipeline | 183.50 -> 203.95 | 10.03 | 11.207 -> 10.089 | 22.40 -> 22.36 |
| document_decision_memo | 182.70 -> 203.58 | 10.26 | 11.257 -> 10.107 | 20.54 -> 20.53 |

Mean repeated-row PP elapsed reduction is 10.10%. Cold-process and first
new-prompt-shape rows remain separate in publicdata. No instrumentation here;
pipe receive timestamps are not HTTP/API TTFT. Ordered two-process A/B is
not interleaved ABBA or population stability. Decode is essentially unchanged.

VERDICT -> Useful arithmetic-preserving model prefill benefit. Still far below
2000input/50output targets; no fullmath/concurrency/cachefamily/shelf promotion.
Native8-row read/write chains still dominate dispatch count. Separate source55
coalescing candidate may reduce them, but wholechain bitwise/operator/model
validation is required. Copy-engine5% gain was a separate experiment and is
not added to this result without a combined matched measurement.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/

- source54-hc-token-natural-v1-run-v1/report.json SHA256
  93fe2f39be192cbefa84c9271ce38c3e09f93a2ef32a169b9d521e24a65aa93c
- source54-hc-token-natural-v1-public-v1.json SHA256
  f5c35f7a12b3a0317c6af24a7f10e18438a70fdd2e1773dca4bdc78580033221
