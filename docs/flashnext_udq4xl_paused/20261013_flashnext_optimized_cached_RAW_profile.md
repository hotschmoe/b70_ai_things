# Optimized cached-window RAW profile

CONFIG -> Actual62 SDK613694ec, HC2/1/1/arithmetic0, pair mirror1/copy0,
PLEdirect/cache65536. Warm freshP2048/GEN8 pin1536 then reused1536/new512
GEN8. One process; CPU API/device recording selected on second request.

COMMAND -> Producer93567 PASS, independent38079 PASS, RAW clock57433 PASS.
Producer2efab8eda92ef1dbd267228b37b30cae65e1caf6835e69083f1b476c3ff7f2ce.
Publicf3390633f5205ecfe45b523153b894e7b1a36260d7b5dd67267b55f3235370a3.
Clockanalysisd90fa944afa029af1a8e93afe95d8a7841e22d96eb3b5c8ffe866b30d175d10a.
Boundunionanalysis2812a2ccc6d99b85af05ec5c37365d1374b533db9bec69489d38617f6a03f2e4.

RESULT -> Exact outputs/work/identity/source/health/normalclose pass. Clock
111293814B and Chrome170118028B stay within1GiB cap. Actual API/device RAW
clock origin and flow endpoint joins hold; query bracket maximum190347ns.
136212 selected device instances and393318 CPU API spans.

| Selected request interval | Seconds |
| --- | ---: |
| Request RAW window | 9.353 |
| Observed any-device union | 3.645 |
| Observed CPU API union | 2.144 |
| API/device overlap | 1.637 |
| Neither recorded category covers interval | 5.201 |

The last row is NOT a GPU-idle measurement or full critical-path allocation.
Native stage handles, internal preparation/sampling, full operation coverage,
continuous RAW/monotonic drift and physical bus bandwidth remain unqualified.
Named HCQ8TokenReuse<4> work sums437/432ms are a kernel lead, not critical
wall time. Queue submit-to-start P95 around16ms is diagnostic with the
recorder active; overlapping waits cannot be added as wall latency.

VERDICT -> Useful qualified diagnostic clock/ownership and observed execution
evidence, not clean speed. Instrumented nativePP8650ms is not a serving
benchmark. A matched untraced same62/cache1536/new512/GEN8 companion is
prepared to measure recorder overhead before kernel or scheduling attribution.
Full fidelity/cache-family/HTTP/concurrency/memory tradeoffs/shelf remain open.
