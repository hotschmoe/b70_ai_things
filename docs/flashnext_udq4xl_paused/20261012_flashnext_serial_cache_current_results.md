# Current Flash-Next serial cache diagnostic, 2026-10-12

CONFIG -> Actual SDK52/original Unsloth UD-Q4_K_XL, pair24/24, prefill512,
context4096, no MTP, mirror staging0/copy-engine0, three prompt-cache slots.
Two owned processes run the same A,A,B,B,A,B sequence, GEN64/natural EOS.
Authentic A2048/B2023 share1536 tokens; both arms explicitly checkpoint/pin
that boundary. Cold arm uses fresh1; reuse arm fresh0 starts empty. Matching
positive pin keeps the cold/reuse arithmetic chunk boundary aligned.
Full first-token logits and PCL body/work diagnostics are enabled in both.

COMMAND -> bin/gpu-run producer29871 PASS720s; independent current-reader
89341 PASS. Authentic tokenizer-only fixtureV2 SHA27685dc7 admitted; new V3
adapter uses the actual SDK52 namespace, preserving earlier CPU failures.

RESULT -> All twelve requests emit64 and finish length. All248320 FP32 first
logits per request are finite and the emitted greedy token attains their
maximum. Six corresponding cold/hit full-head byte comparisons and all
output ID/finish comparisons are exact; within-role repeats are exact too.
The empty cache's first request reuses0. Every subsequent hit reuses1536,
evaluates exactly512(A) or487(B) new prompt rows, with exact completed work
coverage on both stage ranges. Pre/post originalfull4/sourceall74/all8,
health/kernel window and normal owned EOF/retirement pass.

| Ordinal | Reused tokens | New prompt rows | Fresh pipe TTFT s | Cache pipe TTFT s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 2048 | 25.802 | 25.768 |
| 2 | 1536 | 512 | 13.940 | 4.738 |
| 3 | 1536 | 487 | 14.120 | 4.582 |
| 4 | 1536 | 487 | 13.776 | 4.581 |
| 5 | 1536 | 512 | 13.963 | 4.746 |
| 6 | 1536 | 487 | 13.771 | 4.579 |

VERDICT -> Bounded current-model serial checkpoint reuse with identical full
heads/output IDs is accepted. Five hit TTFTs4.58-4.75s versus matched warm
fresh13.77-14.12s are about66% lower, under the SAME instrumentation. These
are instrumented native-pipe diagnostics, not clean HTTP/API latency or a
2000tok/s prefill claim; reused tokens are not newly evaluated throughput.
The first request remains process-cold and does not establish a cache hit.

No full cache-family qualification: RAM parking, persistence, arbitrary
history/depth, eviction, cancellation, independent sessions, stale identity,
restart and real simultaneous streams remain open. Full original-model
numerical fidelity, physical critical-path/tier bandwidth, clean natural
completion and verified shelf remain open. A useful clean cache latency run
and actual service integration follow this restored-state proof.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/

- Producer source52-serial-cache-latency-v3-run-v1/report.json SHA256
  fb01b62f3e45f36cf214deb97ba7cf21e935a8a6002ca6f2a1ac6d027e244825
- Public source52-serial-cache-latency-v3-public-v1.json SHA256
  ba7c99be664d8f480eb6148cdc325f9d5e8ad3b54d25b353736cfdc49da4092a
- Reader source plan SHA256
  84287781abfa9039ef747e23873043a907211bb94bd01bbd0e4b4ba73e1d5d37
