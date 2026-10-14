# Capture-free current serial cache latency, 2026-10-12

CONFIG -> Exact SDK52/main Unsloth UD-Q4_K_XL, pair24/24, prefill512/context4096,
noMTP/mirror0/copy0, matched1536 pin, authentic A2048/B2023 sequenceA,A,B,B,A,B.
Two owned processes: every cold request fresh1, cache arm fresh0 starts empty.
GEN64/naturalEOS/cap64. SFD/fullhead/PCL/PREFIX capture flags OFF, no profiler;
normal native cache notices remain. Same inputs/settings as accepted diagnostic.

COMMAND -> bin/gpu-run producer50629 PASS762s; independent reader29925 PASS.
Genuine root command construction56150 passed before lease.

RESULT -> All twelve emit64/length. OutputIDs/finish repeat by role and match
cold/hit exactly; actual native RESUME/REUSED/DONE counters show0first then
1536x5, with512(A)/487(B) new rows. Originalfull4/sourceall74/all8/health/
kernel/normalownedretirement pass. No firsthead or stagebody capture here;
those remain authority only for the separate prior diagnostic, not this run.

| Ordinal | Reused | New rows | Fresh pipe TTFT s | Cache pipe TTFT s | Cache completion s | Hit gap P50/P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 2048 | 34.818 | 26.002 | 28.843 | 44.35 / 52.06 |
| 2 | 1536 | 512 | 14.021 | 4.750 | 7.540 | 43.80 / 51.24 |
| 3 | 1536 | 487 | 14.726 | 4.592 | 7.495 | 44.58 / 54.16 |
| 4 | 1536 | 487 | 13.860 | 4.576 | 7.387 | 43.40 / 52.27 |
| 5 | 1536 | 512 | 14.016 | 4.753 | 7.548 | 44.08 / 51.36 |
| 6 | 1536 | 487 | 13.858 | 4.580 | 7.398 | 43.72 / 52.11 |

Repeated warm fresh TTFT13.858-14.021s versus corresponding hits4.576-4.753s:
about66-67% lower. First process-cold and first new-shape B remain separate,
not pooled into a headline distribution. Cache completion7.39-7.55s for64
capped tokens. Newly evaluated512/487 rows are the throughput denominator;
reused1536 tokens must not inflate a2000tok/s prefill claim.

VERDICT -> Current native capture-free serial cache latency benefit is observed
with matched outputs/work counts. This is one ordered two-process comparison,
not interleaved ABBA, clean HTTP/API, naturally completed task, fullcachefamily,
originalfullmath, real1/2/4/6 simultaneous streams or verified shelf. Cache
identity/restart/eviction/cancel/parking and broader histories remain required.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/

- Producer source52-clean-serial-cache-latency-v1-run-v1/report.json SHA256
  f708548c1d92e78aab43be99b5295c2b13a052638b3ae4eb5851da5877c5ea6e
- Public source52-clean-serial-cache-latency-v1-public-v1.json SHA256
  a593c5ad713debcd04f37feed564abf1efbbd21b212655e1785acd33bd3ac7b7
