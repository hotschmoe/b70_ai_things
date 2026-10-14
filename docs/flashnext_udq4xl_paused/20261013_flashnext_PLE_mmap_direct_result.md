# Existing PLE mmap/direct loaded comparison

CONFIG -> Exact source61 SDK, pair24/24, UD-Q4_K_XL, P2048/pref512,
same two loaded A512/C64 cohorts per process. Cooperative scheduling/cap1,
share0.5, copy0/mirror0 and host-only trace enabled in both arms. Sole model
configuration change: --ple-io mmap/direct; rowcache65536/inflight256 fixed.

COMMAND -> Actual32123 GPU-child/CPU-closing-scan experiment; root94846
independent reader. Both model phases exit0 and complete four histories.

RESULT -> Producer e1ba52b4b85cb00ca764e9785c12a3ce9f47dedee97cce52f3cbdc646e71ff8a;
public 9fdf9cc610c7acf2428c49ce6f343135cf4d9af16aa3c9be66de8d7717b37f1f.
Complete histories and fresh repeats match; sixteen owner/stage traces,
source/all8/current pre/post whole identity/health/typed close pass.

| Selected stage0 PLE future-get interval | mmap ms | direct ms |
| --- | ---: | ---: |
| First cohort long request | 1297.674 | 0.005 |
| First cohort incoming request | 1580.601 | 0.004 |
| Repeated long request | 0.004 | 0.003 |
| Repeated incoming request | 0.004 | 0.004 |

Observed first-cohort long TTFT30.853s ->26.039s, incoming31.076s ->26.916s.
Repeated long14.087s ->14.048s, incoming26.909s ->26.775s. These are same-host
request/receive intervals, not HTTP latency or physical GPU execution times.

VERDICT -> The selected Direct futures were ready when collected; this does
not imply zero IO cost. Mmap bypasses the configured bounded row cache;
Direct uses it and batched aligned reads. Initial unarmed model work and
different residency can warm both paths. No cold-cache or cache-hit-count
claim: native serve does not print the existing non-serving IO counters.

This ordered, instrumented comparison is not a clean causal speed result.
The invalid device-free V4 compiler attempt overlapped the mmap arm, adding
unmatched host load. Report clean_performance_qualified=false remains correct.
Use a capture-free natural-workload ABBA before speed promotion. No2000/50,
full model mathematics, cache-family, API or shelf qualification follows.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source61-ple-io-v1-split-run-v1
