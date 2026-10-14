# Flash-Next prefill2048 natural candidate, 2026-10-12

CONFIG -> Accepted SDK50 original Unsloth UD-Q4_K_XL, balanced24/24 B70s.
Same authentic three P2048 prose/code/document prompts and fresh repeats as
the natural512 baseline; only prefill512 ->2048. Context4096/spec2/no MTP,
mirror staging1/ESIMD0/grouped oneMKL0/cacheOFF/short first chunk0. Six
natural-EOS requests, cap512, no padding/forced output. Candidate is capacity-
supported by the source audit, not predicted to be faster.

COMMAND -> bin/gpu-run producer50219 PASS638s; independent naturalv1
reader82505 PASS. New output root natural-p2048-pref2048-source50-v1-run-v1.

RESULT -> Exactly one observed PP2047 step per request, with2048 actual input
rows including the final verifier token. No stepdown/fallback observed. Actual
resident expert slots17163 versus17967 at512 (8757 primary versus9101), and
reported loaded primary free2453MiB versus2071MiB. All six emitted512 with
length finish. Fresh-state per-prompt repeats match. Pre/post originalfull4,
source/all8, health, kernel window and owned retirement passed.

| Ordinal | Case | PP tok/s | Decode tok/s | Pipe TTFT s | Gap P50/P95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | prose_operations_review | 96.66 | 14.05 | 23.580 | 56.89 / 114.02 |
| 2 | prose_operations_review | 173.30 | 21.34 | 11.864 | 46.18 / 55.08 |
| 3 | code_event_pipeline | 156.33 | 17.93 | 13.144 | 53.05 / 84.23 |
| 4 | code_event_pipeline | 172.17 | 21.99 | 11.939 | 42.63 / 59.62 |
| 5 | document_decision_memo | 154.62 | 14.96 | 13.292 | 61.82 / 106.98 |
| 6 | document_decision_memo | 172.24 | 20.50 | 11.937 | 48.08 / 58.97 |

Repeated warm PP173.30/172.17/172.24 versus the earlier512 baseline
185.17/183.93/183.06: about6.7% more PP elapsed time. This is a separate
ordered run, not an interleaved ABBA. Repeated decode21.34/21.99/20.50;
input-first and process-cold results stay separate in the record.

A separate exact-output comparison found all six output ID sequences DIFFER
from the baseline512 run. No full-head/intermediate comparison was collected
here, so the numerical cause and quality implication are unqualified. Larger
chunks change arithmetic batching and residency; neither is isolated as the
cause. Native repeat coherence alone is not cross-configuration fidelity.

VERDICT -> Do not promote2048. Capacity alone did not improve latency. Keep
prefill512 for the current path; inspect real prefill transfer/kernel timelines
before more chunk tuning. Complete prefix cache/current model fidelity/real
concurrency/critical path/traffic/shelf and2000/50 remain open.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/

- Producer report SHA256
  9bbf16259469953553ef152ecfefcec00e5e96416f5d4f2e5e460233d6a7116a
- Public natural-p2048-pref2048-source50-public-v1.json SHA256
  48016420dc5058a638704c1e7530a8ad0ad5e1dde6a08c495807db34c03cdd89
