# Whole HC prefill coalescing matched model result, 2026-10-12

CONFIG -> Original Unsloth UD-Q4_K_XL fresh55 pair24/24, P2048/
pref512/ctx4096; three prose/code/document prompts, fresh-state repeats,
GEN512/natural EOS. HC tile2 fixed, mirror1/copy0/grouped0/ESIMD0/
noMTP/cacheOFF/short first chunk0. Sole HC_PREFILL_COALESCE0/1.
Original Q8/F32 weights, arithmetic and downstream precision unchanged.

COMMAND -> Fresh74/all8 NoDri build62469 PASS309s; whole-chain synthetic
bitwise/changed-input replay pass; producer76995 PASS1004s; independent
reader62892 PASS. Full4/source/kernel/percard/compiledcollective health
and normal owned process/container retirement passed.

RESULT -> All12 complete512 outputs and finish match across arms and
fresh-state role repeats. Every request ended at length cap; no completed
task claim. Repeated warm cases:

| Case | PP tok/s off -> on | PP elapsed reduction % | Pipe TTFT s off -> on | Decode tok/s off -> on |
| --- | ---: | ---: | ---: | ---: |
| prose_operations_review | 208.14 -> 241.47 | 13.80 | 9.885 -> 8.527 | 21.66 -> 21.64 |
| code_event_pipeline | 205.71 -> 238.11 | 13.61 | 10.003 -> 8.648 | 22.35 -> 22.35 |
| document_decision_memo | 205.46 -> 237.52 | 13.50 | 10.014 -> 8.669 | 20.52 -> 20.49 |

Mean repeated-row PP elapsed reduction: 13.64%. First/cold rows are
retained separately in public evidence. Decode is essentially unchanged.
Native prompt time and pipe receive TTFT differ; these are not HTTP timings.
Ordered A/B is not ABBA or repeated-launch population stability.

VERDICT -> Accepted model-level arithmetic-preserving prefill gain.
This comparison already holds prior HC token reuse on in both arms; do
not add independently measured copy-engine gains. The next profile should
measure this current configuration before choosing another kernel change.
No 2000/50 result, full original numerical gate, concurrent service,
cache-family qualification or serving shelf promotion is implied.

Evidence root: F17-source37-20261010 under the campaign results directory.
Producer report SHA232c5dc2fb07f4508cb547e56cc24b9e3069fc119cde6ce9a695273c3c54a81a.
Independent public SHAb5e9361d8e4483f75bd1fae3f2e1442c744c5d7f009fe475c9ad0852fd3a0530.
