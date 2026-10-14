# Teacher-prefill head omission: first paired result

CONFIG -> Exact original34 MTP, paired24/24 target, P2048/GEN512,
context4096, normal reference cuts, fixed6500 profile/cache0, same weight
and buffer reserve. Both arms enable MTP. Only teacher-head-skip0/1 changes.
Decode, recurrent/QSA/indexer/FFN/next-hidden/settle operations stay unchanged.

COMMAND -> Actual19552 PASS. New baseline head0 was individually validated
before head1. Root independently replayed all52 saved artifact hashes, raw
512-token histories, first-proposal head counters, residency, pre/post five
identity rows, normal container terminal/creatorEOF/removal and source closure.
Strict per-card/compiled collective health and kernel checks passed.

RESULT -> Both full512 outputs/EOS and residency match each other and the
closed normal-reference V6 parent. Both229 accepted/53 rejected windows.
At first proposal both teacher_rows2047/cells2048/fresh_head1; baseline
calls2048/skips0, candidate calls1/skips2047. The discarded head work is
actually omitted, while the first consumed head is newly computed.

| Teacher head skip | Prompt elapsed ms | Decode elapsed ms |
| --- | ---: | ---: |
| 0 | 59761.300 | 55235.772 |
| 1 | 37897.617 | 30620.117 |

Prompt elapsed fell36.585percent in this ordered pair; unchanged decode
also fell44.565percent. That negative control demonstrates substantial timing
variation. Do not assign the prompt difference wholly to head omission, or
claim a repeatable speed/TTFT benefit from this pair. These are native CLI
prompt intervals, not HTTP client TTFT. An unchanged-ELF ABBA0/1/1/0 runtime
is prepared to validate timing with all four raw measurements.

Producer external:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/original-mtp53-teacher-head-natural-v2-run-v1/report.json
SHA25681691b40a02dc56b308d2939e464d7b43279e438c0689c4a64fd5b9d66ad277d.
Independent sibling original-mtp53-teacher-head-natural-v2-public-v1.json
SHA256ac0dfa8329a7af2c96be5bf467a5453f99fcf140bafa386db098f8f29472eb5a.

VERDICT -> Actual unused-head omission and first-case full-history fidelity
are accepted. Clean latency benefit, three-role/repeated fidelity, private-slot
runtime/caching, API/concurrency and shelf remain required. Literal forced
controls contain no teacher rows and were not repeated for this isolated
prefill change; no fabricated new forced-state proof is claimed.
