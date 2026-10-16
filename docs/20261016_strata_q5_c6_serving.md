# S139: Q5 full-head C6 runtime-word serving discriminator

CONFIG -> Two fresh processes on the SAME new S138 executable
`6b37aef71607a4217192165523f413063c03d5a3cf042c6dd1031800ca2a6eb2`.
Only STRATA_Q5_HEAD_RUNTIME_ONES_C6 differs:0 versus01010101. C2 runtime word,
owner queues, parallel argmax, whole-pair graph, probability skip, Q6/A4/center,
QSA/value/IQ2 variants remain OFF. Fixed suffix0/lookup0/spec4/MTPmax4/min-p0,
batch2/depth2/paired head ON, EP149/363 with all24576 experts resident,
two200k context allocations and32GiB conversation parking remain common.
TRACE1 diagnostic logging is common; CPU boundary ring/profiler are absent.
The S138 narrow actual host-marker check passed before this serving stage.

COMMAND -> One107-prompt/256-output C2 warm pair per arm, then three fresh
107-prompt/1500-output C2 pairs and bounded arithmetic/thinking/tool functionals.
OFF completed first, then ON; no old control or S134 result was reused. Every
ON pair immediately matched its corresponding fresh control before proceeding.
Timed requests used CPU-quiet conditions inside the existing whole-server GPU
lease. Root reused closed S136 both-card/compiled post-health after explicit
no-intervening-GPU validation; full post-health was retained for each arm.
No retry, reset, retime, extra baseline, full-cache campaign or daily restore.

RESULT -> Comparator exited0, PASS_MATCHED_DISCRIMINATOR. Warm and all three
timed pairs matched request, complete text/SSE content, usage, output cap,
cache/prompt/output counts, offered/accepted drafts, windows/pair/solo counts,
reported rows and logged captures. Request admission overlapped. Raw token IDs
were not observed, and rounded native row reports are not unprinted exact
per-window row sequences. All functionals, normal QUIT, wrapper exits0,
prestop Running/non-OOM, full CID+NAME removal, per-card and compiled collective
post-health passed for both arms. Immediate control-parity receipts are retained.

| Timed client metric | OFF | C6 ON |
| --- | ---: | ---: |
| Aggregate output tok/s, median | 103.072328 | 104.393157 |
| Pair wall seconds, median | 29.105775 | 28.737516 |
| Stream0 decode tok/s, median | 53.474006 | 54.344196 |
| Stream1 decode tok/s, median | 55.862589 | 56.688704 |
| Stream0 TTFT seconds, median | 0.919054 | 0.925228 |
| Stream1 TTFT seconds, median | 2.113088 | 2.096618 |

Change of aggregate medians is+1.281458379%. Individual repeat throughput
changes are+1.281458379%,+1.602089334%,+1.400550244%; their median is+1.400550244%.
Their pair-wall changes are-1.265244794%,-1.576827154%,-1.381205764%.
All samples are retained. Client decode/wall and SSE content-event gaps are
separate observables; SSE events can contain multiple tokens, so event-gap
p50/p95/max in comparison.json are not native token ITL or device service.

The ON C6 host-selection marker appeared exactly once by warm and at final
closure; OFF had none and old C2 marker was absent in both. Captures matched,
including the six-row batch layout. This proves observed eligible HOST selection
with successful work, not every shape/call or independent GPU dispatch counts.
Actual argv/environment, loaded-library identities and map hashes are retained
as explicit projections; raw maps and SSE remain external.

VERDICT -> Finite matched discriminator passed with descriptively positive
changes in this one OFF-then-ON sequence. Each arm is a distinct process and
three repeats are nested in that process, not independent population replication.
Keep the source default OFF; no promotion, stability/noise band,150 tok/s target,
causal integer-instruction attribution or broad model-context claim follows.
S132's private C6 fixture had a much larger measured change in its separate
build/context; it is descriptive prior evidence, not the main serving result.
S134 remains its failed marker/null comparison. S138 fixes the demonstrated
host marker binding, without retroactive qualification or device-ISA equality.
Actual post-result model decisions were pending at this curation cutoff.

The compact payload retains exact controllers/configs, all eight warm/timed
pair records and work logs, immediate ON parity, functional verdicts, normal
closure/removal/health and source/SHA index. Larger response/map records remain
external; no binaries/model weights were read or rehashed for this curation.

[Compact S139 evidence](../strata/flash-next/lowlevel_probe/q5_c6_serving_s139/README.md).
