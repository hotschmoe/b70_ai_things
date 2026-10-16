# Q6 aligned-load serving discriminator, 2026-10-16

The guarded A4 candidate passes matched C2 serving checks with a modest
positive observation. Aggregate median output rate rises from 102.892755 to 103.332800
tok/s (+0.427673%). Keep default OFF: one fixed OFF->ON sequence and three
nested pairs do not establish a qualified speed or stability gain. The run
started 2026-10-15 and closed 2026-10-16.

CONFIG -> Same main 064b63ce75d7568db541af5cf24eee07f41d0e54ea49813af9958a816d75d5ca
in both fresh arms; STRATA_Q6_A4=0/1 is the sole candidate difference. Whole-pair
graphs, CENTER, IQ2 multi, QSA/value and other candidates OFF. Fixed native and
HTTP policy: suffix0/lookup0/spec4/mtpmax4/minp0; depth2/headpair ON. EP149/363,
all experts resident,2x200k context and32GiB/four parked conversations retained.
The new unique kernel is guarded to canonical C6/A2/RPW1/SG32/PF0/CENTER0,
6144->2560 or2560->10240, even blocks per row and runtime base alignment4.
Original kernel text/fallbacks remain intact; no actual pointer value claimed.

COMMAND -> Actual source claude-opus-5-5(29.275s),claude-fable-5-1(45.866s)
and root full-read; one 220 s CPU-only incremental native-TU/archive/main build.
Actual main host Info/cache/guard/name pointers and argument forwarding are
bound independently. Reviewed computational prefixes match original S102/S85
16384B, CENTER S87 (10688 B) and A4 S104 (15344 B), with metadata/options/constants/
relocations equal. Original/A4 each have 8 raw differing text bytes, all after
EOT; raw image/text identity remains false. CENTER whole image/text is exact.
Observed SIMD32/GRF128/eu8 only; absent barrier/SLM/spill/scratch fields unknown.
Reuse avoids a new decoder and another 114-snapshot private run.

Each arm: one 107-prompt/256-output C2 warm pair, then three fresh 107/1500 C2
pairs and existing arithmetic/thinking/tool functionals. GPU pair leased for
server lifetime; CPU quiet only for measurement windows. Contextual prehealth
reuses the last coordinated S106 API post-health, same boot/runtime and no
intervening coordinated GPU work per root validation. Each arm gets normal
QUIT, full owned CID/name absence and fresh per-card+compiled collective
post-health. No retry, daily restore or standalone cache qualification.

RESULT -> All four pair comparisons pass full text/SSE content, requests,
usage, per-request draft/accepted work, windows/paired/solo, printed rows and
observed capture layouts. Functionals, identity/loader provenance, teardown
and health all pass. ON logs contain exactly one eligible HOST-selection marker when checked
after warm; OFF none; pair-graph marker absent both. This does not count shapes or
independently prove GPU dispatch. Raw native token IDs were not observed.

| Median client metric, three timed pairs | OFF | ON |
| --- | ---: | ---: |
| Aggregate output tok/s | 102.892755 | 103.332800 |
| Stream0 decode tok/s | 53.467855 | 53.610680 |
| Stream1 decode tok/s | 55.853516 | 55.987458 |

Per-pair aggregate changes: +0.427673%,+0.275745%,+0.395947%. These are nested,
fixed-order observations; the aggregate-median ratio is not a paired-median
statistic or independently replicated estimate.

| SSE event-gap summaries, ms | OFF p50 | ON p50 | OFF p95 | ON p95 |
| --- | ---: | ---: | ---: | ---: |
| Stream0 | 0.2083 | 0.2137 | 47.5363 | 47.4363 |
| Stream1 | 0.2066 | 0.2277 | 47.5416 | 47.4198 |

Entries are medians of three per-request percentile summaries, not pooled
quantiles. SSE content events can contain multiple tokens: this is not native
token ITL or GPU service latency. TRACE1 prompt-start logging is common; CPU
ring/profile absent. Rounded native average rows do not expose every row.

VERDICT -> Retain opt-in source/evidence, default OFF. No 150 tok/s claim,
population/stability qualification, standalone cache rerun, or attribution to
traffic/occupancy/idle time. Actual postreviews by `claude-opus-5-5` (41.118 s) and
`claude-fable-5-1` (78.299 s) agree with retaining default OFF and closing this
discriminator. Both prioritize the separate S112 multi-workgroup greedy
selection private oracle, then S111 probability omission as a fallback. These
remain source proposals; neither model establishes a serving gain. Root checked the pinned compiler: the barrier uses workgroup memory scope.
Preparation of an isolated private clone with an explicit leader device acquire
is authorized; compiled and GPU gates remain pending. The live kernel is unchanged. No further build, probe or promotion follows automatically.

[Exact source, build, compiled bindings and bounded runtime evidence](../strata/flash-next/lowlevel_probe/q6_a4_main_serving_s107/README.md).

[Pinned SDK check and next-source decision](../strata/flash-next/lowlevel_probe/q6_a4_main_serving_s107/postreview/root-s112-ordering-followup.json).
