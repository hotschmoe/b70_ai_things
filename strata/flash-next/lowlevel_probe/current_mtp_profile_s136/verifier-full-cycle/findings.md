# S136: verifier ranking after full saved-cycle reconciliation

CONFIG -> One all-OFF instrumented 64-token C2 request pair, FIRST retained
source-compatible primary-device cycle3, 5151 bounded events, no truncation.
The original whole analyzer exited -2 after saved artifacts; global coverage was
not completed. This note does not turn that attempt into an analyzer PASS.

COMMAND -> `python3 s136-verifier-bounded-ranking/derive.py` reads ONLY the
5,255,606-byte saved cycles.json, asserts its existing SHA, and sweeps sorted
interval endpoints. Actual terminal0, 0.093 seconds. No raw trace, CPU/peer clock
join, model/ELF reads, build, GPU, decoder or model consultation. Source reads
are the unchanged verifier and existing S125/S44 small evidence.

RESULT -> The missing budget is real in the retained timeline, but it is not
all hidden MTP arithmetic. Do not select another kernel by treating j1 as the
whole MTP path or by treating absent trace events as physical idle.

| Same-clock interval | Span ms | Recorded union ms | Uncovered ms |
| --- | ---: | ---: | ---: |
| Previous C6 end -> first C2 start (LEFT) | 11.600500 | 4.880681 | 6.719819 |
| First C2 vocabulary head | 1.209583 | 1.209583 | 0 |
| First head end -> both selects complete | 0.289479 | 0.276038 | 0.013441 |
| Both selects -> second C2 start (j1 compatible) | 1.311688 | 1.095440 | 0.216248 |
| Second C2 vocabulary head | 1.207708 | 1.207708 | 0 |
| Second head end -> both selects complete | 0.289479 | 0.274789 | 0.014690 |
| Both selects -> following C6 start (verifier compatible) | 38.569313 | 34.849607 | 3.719706 |
| Following C6 vocabulary head -> saved cycle end | 2.255500 | 2.255416 | 0.000084 |
| TOTAL | 56.733250 | 46.049262 | 10.683988 |

Offsets use saved primary-device timestamps relative to cycle start. Adding
short durations after subtraction avoids rounding at epoch-sized floats. The
0.062/0.063 us differences from earlier 1.311750/38.569250 arithmetic, and the
0.084 us final suffix, are timestamp precision, not extra operations.

LEFT includes 2338 `zeCommandListAppendEventReset` intervals with 2.135442 ms
reset-only coverage, plus previous-window commit kernels and the depth0 fronts.
The largest absent-record spans are 1.731980 ms (+5.385520 -> +7.117500) and
0.702063 ms (+9.227937 -> +9.930000). Tooling origin/perturbation of resets is
UNKNOWN. The forensic agent owns the source-conditioned LEFT call mapping;
see sibling `s136-headfree-bounded-attribution`. It is incorrect to allocate
LEFT wholesale to MTP forwards.

The verifier-compatible region begins at second selects141201/141202 ending
cycle+15.908437 ms. Its FIRST retained event is ep_bump141540 at +19.225500 ms:
3.317063 ms with no retained primary event. Beyond that first gap, total absent
coverage is another 0.402643 ms. This is a stronger readiness question than
blindly calling the entire 38.569 ms arithmetic. No CPU cause, peer wait cause,
owner identity, GPU stall or hardware-idle conclusion follows from this clock.

## Actual verifier operation ranking

These are sums of measured operation durations inside ONE saved interval,
not medians, disjoint critical-path costs or attainable speedups. Different
families overlap and roles remain source-conditioned.

| Family | Calls | Service ms | Coverage with only this family active ms |
| --- | ---: | ---: | ---: |
| HC norm/down/reduce/up | 388 | 6.159098 | 6.158148 |
| Routed GU | 48 | 4.411336 | 4.411067 |
| Q6 dense | 71 | 3.818719 | 3.767660 |
| QSA | 480 | 3.448338 | 3.447656 |
| Routed down | 48 | 3.158937 | 3.158897 |
| EP wait | 48 | 3.082584 | 1.160050 |
| Q4 dense | 45 | 2.313627 | 2.066950 |
| GDN | 180 | 1.505239 | 1.505117 |

Other MMVQ service5.780543 ms and other operations3.096897 ms are retained in
facts.json. Across all families, multi-family coverage1.923833 ms is explicitly
kept unallocated. In particular EP wait largely overlaps useful work; removing
its 3.083 ms duration is not an available 3.083 ms saving.

The largest Q6 shape rows are native_q6k_wide_a2 C6 grid640 (29 calls,
1.752695 ms) and grid2560 (14 calls,1.154160 ms), together2.906855 ms. Native
RPW4 grid maps these to M2560 and M10240. Existing S41/S44 tensor headers map
N6144/M2560 to attention/GDN output-compatible projections and N2560/M10240 to
qkv-compatible projections. Names/grid do NOT prove each event's tensor.
Other Q6 shapes account for0.911864 ms; do not silently widen the candidate.

## Source seam and one candidate disposition

Pinned verify.cpp e97ce93a... (full SHA/pins below) shows actual dependencies:
HC read produces `mixed_` and, when fused quantization is selected, `xq_`;
GDN qkv consumes xq_ and writes qkv_L_ (L1013-1030). GDN recurrence/norm
produces y_/optionally xq_; output projection consumes it and writes bo_
(L1064-1070). QSA attention gate produces attn32_, quantization writes xq_,
output projection writes bo_ (L1278-1283). HC half1 reads bo_ with apply=true
and produces mixed_/optionally nat_xq_ (L1289). Routed GU/down consumes nat_xq_,
uses hit_scratch_, writes parts_; EP return precedes combined consumption
(L1474-1498). Existing quant-producing fusion means there is not always a
separate activation-quantization launch to remove.

ONE distinct kernel mechanism remains the parked S125 exact Q6 pre-expansion
plus multi-output signed-int8 DPAS. It is broader than the closed Q6 A4/load
shuffle, packed-Q6, GRF/SG tuning and single-SG serial-K DPAS. It preserves each
native-lane pair of dot16 results and FP parenthesization, uses parallel native
K partitions, then reproduces the original XOR tree in a second kernel.
However it adds offline-format ownership, two-launch dependency, INT32 tile
SLM/barrier, global32-partial scratch and extra retained weight memory. It is
not a small unproven direct `K32 dot` replacement.

For ONE real S44 `blk.3.attn_output.weight` N6144/M2560/type14, original bytes
12,902,400; pre-expansion adds16,834,560 bytes; native-lane scratch1,966,080
bytes. Full29-output deployment alone adds488,202,240 bytes, before scratch;
current serving headroom is UNKNOWN. Original bytes must remain for existing
prefill/fallback. N2560/M10240 uses7,864,320 scratch bytes. This materially
raises integration cost relative to a single runtime operand change.

Cheapest meaningful kernel discriminator IF selected later: ONE representative
S44 output shape, production GPU-produced Q8 rows, direct public Q6 control,
then preexpanded candidate plus final native-XOR kernel counted together.
Compare exact signed-code/scale/d extraction, per-native-lane INT32 dots and
all consumed FP partials/final output, guards/read-only/tails, A/B/A freshness;
only then completed-two-kernel timing. Preserve native half-scale products,
pair-parenthesized FP update/K order and XOR16/8/4/2/1; strict mismatch stops.
No broad shape sweep, no all-model packing, no compiled-ISA framework before
a private GPU discriminator. Existing S44 offsets are references, not freshly
read or rehashed model identity. No candidate source implementation is
prepared here: S125's tile/reduction and layout are still source feasibility.

VERDICT -> Rank verifier dense work as a real larger pool, but DO NOT yet
select S125 production work or a paired-forward rewrite from this record.
The largest unresolved end-to-end lead is the same-clock LEFT6.720 ms and
post-pair3.317 ms absent-record readiness, with profiler-reset perturbation
unresolved. Root should choose whether to explain those source/lifecycle
boundaries first or deliberately parallelize ONE private Q6 output probe.
Existing HC/GU mechanisms and rejected tuning are not reopened by this census.
No serving forecast, physical-idle inference or claim of a current GPU win.
