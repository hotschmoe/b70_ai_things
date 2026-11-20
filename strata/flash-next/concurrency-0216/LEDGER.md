# Flash-Next 0.2.16 concurrency ledger

Status: B1-B7 captured; B8 could not load; operation census remains incomplete in the ways listed below. Raw traces and client markers are under
`/mnt/vm_8tb/b70/results/conc-0216/`. The original candidate binary and DD
settings are the control. Device service durations overlap and are not an
additive wall-time or critical-path budget. Unittrace changes execution timing.
Byte counts, source/ISA bindings, precise causal joins and several scenarios
remain to be completed.

| Scenario | Capture status | Ledger |
| --- | --- | --- |
| B1 C1 decode | Both-device replay visible, client-bounded interval; window breakdown pending | `ledger/b1/operations.csv`, `summary.json` |
| B2 C2 decode | Closed retry; client-bounded two-stream decode | `ledger/b2/operations-closed.csv`, `summary-closed.json` |
| B3 cached C4 decode | Closed retry; client-bounded four-stream decode | `ledger/b3/operations-closed.csv`, `summary-closed.json` |
| B4 cold C4 | Closed retry; whole concurrent request batch | `ledger/b4/operations-closed.csv`, `summary-closed.json` |
| B5 >=128K restore | Forced return passed; traced pair includes full responses and may omit graphs captured while paused | `ledger/b5/operations.csv`, `summary.json` |
| B6 32K prefill | Closed retry; request start to first content | `ledger/b6/operations-closed.csv`, `summary-closed.json` |
| B7 C8 | Eight actual slots with resident KV 32768; closed diagnostic and unprofiled short-prompt runs | `ledger/b7/operations.csv`, `unprofiled-latencies.csv` |
| B8 C16 | Did not load at 32768 or minimum 20480 resident cells | `ledger/b8/load-attempts.json` |

B1 contains 82 Q5_K verifier-head events in the selected decode interval.
Per-event totals divided by 82 are rough per-window service averages, including
edge windows and overlapping device work. They are ranked observations, not
proven removable critical-path savings.

| Rank | Card 0 recorded kernel family/shape | Calls | Total ms / 82 windows |
| ---: | --- | ---: | ---: |
| 1 | HC Q8 down, 4 rows | 7954 | 1.675 |
| 2 | KV append Q8 steps | 1031 | 1.354 |
| 3 | HC Q8 up, 4 rows | 7954 | 1.256 |
| 4 | Q5_K KREG verifier head, 4 columns | 82 | 0.939 |
| 5 | Q6_K KREG, 4 columns, first shape | 2378 | 0.857 |
| 6 | Q2_0 routed down, word expansion | 2460 | 0.713 |
| 7 | IQ3_S KREG2, 4 columns | 820 | 0.671 |
| 8 | Q6_K KREG, 4 columns, second shape | 1148 | 0.637 |
| 9 | IQ2_XXS routed gate/up | 738 | 0.635 |
| 10 | IQ3_S routed gate/up row reuse | 1066 | 0.627 |

Card 1's `ep_wait` occupies 1451.8 ms in this 2.619 s profiled client interval
(3936 calls, 17.705 ms per 82 head-bounded windows); card 0 has 3936 such
calls totalling 0.324 ms per window. This is recorded spin service, not
productive compute or an independently measured idle interval. The per-layer
direction and overlap still need explicit reconstruction. See raw per-operation tables for distributions,
launch geometry and other kernels. Clock brackets do not establish exact
cross-card causal order. The 256-token profiled run (97.38 tok/s) and separate
1500-token unprofiled C1 control (134.25 tok/s) differ in length and cannot
isolate profiler overhead.

Open questions: exact per-layer wait direction; full source-to-shape/byte
binding; host CPU attribution; NVMe and host-mapped physical traffic;
per-kernel hardware metrics; complete B5 graph capture and causal first-token
clock joins. The prior 3.4 GiB multi-trace is
unclosed: all corresponding operation counts may omit buffered records.

B5 forced AB returned 132243 cached tokens and read 31 new tokens each. Native
restore records were 217.5 and 216.1 ms. Profiled client TTFT was 1.453 and
7.015 s; the second admission followed a recorded 3479.4 ms slot park,
170.5 ms copy and 330.5 ms draft-first. A main-session park took 462.4 ms.
These profiled, overlapping quantities do not form a complete additive budget.

## Scenario interpretation and limits

Ranked recorded service operations for all seven captured scenarios are in
[OBSERVED-OPERATIONS.md](OBSERVED-OPERATIONS.md). Each scenario also has an
`explicit-transfers.csv` derived from named Level Zero copy events. Its byte
counts are logical operation sizes, not measured physical PCIe traffic;
kernel-mediated EP, mapped-memory and KV transfers are not included in those
copy rows. Top service totals are not proven critical paths.

1. C1 uses a solo four-row speculative window; C2/C4 use depth-1 batch windows.
   The earlier unprofiled 0216 coding run produced 1500 tokens with 1443
   drafts offered, or about 3.12 output tokens per solo window. A batch slot
   has at most two output rows per window, and acceptance and admission gaps
   further affect client rate. The closed diagnostic trace confirms per-row
   attention: B2 contains 135 four-row and one two-row verifier heads and 6504
   QSA chunk/resolve launches (135*48 + 24); B3 contains 283 eight-row, 13
   six-row and six two-row heads and 28248 of each (283*96 + 13*72 + 6*24).
   Thus more rows increase attention and resolve launches; expert union, host
   commit, emit and other work also grow. Their independent critical-path
   shares are not yet measured.
2. Cold C4 is admitted serially through the Python control lock and a single
   main prompt session; decode is served between prompt chunks. The unprofiled
   original-binary fixture produced sorted TTFTs 18.79, 40.73, 62.70 and
   84.60 s. This is an observed serialization pattern, not a proof that no
   decode overlaps prefill. Concurrent prompt execution would require a
   scheduling and state-management change.
3. In the earlier unprofiled forced-return run, median TTFT was 2.083 s and
   median native restore was 211.2 ms. One 1.443 s first return had an
   outgoing park of 435.1 ms, restore of 210.2 ms, and 24+6 tail tokens read
   in 153.1+43.2 ms, inside an 889 ms native prompt interval. The remaining
   roughly 0.5 s outside that interval is not separately attributed to Python
   versus other host/client work. A paired return can additionally queue behind
   promotion, including recorded 422-1509 ms parks and roughly 160 ms copies.
   The B5 profiled pair independently passed no-recompute checks but must not
   be used as an unprofiled latency measurement.
4. Old-night versus 0216 matching logs show verify-run time rising with rows;
   the current trace establishes the per-row streamed-KV dispatch that the
   old source's batched path avoided. This is a strong candidate explanation,
   not an isolated regression measurement: other port, admission and runtime
   changes remain possible until a default-off A/B is tested.

Hardware metric enumeration on card 0 returned no metric groups (status
0x7ffffffe); no compute or memory-bandwidth counter claim is made. `py-spy`
was unavailable and host CPU samples were not collected. Logical estimates
for weights, host-staged traffic, PLE page-cache/NVMe reads and individual
kernel bandwidth are not a complete ledger. The closed retry used a 600-second
native watchdog solely to permit the 270-second profiler flush. The old
unclosed trace is retained as partial evidence, and the watchdog abort was
followed by reset, both-card health and compiled two-rank collective health.

C8 has eight slots at 200K maximum context and 32768 resident cells; the
unprofiled two-repeat short-prompt aggregate was 119.46/136.70 tok/s, with
TTFT p50/p95 4.60/10.15 s across 16 requests. Median per-stream content-event
gap p50/p95 was 80.81/83.69 ms; this is not exact token ITL. In the separate
profiled all-eight-active interval, 539 verifier heads were eight rows;
head-to-head p50/p95 was 58.20/240.35 ms, a diagnostic including waits and
possible capture effects. Paired draft heads were disabled at eight slots.
C16 did not load because the peer experts did not fit, even at the engine
minimum resident KV; latency and throughput are not measured.

## Source-derived logical data sizes

For the compiled Flash-Next geometry (24 query heads, 2 KV heads, dimension
256, page size 4), one complete INT8 streamed K/V page contains 4096 code
bytes plus 128 FP16 scale bytes, or 4224 bytes. This follows `kv_block_bytes`
in `engine/sycl/src/kernels/cuda/kv_stream.dp.cpp`. One resolve miss therefore
requires that much K/V payload at the source level; actual model miss counts,
transaction granularity and physical PCIe/VRAM traffic have not been measured.

At the maximum selection width of 2051 distinct cells, the selected INT8 K/V
codes and scales name 2165856 logical bytes per QSA layer and row, or 25990272
bytes across the 12 QSA layers. Short contexts name fewer cells. This counts
each selected cell once; it does not estimate duplicated head loads, cache
hits, page-copy padding, mapped writes or actual link traffic.

The original engine log identifies the Q5_K full-vocabulary head payload as
437043200 bytes (248320 rows x 2560 inputs x 176/256 bytes). The B1 trace has
82 full-head launches averaging 939.36 microseconds; one logical pass over
that weight payload divided by this *profiled* event duration is about
465 GB/s, or 76.5% of the nominal 608 GB/s figure. This is a calculated
logical payload rate, not a hardware bandwidth measurement: caches, repeated
loads, activation/output traffic, concurrency and profiler overhead are not
resolved. Smaller Q5_K kernels with the same template name have different
shapes and are not included in that head calculation.

The exact pack manifest `native_experts.txt` contains 48 per-layer expert
layouts, reproduced in `ledger/expert-layout.csv` with gate/up and down quant
types. A single expert blob ranges from 1305600 to 2329600 bytes by layer;
those 48 per-expert blob sizes sum to 83814400 bytes. Ten distinct selected
experts per layer therefore name 838144000 bytes of expert blob payload across
one complete 48-layer row. This is a logical footprint, not a claim that those
bytes crossed VRAM or PCIe: cache behavior, four/two/one-row reuse, expert
union, padding, and per-card attribution remain to be measured. The manifest
contains 512 experts per layer; 512 times the 48-layer sum matches its stated
42912972800-byte total.

## Stage C streamed-KV batching, interim

The default-off candidate has synthetic bitwise row/streamed-KV parity and
clean matched C1/C2/C4 runs with both-card post-health. Its C1 and C2
OFF/ON text hashes match, but all four C4 outputs differ across the OFF/ON
servers, with each server internally repeatable. Earlier original and
candidate-OFF servers also differed, so this observation does not establish
either safety or causation. The ON path and end-to-end output must be checked
further before proposing a daily-driver change.

C1 OFF/ON repeat rates were 134.32/132.56 and 134.64/132.45 tok/s. C2
median stream was 71.30 OFF and 76.13 ON. C4 OFF/ON aggregate runs were
89.82/89.03 and 149.06/147.65 tok/s, while TTFT p50/p95 changed from
4.45/8.61 to 0.89/1.59 s; admission variance makes the aggregate delta
non-isolated. The corresponding median per-stream SSE content-event-gap
p50/p95 was 44.29/45.86 ms OFF and 40.37/41.93 ms ON; an event can carry
multiple tokens. See `ledger/c-comparisons.csv` and JOURNAL.md. Neither
near-200K recovery nor the candidate's C8 operation has yet been qualified.

The B5 profiled whole-pair interval recorded seven card-0 `D2M` copy events
of 117669888 bytes each, totaling 823689216 logical bytes and 184.08 ms of
summed device service, plus two `M2D` events of the same per-call size,
totaling 235339776 logical bytes and 32.48 ms. Unittrace named their queue
`L0 Compute Engine<0,0>`. These raw profiler labels do not identify physical
PCIe bytes or prove which single restore or park each event served; the
interval includes both complete responses and some graphs can be missing.
The full per-size and per-card event tables are in `ledger/b5/explicit-transfers.csv`.

The eight cached-C4 native admissions in each arm are tabulated in
`ledger/c-c4-admissions.csv`. The elapsed `draft_first` median/range was
453.4/11.0-595.2 ms for original, 795.6/11.1-2858.8 ms OFF and
21.7/20.1-53.7 ms ON, while median recorded park was about 102-106 ms. These
are logged subintervals, not proof that the attention switch caused the
admission difference. The host elapsed interval includes the draft-first call
and its waits; the mechanisms and full request-to-slot mapping remain open.

Source inspection further limits the `draft_first` interpretation:
`engine/sycl/src/core/mtp.cpp:2157` synchronously waits for residual staging
before `draft()`, and `draft()` can capture graphs, perform PLE prefetches,
submit draft graphs and wait for their completion. The logged interval is
therefore not an isolated MTP GPU kernel duration; separating queue backlog,
capture, PLE and execution requires instrumentation or an aligned trace.

The EP flag direction is identifiable from source even though its critical
path is not: in `engine/sycl/src/core/verify.cpp:1823`, the peer queue waits
for the primary card's inbox flag after the primary pushes the layer input;
at line 1937, the primary waits for the peer's return flag before combining.
The recorded C1 peer `ep_wait` service is consequently waiting for the
primary-side inbox; the much smaller primary `ep_wait` service waits for the
peer return. Spin duration alone cannot prove whether moving the shared expert
would shorten the window. A bounded per-layer parser is prepared for the
closed C1 trace; its ordinal grouping and cross-device clock limits must be
validated before use.

B5's log explicitly confirms that `STRATA_NIGHT_PARK_MOVE` and
`STRATA_SLOT_COPY_DIRECT` were enabled. `STRATA_SLOT_CKPT_KV_DIRECT` was
configured, but has no runtime dispatch marker in this source. Enablement
does not establish that every checkpoint used the move or direct-KV branch,
and the old-night versus 0.2.16 park and draft-first regression is not yet
causally explained.

## Profiled scaling observations

The table divides all recorded card-0 operations whose names contain the
indicated family by full-vocabulary verifier-head counts in the same
client-bounded trace interval. The name filter includes multiple kernel
shapes and can include edge, admission or draft events; it is not an exact
per-window causal breakdown. Device service can overlap. The runs used
different contexts and C8 also used a different resident KV budget, so these
figures do not isolate concurrency or represent unprofiled latency.

| Trace | Heads | QSA vector family ms/head | GDN multi ms/head | Card 0 EP wait ms/head | Card 1 EP wait ms/head |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 C1, short | 82 | 0.395 | 1.080 | 0.324 | 17.705 |
| B2 C2, short | 136 | 1.211 | 1.097 | 0.355 | 22.967 |
| B3 cached C4, ~20K | 302 | 2.946 | 2.151 | 0.493 | 37.980 |
| B7 C8, short, 32768 resident | 539 | 2.402 | 2.223 | 0.777 | 34.772 |

The narrower B2/B3 QSA launch accounting in Scenario interpretation above
selects a specific verifier kernel shape, hence its counts differ from this
whole-family table. These observations show where recorded service grows;
they do not establish additive critical-path contributions or physical bytes.

The cold-C4 serialization has two source-level constraints beyond the single
control lock. `terminal/serve/server.py:1865` asks a prompt to yield only
when a waiter has at most half its token length. The four ~19798-token
fixture prompts are approximately equal, so that test does not request a
yield among them. In `engine/sycl/src/program/generate.cpp:11620`, a requested
yield is also accepted only when the remaining read is greater than both the
chunk size and `short_read`; after the first 12288-token chunk, only about
7510 tokens remain in this fixture, below the 12288 chunk size. Existing
active slots can still decode between prompt chunks under
`STRATA_BATCH_DECODE_SHARE` (default 0.5); that interleaving is not parallel
prefill of the waiting prompts. Different prompt lengths and chunk settings
can take different branches, so this explains the fixture rather than every
possible workload.

Native unprofiled batch summaries isolate the scheduled window from the
client's admission wall time: C2 OFF averaged 25.12/24.98 ms and ON averaged
24.09/24.13 ms. Both report 3.70/3.75 kept rows per window; those are accepted
output rows, not the verifier's processed row counts. The
reported run component changed from 22.35/22.21 to 21.33/21.36 ms; commit
and emit remained 1.41/1.36 ms in these logs. These are two episode averages,
not p50/p95 or an individual kernel measurement. C4 averages and their
different row counts are in `ledger/c-native-batch-summaries.csv`; its output
identity and admission differences still prevent a promotion claim.

C8 is eight overlapping client requests and eight configured slots, not eight
sequences in a single verifier window. At batch MTP depth 1 the SYCL
`batch_step` packs two rows per active slot into at most eight rows, and rotates
the starting slot when more than four are active
(`engine/sycl/src/program/generate.cpp:10026-10087`). Thus at most four
streams participate in one verifier window. The unprofiled C8 episodes
reported 7.34/7.38 average kept output rows and 42.09/41.83 ms per scheduled window;
client-level aggregate and tail latency also include scheduling and admission.
The profiled all-eight-client-overlap interval contained eight-row verifier
heads, which by itself does not mean eight streams per engine window.

Widening a verifier window to fit eight depth-1 streams is not a safe
configuration-only change on this SYCL port. The common window limit defaults
to eight, but `kernels/cpu/expert.hpp` also fixes `MAXT = 8` and verifier
initialization rejects a larger value; the SYCL `run_slots` and
`batch_launch` helpers contain fixed `rows[8]` arrays. Batched QSA rows are
also capped at eight, and several optimized paths explicitly guard eight
rows. An experimental wider build requires deliberate source porting, buffer
and kernel audits, overflow tests, parity and a fresh C1 regression gate;
changing `STRATA_VERIFY_MAX_T` alone would not establish correctness. This
is a source finding, not evidence that a wider window improves throughput.

Three configured switches have narrower scope than their names might suggest:

- `STRATA_MTP_PARALLEL_ARGMAX` requires the full 248320-token head and no
  subset draft head; this config binds the 40525-token draft vocabulary, so
  that dispatch is ineligible.
- `STRATA_BATCH_MTP_PAIR_GRAPH` is selected only when the pair has exactly
  two draft steps. The depth-1 slots cap drafts at one, so the depth-1 runs
  cannot take it; depth 2 is worth testing as a distinct configuration.
- `STRATA_GDN_COMMIT_LAYERS` requires two owners with exactly three rows each
  and six rows total. Depth-1 groups have two rows per active owner, so this
  fast path is ineligible in those batch windows. Likewise, the grouped GDN
  verify kernel calls `gdn_step_norm_multi_kernel<true, false>`; the printed
  256-GRF switch marker does not mean the grouped form uses its 256-GRF
  variant. The non-grouped path may still use it.

These follow the checked-in dispatch guards and observed startup/graph logs;
they do not by themselves show that enabling any alternative is faster.

For the two cached C4 episodes, subtracting the rounded native sum of
window run+commit+emit durations from the corresponding batch wall interval
leaves approximately 6.74/6.62 s OFF and 1.12/1.20 s ON. This residual
includes admissions and other work outside those measured windows, and the
window totals are rounded; it is not an isolated admission cost. Together
with the recorded first-draft and park intervals, it shows why the client
aggregate difference cannot be assigned entirely to faster batch windows.

A stronger control for the C4 identity issue: the earlier Stage A baseline
and later Stage C control used the exact same original binary, DD settings and
C4 fixture, yet all four C4 texts differed across those server runs. Their C2
outputs matched. The runs differed in prior sequence (one versus two C1
repetitions) and arrival order. This rules out the new switch as the sole
necessary condition for cross-server C4 text differences, but does not prove
that the switch is numerically equivalent under matched state and schedule.

## Matched eight-slot switch comparison

Both arms used eight actual slots, 32768 resident INT8 KV cells, a 200K
context setting and the same 148/364 expert placement. Candidate ON ran first
and original second; each had two C1, C2, C4 and C8 short-prompt repetitions,
with identity, arithmetic, exact corresponding output hashes, teardown and
both-card post-health passing. C1 original 133.74/134.28 and ON 133.84/134.33
tok/s. C8 decode-window aggregate original 152.99/156.09 and ON
163.74/164.11 tok/s, approximately 6.1% higher in this bounded matched run.
C8 TTFT p50/p95 across 16 requests was 2.31/4.09 s original and 2.28/4.08 s
ON. Median per-stream SSE event-gap p50/p95 was 80.79/83.52 ms and
76.02/78.76 ms; events can include multiple tokens. Raw per-stream values are
in `ledger/c8-comparisons.csv`. These are short requests, not proof of eight
near-200K contexts or long-term stability. Eight clients share at most four
sequences per depth-1 verifier window.

## Batch MTP depth comparison

With the original binary and four-slot DD settings, two-repeat C1 median
was 133.25 tok/s at depth 1, 133.47 at depth 2 and 133.53 at depth 3.
C2 median stream rate was 70.81, 82.75 and 75.00 tok/s respectively.
Cached C4 median stream rate was 31.80, 32.62 and 27.58 tok/s; C4 TTFT
and output hashes varied across servers, so its attribution is limited.
C1/C2 corresponding outputs matched, identity and arithmetic passed, both
servers shut down cleanly and both-card post-health passed. Depth 2 activated
the paired two-step graph and the two-owner, three-row GDN commit. Each SSE
event can carry multiple tokens; especially at depth 2/3, near-zero event-gap
p50 reflects burst delivery and is not an individual-token compute time.
The combined depth-2 plus streamed-attention candidate, long-context recovery
and controlled C4 identity have not yet been qualified.

## Final Stage C qualification update

The combined depth-2 plus streamed-attention candidate passed a bounded
near-200K exercise on four slots, 65536 resident INT8 KV cells and 32 GiB
RAM parking. Six initial prompts each held 199501 tokens. The two A/B and C/D
follow-up pairs decoded concurrently; after two more histories forced slot
exchange, four A/B/C/D returns each restored 199839 tokens from RAM, read 31
new tokens, replayed no previous tail and returned their correct markers.
The four explicit forced-exchange restores each took 307.2-309.9 ms in native
logs. Actual RAM-return TTFT p50/p95 was 3.57/7.69 s across six observations;
that bounded distribution is not a steady-state stability claim. Both cards
passed post-health after clean teardown. The summary and individual request
logs are under `/mnt/vm_8tb/b70/results/conc-0216/c-combined-context/`.

Controlled ordered C4 on the same running combined server reproduced all
four seed hashes in both overlapping repetitions. It does not establish why
cross-server C4 hashes differed, or whether OFF/ON are equivalent under the
same initial state. The complete Stage C results and conditional DD config
diff are in JOURNAL.md. Stage B's top-ten tables in OBSERVED-OPERATIONS.md
remain recorded service, not proven top-ten critical-path costs; hardware
metrics, complete traffic and host CPU attribution remain incomplete.
