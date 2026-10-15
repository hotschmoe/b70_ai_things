# Flash-Next goals and measured status

Set by the user on 2026-10-13. One or two users. Backend focus: Strata.
vLLM parked, ready to port. 27B INT4 AutoRound on vLLM is the fallback.

The [2026-10-15 backend investment review](20261015_backend_investment_review.md)
records the expanded C1-C4 platform contract, independent Opus/Fable
recommendations and source-checked contribution options. It does not
change the active backend or qualify a replacement.

| Goal | Target | Measured (2026-10-14, bounded screens) | Gap |
|---|---|---|---|
| Prefill | 2000+ tok/s | median 1952-2004 tok/s at 32k, 1260-1337 at 4k across same-behavior screens (IQ3_XXS, v0.1.44.1, split27, spec4; 3 prompts each) | 32k near target, not a stable 2000+ floor; short prompts below |
| Decode | 150+ tok/s per stream | median 83.8 coding, 64.5-65.7 after 4k, 64.9-65.0 after 32k, spec4 (at most 3 drafts) | about 2x short |
| Context | 200k per stream, 2 streams | IQ3_XXS: 2 x 200000 slots with spec4 (at most 3 MTP drafts); both retrieved three needles in 199501-token prompts | capacity screen passed; sustained load remains |
| KV eviction to RAM | restore without recompute under load | both GPU slots overwritten, then concurrent returns reused 199555/199582 tokens; 54/27 new tokens read. RAM restore 430/422 ms, client TTFT 1.97/4.40 s (parking16384 MiB/4 entries) | bounded forced-eviction test passed; sustained use remains |
| Cached prefix latency | Diagnostic reference: p95 2 s TTFT, including RAM restore; not a hard target or gate (user, 2026-10-15) | S20: four forced RAM returns reuse199839 held tokens/read31 new/0replay; TTFT2.758-6.423s (four samples) | Once recovery avoids recompute, prioritize concurrency, decode and MTP |

Latest candidate status (2026-10-15): S20 passes all four full-length RAM
returns without retained-history replay. S21 depth2 concurrent coding is
98.901 aggregate tok/s versus93.064 depth1 (three pairs;6.272%); median
client decode52.737 tok/s per stream, still below150. Full-cache recovery
and deeper-MTP throughput are separately bounded qualifications. No daily
shelf promotion; clean serving control remains retained below.

S28 fixes the bounded EP publication oracle with explicit barrier scope:
scoped mirror and actual native EP TU each65/65 PASS, and frozen engine
152055a6 passes concurrent output prefixes, drafting, functionals and final
health. No new speed or full-cache qualification. S26 head-pair remains
OFF/unqualified. See [S28 evidence](20261015_strata_ep_publication_scope.md).

S29 now tests that head-pair option: same-binary C2 coding median97.881
OFF ->102.279 ON aggregate tok/s (+4.493%, three pairs), all six outputs
match. Client per-stream medians53.399/55.620, still below150. Functionals
and both teardown/health closures pass; full depth2/headpair RAM recovery
is pending. A batch->solo draft-counter loss is identified, so complete
acceptance comparisons are invalid until corrected. See stageS29.

S30 corrects the Python batch-to-solo draft carry loss: six CPU tests and
the actual native-completion/API sum oracle pass. The same frozen152055a6
headpairON/depth2 candidate now passes full context/cache correctness:
four distinct199501-token prompts, two near-full active256-token replies,
and four forced RAM returns reuse199839 held tokens/read31new/replay0.
Draft KV byte-readback checks pass. Configured parking32GiB/4entries;
peak sampled nativeRSS16.255GiB, container16.244GiB, zero swap/OOM.
Native restores663.3-685.1ms; clientTTFT2.992-6.918s, diagnostic only.
One fresh32719-token arrival during decode still causes a7.753s content
gap and29.564s TTFT. These correctness runs include a counter observer
and snapshot verification; they do not replace clean S29 speed evidence.
Final teardown/health closure is recorded with stageS30. No daily promotion.

S31 supplies the previously missing current paired-head operation trace:
18C2head calls reconcile9 native paired windows; five reviewed replay
cycles each have twoC2heads (2.416ms service), oneC6head (2.262ms), and
9 actual L0 AppendCommandListsExp calls. Both32-token replies and final
health pass. These are diagnostic timings, not a new throughput result.
Larger recorded gaps after draft selection and before mapped-input copies
need uninstrumented boundary verification; the older standalone GR gap
does not describe serving. See the dated paired-head profiling report.

S32 adds buffered CPU boundary timestamps without unitrace, frozen595f2ff5.
Both128-token prefixes and functionals/normalstop/finalhealth pass; exact
1684records/dropped0. Forty reviewed replaywindows: PLEgather median0.829ms,
verifiersubmit55us, pairedsubmit45us, pairedmappedstores0.316us. Thirty-eight
adjacent joins: pairwaitreturn->nextverifiersubmit0.866ms/p951.112ms.
GPUcompletion and CPUwaitreturn remain unjoined; no physicalidle/speedclaim
or new fullcache qualification. See [S32 report](20261015_strata_host_boundary_profile.md).

Quality: on 2026-10-14 Flash-Next IQ3_XXS beat the 27B INT4 8-0 on paired
langs + sysadmin tasks (41/43 vs 33/43); see the A/B brief, Outcome 2.

Retained clean control config: strata/flash-next/serve-iq3xxs-200k.json, clean
upstream 296b59d2/v0.1.44.1. Peak sampled resident VRAM 28.22/29.27 GiB;
all 24576 main experts in VRAM. Health, concurrent coherence, reasoning,
tool round trip and clean teardown passed. The soak observer is stopped at user request; multi-day stability is
still unqualified. Intake observed uptime was 10.7 h on the old config.

Matched concurrent coding screens: layer spec3/spec4 total63.56/63.94
output tok/s, EP spec3 69.99 (35.8-37.1 per stream). Three two-request
windows per arm,107 input/1500 output each; total includes prefill and
admission. All six outputs match. Every overlap response offers ZERO MTP
drafts. Upstream --batch-mtp is default-off and rejects layer splits;
its EP combination remains untested. This limits the value of depth tuning
for two users. Single-stream quiet layer spec3 is77.7 coding/64.7 prose;
spec4/.5 is83.8/65.7, and cutoff0.3/0.7 gives no broad gain. Retain spec4,
which includes the seed and offers at most three MTP drafts.

EP #1798/#2205 on head296b59d2 records91.9 coding and71-72 prose decode
with spec4, but only578-579 prefill at4K/32K, so it is not promoted.
A4K trace puts52.8% of prefill in combine (peer wait/row collection included).
Docker IPC launcher fix: https://github.com/Niko1221/Strata/pull/2468.
A transfer microprobe crashed before large timing; rebind/per-card/compiled
collective health recovered, and no native copy-path patch was promoted.
A new tool continuation fails on EP/spec3 and passes on layer/spec4;
remaining functional2x2 arms are needed to isolate its cause. This further
blocks EP/spec3 promotion. The original EP/spec4 tool probe did pass.
S5 measures large peer reads at3.32 GB/s; DMA is slower (1.65), as are
wider uncached loads. Opt-in FP32 peer reduction improves matched EP
prefill4K/32K from570/576 to1037/1106 tok/s; an8192 chunk reaches1265
at32K, all experts resident.26 numerical cases and functional/teardown
health gates pass; generated text differs later due to regrouping. Still
below2000, no full-context candidate or daily-driver promotion. Source PR
https://github.com/recutita/Strata/pull/1 targets the author branch behind
upstream#2205. S7 rejects asynchronous CPU submission (slower), and exact
workspace plus12288-token chunks reaches1313.4 tok/s at32K (three fresh
prompts; TTFT24.983s). All experts resident and functional/teardown health
pass. This is an experimental control, not full200k qualification. Next
measure PUSH peer returns and individual operation latency. S8 standalone
256MB PUSH is22.84ms/11.22GB/s versus uncached read77.25ms/3.32GB/s;
80 freshness checks and post-health pass. S9 same-binary PUSH0/1 gives
4K1047 ->1214 and32K1308 ->1531 tok/s, native32K prefill25.013 ->21.379s,
client TTFT25.077 ->21.444s. Nine matched outputs exactly equal,52 numerical
cases and functional/teardown health pass. Coding decode88.9 both arms.
Still below2000; candidate full200k and daily promotion remain unqualified.
PUSH source update78cea7a1 is on the same stacked author-branch PR.
S10 now has complete per-operation prefill and decode traces. Return-path
GPU service: pull reduction/read0.64/11.08ms -> PUSH reduction+write/local
read4.23/0.31ms per4044-token layer. Attention is612ms at4K and uses32-query
batches (126 full batches plus a tail). S11 query batches128/256 give
less than1% gain, so the research knob is reverted. One traced four-token decode window shows peer waiting19.60ms
versus row returns1.06ms; overlapping primary work must be profiled next.
Explicit stop/flush and tracing during graph capture are required for valid
coverage. Profiler overhead is large; these are diagnostic, not benchmarks.
S12 lane-cell score variant gives no broad gain (4K1221/32K1515); leave
it off. Final clean main still equals refreshed origin/main296b59d2 and is
serving the qualified split27/spec4/two200k configuration. Final stable
identity, concurrent arithmetic, reasoning, tools and27B check pass.
Attention value gather/accumulation and primary dense/shared decode work
remain the next operation-level targets; no extra candidate promotion.
Four full RAM snapshots project14.47 GiB and a fifth during swap18.08 GiB;
24-32GiB parking is appropriate to test two active/two parked conversations.
Four DISTINCT full200k conversations are unqualified; production retains
16GiB/4 slots and two active GPU streams. IQ3_S and concurrent MTP remain
open.
Final layer spec4 is serving :18085, observer off, final functional gates
and post-experiment GPU health pass. The only config update is log path.


S13 value-transpose/four-load pipeline passes36 exact INT8 output/scratch
cases but is neutral; no production patch. S14 tests existing opt-in EP
batch MTP on frozen PUSH-s9. CPU-quiet paired coding total68.72 ->93.89
output tok/s, roughly49-51 per stream; initial three-window medians69.32
->93.42. All24576 experts stay resident, functional and teardown health
pass. Solo spec4 remains at most three drafts; the existing batch path
limits each slot to ONE proposal. Native3.74-3.80 emitted rows/window
proves acceptance with two slots, but HTTP draft counters incorrectly stay
zero because slot BDONE lacks counts. Earlier ordinary-batch zero-draft
findings remain valid. New source accounting fix remains open work.
Mixed long-prefill arrival still interrupts decode for about7.8s; incoming
32K TTFT29.25s. This candidate is not full200k/four-distinct qualified or
promoted. The prepared four-chat harness requires32GiB parking and bounded
RAM sampling; it has not run. Dense/GR microprobes are built, no knob tested.

Evidence: docs/20261014_flashnext_strata_optimization.md, JOURNAL.md S1/S2/S3/S4/S5/S6/S7/S8/S9/S10/S11/S12
entries dated 2026-10-14, and raw results under
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_opt_20261014.
EP/MTP raw results: /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_ep_mtp_20261014.
The original 2026-10-13 measurements remain in the A/B armB report and
JOURNAL.md.

S15 bounded dense/GR knob screens find no broad candidate. Disabling the
shared-expert stream gives93.61 ->92.81 total concurrent tok/s; all six
response hashes and functional/teardown health pass. Keep default execution,
no promotion. Claude Fable5.1/Opus5.5 consultation and loop-cost decisions
are recorded with the stage evidence. Candidate cache qualification is next.

S16 adds correct batch draft accounting (upstream Strata PR2511), with6/6
matched response hashes and93.77 total tok/s. Two full200k slots actively
decode256-token replies and four distinct199501 prompts retrieve needles.
However, stricter RAM exchange FAILS: A replays255 retained reply tokens,
B loses its full prefix and is cancelled during re-prefill. Full four-chat
RAM/no-recompute qualification is incomplete. Partial failed-exchange peak
RSS18.70GiB, zero swap/OOM; this is not a full-exchange peak. Cached return
TTFT1.44-13.10s fails the provisional2s working target. Earlier short-reply
screens did not exercise preservation of a256-token batched reply. Fix
cache lifecycle with small contexts before another full run. Daily clean
layer remains recoverable; no routine restoration between future tests.


S19 cache candidate b09c9bf3 fixes both loss modes in the small oracle:
4x3997-token distinct chats,256 replies,two-active proof,then four forced
RAM returns all reuse4335 held tokens/31newread/0replay. Four draft-KV
readbacks and10/10 available baseline hashes pass; functional/teardown
health pass. Full200k remains untested for this patch, so the RAM goal is
not newly qualified. Small-context p95 RAM1.758s/resident2.161s (4samples
each, diagnostic) does not establish the provisional full-size2s target.
Servers stopped between experiments; no routine daily restoration. Next:
separate8line snapshot-buffer reuse, cheap oracle, then one final full4x200k
qualification against absolute goals. No unmatched relative speed claim.
Details: stage doc S19, raw cachefix_20261014/cache-candidate/s19-report.json.


S20 full recovery PASS: four199501-token conversations, two active
200000-token slots,32GiB/four-entry parking. All four forced RAM returns
reuse199839 held tokens/read31 new/zero retained-tail replay; needles and
two-active proof pass. Native full restores419.4-430.3ms, end-to-end TTFT
2.758-6.423s (four samples), diagnostic only. Peak native RSS16.19GiB,
zero swap/OOM. Identity/math/reasoning/tools pass; teardown evidence in S20
stage journal/report. This qualifies bounded latest-text-chat RAM recovery;
no daily shelf promotion, broader branch/image/CUDA coverage untested.
Fresh full-length prefill1153.5-1174.4 tok/s and long-context decode remain
below goals. Pivot to concurrency/decode/MTP; defer latency-only work.


S21 opt-in STRATA_BATCH_MTP_DEPTH=2 improves matched two-stream coding
median93.064 ->98.901 aggregate tok/s; all six outputs per arm match S16,
plus both9707-token outputs match across depths. Depth3 screen93.156 is
rejected. Functional/teardown evidence is in S21 report/journal; no full RAM
repeat or daily-shelf promotion. Main verifier and serial drafting costs
now guide the next stage, rather than a2s cached-latency target.
