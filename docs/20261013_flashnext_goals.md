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

S33 tests a default-OFF4096-token scheduling quantum at fixed12288allocation.
Both matched32769-token mixed state screens,1500-token decoder references,
32-new-token cached returns and functionals/teardown/health pass. Largest
content gap7.824->4.059s improves, but incomingTTFT29.526->37.581s and
decodercompletion41.619->46.612s worsen. KeepQ0; no speed winner or new
full200kqualification. The growing gather event-region includes activation
gather and hostpeer enqueueing; MMQ nativeweightgather is not built here.
See [S33 tradeoff](20261015_strata_prefill_quantum.md). Skipmidpoint sweep;
isolate actual EP/gather/dequant and primarydense work next.

S34 reuses the historical S10 trace instead of starting another profiler.
Across 48 layers, activation gather service has median 0.149 ms, followed
by a 52.987 ms median interval without recorded primary GPU operations.
Corresponding host spans contain 58,716 peer kernel appends. This is old,
instrumented evidence with different expert ownership, not current latency
or proof of physical GPU idle. The existing per-layer async dispatch arm
was already slower; do not repeat it. Next, test a distinct mechanism:
one peer activation gather into existing scratch, followed by reverse-order
expert processing with unchanged arithmetic. Its standalone correctness
oracle is prepared, not yet built or run. See the
[S34 dispatch audit](20261015_strata_ep_prefill_dispatch.md).

S35 passes the standalone one-gather/reverse-expert alias oracle at actual
T=4096 and 12288, with ten real expert format pairs and synthetic routing.
All 297,369,600 FP32 values match bit for bit; gather bytes, finiteness,
nine buffer guards and unused-capacity poison pass. Both card-1 health
and teardown closures pass. Cold/warm order prevents a performance claim.
The real IPC serving path still needs guarded integration and a matched
comparison; no new full-cache qualification or promotion. See the
[S35 numerical report](20261015_strata_peer_gather_alias.md).

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


S38 guarded peer-input alias passes actual IPC serving: all48 layers activate,
13 exact request/output/token/cache matches, reasoning/tools and both-arm
normal teardown/per-card/compiled10 health PASS. Same9d30ad8b OFF/ON medians:
4K1303.0->1326.2 (+1.78%),32K1546.1->1556.1 (+0.65%),C2 aggregate
104.516->104.017 (-0.48%). Three observations each, fixed order; no stability
or significance claim. Keep aliasOFF and qualified152055a6/headpairON/depth2.
No full-context rerun or daily promotion for this small prefill-only change.
Current CPU submission spans are now measured, without a GPU-clock join.
Move to unchanged-kernel GRF numerical preparation; targets remain open.
See docs/20261015_strata_peer_gather_serving.md and raw peer-gather-serving-s38.


S39 existing HC GRF256 knob passes48 full snapshot pairs/35,893,248 bytes
across T1/2/6 apply/inject cases and T2/6 changed-input graph replay. Actual
selected DOWN/UP branches run1260 times each/mode; tool reports128vs256
registers and private/spill0 in both. GRF256 is slowerall12 instrumented
cases (4.62-17.62% whole-read), so keep default/no serving or200k repeat.
One29s adapter-only build, two39/47s card1quiet windows, all pre/posthealth/
normalteardown PASS. No clean or serving regression claim. Main serving
winner remains152055a6/headpairON/depth2; absolute targets remain unmet.
See docs/20261015_strata_grf_resource_probe.md.


S41 existingQ6packing failsfirstC6bitwisegate (maxabs9.54e-7), not a proof
of invalidquantizeddotmath. IndependentCPU114outputs/19selectedrows support
rounding explanation, no wholematrix or modelquality qualification. Keep
winner152055a6/headpairON/depth2/strictgateFAILED; no new tolerance or speed
claim. One103s adapterbuild/one36s card1window incl healthyteardown; no
fullengine/serve/cache repeat. Source-derived numericalpolicy next, targets
remain unmet. See docs/20261015_strata_q6_dense_packed_probe.md.


S42 grouped peer GU fails strict equality; independent operand accuracy and
speed remain unqualified. S43's CPU reference confirms source-bound
compliance for both first-matrix Q6 C6 outputs, without another GPU run.
These are primitive diagnostics. Keep S41's strict failure and qualified
152055a6/headpairON/depth2; absolute serving targets remain unmet.
See [S42](20261015_strata_grouped_gemm_probe.md) and
[S43](20261015_strata_q6_cpu_reference.md).

S84/S85 add private primitive evidence, without changing serving status:
long/full256-query scoring+top-k improves while short contexts regress;
Q6 C6 centering passes strict gates and reduces two repeated-weight projection
times by22.8/22.1percent. A defaultOFF main-source proposal is prepared, with
matched full-model comparison pending. Absolute targets remain unmet.
[Source, ISA and numerical/timing evidence](20261015_strata_qsa_tiling_and_q6_centering.md).


S87 same-new-binary two-card Q6 centering OFF/ON passes all four paired
output/native-work/capture gates, functionals, normal QUIT and post-health.
Three timed C2 pairs median103.088->104.279 aggregate tok/s (+1.155%);
per-stream decode53.516/55.867->54.119/56.475. Retain default-OFF opt-in,
no daily promotion or new full-cache qualification. One OFF-then-ON process
sequence, not independent process replication. [S87 result](20261015_strata_qsa_tiling_and_q6_centering.md#s87-guarded-q6-serving-comparison-passes-with-a-small-bounded-gain).


S88 adds the missing16K strict scorer+top-k boundary case (median paired
-3.661%, primitive only), enabling an earlier4097-block guard. S89 actual-main
same-binary QSA0/1 then passes all output/work/functionals/normalteardown/health
gates. Three32K pairs have median prompt-time change-0.547%; native throughput
medians1547.5->1559.2tok/s and clientTTFT21.213->21.054s. The inactive4K path
is mixed/null, around1300tok/s. This is one fixed-order processsequence, not
causal/stability/default/shelf promotion. Keep defaultOFF opt-in; Q6 is OFF in
both arms and S87 remains separate.2000prefill/150stream goals remain open;
S30 RAM recovery is not rerun. Next: one current32K operation/readiness trace.
[S88/S89 evidence](20261015_strata_qsa_tiling_and_q6_centering.md).


S90 completes the separate current32K ON request profile: actual768 tiled
scorer calls and12264 full32 attention chunk/merge calls reconcile current
source. One cold instrumented32728/32 request includes capture/admission/
prefill/decode; it is not clean speed or an S89 delta. Attention-family
recorded service6.473s and host-copy14.404s spans are non-additive and do not
prove removable idle/traffic. Targets remain open; retained S30 recovery is
unchanged. [Current operation/readiness profile](20261015_strata_current32k_profile.md).


S91 prevents an invalid private attention comparison before GPU execution:
renamed host functions still share kernel-info state. The exact retained
S13 probe has the same issue, so its distinct value-candidate correctness
and neutrality attribution is unsupported. Old logs remain intact; no new
speed result or target achievement. Corrected S92 binding is source-only,
with strict checks retained. [Binding audit](20261015_strata_attention_binding_audit.md).


S92 resolves the private attention host-binding defect and passes all 36
strict primitive cases. Full32 unmasked width2051/cap2112 paired chunk+merge
times improve8.469% (page4) and9.182% (page512); masked32/width1999 regresses
2.392%. One process/three nested seeds, no serving gain. Keep a private lead;
Actual post Opus5.5/Fable5.1 accept/rootfullread. The separate exact-cap2051
three-seed numeric-only gate passes strict bytes/memory/identity/health;
explicit defaultOFF staged-prefill main integration/serving remain separate,
without a serving result here. Targets and S30 recovery remain unchanged. [S92 evidence](20261015_strata_attention_value_pipeline.md).


S93 closes as FAILED_ACTIVATION, with no candidate serving speed result.
ProductionCAP2051 strict bytes and actual main host/image binding pass, but
current resident KvPlan mode0 makes the stagedmode1 caller guard false.
OFFsixprefills/functionals and ONwarm+fourprefills match completed outputs/
work; missing candidate marker stops ON before its remaining work. Both
normal teardown/fullCID+NAMEabsence/percard+compiledhealth pass. DefaultOFF
retained; S94 resident caller proof is source-only. Targets and S30 recovery
remain unchanged. [S93 activation finding](20261015_strata_attention_value_serving.md).


S94 corrects resident caller eligibility and observes candidate HOSTmarker1,
but first ON4K fails native-work parity: offered drafts207OFF/205ON despite
matching request/text/SSE/usage/output and all other work counts. No speed
summary, remaining ON tests or retry. Bothnormalteardown/fullCID+NAMEabsence/
percard+compiledhealth pass. KeepdefaultOFF/unqualified; cause unresolved,
no numericalfailure/benignnondeterminism/regression claim. Pivot separate
currentC2profile with candidatesOFF. Goals/S30 recovery remain unchanged.
[S94 work gate](20261015_strata_attention_value_resident.md).
Actual S94post Opus34.340s/Fable20.127s/rootfullread acceptpark/pivot. Raw
S93inactive0/1 offered205/205 vsS94OFF207/ON205 shareaccepted136/outputSHA;
differentbinary/processcontext, not matchedOFF/OFF/control or benigncause.
No retroactivegate relaxation or target/speed claim.


S95 completes one current C2 diagnostic profile with all candidatesOFF,
normal teardown and both-card/compiled health PASS. Twelve structurally
reviewed GPU cycles place allnonvocabularyMMVQ12.680488ms above other named
families; this is instrumented service, not attainable wall savings. CPU
14consecutive replay handoffs median2.36us do not select a hosthandoffpatch.
Two extra wholeC6heads remain unattributed; native CPU/GPU/peer clocks stay
unjoined. Next is one routed-expert integerdot/load/decode SOURCE audit, no GPUcandidate
or throughput/target/promotion gain. Goals/S30recovery remain unchanged.
[S95 diagnostic](20261015_strata_current_c2_profile.md).
Actual S95post Opus38.253s/Fable49.759s/rootfullread accept boundedcensus and
ONEsource-only routed-expert integerdot/load/decode-reuse audit. Routed
GU/down7.504ms and MMVQ12.680ms are disjoint name predicates, not nested.
Attention13tagbinding is not expertISA; no exhaustiveallmainISA coverage or
integeralgebra-only bitguarantee. No newGPU/build or throughput claim.


S97/S98 identify repeated routed IQ2_XS decoding in actual compiled code and
qualify a private shared-preparation primitive:88 strict snapshots and health
pass; warm mixed GU intervals87.80415->52.18525us/call(-40.5663%). Singleton
results are small/mixed. This one-card two-expert fixture does not establish
end-to-end decode/prefill gains, real route reuse, or a default/shelf change.
S30 recovery and all serving targets remain unchanged.
[Source, compiled mechanism and bounded result](20261015_strata_routed_iq2_reuse.md).


S100 closes the default-OFF IQ2_XS serving comparison: all four pairs match
text/SSE/usage/native work/drafts/captures; functionals, normal teardown and
full pinned health pass. Aggregate median103.016058->103.333098tok/s
(+0.307757%); per-stream53.481/55.845->53.730/56.023. One OFF-then-ON
sequence/three nested pairs, not a general gain. Retain defaultOFF/no rerun
or shelf/cache promotion. S98's40.6% warm kernel reduction does not predict
serving throughput. [S100 evidence](20261015_strata_routed_iq2_reuse.md).
S101 actual Opus5.5/Fable5.1 prepare independent Q6-load and EP-readiness
fusion leads; root accepts source audits only, with arithmetic/visibility
and critical-path limits. Graph-boundary follow-up is tracked separately.
[S101 recommendations](20261015_strata_next_kernel_candidates.md).
Throughput targets remain unmet; S30 RAM recovery is retained without a new claim.

S106 whole-pair graphs close with matched text/native work/functionals and
normal teardown/full health. Clean aggregate102.914658 ->102.976265tok/s
(+0.059862%, one OFFthenON sequence/three nested pairs) does not resolve a
serving gain. Separate main host-API summaries show submissions231 ->126,
consistent with six-to-one depth2 coarsening under bounded source conditions;
no per-call/device-idle or causal-speed proof. Keep defaultOFF/no repeat or
fullcache campaign. S104 Q6 aligned loads pass114 strict private snapshots
with warm nested improvements4.85%/2.20%; main source proposal prepared only,
no serving gain or target change. S30 recovery remains retained.
[S106 graph result](20261015_strata_graph_capture_and_bottlenecks.md),
[S104 private Q6 lead](20261015_strata_q6_aligned_loads.md).


S112 passes48 strict private greedy-selection cases on one B70. Completed
two-owner host chains average165.55->116.41us in two nestedABBA rounds;
this includes common probability/select work and establishes no serving gain.
Original production/default remains unchanged. Next S113 prefill fixed-policy
discriminator needs no rebuild; separate defaultOFF MTP integration is source
preparation only. Targets remain unmet and S30RAMrestore remains retained.
[S112 evidence and limits](20261016_strata_greedy_top1.md).


S113 completes the new fixed-policy attention-value serving discriminator:
4K native prefill median1294.9->1299.2tok/s(+.3321%),32K1548.9->1590.6
(+2.6922%). Allwarm/six-timed response/work/capture gates,functionals,identity,
normalteardown and fullhealth pass. Three nested fixed-order observations
per shape remain descriptive; keepdefaultOFF/no immediate repeat/cachecampaign.
Bothprefilltargets remain unmet; S94 workfailure/causeunresolved is preserved.
S30boundedRAMrestore remains retained, no new C2decode/context qualification.
[S113 evidence and limits](20261016_strata_attention_value_fixed_policy.md).


S115 selected original HC UP/reduce ISA confirms repeated nonlinear setup
before the group0-only global store. Root authorized one private producer
activation oracle SOURCE PREP (S116); no candidate build/runtime/performance
result. Keep extra+0/division/nativeexp/store-load and full intermediate/final
strict checks. Static workgroups/opcodes do not forecast speed. All targets
and retained S30 RAM recovery remain unchanged.
[S115 compiled mechanism and limits](20261016_strata_hc_lowrank_producer.md).


S114 default-OFF parallel MTP argmax main integration passes matched warm and
three C2 pairs, functionals, identity, normal teardown and full health. Same
binary aggregate103.073716->103.673778tok/s(+.582168%); perstream53.485/55.836
->53.858/56.241. FixedOFFthenON/nestedpairs, descriptive only. KeepdefaultOFF,
no immediate repeat/cachecampaign/shelf promotion. Probability omission is
still unexecuted source prep, a separate mechanism. Next small private HC
producer oracle build/compiled audit; no forecast gain. Targets still unmet
and S30 RAM recovery retained. [S114 evidence](20261016_strata_mtp_parallel_argmax.md).


S116 private HC producer relocation passes12base/72eager+72capturedroutes,
36historicalanchors/intermediate-finalbytes/specials, normalteardown and
freshcard1health. CompletedHOSTfour-kernel medians73.766->72.593us; two
nestedABBArounds inoneprocess/all8samples retained, notservinggain. Keep
privateproof, deferdedicatedmainbuild/serve; no newdefault/context/cache
qualification. Targetsremainunmet/S30RAMrestore retained.
[S116 result and limits](20261016_strata_hc_lowrank_producer_oracle.md).


S139 corrected C6 head runtime-word candidate passes fresh same-binary
C2 serving: aggregate103.072328->104.393157tok/s(+1.281458%); stream
decode53.474006/55.862589->54.344196/56.688704. Exact output/work,
functionals, normal teardown and full health match. One fixed OFFthenON
sequence/three nested pairs, defaultOFF;150tok/s perstream remains unmet.
No new context/RAM qualification: retained S30 recovery remains the bounded
full-size evidence. [S139 result](20261016_strata_q5_c6_serving.md).
S136 full bounded-cycle readiness/commit attribution and actual
Opus5.5/Fable5.1 select one existing-kernel QSA commit grouping private
GPU oracle next, before broader Q6/HC work. Its result is still pending.
[Decision and each actual agent's reasoning](20261016_strata_mtp_commit_next_decision.md).
