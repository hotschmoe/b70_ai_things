# S95 current C2 operation and readiness profile

Date: 2026-10-15. One bounded profile closed with normal teardown and health
PASS. Nonvocabulary MMVQ is the largest named primary-GPU service family.
The selected next step is one source-only routed-expert integer-dot/load/
decode-reuse audit, not another prefill kernel or a new GPU arm. Service does not
establish a clean gain or removable wall-time budget.

CONFIG -> Frozen audited d94e96ca main, EP149/363, all24576 experts resident,
two200k INT8 slots, depth2/headpairON, HCcommon/RPW1/A2/PF0. VALUE0/QSA0/
Q6center0 and explicit XMX0/ROT0. TRACE1/HOST_BOUNDARY1, VERIFY_PROFILE
absent; no native/kernel/build/model change. One ordered107prompt/64output
coding pair, no generation warmup, profiler injected before first capture.

COMMAND -> Actual source consultations `claude-opus-5-5`54.125s and
`claude-fable-5-1`80.069s/root full-read closed controller conditions. Root
fixed the output hash key, bounded profiler control to30s and supplied a
byte-exact local S32 parser before GPU execution. A later analysis-only
interval sweep passed16 CPU boundary-reference checks; original source and
diff are retained. One raw JSON parse after stop took33.181s; CPU-ring
parsing took0.066s. No extra GPU run or raw reparse was needed.

RESULT -> Raw trace269853744B, SHA256
`1139c4c8cc9a4b0737a6c57e67a8b717151ec4276714fa355fb48e3cd9ab3583`:
1663171 events,723519 complete/selected events,859 canonical operation CSV
rows. Both output prefixes/text/SSE/usage/finite-work and paired overlap
pass. Native summary has24windows,21paired/2solo; both streams offer46
drafts and accept39/40. CPU ring has820records/dropped0, complete declared
counts/final window_end and idle slots; no footer exists. Normal QUIT,
full CID+NAME absence, wrapper0 and both-card/compiled10 post-health pass.
No functional campaign was requested; S94 OFF functionals remain separate.

Twelve consecutive reviewed GPU cycles5..16 each show H2-select-select,
H2-select-select,H6,48 identically named EP waits and9 fully contained L0
submissions. Their instrumented cycle median is57.067ms. Selection used
structure/capture rules, not fastest durations. All cold/capture/solo/tail
records remain external in the full5.6MB cycles file; the whole CSV is not
an exact prefill/decode phase attribution.

| Primary-GPU named family | Median recorded service per reviewed cycle |
| --- | ---: |
| All nonvocabulary MMVQ | 12.680488ms |
| Routed native GU/down | 7.504495ms |
| Four HC kernels | 6.718818ms |
| Three vocabulary heads | 4.664113ms |
| EP wait kernel | 3.068313ms |

The MMVQ definition extends native_mmvq-only10.745196ms with IQ3_S
launch_mmvq_multi_nc21, roughly1.935ms/55calls. Names/grids/types do not
identify every weight or shared projection. These diagnostic service sums are non-additive; overlap is not independently
established. They cannot be added to CPU spans or translated into attainable
wall savings.
Routed GU/down7.504495ms is DISJOINT from the MMVQ12.680488ms predicate:
native_gu_port/native_down_port names versus mmvq names. It is not a subset
or a percentage of that MMVQ pool. Their wall-time overlap remains unknown.

CPU first-role evidence conservatively selects16 replay windows, with14
eligible consecutive same-dump transitions. Window_end to next window_enter
median is2.36us. This does not establish a new host-handoff problem or
justify repeating S79/removing readiness waits. Submit/wait spans include
instrumentation and device work. GPU C2heads42 reconcile21 paired rounds,
but whole GPU C6heads24 versus CPU22S6+2S3 leave two C6 heads unattributed.
Do not call them dummy warmup, admission or capture work, or equate GPU cycle
indices with CPU window IDs.

Before/after clock correction drifts305.324us. Native C++ steady_clock
identity remains unverified: no native CPU/GPU or cross-card timestamp join.
Busy/service/spin/API spans are non-additive. Gaps are recorded coverage,
not physical idle, PCIe traffic, bandwidth or causal removable delay. One
31prompt/4output arithmetic canary triggered the already-stopped profiler's
flush; it is excluded from pair/ring/native pair work, while process PLE
aggregates include it.

VERDICT -> Retain this bounded operation census and source limits. Prior
shared-streamOFF rejection and S55/S65/S73/S75/S87 results remain separate;
do not repeat their mechanisms based on this ranking. No candidate/build/
GPU arm, throughput, stability, cache/default/shelf promotion or S94 cause
is established. Actual postconsultations `claude-opus-5-5`38.253s and
`claude-fable-5-1`49.759s accept the census and source-only routed-expert
audit; root read both full answers. Both models incorrectly nested the
disjoint routed/MMVQ pools; their nesting/payoff statements are rejected.
The prior13-tag/four-image binding proof covers attention, not expert ISA,
and no exhaustive all-main ISA extraction exists. The next source audit
must bind its own relevant actual kernel/image and prove integer domain,
overflow, signedness/sign-extension and unchanged FP conversion/scale/
reduction order before proposing a candidate. Integer algebra alone is
not a blanket bit-exact guarantee. Existing correct dot/load/hoisting
codegen may close a lever without a build. No rounding workaround, new
build or GPU test is selected by this record.
Absolute targets and S30 RAM recovery remain unchanged.

[Small records](../strata/flash-next/lowlevel_probe/current_c2_profile_s95/README.md)
include exactly one [canonical859-row operation CSV](../strata/flash-next/lowlevel_probe/current_c2_profile_s95/runtime/analysis/gpu/operations.csv).
Raw root: `/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.
Authority: `current-c2-profile-s95-root-runtime-decision.json` and
`current-c2-profile-s95-analysis-followup/{facts,extended-mmvq-and-cold-limits}.json`.
Postdisposition: `current-c2-profile-s95-post-reviews-20261015/root-post-resolution.json`.
