# Strata private HC low-rank producer oracle, 2026-10-16

S116 passes its bounded private strict oracle:12 base cases,36 historical
anchors,72 eager+72 captured route calls and all synthetic diagnostics.
Normal teardown and fresh card1 post-health pass. Production remains unchanged.
The eight completed-host observations are descriptive; this is no serving
qualification or general all-input correctness result.

CONFIG -> n_embd2560/hc4/hc_lr320, normal C16 UP, sliced/non-MORE down,
non-ILP/default GRF, SG32, nsplit=1. T1/2/6 x apply0/1 x inject0/1 gives12
ordinary base cases. Six unique private classes retain norm ->down ->reduce
->UP boundaries. Candidate reduce keeps raw part/injection and computes owner
lo; candidate UP stages that lo into the original SLM/barrier before retained
dot/FMA/butterfly/gate/residual/mean source arithmetic.

COMMAND -> One CPU-leased private build in pinned e41/DPC2026.1 with precise
FP/noDRI/networknone; selected host/image audit and two candidate-only offline
decodes; actual Opus/Fable source and compiled reviews; then ONE card1 process
on physical0000:44:00.0 with quiet cpu-run outside gpu-run. Strict cases precede
conditional two nested ABBA rounds. No retry/reset, serving/model call or
routine server restore. Wrapper/native/wait/rm/analyzer/identity/driver/tee
all exit0; wrapper wall28s. Reused prehealth is explicitly bound to root's
same-boot/no-intervening-GPU S114 full posthealth proof; post-card1 health is
fresh. No new collective measurement is claimed for this single-card fixture.

RESULT -> Every ordinary case checks changing A-B-A eager/captured results,
raw part2/part/injection, candidate lo-before-UP against control lo-after-UP,
all final outputs, tails/guards/read-only inputs and disjoint route ownership.
The72 eager/72 captured counts are native internally checked coverage totals;
12 per-case messages and36 unique historical anchor checks are exported.
The12 synthetic reduce->UP route calls and one direct-control-only UP probe
also pass. There are no HC counters or counter-reset recovery logic.

Actual initial/restored synthetic raw bits include +/-minimum subnormal,
+/-maximum subnormal, +/-minimum normal,+/-1, infinities and two quiet-NaN
payloads. Both initially supplied zero signs reduce to +0 in the displayed
raw-part sample. Thus candidate cross-context raw -0 is not established by
this fixture. Candidate pre-UP lo includes signed-zero results for selected
subnormal inputs. Changed/restored raw and lo bits are retained. Direct
control-UP raw -0 diagnostics remain control-only; they do not close this
candidate coverage limit or prove universal nonfinite behavior.

Completed host time for one warmed T6/apply1/inject1 four-kernel graph read,
including submission and completion wait, in actual ABBA order:

| Round | Control A (us) | Candidate B (us) | Candidate B (us) | Control A (us) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 91.264 | 77.828 | 72.558 | 73.771 |
| 1 | 73.761 | 72.628 | 71.165 | 73.079 |

Round means82.5175 ->75.193us (-8.8763%) and73.420 ->71.8965us (-2.07505%);
four-point medians73.766 ->72.593us (-1.59016%). The first control91.264us is
retained. One seeded process and two nested rounds do not establish replicated
speedup, isolated UP service, device cycles, bandwidth or model gain. No
additional timing run was needed to produce a cleaner number.

The822704B executable SHA is33e390536480f83f169a63e1f9d2f2e06d8e341279a956c84a29a255951c46d3.
Build wrapper23s/compile10.830s/AOT1.749s/host-link0.199s. Actual six distinct
Info/name/image bindings and12 cache/guard cells close the selected audit.
Original control UP/reduce computational prefixes82848/16272B, normalized
execution/payload metadata, options, constants and raw+decoded relocations
match [S115](20261016_strata_hc_lowrank_producer.md). Complete image/text bytes
differ after EOT; norm/down have exact retained source projections without a
new old-binary audit.

Candidate reduce materializes raw r30:f at15224. Extra+0 at16240 uses acc0:f;
scale0.25 at16256, native-exp lowering16320, +1 at16360 and final division
16432/16624 feed owner-lo store17648. Raw-part store16384 is scheduled AFTER
early activation using the same raw r30:2 payload before reuse. Old UP extra
+0 uses a GRF. Matching :f encodings and mul0.25 are not universal accumulator,
store-forward or division equivalence proofs. The observed fixture outputs
pass; this does not prove every GRF dependency or every possible FP input.

Candidate UP owner-lo load1568 ->SLM store1616 ->sync.bar1840 has no preceding
FP arithmetic. All1765 downstream register-erased FP arithmetic signatures
retain order/types/regions/operand positions/immediates/accumulator names.
This is corroboration, not SSA or whole-UP byte identity. UP static exp17 ->16
and reduce adds1; static sites are not executed SFU counts or removable time.
All six images use GRF128/SIMD32/eu_thread_count8 and the same correctly-rounded
divide/sqrt option. UP SLM12288, metadata barrier_count1, actual2 sync.bar.
Missing SLM/barrier/spill/scratch fields remain UNKNOWN. Raw CR0 mask0x400004C0:
old/control reduce280, candidate reduce304, UP328; no mode interpretation.

Actual source consultations: claude-opus-5-5/68.062s and
claude-fable-5-1/86.535s, both exit0/CPU_BUILD_GO. Actual compiled consultations:
claude-opus-5-5/50.087s and claude-fable-5-1/68.108s, both exit0/GPU_GO after root
read. Their exact text, actual model/session metadata and root corrections are
retained. Actual post-run consultations were claude-opus-5-5/52.551s and
claude-fable-5-1/48.835s, both exit0. Both retain the private proof and recommend
no dedicated main build or serving comparison; root agrees. No private repeat,
cache campaign or promotion is selected. A future otherwise-required build
would still need its own source/compiled/serving checks, not free approval.

Root rejects a cold-first/artifact explanation for91.264us: its cause remains
unproved. No calibrated serving noise band or rigorous clean-wall upper bound
is inferred from historical instrumented family medians. Opus's config-only
depth3 rerun is not selected: S21 already rejected it. The cited64 output tokens
/24 windows aggregate TWO streams and cannot establish a per-stream3-token
window ceiling. The next selected task is dense non-vocabulary MMVQ attribution
from existing small census/source records; no new GPU experiment is queued.

VERDICT -> PASS_PRIVATE_STRICT_FIXTURES_ONLY; cleanup/identity/card1-health
closed; actual postreviews/root disposition closed. No promotion, generic
all-input proof or serving forecast. The narrow raw -0 coverage limit and single-process timing limits remain explicit.

[Curated S116 evidence](../strata/flash-next/lowlevel_probe/hc_lowrank_producer_s116/README.md)
contains full private source/build recipes, receipts, bounded binding/FP
excerpts, runner/analyzer, native results and closure. Exact/derived copies
have hashes and recipes. Binary/full-ISA/27 MB anchor payloads remain external
with existing identities; old S115 evidence is linked instead of duplicated.
