# Next decode candidate: paired routed GU tile, source only

CONFIG -> Rank ONE conditional candidate behind S116 HC: a SYCL LN8 gate/up
producer tile for32 matched FF indices, reusing the same activation operands
for both independently ordered dot chains and publishing their32-value Q8
SwiGLU block locally. Scope initially GU17/IQ2_XXS, N2560/FF640, local256/SG32,
single entry at a time, current source-default S100 modeOFF. No live patch,
build, GPU, profiler, raw trace parse or new model consultation occurred.

COMMAND -> Read current `iq_kernels.dp.cpp` and verifier callsites plus the
existing small S95 canonical operation CSV. Exact source hashes/line snippets
and selected original CSV rows are alongside this note. Source names and
whole-collection service evidence do not prove per-weight/runtime dispatch.

RESULT -> Current routed GU produces gate and up in different work-groups:
`native_gu_kernel`2045..2082 maps row<640 to gate, row>=640 to up; LN8 gives
32 rows/WG and40 work-groups along the row axis. `row_dot_lanes`810..819 uses
the same entry activation separately for each row. In `vec_dot_iq2_xxs_q8_1`
144..174 each weight part loads8 activation int32 words plus its half scale;
the two projections use different weights but the same activation coordinates.
The weights, grid/sign decode and DP4A work cannot be shared between them.

`native_expert_grouped`5353..5365 owns gate/up/h/hq scratch planes; after GU,
the current SYCL path ALREADY fuses SwiGLU+Q8 (`swiglu_q8_1_entries_kernel`
2560..2594, launched5488..5498), then down reads hq only. Proposed tile:
256 threads still form32 independent LN8 row teams; teamr computes gate[r]
and up[r] with two accumulators, same k order, one activation operand load.
At entry completion team leaders place the32 FP32 gate/up pairs in bounded
SLM, all threads barrier, one SG32 maps lane=FFindex and invokes the existing
SwiGLU expression and unchanged `q8_1_store`. Twenty such tiles cover FF640;
the following existing down launch/queue edge stays. No global counter,
root-wide barrier, extra queue or routing/EP change is needed by this design.

Why this is distinct: S100 reuses one weight decode across multiple routed
entries; this candidate reuses one entry's activation across TWO different
weight rows and removes their producer/consumer global materialization. No
IQ2 entry batching, Q6 decode/layout, SG/GRF/MORE knob, grouped GEMM, sampler or
submission-graph change. A HIP-only `native_gu_fused_kernel`5246 already exists
for TG18/21/22/23, but uses a different LN32/LDS/multi-entry scheme and is not
a validated SYCL implementation or arithmetic template for this LN8 candidate.

The S95 primary-device collection contains GU17/LN8 `{40;60}` local256
240 events, median98.6975us; its already-fused SwiGLU/Q8 `{150}` local256
1152 events, median3.333us. These are whole-collection name/grid rows, not
clean per-layer or steady-cycle matched measurements. Routed GU/down family
reviewed-cycle service7.504495ms is DISJOINT from broad MMVQ12.680488ms and
is not an attainable saving. Removing the3us consumer alone is a weak lead.
The potential larger mechanism is activation-send/instruction reuse inside
the long GU producer and earlier per-tile hq readiness, with no weight-byte
reduction. Actual cached traffic, issue limits, occupancy and critical-path
overlap are UNKNOWN; the tile can easily lose from half as many WGs, doubled
live accumulators and a barrier per entry. No forecast follows.

Arithmetic/lifetime gates -> Keep each IQ2 integer dot sequence, signed
grid/sign handling, `sumi*ls/8` truncation, `(half_weight_d*half_x_d)*sumi`,
per-lane k-stride8 and XOR4/2/1 sum tree exactly. Interleave gate/up chains
without reassociation or contraction. The current rounded FP32 gate/up stores
are a materialization boundary: use explicit FP32 SLM stores/loads before
SiLU, not a claim that arbitrary forwarding is bit-equivalent. Preserve
contract(off), native exp, division/multiply order, signed zero/subnormal/
nonfinite handling, and the Q8 SG32 XOR16/8/4/2/1 max/sum tree, quant rounding
and half ds conversion. Changed FP forms/control mode are HOLD, no tolerance.

Every WG must finish each group's same entry loop uniformly before barrier;
empty groups exit uniformly first, tails cannot skip a barrier. Index hq by
original entry e, not ent_tok[e] or ent_dst[e]. Runtime resident planning
creates groups/entry order; do not assume expert counts from launch capacity.
Preserve cap_entries scratch layout/API and phase1/phase2 semantics. Gate/up
are private function scratch, but future caller/diagnostic readers must be
audited before omitting stores in production. Existing local `hit_scratch_`
and peer `ep_scratch1_` are passed on distinct queues; keep their existing
ownership, scratch capacity and in-order producer/down lifetime, not a new
generic no-alias claim. Verifier1470..1484 supplies the local plan/xq/scratch;
1435..1448 supplies peer plan/inbox/scratch. Do not change EP waits/copies or
claim cross-card clock/physical idle evidence.

VERDICT -> Queue source/compiled feasibility only; S116 remains higher
priority. Cheapest first discriminator: ONE selected existing main GU17/LN8
host/image binding and offline ISA read, with retained hq consumer source,
to establish actual activation sends/extracts and resources. No all-image
walk or new trace. If the likely saving is only the existing3us consumer or
register/barrier costs dominate, close this lever without a private build.
If worthwhile, ONE private strict fixture should compare the original
GU+SwiGLU/Q8 versus this tile: diagnostic gate/up bits, hq bytes, unchanged
down outputs, read-only inputs/guards/unused entries, zero/one/multiple groups,
eager and A/B/A captured fresh inputs. First synthetic IQ2 extreme/sign/scale
cases plus exact retained real-format blobs if already available; no model
read or broad sweep. Actual selected control/candidate FP/operand/resource
audit precedes GPU. Only a later matched WHOLE grouped-stage time can decide
net readiness benefit; isolated byte counts or service medians are insufficient.
