# Native layer3 QSA target observer proposal

CONFIG -> Source-only audit of the consumed source37 and fresh source39 SDKs.
The proposed observer is default OFF and supplies native comparison targets for
the independently owned QSA V4 experiment. Captures are never math inputs.

COMMAND -> Read only tracked/consumed source, headers and build receipt metadata;
compare complete source bytes and the delimited QSA call branch. No model arrays,
weights, compilation, Docker, GPU, backend patch or running engine were touched.

RESULT -> Five QSA kernel sources and seven relevant interface/state headers are
byte-identical across source37 and source39. The 16,368-byte QSA branch in
`Verifier::record_window` is byte-identical, SHA
`0fe3640ce2df55d0ae052b03036a001ebeec55f8dd1ce64bb5f807b83de93269`.
Whole `verify.cpp` differs: source39 adds batch-direct observer graph selection,
span naming and graph retirement. `generate.cpp` also differs. Exact source
paths/hashes and receipt identities are in the accompanying source audit JSON.

VERDICT -> External capture points can expose the real projections, fused
norm/RoPE outputs, public indexer/KV state and attention/gate outputs. They do
not expose internal norm arguments, rsqrt values, trigonometric registers,
attention scores or softmax registers. No source37 runtime proof transfers to
source39 or to a new observer build. The unexecuted V4 helper cannot establish
the cause of the prefix4 head discrepancy from its source alone.

## Proposed scope and startup admission

Use a new bounded startup-only flag and configuration, with enablement only on
the exact value `1`. OFF must create no object, allocation, copy, wait, output or
extra capture branch. ON requires an exclusive absolute output directory,
current plan/source/model binding digest and a bounded request roster. Do not
reuse frozen P30/NUM10 fields or reinterpret their scope.

The first proposed case is fresh serial batch0 prefix4, IDs
[248045, 8678, 198, 15666], with the original admitted maxctx2048, prefill64,
spec2/short-read64, native HC/QSA/indexer/RoPE, FP16 KV and a single generated
token. Prove the actual fresh/reset/full-read/no-reuse request and actual
accepted row counts. Admission derives the prompt windows from source and
accepted IDs: prompt p0/T2, prompt p2/T1, target p3/T1. Alternative reread,
prefill, split-group, batch-slot, KV rotation/INT8/Q4/hybrid, RoPE table/scaling,
control-vector and eager routes are outside this first case. EAGER presence,
including zero, must be rejected. Do not manufacture a T2 window if the real
caller takes another route.

One-card owns layer3 in stage [0,48). A 32/16 pair owns layer3 only in stage0
[0,32). Stage1 [32,48) has an explicit zero data quota, no observer allocation,
and a source-bound stage/request/route acknowledgement. Its zero quota is not
a missing frame waiver. Both stage inventories, final handoff/head ownership,
all48 plus full-head numerical coverage and normal teardown remain mandatory. A different
layer split requires a new actual-owner admission.

## Actual call ordering and capture points

The following sequence is the identical source37/39 QSA branch. Capture copies
must use its actual `tb`, `te`, `n`, `qi`, `st`, pointers and caller queue `cs`.
Do not infer a slot owner, physical page or scratch row from model geometry.

| Point | Native source operation | Proposed output target |
| --- | --- | --- |
| HC input/read | `gr_read_group(0, pending, ...)`; native HC submits each row | Actual HC residual operand and mixed F32 row before QSA or FFN scratch reuse |
| Input packet | `if (!q8_attn) native_quantize_q8_1(xm,xq_,N,n,cs)` | Complete native mixed Q8_1 packet, including D/S and codes |
| Raw indexer key | BF16 single or multi GEMV into `idx_raw_L_ + qi*MT*ID` | Actual projected indexer key before native append |
| K/V projection | `native_mmvq(wk,...)`, then `native_mmvq(wv,...)` | Full K and V F32 projection outputs before normalization/storage |
| K normalization/RoPE | `norm_rope` fused native call when supported | Full post-fused K output; no pre-RoPE norm or rsqrt claim |
| KV append | FP16 `kv_append_step` for each actual row | Real page table and bounded native K/V pool snapshots |
| Indexer append | `native_qsa_indexer_append_steps` or row calls | Real tail/dead/active pooled rows/block position before and after the window append |
| Q/gate projection | `native_mmvq(wq,...)` into `qfull_` | Full [query,gate] F32 rows before query normalization |
| Q normalization/RoPE | Fused call with input stride 2*HD=512 | Full `qcur_` after the actual fused call |
| Indexer query projection | BF16 single/multi GEMV into `qidx_` | Actual [4,128] projected query before its in-place norm/RoPE |
| Indexer query norm/RoPE | `norm_rope(qidx_,...)` | Actual post-fused [4,128] query |
| Selection | `qsa_block_scores`, `qsa_block_topk` | Actual step and active selected IDs; independent own IDs remain math inputs in V4 |
| Residency | `qsa_kv_resolve` | Actual resolved page-table mapping and native pool source bounds |
| Attention | `qsa_decode_attn_batch` on the real pools | Full resident attention F32 output before gating |
| Gate | `native_qsa_gate_apply` | Full gated F32 output before Q8_1 quantization |
| Output packet/projection | Quantize gated output, then `native_mmvq(wo,...)` | Complete gated Q8_1 packet and [2560] output-projection F32 row before HC write |

Also copy the actual four gamma vectors as model output targets, and record
weight/native-type descriptors, epsilon, resolved RopeScaling, native switches,
`qb`, `fuse_nr`, `dec_batch`, group count, KV mode and source stage bounds.
Gamma capture is not permission to feed captured weights into the original
reference. The source upload390 roster alone is not new QSA projection proof.

For native HC, `gr_read_group` returns false deliberately, so this profile
quantizes `mixed_` into `xq_`. Within a group, packet row zero corresponds to
`tb`; its source offset is `(t-tb)*(N/32)*36`, not `t*(N/32)*36`. F32 projections
use the actual row offsets in their T-sized scratch buffers. `xq_` is reused for
the larger gated packet later; both packets must be copied at their own points.

In the qb path K rows use positions `pos_k`, Q uses `pos_`, and indexer queries
use `pos_i`. In the row path K/indexer calls use the first appropriate entries
of `pos_ + t*NH`. Record the actual position values and provenance; do not
replace them with guessed equal-text positions. These actual paths are equal
for the admitted text profile only after the recorded values prove it.

## State, residency and quota discipline

Actual `QsaState` carries `n_pages`, `n_slots`, `max_cells`, `kv_mode`,
`kv_elastic`, pool pointers, page table and indexer row capacity. `page_table`
maps logical to physical pages; plain FP16 geometry does not itself prove that
logical page0 is physical page0. Capture the real mapping and either the bounded
physical pool span or a separately identified pure copy/gather of its mapped
cells. No clamp, fabricated identity mapping or zero replacement is allowed.
Selected IDs and pool contents are targets only, never V4 helper operands.

The simplest first observer may require actual resident kv_mode0 with a proved
identity table and sufficient slot capacity. Publish its first physical page
and page_table[0], and qualify a logical cell view only if the actual mapping is
zero. If the real matched profile is streamed, do not silently reinterpret that
view: use a new bounded physical-pool copy or validated copy/gather proposal.
Bounds must come from actual allocations and capacities, with checked size
arithmetic. The maxctx2048 upper bound is finite; it is not measured VRAM usage.

Indexer state observed after append is forward state at that point. It must not
be called accepted-row-committed state before the real commit occurs. Record
actual accepted counts and the next window's before state. Copy only initialized
active pooled rows, or prove the real session-zero initialization before calling
padding zero. At four cells the active final pool includes the completed row
and spare dead row; root's independent own state has shape [2,128].

Per-row projections/outputs have exact typed extents: mixed2560, K/V512,
Q/gate12288, Q6144, indexer key128, indexer query512, attention/gated6144 and
output projection2560 F32 values. Mixed/gated packets are 2880/6912 bytes per
row. State and gamma fields are separate window/model quotas. Each owner has
three window frames totaling four rows, with complete fields for every active
row and exactly one state snapshot per declared point. A nonowner has exactly
three explicit zero-quota window acknowledgements. Publish all checked byte
quotas before allocations; log actual allocated/freed observer bytes separately
from source39 logical cache accounting. Do not claim physical device peaks from
those counters.

## Graph, synchronization and ownership requirements

Allocate stable per-stage capture storage before recording any graph. Enablement
and pointer membership are process constants. Do not bake a host per-request
eligibility condition into graph recording and later expect another request to
gain missing copy nodes. Use dedicated observer graphs, or fresh enabled
ordinary graphs with explicitly registered copy nodes and strict lifetime
retirement; the implementation choice must be reviewed before patching.

Capture each transient value immediately after its producer, using queued
device-to-device copies on the same caller queue and before buffer reuse.
Do not call standalone normalization, another GEMV, exp or trigonometric
functions to invent a missing internal witness. Copying bytes adds observation
work and dependencies; only matched ON/OFF full-vector controls can prove its
numerical neutrality. Avoid a queue wait at every hook.

Register complete row slices while recording each actual T1/T2 and AR/doorbell
variant. Reject duplicate, omitted, wrong-stage or overlapping fields. Write a
fresh request/window/stage nonce into device control storage outside recording
before each replay, and copy/stamp it after all observer writes. After the
actual model window returns, wait for the owning queue/event, verify nonce and
complete roster, and then dump via exclusive nonsymlink files. No stale warm
frame, earlier request or graph-initialization frame may satisfy the target.

Keep graph references, source/position buffers and capture allocations live
through completion. Destroy every graph containing capture pointers before
freeing them. Both stage queues, actual stdout EOF/readers and owned containers
must retire before leases or post-health/source scans proceed. Cross-stage dump
coordination must not treat stage0 completion as the stage1 head terminal.

## Required qualification before any fidelity conclusion

Prepare a new source39-compatible default-OFF patch and fresh tracked SDK build,
source/header/ABI/ELF/upload receipts, then genuine one-card and paired baseline
admission. The identical kernel bytes above justify a deliberate port design,
not reuse of a source37 binary or source37 runtime verdict.

Matched OFF and ON use the exact same IDs, full argv/env/math recipe, model and
tokenizer/source identities, topology, cache/reset counters, context and expert
residency. Prove first-head and all48 full raw numerical equivalence for all
four positions, original requests/outputs and normal teardown. Include observer
OFF resource/marker absence, actual ON positive ownership/quotas and negatives
for stale nonce/request, source/model mismatch, wrong row/stage, missing field,
size overflow, invalid mapping, incomplete publication and forced cleanup.

Retain exact per-card and compiled P2P0 health, raw kernel journal checks,
lease/command/inspection/EOF receipts, current source/library snapshots and new
ordered publisher-four/page brackets. A read-only reader must recollect all
raw fields and original metadata, repeat complete current joins after comparison
and keep partial or failed collectors false. Only then compare native targets
to fresh independently owned V4 outputs. Preserve every difference and every
prior reference; assign no tolerance, root cause, speed gain or stability claim.

Remaining internal unknowns are the fused RMS sum/argument/rsqrt, analytic
pow/sin/cos register values, MMVQ/BF16 GEMV accumulation ordering, per-lane
attention dot partials, score/max/exp/softmax registers and contraction/rounding
decisions. A future in-kernel observer would require another default-OFF source
change and independent ON/OFF requalification. Public function outputs and
standalone shadows must not be relabeled as those internal native values.
