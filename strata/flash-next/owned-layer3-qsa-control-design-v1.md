CONFIG -> NEW bounded original layer3 QSA control proposal, separate from V3.
Source/metadata localization receipt9b018661 binds the preserved FAILED V2 run.
No normalized/native/own matrix payload was read for localization. No material
threshold was fitted. Actual V3 snapshot correction is unchanged and independent.

COMMAND -> Source/CPU only now:
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest \
 test_owned_layer3_qsa_contract_cpu_v1 -q
Root later produces original own projections and compiles/runs a new bounded
primitive helper under the established lease/image/source/health/EOF/full4
guards. This contract alone does not authorize a runtime or transfer proof.

RESULT -> Earliest last-token phase amplification is layer3 (first QSA layer):
token3 input NMSE2.84296e-14 -> attention8.64466e-10 -> FFN3.38796e-7.
The token0 layer3 attention remains1.43375e-14; its FFN rises6.77436e-13 and
layer4 FFN3.07839e-10. Token1/2 inputs stay about4e-14 through layer4, then
layer5 GDN attention reaches3.92759e-9/2.18982e-8, consistent with a possible
earlier temporal-state contribution, not proof of a local layer5 cause.
Head NMSE.007888 remains exploratory; original parent failed its full ledger.

Actual verifier source dispatches fused native RMS+RoPE when enabled/usable,
FP16 resident K/V, qsa_decode_attn_batch, native F32 gate and Q81 out projection.
The --native preset enables native QSA/indexer/RoPE/router. Exact original args,
environment and compiled source must prove that dispatch in the future case;
unsupported KV formats, RoPE scaling/table/position profiles and alternate
attention lane-cell flags must be rejected or explicitly separately declared.

QSA128/256-column norm uses256 threads: per-thread square partial,32-lane XOR,
eight warp sums and another32-lane XOR, F32 mean/epsilon, sycl::rsqrt and
(scale*input)*gamma. This is a different reduction/multiply order from HC.
Fused norm+RoPE preserves that order before float rotary operations. The current
own QSA estimate uses FP64 mean/sqrt/NumPy trig and does not emulate those
device operations. Device rsqrt/pow/trig stay unqualified until measured on
the exact OWN arguments with unchanged flags, without ULP/value lookup.

Native decode attention has24 query heads,2 KV heads,256 dimensions,12 queries
per KV and64-cell chunks. The ordinary score path sums eight products perlane
then XOR32; precise compilation may contract its unannotated multiply-adds.
That lowering is not inferred. Softmax uses native exp, FP32 subgroup reduction,
and values accumulate with explicit sycl::fma; chunk reduction also uses FMA.
Native QSA gate uses FP32 reciprocal/native exp. Current own NumPy attention
and gate are broader-precision estimates. No scratch qsa_attend implementation
is mistaken for the actual resident qsa_decode_attn_batch path.

Proposed independent replay: all four ORIGINAL inputs pass owned HC/GDN/PLE
upstream, and layer3 projects own Q-full/K/V/indexer from original weights.
Begin with own zero KV/tail/dead/pool state; retain all four own cells in causal
order. Geometry topk2048+3 exceeds this bound, so IDs0..committed-1 are required;
native selected IDs are never inputs and no routing policy changes are needed.
Record own projection, norm arguments, normalized/rotary values, actual FP16
pool bytes, indexer state, scores/max/exp/sum, FMA attention, gate and complete
Q81 output packet. Existing native values are OUTPUT TARGETS ONLY. Additional
native layer3 observation fields require a new default-off source-bound observer
and fresh qualification; existing residual-only P30 does not expose them.

The7 CPU controls establish bounded source geometry, known-answer two-stage
reduction, multiplication order, FP16 storage, all-cell causal coverage and
fail-closed own-zero-state/no captured-operand/ID/no tolerance provenance.
They confer no intrinsic, norm/RoPE/decode, selection or wholemodel authority.

Token0 layer3/layer4 FFN is a separate later seam: preserve own HC mixed input,
original BF16 router/shared-gate storage, own logits/IDs/weights and expert
outputs. Native router uses native exp and F32 XOR/top10 normalization; current
own FFN uses NumPy exp/reductions. Compare own IDs as outputs, never substitute
native IDs, logits, hidden values or combine results into fullmodel math.
Explicit native combine/hidden contraction and exact dispatched shapes must
be inspected before that separate control, not selected from error metrics.

VERDICT -> Concrete source contract/provenance proposal only. No model reads,
GPU, compiler, intrinsic measurement, fitted tolerance or causal primitive PASS.
V3 own-HC experiment and all earlier sources/evidence remain unchanged.
